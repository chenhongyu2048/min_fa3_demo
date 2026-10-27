// Copied and trimmed from include/min_fa3_varlen_scheduler.h.
// DCP_MEGA replaces dynamic coordinate decoding with a host-built attention
// descriptor queue while preserving the producer/consumer hand-off protocol.

#pragma once

#include <cassert>
#include <cstdint>

#include "cutlass/arch/barrier.h"

#include "dcp_mega_min_fa3_varlen_params.h"
#include "mega_ring_semaphore.cuh"
#include "min_fa3_named_barrier.h"
#include "min_fa3_varlen_scheduler.h"

namespace flash {

template <int kBlockM, int kBlockN,
          int NumMmaThreads = 2 * cutlass::NumThreadsPerWarpGroup,
          int NumProducerThreads = cutlass::NumThreadsPerWarp,
          bool Split = false,
          int DCPSize = 1,
          int CommHeads = 4>
class DCPMegaVarlenTileScheduler {
    static_assert(DCPSize == 1 || DCPSize == 2 || DCPSize == 4 || DCPSize == 8);
    static_assert(CommHeads == 4 || CommHeads == 8);
    static constexpr int NumThreads = NumMmaThreads + NumProducerThreads;
    static constexpr cutlass::arch::ReservedNamedBarriers EmptyBarrier
        = cutlass::arch::ReservedNamedBarriers::StreamkBarrier0;
    static constexpr cutlass::arch::ReservedNamedBarriers FullBarrier
        = cutlass::arch::ReservedNamedBarriers::StreamkBarrier1;

public:
    using SharedStorage = int2;
    static constexpr bool EnableMegaRing = false;
    static constexpr bool EnableChunkedSegments = false;
    static constexpr bool CollectMegaRingStats = false;
    static constexpr bool EnableQueuedInitialWork = true;
    struct Params {
        int32_t* attention_dynamic_counter;
        min_fa3_varlen_demo::dcp_mega::AttentionWorkDesc const* descriptors;
        int32_t const* q_dependencies;
        int32_t const* q_ready;
        int32_t* attention_done;
        int32_t const* chunk_sequence_splits;
        int32_t const* history_sequence_splits;
        int attention_count;
        int compute_block_offset;
        int num_compute_ctas;
        int32_t const* dynamic_metadata = nullptr;

        CUTLASS_DEVICE Params resolve() const {
            Params result = *this;
            if (dynamic_metadata != nullptr) {
                using namespace min_fa3_varlen_demo::dcp_mega;
                auto const& header = *reinterpret_cast<MetadataHeader const*>(dynamic_metadata);
                result.descriptors = reinterpret_cast<AttentionWorkDesc const*>(
                    dynamic_metadata + header.attention_offset);
                result.q_dependencies = dynamic_metadata + header.q_dependencies_offset;
                result.chunk_sequence_splits = dynamic_metadata + header.chunk_splits_offset;
                result.history_sequence_splits = dynamic_metadata + header.history_splits_offset;
                result.attention_count = header.attention_count;
            }
            return result;
        }
    };

    static Params to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(args.tile_count_semaphore != nullptr);
        assert(args.mega_ring_ring_sizes != nullptr);
        return {
            args.tile_count_semaphore,
            reinterpret_cast<min_fa3_varlen_demo::dcp_mega::AttentionWorkDesc const*>(
                args.mega_ring_ring_sizes),
            args.num_m_blocks_ptr,
            args.mega_ring_kv_ready_counts,
            args.mega_ring_completed_tiles,
            args.num_splits_dynamic_ptr,
            args.num_nheads_in_l2_ptr,
            args.virtual_grid_blocks,
            args.compute_block_offset,
            args.num_blocks,
        };
    }

    static dim3 get_grid_shape(Params const&, int num_sm) {
        return {static_cast<uint32_t>(num_sm), 1, 1};
    }

    struct WorkTileInfo {
        int descriptor_idx;

        CUTLASS_DEVICE bool is_valid(Params const& params) const {
            return descriptor_idx >= 0 && descriptor_idx < params.attention_count;
        }

        CUTLASS_DEVICE int kind(Params const& params) const {
            return params.descriptors[descriptor_idx].kind;
        }

        CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            auto const& desc = params.descriptors[descriptor_idx];
            int split_idx = desc.split_idx;
            if constexpr (Split) {
                int32_t const* sequence_splits
                    = desc.kind == min_fa3_varlen_demo::dcp_mega::kHistory
                    ? params.history_sequence_splits
                    : params.chunk_sequence_splits;
                int const actual_splits = sequence_splits[desc.batch_idx];
                split_idx |= actual_splits << 16;
            }
            return {desc.m_block, desc.head, desc.batch_idx, split_idx};
        }
    };

private:
    SharedStorage* work_info_smem_;

    CUTLASS_DEVICE void wait_for_q_dependency(
        Params const& params,
        int descriptor_idx) const {
        if (descriptor_idx >= 0 && descriptor_idx < params.attention_count) {
            auto const& desc = params.descriptors[descriptor_idx];
            if (desc.kind == min_fa3_varlen_demo::dcp_mega::kHistory) {
                for (int dep = 0; dep < desc.q_dependency_count; ++dep) {
                    int const task_id = params.q_dependencies[
                        desc.q_dependency_begin + dep];
                    min_fa3_varlen_demo::mega_ring::wait_until_at_least_acquire(
                        params.q_ready + task_id, DCPSize);
                }
            }
        }
    }

    CUTLASS_DEVICE WorkTileInfo claim_initial(Params const& params) const {
        int descriptor_idx = params.attention_count;
        if (threadIdx.x % NumProducerThreads == 0) {
            descriptor_idx = int(blockIdx.x) - params.compute_block_offset;
            wait_for_q_dependency(params, descriptor_idx);
        }
        descriptor_idx = __shfl_sync(0xffffffff, descriptor_idx, 0);
        return {descriptor_idx};
    }

    template <typename OnClaim>
    CUTLASS_DEVICE WorkTileInfo claim_next(
        Params const& params,
        OnClaim&& on_claim) const {
        int descriptor_idx = params.attention_count;
        if (threadIdx.x % NumProducerThreads == 0) {
            descriptor_idx = params.num_compute_ctas
                + atomicAdd(params.attention_dynamic_counter, 1);
            if (descriptor_idx < params.attention_count) {
                on_claim(WorkTileInfo{descriptor_idx});
                wait_for_q_dependency(params, descriptor_idx);
            }
        }
        descriptor_idx = __shfl_sync(0xffffffff, descriptor_idx, 0);
        return {descriptor_idx};
    }

public:
    CUTLASS_DEVICE explicit DCPMegaVarlenTileScheduler(
        SharedStorage* smem_scheduler)
        : work_info_smem_(smem_scheduler) {}

    CUTLASS_DEVICE void init_consumer() const {}

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            WorkTileInfo work = claim_initial(params);
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                work_info_smem_->x = work.descriptor_idx;
            }
            flash::named_barrier_arrive(
                NumThreads,
                FullBarrier);
            if (!work.is_valid(params)) {
                flash::named_barrier_sync(
                    NumThreads,
                    EmptyBarrier);
            }
            return work;
        } else {
            return get_next_work<false>(params, {0});
        }
    }

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_initial_work_from_queue(
        Params const& params) const {
        return get_initial_work<IsProducerWarp>(params);
    }

    template <typename OnClaim>
    CUTLASS_DEVICE void prefetch_next_work(
        Params const& params,
        WorkTileInfo& current_work,
        OnClaim&& on_claim) const {
        current_work = claim_next(params, static_cast<OnClaim&&>(on_claim));
    }

    CUTLASS_DEVICE void publish_completion(
        Params const& params,
        WorkTileInfo const& completed_work) const {
        auto const& desc = params.descriptors[completed_work.descriptor_idx];
        min_fa3_varlen_demo::mega_ring::store_release(
            params.attention_done + desc.completion_id, 1);
    }

    CUTLASS_DEVICE static int initial_descriptor_idx(Params const& params) {
        return int(blockIdx.x) - params.compute_block_offset;
    }

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_next_work(
        Params const& params,
        WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            int descriptor_idx = __shfl_sync(
                0xffffffff, current_work.descriptor_idx, 0);
            flash::named_barrier_sync(
                NumThreads,
                EmptyBarrier);
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                work_info_smem_->x = descriptor_idx;
            }
            flash::named_barrier_arrive(
                NumThreads,
                FullBarrier);
            if (!WorkTileInfo{descriptor_idx}.is_valid(params)) {
                flash::named_barrier_sync(
                    NumThreads,
                    EmptyBarrier);
            }
            return {descriptor_idx};
        } else {
            flash::named_barrier_sync(
                NumThreads,
                FullBarrier);
            int const descriptor_idx = work_info_smem_->x;
            flash::named_barrier_arrive(
                NumThreads,
                EmptyBarrier);
            return {descriptor_idx};
        }
    }
};

}  // namespace flash
