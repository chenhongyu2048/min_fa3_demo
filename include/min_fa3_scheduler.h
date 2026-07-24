// Copied and trimmed from Hopper forward sources:
// - hopper/tile_scheduler.hpp
// Trimmed down to the scheduler variants used by the minimal SM90 forward demo:
// causal uses DynamicPersistentTileScheduler, non-causal uses StaticPersistentTileScheduler.

#pragma once

#include <cassert>

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

#include "min_fa3_named_barrier.h"
#include "utils.h"

namespace flash {

struct TileSchedulerArguments {
    int const num_blocks, num_head, num_batch, num_splits;
    int const qhead_per_khead;
    int const seqlen;
    int const seqlen_k, headdim, headdim_v, element_size;
    int* const tile_count_semaphore = nullptr;
    int const* const cu_seqlens = nullptr;
    int const* const seqused = nullptr;
    int const* const num_splits_dynamic_ptr = nullptr;
    int const* const num_m_blocks_ptr = nullptr;
    int const* const varlen_batch_idx_ptr = nullptr;
    int const* const num_nheads_in_l2_ptr = nullptr;
};

template<bool Split=false>
class StaticPersistentTileScheduler {
public:
    using SharedStorage = int;
    // MEGA_RING: default fixed-layout scheduler does not expose ring-step metadata.
    static constexpr bool EnableMegaRing = false;
    static constexpr bool EnableChunkedSegments = false;
    static constexpr bool CollectMegaRingStats = false;
    static constexpr bool EnableQueuedInitialWork = false;

    struct Params {
        int total_blocks;
        cutlass::FastDivmod m_block_divmod, head_divmod;
        cutlass::FastDivmod nsplits_divmod;
    };

    static Params to_underlying_arguments(TileSchedulerArguments const& args) {
        return {args.num_blocks * args.num_head * args.num_batch * (!Split ? 1 : args.num_splits),
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head * (!Split ? 1 : args.num_splits)),
                cutlass::FastDivmod(!Split ? 1 : args.num_splits)};
    }

    static dim3 get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE bool is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int block, bidh, bidb;
            bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(block, tile_idx));
            int split_idx = 0;
            if constexpr (Split) {
                bidh = params.nsplits_divmod.divmod(split_idx, bidh);
            }
            return {block, bidh, bidb, split_idx};
        }
    };

    CUTLASS_DEVICE StaticPersistentTileScheduler(SharedStorage* const smem_scheduler) {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE void init_consumer() const {}
    CUTLASS_DEVICE void prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {current_work.tile_idx + int(gridDim.x)};
    }
};

template<int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp,
         bool Split=false, bool PackGQA=false, bool WarpSpecialized=true>
class DynamicPersistentTileScheduler {
    static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
    static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;

public:
    using SharedStorage = int;
    // MEGA_RING: default fixed-layout scheduler does not expose ring-step metadata.
    static constexpr bool EnableMegaRing = false;
    static constexpr bool EnableChunkedSegments = false;
    static constexpr bool CollectMegaRingStats = false;
    static constexpr bool EnableQueuedInitialWork = false;

protected:
    SharedStorage* const tile_count_smem;

public:
    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const m_block_divmod, head_divmod;
        cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
        cutlass::FastDivmod const l2_minor_residual_divmod;
        int const num_hb_quotient;
        int* const tile_count_semaphore;
    };

    static Params to_underlying_arguments(TileSchedulerArguments const& args) {
        long long const size_one_kv_head = long(args.seqlen_k) * long(args.headdim + args.headdim_v) * long(args.element_size);
        int const size_l2 = 32 * 1024 * 1024;
        auto find_log2_floor = [&](int n) { return 31 - cutlass::clz(n); };
        int const swizzle = (size_l2 < size_one_kv_head ? 1 : (1 << find_log2_floor(size_l2 / size_one_kv_head)))
            * (PackGQA ? 1 : args.qhead_per_khead);
        int const num_hb_remainder = (args.num_head * args.num_batch) % swizzle;
        int const num_split_blocks = args.num_blocks * (!Split ? 1 : args.num_splits);
        assert(args.tile_count_semaphore != nullptr);
        return {num_split_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(swizzle), cutlass::FastDivmod(swizzle * num_split_blocks),
                cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
                (args.num_head * args.num_batch) / swizzle,
                args.tile_count_semaphore};
    }

    static dim3 get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE bool is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int block, bidh, bidb;
            int l2_mod, bidhb, bidhb_residual;
            bidhb = params.l2_major_divmod.divmod(l2_mod, tile_idx);
            if (bidhb < params.num_hb_quotient) {
                block = params.l2_minor_divmod.divmod(bidhb_residual, l2_mod);
            } else {
                block = params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
            }
            bidb = params.head_divmod.divmod(bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);
            int split_idx = 0;
            if constexpr (Split) {
                split_idx = params.m_block_divmod.divmod(block, block);
            }
            block = params.m_block_divmod.divisor - 1 - block;
            return {block, bidh, bidb, split_idx};
        }
    };

    CUTLASS_DEVICE DynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : tile_count_smem(smem_scheduler) {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE void init_consumer() const {
        if (WarpSpecialized || cutlass::canonical_warp_idx_sync() > 0) {
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
        }
    }

    CUTLASS_DEVICE void prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0);
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            if (threadIdx.x % NumProducerThreads == 0) {
                *tile_count_smem = current_work.tile_idx;
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            return {new_tile_idx};
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            int tile_idx = *tile_count_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            return {tile_idx};
        }
    }
};

}  // namespace flash
