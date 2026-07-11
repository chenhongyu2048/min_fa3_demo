// Copied and trimmed from Hopper scheduler source:
// - hopper/tile_scheduler.hpp

#pragma once

#include <cassert>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/fast_math.h"

#include "hopper_compat/utils.h"
#include "min_fa3_named_barrier.h"

namespace flash {

struct TileSchedulerArguments {
    int const num_blocks, num_head, num_batch;
    int const qhead_per_khead;
    int const seqlen;
    int const seqlen_k, headdim, headdim_v, element_size;
    int const* const cu_seqlens = nullptr;
    int const* const seqused = nullptr;
    int* const tile_count_semaphore = nullptr;
    // MEGA_RING_BWD: optional multi-step causal-zigzag scheduling metadata.
    int const ring_world_size = 1;
    int const ring_rank = 0;
    int const num_comp_sm = 0;
    int const* const half_cu_seqlens = nullptr;
};

template <bool Varlen, int kBlock = 128, bool Persistent = false>
class SingleTileBwdScheduler {
public:
    static constexpr bool IsPersistent = Persistent;
    using SharedStorage = int;

    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const block_divmod, head_divmod;
        int const num_blocks, num_head, num_batch;
        int const seqlen;
        int const* const cu_seqlens;
        int const* const seqused;
    };

    static Params to_underlying_arguments(TileSchedulerArguments const& args) {
        return {args.num_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                args.num_blocks, args.num_head, args.num_batch, args.seqlen,
                !Varlen ? nullptr : args.cu_seqlens,
                !Varlen ? nullptr : args.seqused};
    }

    static dim3 get_grid_shape(Params const& params, int sm_count) {
        if constexpr (Persistent) {
            return {uint32_t(cutlass::fast_min(params.total_blocks, sm_count))};
        } else {
            return {uint32_t(params.num_blocks), uint32_t(params.num_head), uint32_t(params.num_batch)};
        }
    }

    struct WorkTileInfo {
        int block_idx = 0;
        int bidh = 0;
        int bidb = 0;
        int tile_idx = 0;
        int ring_step = 0;

        CUTLASS_DEVICE bool is_valid(Params const&) const { return bidb >= 0; }

        CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const&) const {
            return {block_idx, bidh, bidb, 0};
        }
    };

    CUTLASS_DEVICE SingleTileBwdScheduler(SharedStorage*) {}
    CUTLASS_DEVICE void set_post_comm(bool) {}

    CUTLASS_DEVICE WorkTileInfo get_work(Params const& params, int tile_idx) const {
        while (tile_idx < params.total_blocks) {
            int bidh, block_idx;
            int bidb = params.head_divmod.divmod(
                bidh, params.block_divmod.divmod(block_idx, tile_idx));
            bool is_valid_tile = true;
            if constexpr (Varlen) {
                int seqlen = params.seqused
                    ? params.seqused[bidb]
                    : (params.cu_seqlens
                        ? params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb]
                        : params.seqlen);
                is_valid_tile = block_idx * kBlock < seqlen;
            }
            if (is_valid_tile) { return {block_idx, bidh, bidb, tile_idx, 0}; }
            tile_idx += int(gridDim.x);
        }
        return {0, 0, -1, tile_idx, 0};
    }

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
        if constexpr (Persistent) {
            return get_work(params, int(blockIdx.x));
        } else {
            WorkTileInfo work_info{int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), 0, 0};
            if constexpr (Varlen) {
                int seqlen = params.seqused
                    ? params.seqused[work_info.bidb]
                    : (params.cu_seqlens
                        ? params.cu_seqlens[work_info.bidb + 1] - params.cu_seqlens[work_info.bidb]
                        : params.seqlen);
                if (work_info.block_idx * kBlock >= seqlen) { work_info.bidb = -1; }
            }
            return work_info;
        }
    }

    CUTLASS_DEVICE void init_consumer() const {}
    CUTLASS_DEVICE void prefetch_next_work(Params const&, WorkTileInfo&) const {}

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (Persistent) {
            return get_work(params, current_work.tile_idx + int(gridDim.x));
        } else {
            return {0, 0, -1, 0, 0};
        }
    }
};

template <bool Varlen, int kBlock, bool SPT = false, bool Persistent = false,
          int NumMmaThreads = 2 * cutlass::NumThreadsPerWarpGroup>
class SingleTileBwdLPTScheduler {
public:
    static constexpr bool IsPersistent = Persistent;
    static constexpr int NumSchedulerThreads =
        NumMmaThreads + 2 * cutlass::NumThreadsPerWarp;
    using SharedStorage = int;
    SharedStorage* const tile_idx_smem;

    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const block_divmod, head_divmod;
        cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
        cutlass::FastDivmod const l2_minor_residual_divmod;
        int const num_hb_quotient;
        int const seqlen;
        int const* const cu_seqlens;
        int const* const seqused;
        int* const tile_count_semaphore;
    };

    static Params to_underlying_arguments(TileSchedulerArguments const& args) {
        long long const size_one_qdo_head =
            long(args.seqlen_k) * long(args.headdim + args.headdim_v) * long(args.element_size);
        long long const size_one_dqaccum_head =
            long(args.seqlen_k) * long(args.headdim) * sizeof(float);
        long long const size_one_head = size_one_qdo_head + size_one_dqaccum_head;
        int const size_l2 = 40 * 1024 * 1024;
        auto find_log2_floor = [&](int n) { return 31 - cutlass::clz(n); };
        int const swizzle = size_l2 < size_one_head
            ? 1
            : (1 << find_log2_floor(size_l2 / size_one_head));
        int const num_hb_remainder = (args.num_head * args.num_batch) % swizzle;
        if constexpr (Persistent) { assert(args.tile_count_semaphore != nullptr); }
        return {args.num_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(swizzle), cutlass::FastDivmod(swizzle * args.num_blocks),
                cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
                (args.num_head * args.num_batch) / swizzle,
                args.seqlen, !Varlen ? nullptr : args.cu_seqlens,
                !Varlen ? nullptr : args.seqused,
                Persistent ? args.tile_count_semaphore : nullptr};
    }

    static dim3 get_grid_shape(Params const& params, int sm_count) {
        if constexpr (Persistent) {
            return {uint32_t(cutlass::fast_min(params.total_blocks, sm_count))};
        } else {
            return {uint32_t(params.total_blocks)};
        }
    }

    struct WorkTileInfo {
        int block;
        int bidh;
        int bidb;
        int tile_idx;
        int ring_step = 0;

        CUTLASS_DEVICE bool is_valid(Params const&) const { return bidb >= 0; }

        CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const&) const {
            return {block, bidh, bidb, 0};
        }
    };

    CUTLASS_DEVICE SingleTileBwdLPTScheduler(SharedStorage* smem_scheduler)
        : tile_idx_smem(smem_scheduler) {}
    CUTLASS_DEVICE void set_post_comm(bool) {}

    CUTLASS_DEVICE WorkTileInfo decode_work(Params const& params, int tile_idx) const {
        if (tile_idx >= params.total_blocks) { return {0, 0, -1, tile_idx, 0}; }
        int l2_mod, bidhb_residual;
        int bidhb = params.l2_major_divmod.divmod(l2_mod, tile_idx);
        int block = bidhb < params.num_hb_quotient
            ? params.l2_minor_divmod.divmod(bidhb_residual, l2_mod)
            : params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
        int bidh;
        int bidb = params.head_divmod.divmod(
            bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);
        bool is_valid_tile = true;
        int num_blocks;
        if constexpr (Varlen) {
            int seqlen = params.seqused
                ? params.seqused[bidb]
                : (params.cu_seqlens
                    ? params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb]
                    : params.seqlen);
            num_blocks = cute::ceil_div(seqlen, cute::Int<kBlock>{});
            is_valid_tile = block < num_blocks;
        } else {
            num_blocks = params.block_divmod.divisor;
        }
        if constexpr (SPT) { block = num_blocks - block - 1; }
        return {block, bidh, is_valid_tile ? bidb : -1, tile_idx, 0};
    }

    CUTLASS_DEVICE WorkTileInfo get_static_work(Params const& params, int tile_idx) const {
        WorkTileInfo work_info = decode_work(params, tile_idx);
        while (tile_idx < params.total_blocks && !work_info.is_valid(params)) {
            tile_idx += int(gridDim.x);
            work_info = decode_work(params, tile_idx);
        }
        return work_info;
    }

    CUTLASS_DEVICE int claim_dynamic_tile(Params const& params, int tile_idx) const {
        WorkTileInfo work_info = decode_work(params, tile_idx);
        while (tile_idx < params.total_blocks && !work_info.is_valid(params)) {
            tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
            work_info = decode_work(params, tile_idx);
        }
        return tile_idx;
    }

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
        if constexpr (!Persistent) {
            return get_static_work(params, int(blockIdx.x));
        } else if constexpr (IsProducerWarp) {
            int tile_idx = int(blockIdx.x);
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                tile_idx = claim_dynamic_tile(params, tile_idx);
            }
            tile_idx = __shfl_sync(0xffffffff, tile_idx, 0);
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *tile_idx_smem = tile_idx;
            }
            flash::named_barrier_arrive(
                NumSchedulerThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            return decode_work(params, tile_idx);
        } else {
            flash::named_barrier_sync(
                NumSchedulerThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            int tile_idx = *tile_idx_smem;
            flash::named_barrier_arrive(
                NumSchedulerThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            return decode_work(params, tile_idx);
        }
    }

    CUTLASS_DEVICE void init_consumer() const {}
    CUTLASS_DEVICE void prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if constexpr (Persistent) {
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                int tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
                current_work.tile_idx = claim_dynamic_tile(params, tile_idx);
            }
        }
    }

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (!Persistent) {
            return {0, 0, -1, 0, 0};
        } else if constexpr (IsProducerWarp) {
            int tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0);
            flash::named_barrier_sync(
                NumSchedulerThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *tile_idx_smem = tile_idx;
            }
            flash::named_barrier_arrive(
                NumSchedulerThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            return decode_work(params, tile_idx);
        } else {
            flash::named_barrier_sync(
                NumSchedulerThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            int tile_idx = *tile_idx_smem;
            flash::named_barrier_arrive(
                NumSchedulerThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            return decode_work(params, tile_idx);
        }
    }
};

// MEGA_RING_BWD: causal zigzag scheduler layered on the copied backward LPT
// scheduler. The base scheduler remains responsible for the original
// n-block/head/batch order; this class only expands it across ring steps and
// filters the front-half KV steps.
template <bool Varlen, int kBlock,
          int NumMmaThreads = 2 * cutlass::NumThreadsPerWarpGroup>
class MegaRingSingleTileBwdLPTScheduler {
    using Base = SingleTileBwdLPTScheduler<Varlen, kBlock, false, true, NumMmaThreads>;

public:
    static constexpr bool IsPersistent = true;
    static constexpr bool EnableMegaRing = true;
    static constexpr int NumSchedulerThreads =
        NumMmaThreads + 2 * cutlass::NumThreadsPerWarp;
    using SharedStorage = int;
    SharedStorage* const tile_idx_smem;
    bool post_comm = false;

    struct Params {
        typename Base::Params base;
        int total_blocks;
        int ring_world_size;
        int ring_rank;
        int num_comp_sm;
        int const* half_cu_seqlens;
    };

    struct WorkTileInfo {
        int block;
        int bidh;
        int bidb;
        int tile_idx;
        int ring_step;

        CUTLASS_DEVICE bool is_valid(Params const&) const { return bidb >= 0; }

        CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const&) const {
            return {block, bidh, bidb, 0};
        }
    };

    static Params to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(args.ring_world_size >= 1);
        assert(args.num_comp_sm >= 1);
        auto base = Base::to_underlying_arguments(args);
        return {base, base.total_blocks * args.ring_world_size,
                args.ring_world_size, args.ring_rank, args.num_comp_sm,
                args.half_cu_seqlens};
    }

    static dim3 get_grid_shape(Params const& params, int) {
        return {uint32_t(cutlass::fast_min(params.total_blocks, params.num_comp_sm))};
    }

    CUTLASS_DEVICE MegaRingSingleTileBwdLPTScheduler(SharedStorage* smem_scheduler)
        : tile_idx_smem(smem_scheduler) {}

    CUTLASS_DEVICE void set_post_comm(bool value) { post_comm = value; }

    CUTLASS_DEVICE WorkTileInfo decode_work(Params const& params, int tile_idx) const {
        if (tile_idx >= params.total_blocks) { return {0, 0, -1, tile_idx, params.ring_world_size}; }
        int const tiles_per_step = params.base.total_blocks;
        int const ring_step = tile_idx / tiles_per_step;
        int const step_tile_idx = tile_idx - ring_step * tiles_per_step;
        Base base_scheduler(nullptr);
        auto base_work = base_scheduler.decode_work(params.base, step_tile_idx);
        if (!base_work.is_valid(params.base)) {
            return {0, 0, -1, tile_idx, ring_step};
        }
        // For steps 1..rank, zigzag consumes only the front half of KV.
        if (ring_step > 0 && ring_step <= params.ring_rank) {
            int half_len = params.half_cu_seqlens
                ? params.half_cu_seqlens[base_work.bidb + 1] - params.half_cu_seqlens[base_work.bidb]
                : 0;
            if (base_work.block * kBlock >= half_len) {
                return {0, 0, -1, tile_idx, ring_step};
            }
        }
        return {base_work.block, base_work.bidh, base_work.bidb, tile_idx, ring_step};
    }

    CUTLASS_DEVICE int claim_dynamic_tile(Params const& params, int tile_idx) const {
        WorkTileInfo work = decode_work(params, tile_idx);
        while (tile_idx < params.total_blocks && !work.is_valid(params)) {
            tile_idx = atomicAdd(params.base.tile_count_semaphore, 1) + params.num_comp_sm;
            work = decode_work(params, tile_idx);
        }
        return tile_idx;
    }

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            int tile_idx = int(blockIdx.x);
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                if (post_comm) {
                    tile_idx = atomicAdd(params.base.tile_count_semaphore, 1) + params.num_comp_sm;
                }
                tile_idx = claim_dynamic_tile(params, tile_idx);
            }
            tile_idx = __shfl_sync(0xffffffff, tile_idx, 0);
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) { *tile_idx_smem = tile_idx; }
            flash::named_barrier_arrive(
                NumSchedulerThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            return decode_work(params, tile_idx);
        } else {
            flash::named_barrier_sync(
                NumSchedulerThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            int const tile_idx = *tile_idx_smem;
            flash::named_barrier_arrive(
                NumSchedulerThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            return decode_work(params, tile_idx);
        }
    }

    CUTLASS_DEVICE void init_consumer() const {}

    CUTLASS_DEVICE void prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
            int tile_idx = atomicAdd(params.base.tile_count_semaphore, 1) + params.num_comp_sm;
            current_work.tile_idx = claim_dynamic_tile(params, tile_idx);
        }
    }

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            int const tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0);
            flash::named_barrier_sync(
                NumSchedulerThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) { *tile_idx_smem = tile_idx; }
            flash::named_barrier_arrive(
                NumSchedulerThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            return decode_work(params, tile_idx);
        } else {
            flash::named_barrier_sync(
                NumSchedulerThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            int const tile_idx = *tile_idx_smem;
            flash::named_barrier_arrive(
                NumSchedulerThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            return decode_work(params, tile_idx);
        }
    }
};

}  // namespace flash
