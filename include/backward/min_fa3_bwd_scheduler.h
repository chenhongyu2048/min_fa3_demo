// Copied and trimmed from Hopper scheduler source:
// - hopper/tile_scheduler.hpp

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"

#include "hopper_compat/utils.h"

namespace flash {

struct TileSchedulerArguments {
    int const num_blocks, num_head, num_batch;
    int const qhead_per_khead;
    int const seqlen;
    int const seqlen_k, headdim, headdim_v, element_size;
    int const* const cu_seqlens = nullptr;
    int const* const seqused = nullptr;
};

template <bool Varlen, int kBlock = 128>
class SingleTileBwdScheduler {
public:
    using SharedStorage = int;

    struct Params {
        int const num_blocks, num_head, num_batch;
        int const seqlen;
        int const* const cu_seqlens;
        int const* const seqused;
    };

    static Params to_underlying_arguments(TileSchedulerArguments const& args) {
        return {args.num_blocks, args.num_head, args.num_batch, args.seqlen,
                !Varlen ? nullptr : args.cu_seqlens,
                !Varlen ? nullptr : args.seqused};
    }

    static dim3 get_grid_shape(Params const& params, int) {
        return {uint32_t(params.num_blocks), uint32_t(params.num_head), uint32_t(params.num_batch)};
    }

    struct WorkTileInfo {
        int block_idx = 0;
        int bidh = 0;
        int bidb = 0;

        CUTLASS_DEVICE bool is_valid(Params const&) const { return bidb >= 0; }

        CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const&) const {
            return {block_idx, bidh, bidb, 0};
        }
    };

    CUTLASS_DEVICE SingleTileBwdScheduler(SharedStorage*) {}

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
        WorkTileInfo work_info{int(blockIdx.x), int(blockIdx.y), int(blockIdx.z)};
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

    CUTLASS_DEVICE void init_consumer() const {}
    CUTLASS_DEVICE void prefetch_next_work(Params const&, WorkTileInfo&) const {}

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_next_work(Params const&, WorkTileInfo const&) const {
        return {0, 0, -1};
    }
};

template <bool Varlen, int kBlock, bool SPT = false>
class SingleTileBwdLPTScheduler {
public:
    using SharedStorage = int;

    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const block_divmod, head_divmod;
        cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
        cutlass::FastDivmod const l2_minor_residual_divmod;
        int const num_hb_quotient;
        int const seqlen;
        int const* const cu_seqlens;
        int const* const seqused;
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
        return {args.num_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(swizzle), cutlass::FastDivmod(swizzle * args.num_blocks),
                cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
                (args.num_head * args.num_batch) / swizzle,
                args.seqlen, !Varlen ? nullptr : args.cu_seqlens,
                !Varlen ? nullptr : args.seqused};
    }

    static dim3 get_grid_shape(Params const& params, int) {
        return {uint32_t(params.total_blocks)};
    }

    struct WorkTileInfo {
        int block;
        int bidh;
        int bidb;

        CUTLASS_DEVICE bool is_valid(Params const&) const { return bidb >= 0; }

        CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const&) const {
            return {block, bidh, bidb, 0};
        }
    };

    CUTLASS_DEVICE SingleTileBwdLPTScheduler(SharedStorage*) {}

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
        int l2_mod, bidhb_residual;
        int bidhb = params.l2_major_divmod.divmod(l2_mod, int(blockIdx.x));
        int block = bidhb < params.num_hb_quotient
            ? params.l2_minor_divmod.divmod(bidhb_residual, l2_mod)
            : params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
        int bidh, bidb;
        bidb = params.head_divmod.divmod(
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
        return {block, bidh, is_valid_tile ? bidb : -1};
    }

    CUTLASS_DEVICE void init_consumer() const {}
    CUTLASS_DEVICE void prefetch_next_work(Params const&, WorkTileInfo&) const {}

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_next_work(Params const&, WorkTileInfo const&) const {
        return {0, 0, -1};
    }
};

}  // namespace flash
