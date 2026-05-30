// Copied and trimmed from Hopper forward sources:
// - hopper/flash_prepare_scheduler.cu
// This file keeps the original Hopper varlen scheduler metadata preparation path,
// trimmed to the forward-only SM90 bf16 head_dim=128 demo.

#include <cub/cub.cuh>

#include "cutlass/arch/barrier.h"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/barrier.h"
#include "cutlass/fast_math.h"

#include "min_fa3_varlen_params.h"
#include "static_switch.h"

namespace min_fa3_varlen_detail {

using min_fa3_varlen_demo::Flash_fwd_params;

// Sort in descending order.
template <typename T>
struct PrepareSortOp {
    __device__ __forceinline__ bool operator()(T const& lhs, T const& rhs) const {
        return lhs > rhs;
    }
};

template <>
struct PrepareSortOp<int2> {
    __device__ __forceinline__ bool operator()(int2 const& lhs, int2 const& rhs) const {
        return lhs.x > rhs.x;
    }
};

template <>
struct PrepareSortOp<int4> {
    __device__ __forceinline__ bool operator()(int4 const& lhs, int4 const& rhs) const {
        return lhs.x > rhs.x;
    }
};

template <int NumWarps, bool Sort>
__global__ void prepare_varlen_num_blocks_kernel(
        int seqlen_q_static,
        int seqlen_k_static,
        int seqlen_k_new_static,
        int const* const cu_seqlens_q,
        int const* const cu_seqlens_k,
        int const* const cu_seqlens_k_new,
        int const* const seqused_q,
        int const* const seqused_k,
        int const* const leftpad_k_ptr,
        int num_batch,
        int num_head,
        int qhead_per_khead,
        int num_sm,
        int num_splits_static,
        cutlass::FastDivmod blockm_divmod,
        cutlass::FastDivmod blockn_divmod,
        int* const tile_count_semaphore,
        int* const num_m_blocks_ptr,
        int* const num_splits_dynamic_ptr,
        int* const varlen_batch_idx_ptr,
        int* const num_nheads_in_l2_ptr,
        bool enable_pdl,
        bool is_causal,
        bool packgqa,
        int max_kvblocks_in_l2) {

    static constexpr int kNumBatchPerWarp = cutlass::NumThreadsPerWarp - 1;
    static constexpr int kSmemSize = 1;
    static constexpr int BLOCK_DIM_X = NumWarps * 32;
    static constexpr int ITEMS_PER_THREAD = 1;
    static_assert(BLOCK_DIM_X * ITEMS_PER_THREAD == NumWarps * 32);
    using BlockMergeSort = cub::BlockMergeSort<int4, BLOCK_DIM_X, ITEMS_PER_THREAD>;

    __shared__ int total_blocks_smem[kSmemSize];
    __shared__ typename BlockMergeSort::TempStorage temp_storage;

    if (enable_pdl) {
        cutlass::arch::launch_dependent_grids();
    }

    if (threadIdx.x < kSmemSize) {
        total_blocks_smem[threadIdx.x] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0 && tile_count_semaphore) {
        *tile_count_semaphore = 0;
    }

    int lane = threadIdx.x % cutlass::NumThreadsPerWarp;

    auto get_num_m_blocks = [&](int batch_idx) {
        int seqlen;
        if (seqused_q) {
            seqlen = batch_idx < num_batch ? seqused_q[batch_idx] : 0;
        } else if (cu_seqlens_q) {
            int cur_cu_seqlen = batch_idx <= num_batch ? cu_seqlens_q[batch_idx] : 0;
            int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
            seqlen = next_cu_seqlen - cur_cu_seqlen;
        } else {
            seqlen = seqlen_q_static;
        }
        if (packgqa) {
            seqlen *= qhead_per_khead;
        }
        return batch_idx < num_batch && lane < kNumBatchPerWarp
            ? blockm_divmod.div(seqlen + blockm_divmod.divisor - 1) : 0;
    };

    auto get_num_n_blocks = [&](int batch_idx) {
        int leftpad_k = batch_idx < num_batch && leftpad_k_ptr != nullptr ? leftpad_k_ptr[batch_idx] : 0;
        int seqlen;
        if (seqused_k) {
            seqlen = batch_idx < num_batch ? seqused_k[batch_idx] : 0;
        } else if (cu_seqlens_k) {
            int cur_cu_seqlen = batch_idx <= num_batch ? cu_seqlens_k[batch_idx] : 0;
            int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
            seqlen = next_cu_seqlen - cur_cu_seqlen;
        } else {
            seqlen = seqlen_k_static;
        }
        int seqlen_new;
        if (cu_seqlens_k_new) {
            int cur_cu_seqlen_new = batch_idx <= num_batch ? cu_seqlens_k_new[batch_idx] : 0;
            int next_cu_seqlen_new = __shfl_down_sync(0xffffffff, cur_cu_seqlen_new, 1);
            seqlen_new = next_cu_seqlen_new - cur_cu_seqlen_new;
        } else {
            seqlen_new = seqlen_k_new_static;
        }
        seqlen = seqlen - leftpad_k + seqlen_new;
        return batch_idx < num_batch && lane < kNumBatchPerWarp
            ? blockn_divmod.div(seqlen + blockn_divmod.divisor - 1) : 0;
    };

    int warp_idx = threadIdx.x / cutlass::NumThreadsPerWarp;
    int batch_cta_idx_offset = int(blockIdx.x) * 992;
    int bidb_start = batch_cta_idx_offset + kNumBatchPerWarp * warp_idx;
    int batch_idx = lane + bidb_start;
    int num_m_blocks = get_num_m_blocks(batch_idx);
    int num_n_blocks = get_num_n_blocks(batch_idx);

    auto get_nheads_in_l2 = [&](int n_blocks) {
        int nheads_in_l2 = n_blocks * 16 <= max_kvblocks_in_l2 ? 16
            : n_blocks * 8 <= max_kvblocks_in_l2 ? 8
            : n_blocks * 4 <= max_kvblocks_in_l2 ? 4
            : n_blocks * 2 <= max_kvblocks_in_l2 ? 2
            : 1;
        if (!packgqa) {
            nheads_in_l2 *= qhead_per_khead;
        }
        return min(nheads_in_l2, num_head);
    };

    int num_splits_dynamic;
    if (int(gridDim.x) > 1 || num_splits_static == 1) {
        num_splits_dynamic = 1;
    } else {
        int total_blocks = num_m_blocks * num_n_blocks;
        #pragma unroll
        for (int i = cutlass::NumThreadsPerWarp / 2; i >= 1; i /= 2) {
            total_blocks += __shfl_down_sync(0xffffffff, total_blocks, i);
        }
        if (lane == 0) {
            atomicAdd(total_blocks_smem, total_blocks);
        }
        __syncthreads();
        total_blocks = total_blocks_smem[0];
        int blocks_per_sm = static_cast<int>(ceilf(float(total_blocks) * 1.1f * float(num_head) / float(num_sm)));
        num_splits_dynamic = std::max(std::min((num_n_blocks + blocks_per_sm - 1) / blocks_per_sm, num_splits_static), 1);
        num_n_blocks = cutlass::ceil_div(num_n_blocks, num_splits_dynamic);
    }

    if constexpr (Sort) {
        if (lane == kNumBatchPerWarp || batch_idx >= num_batch) {
            num_n_blocks = INT_MIN;
        } else if (is_causal) {
            num_n_blocks = num_n_blocks * blockn_divmod.divisor - num_m_blocks * blockm_divmod.divisor;
        }
        int4 batch_coords[ITEMS_PER_THREAD];
        batch_coords[0] = make_int4(num_n_blocks, num_m_blocks, num_splits_dynamic, batch_idx);
        BlockMergeSort(temp_storage).Sort(batch_coords, PrepareSortOp<int4>());

        if (is_causal) {
            batch_coords[0].x = blockn_divmod.div(batch_coords[0].x + batch_coords[0].y * blockm_divmod.divisor);
        }

        batch_idx = batch_cta_idx_offset + threadIdx.x;
        if (batch_idx < num_batch && threadIdx.x < 992) {
            if (num_nheads_in_l2_ptr) {
                num_nheads_in_l2_ptr[batch_idx] = get_nheads_in_l2(max(batch_coords[0].x, 1));
            }
            num_m_blocks_ptr[batch_idx] = batch_coords[0].y;
            num_splits_dynamic_ptr[batch_idx] = batch_coords[0].z;
            varlen_batch_idx_ptr[batch_idx] = batch_coords[0].w;
        }
    } else {
        if (batch_idx < num_batch && lane < kNumBatchPerWarp) {
            if (num_nheads_in_l2_ptr) {
                num_nheads_in_l2_ptr[batch_idx] = get_nheads_in_l2(max(num_n_blocks, 1));
            }
            num_splits_dynamic_ptr[batch_idx] = num_splits_dynamic;
            num_m_blocks_ptr[batch_idx] = num_m_blocks;
        }
    }
}

}  // namespace min_fa3_varlen_detail

namespace min_fa3_varlen_demo {

void prepare_varlen_num_blocks(Flash_fwd_params& params,
                               cudaStream_t stream,
                               bool packgqa,
                               int blockM,
                               int blockN,
                               bool enable_pdl) {
    int qhead_per_khead = cutlass::ceil_div(params.h, params.h_k);
    int num_warps = cutlass::ceil_div(params.b, 31);
    int num_ctas = cutlass::ceil_div(params.b, 31 * 32);
    int const size_l2 = 8 * 1024 * 1024;
    int const element_size = params.is_e4m3 ? 1 : 2;
    int const size_one_kvblock = blockN * (params.d + params.dv) * element_size;
    int const max_kvblocks_in_l2 = size_l2 / size_one_kvblock;
    BOOL_SWITCH(params.varlen_sort_batches, Sort, [&] {
        NUM_WARP_SWITCH(num_warps, NumWarps, [&] {
            min_fa3_varlen_detail::prepare_varlen_num_blocks_kernel<NumWarps, Sort><<<num_ctas, 32 * NumWarps, 0, stream>>>(
                params.seqlen_q,
                params.seqlen_k,
                0,
                params.cu_seqlens_q,
                params.cu_seqlens_k,
                nullptr,
                params.seqused_q,
                params.seqused_k,
                params.leftpad_k,
                params.b,
                !packgqa ? params.h : params.h_k,
                qhead_per_khead,
                params.num_sm,
                params.num_splits,
                cutlass::FastDivmod(blockM),
                cutlass::FastDivmod(blockN),
                params.tile_count_semaphore,
                params.num_m_blocks_ptr,
                params.num_splits_dynamic_ptr,
                params.varlen_batch_idx_ptr,
                params.num_nheads_in_l2_ptr,
                enable_pdl,
                params.is_causal,
                packgqa,
                max_kvblocks_in_l2);
        });
    });
}

}  // namespace min_fa3_varlen_demo
