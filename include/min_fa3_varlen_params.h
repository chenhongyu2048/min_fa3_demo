// Copied and trimmed from Hopper forward sources:
// - hopper/flash.h
// - hopper/flash_api.cpp (host-side varlen parameter initialization style)
// Params in this file are copied from the original Hopper forward params path
// and trimmed down for the minimal SM90 bf16 head_dim=128 varlen forward-only demo.

#pragma once

#include <cuda.h>
#include <cstdint>

namespace min_fa3_varlen_demo {

struct Qkv_params {
    using index_t = int64_t;

    // The QKV matrices.
    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t v_dim_stride;

    // The number of heads.
    int h, h_k;
};

struct Flash_fwd_params : public Qkv_params {
    using index_t = int64_t;

    // The O matrix (output).
    void* __restrict__ o_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the softmax sum.
    void* __restrict__ softmax_lse_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, d;
    int seqlen_q_rounded, seqlen_k_rounded, d_rounded;
    int total_q, total_k;
    int b_k;
    int dv, dv_rounded;

    // The scaling factor for the kernel.
    float scale_softmax;

    // Array of length b+1 holding starting offset of each sequence.
    int* __restrict__ cu_seqlens_q;
    int* __restrict__ cu_seqlens_k;
    int* __restrict__ leftpad_k;

    // If provided, the actual length of each q/k sequence.
    int* __restrict__ seqused_q;
    int* __restrict__ seqused_k;

    // Local window size. This demo only uses the causal special case or full attention.
    int window_size_left, window_size_right;
    int attention_chunk;

    bool is_bf16;
    bool is_fp32;
    bool is_e4m3;
    bool is_causal;
    bool is_local;

    int num_splits;

    int* __restrict__ tile_count_semaphore;
    int* __restrict__ num_m_blocks_ptr;
    int* __restrict__ num_splits_dynamic_ptr;
    int* __restrict__ varlen_batch_idx_ptr;
    int* __restrict__ num_nheads_in_l2_ptr;
    bool skip_scheduler_metadata_computation;
    bool varlen_sort_batches;
    int tile_count_semaphore_offset;
    bool head_swizzle;
    bool prepare_varlen_pdl;

    int arch;
    int num_sm;
};

void prepare_varlen_num_blocks(Flash_fwd_params& params,
                               cudaStream_t stream,
                               bool packgqa,
                               int blockM,
                               int blockN,
                               bool enable_pdl);

void run_min_fa3_varlen_fwd(Flash_fwd_params& params, cudaStream_t stream);

template <bool IsCausal>
void run_min_fa3_varlen_sm90(Flash_fwd_params& params, cudaStream_t stream);

}  // namespace min_fa3_varlen_demo
