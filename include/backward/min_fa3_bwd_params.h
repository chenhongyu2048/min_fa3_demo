// Copied and trimmed from Hopper backward parameter sources:
// - hopper/flash.h
// - hopper/flash_api.cpp

#pragma once

#include <cuda.h>
#include <cstdint>

namespace min_fa3_backward {

struct Qkv_params {
    using index_t = int64_t;

    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ v_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    int h, h_k;
};

struct Flash_fwd_params : public Qkv_params {
    using index_t = int64_t;

    void* __restrict__ o_ptr;
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;
    void* __restrict__ softmax_lse_ptr;

    int b, seqlen_q, seqlen_k, d;
    int seqlen_q_rounded, seqlen_k_rounded, d_rounded;
    int total_q, total_k;
    int dv, dv_rounded;

    float scale_softmax;
    int window_size_left, window_size_right;

    int* __restrict__ cu_seqlens_q;
    int* __restrict__ cu_seqlens_k;
    int* __restrict__ seqused_q;
    int* __restrict__ seqused_k;

    bool is_bf16;
    bool is_causal;
    bool is_local;

    int arch;
    int num_sm;
};

struct Flash_bwd_params : public Flash_fwd_params {
    using index_t = int64_t;

    void* __restrict__ do_ptr;
    void* __restrict__ dq_ptr;
    void* __restrict__ dk_ptr;
    void* __restrict__ dv_ptr;

    void* __restrict__ dq_accum_ptr;
    void* __restrict__ dk_accum_ptr;
    void* __restrict__ dv_accum_ptr;

    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    void* __restrict__ dsoftmax_sum;
    void* __restrict__ softmax_lse_log2_ptr;

    int* __restrict__ dq_semaphore;
    int* __restrict__ dk_semaphore;
    int* __restrict__ dv_semaphore;

    bool deterministic;
};

void run_min_fa3_bwd(Flash_bwd_params& params, cudaStream_t stream);

}  // namespace min_fa3_backward
