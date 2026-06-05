// Copied and trimmed from Hopper forward sources:
// - hopper/flash.h
// - hopper/flash_api.cpp (host-side initialization style)
// Params in this file are copied from the original Hopper forward params path
// and trimmed down for the minimal BSHD bf16 hdim128 forward-only demo.

#pragma once

#include <cuda.h>
#include <cstdint>
#include <optional>

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
    int dv, dv_rounded;

    // The scaling factor for the kernel.
    float scale_softmax;

    // Local window size. This demo only uses the causal special case or full attention.
    int window_size_left, window_size_right;
    int attention_chunk;

    bool is_bf16;
    bool is_causal;
    bool is_local;

    int num_splits;
    int* __restrict__ tile_count_semaphore;

    int arch;
    int num_sm;
};

// By default the launch grid is computed from get_grid_shape(...).
// When provided, manual_block_count overrides the 1D grid.x thread-block count.
void run_min_fa3_fwd(
    Flash_fwd_params& params,
    cudaStream_t stream,
    std::optional<int> manual_block_count = std::nullopt);
