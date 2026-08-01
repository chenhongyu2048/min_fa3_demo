// Copied and trimmed from Hopper forward sources:
// - hopper/flash_api.cpp
// Dense read-only KV-cache binding for SM90 BF16 D=128 decode/chunk prefill.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "heuristics.h"
#include "min_fa3_kvcache_launch.h"

namespace py = pybind11;

namespace {

using min_fa3_varlen_demo::Flash_fwd_params;

constexpr int kHeadDim = 128;
constexpr int kBlockM = 128;
constexpr int kCausalBlockN = 128;
constexpr int kNoncausalBlockN = 176;
constexpr int kPrepareVarlenMaxBatches1Cta = 992;

int round_multiple(int value, int multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

void check_bshd(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must have dtype torch.bfloat16");
    TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B, S, H, 128]");
    TORCH_CHECK(tensor.size(3) == kHeadDim, name, " must have head_dim 128");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous BSHD");
}

void check_packed_qkv(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must have dtype torch.bfloat16");
    TORCH_CHECK(tensor.dim() == 3, name, " must have shape [total_tokens, H, 128]");
    TORCH_CHECK(tensor.size(2) == kHeadDim, name, " must have head_dim 128");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous [total_tokens, H, 128]");
}

void check_cu_seqlens(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must have dtype torch.int32");
    TORCH_CHECK(tensor.dim() == 1, name, " must have shape [B + 1]");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.numel() >= 2, name, " must have shape [B + 1] with B >= 1");
}

void check_cu_seqlens_host(
    const torch::Tensor& tensor,
    const torch::Tensor& device_tensor,
    const char* name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must have dtype torch.int32");
    TORCH_CHECK(tensor.dim() == 1, name, " must have shape [B + 1]");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.numel() == device_tensor.numel(),
                name, " must have the same length as its CUDA cu_seqlens tensor");
}

int validate_host_cu_seqlens(
    const torch::Tensor& tensor,
    int64_t total_tokens,
    const char* name) {
    int const batch_size = static_cast<int>(tensor.numel() - 1);
    int const* values = tensor.data_ptr<int>();
    TORCH_CHECK(values[0] == 0, name, " must start with 0");
    int max_seqlen = 0;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        TORCH_CHECK(values[batch_idx + 1] > values[batch_idx],
                    name, " must be strictly increasing; sequence ", batch_idx,
                    " has non-positive length ", values[batch_idx + 1] - values[batch_idx]);
        max_seqlen = std::max(max_seqlen, values[batch_idx + 1] - values[batch_idx]);
    }
    TORCH_CHECK(values[batch_size] == total_tokens,
                name, "[-1] must equal the packed tensor's total token count. Got ",
                values[batch_size], " vs ", total_tokens);
    return max_seqlen;
}

int choose_num_splits(const Flash_fwd_params& params) {
    int const block_n = params.is_causal ? kCausalBlockN : kNoncausalBlockN;
    int const qhead_per_khead = params.h / params.h_k;
    int const num_m_blocks = (params.seqlen_q * qhead_per_khead + kBlockM - 1) / kBlockM;
    int const num_n_blocks = (params.seqlen_k + block_n - 1) / block_n;
    int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * 2;
    // Varlen dynamic split uses one long sequence as the upper-bound workload,
    // matching get_num_splits in the official Hopper API.
    int const total_mblocks = params.h_k * num_m_blocks;
    return num_splits_heuristic(
        total_mblocks,
        params.num_sm,
        num_n_blocks,
        num_m_blocks,
        size_one_kv_head,
        params.is_causal,
        128);
}

bool choose_pack_gqa(const Flash_fwd_params& params) {
    if (params.num_splits > 1) {
        return true;
    }
    if (params.h == params.h_k) {
        return false;
    }
    return should_pack_gqa(
        params.cu_seqlens_q != nullptr || params.seqused_q != nullptr,
        params.seqlen_q,
        params.h / params.h_k,
        kBlockM);
}

Flash_fwd_params make_kvcache_params(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& cache_seqlens,
    torch::Tensor& out,
    torch::Tensor& softmax_lse,
    torch::Tensor& scheduler_metadata,
    torch::Tensor& out_accum,
    torch::Tensor& softmax_lse_accum,
    int requested_num_splits,
    bool is_causal) {
    Flash_fwd_params params{};

    params.is_bf16 = true;
    params.is_fp32 = false;
    params.is_e4m3 = false;

    params.q_ptr = q.data_ptr();
    params.k_ptr = k_cache.data_ptr();
    params.v_ptr = v_cache.data_ptr();
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k_cache.stride(0);
    params.v_batch_stride = v_cache.stride(0);
    params.q_row_stride = q.stride(1);
    params.k_row_stride = k_cache.stride(1);
    params.v_row_stride = v_cache.stride(1);
    params.q_head_stride = q.stride(2);
    params.k_head_stride = k_cache.stride(2);
    params.v_head_stride = v_cache.stride(2);
    params.v_dim_stride = v_cache.stride(3);

    params.o_ptr = out.data_ptr();
    params.o_batch_stride = out.stride(0);
    params.o_row_stride = out.stride(1);
    params.o_head_stride = out.stride(2);
    params.softmax_lse_ptr = softmax_lse.data_ptr();

    params.b = static_cast<int>(q.size(0));
    params.seqlen_q = static_cast<int>(q.size(1));
    params.seqlen_k = static_cast<int>(k_cache.size(1));
    params.h = static_cast<int>(q.size(2));
    params.h_k = static_cast<int>(k_cache.size(2));
    params.d = kHeadDim;
    params.dv = kHeadDim;
    params.seqlen_q_rounded = round_multiple(params.seqlen_q, kBlockM);
    params.seqlen_k_rounded = round_multiple(params.seqlen_k, kBlockM);
    params.d_rounded = kHeadDim;
    params.dv_rounded = kHeadDim;
    params.total_q = params.b * params.seqlen_q;
    params.total_k = params.b * params.seqlen_k;
    params.b_k = params.b;
    params.scale_softmax = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.leftpad_k = nullptr;
    params.seqused_q = nullptr;
    params.seqused_k = cache_seqlens.data_ptr<int>();

    params.is_causal = is_causal;
    params.is_local = false;
    params.window_size_left = params.seqlen_k - 1;
    params.window_size_right = 0;
    params.attention_chunk = 0;

    auto* properties = at::cuda::getCurrentDeviceProperties();
    params.arch = properties->major * 10 + properties->minor;
    params.num_sm = properties->multiProcessorCount;

    params.num_splits = requested_num_splits == 0 ? choose_num_splits(params) : requested_num_splits;
    params.pack_gqa = choose_pack_gqa(params);

    params.skip_scheduler_metadata_computation = false;
    params.varlen_sort_batches = true;
    params.head_swizzle = params.is_causal;
    params.prepare_varlen_pdl = params.b <= kPrepareVarlenMaxBatches1Cta;

    int const b_rounded = round_multiple(params.b, 4);
    int const num_prepare_batch_vectors = 2 + 1 + (params.head_swizzle ? 1 : 0);
    int const head_swizzle_offset = b_rounded * 3;
    int const tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors;
    int* metadata_ptr = scheduler_metadata.data_ptr<int>();
    params.num_splits_dynamic_ptr = metadata_ptr;
    params.num_m_blocks_ptr = metadata_ptr + b_rounded;
    params.varlen_batch_idx_ptr = metadata_ptr + b_rounded * 2;
    params.num_nheads_in_l2_ptr = params.head_swizzle ? metadata_ptr + head_swizzle_offset : nullptr;
    params.tile_count_semaphore = metadata_ptr + tile_count_semaphore_offset;
    params.tile_count_semaphore_offset = tile_count_semaphore_offset;

    if (params.num_splits > 1) {
        params.oaccum_ptr = out_accum.data_ptr();
        params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
        params.oaccum_split_stride = out_accum.stride(0);
        params.oaccum_batch_stride = out_accum.stride(1);
        params.oaccum_head_stride = out_accum.stride(2);
        params.oaccum_row_stride = out_accum.stride(3);
        params.lseaccum_split_stride = softmax_lse_accum.stride(0);
        params.lseaccum_batch_stride = softmax_lse_accum.stride(1);
        params.lseaccum_head_stride = softmax_lse_accum.stride(2);
    }

    return params;
}

Flash_fwd_params make_kvcache_varlen_params(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    torch::Tensor& out,
    torch::Tensor& softmax_lse,
    torch::Tensor& scheduler_metadata,
    torch::Tensor& out_accum,
    torch::Tensor& softmax_lse_accum,
    int effective_num_splits,
    bool is_causal) {
    Flash_fwd_params params{};

    params.is_bf16 = true;
    params.is_fp32 = false;
    params.is_e4m3 = false;

    params.q_ptr = q.data_ptr();
    params.k_ptr = k_cache.data_ptr();
    params.v_ptr = v_cache.data_ptr();
    params.q_batch_stride = 0;
    params.k_batch_stride = 0;
    params.v_batch_stride = 0;
    params.q_row_stride = q.stride(0);
    params.k_row_stride = k_cache.stride(0);
    params.v_row_stride = v_cache.stride(0);
    params.q_head_stride = q.stride(1);
    params.k_head_stride = k_cache.stride(1);
    params.v_head_stride = v_cache.stride(1);
    params.v_dim_stride = v_cache.stride(2);

    params.o_ptr = out.data_ptr();
    params.o_batch_stride = 0;
    params.o_row_stride = out.stride(0);
    params.o_head_stride = out.stride(1);
    params.softmax_lse_ptr = softmax_lse.data_ptr();

    params.b = static_cast<int>(cu_seqlens_q.numel() - 1);
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.h = static_cast<int>(q.size(1));
    params.h_k = static_cast<int>(k_cache.size(1));
    params.d = kHeadDim;
    params.dv = kHeadDim;
    params.seqlen_q_rounded = round_multiple(max_seqlen_q, kBlockM);
    params.seqlen_k_rounded = round_multiple(max_seqlen_k, kBlockM);
    params.d_rounded = kHeadDim;
    params.dv_rounded = kHeadDim;
    params.total_q = static_cast<int>(q.size(0));
    params.total_k = static_cast<int>(k_cache.size(0));
    params.b_k = params.b;
    params.scale_softmax = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
    params.leftpad_k = nullptr;
    params.seqused_q = nullptr;
    params.seqused_k = nullptr;

    params.is_causal = is_causal;
    params.is_local = false;
    params.window_size_left = max_seqlen_k - 1;
    params.window_size_right = 0;
    params.attention_chunk = 0;

    auto* properties = at::cuda::getCurrentDeviceProperties();
    params.arch = properties->major * 10 + properties->minor;
    params.num_sm = properties->multiProcessorCount;

    params.num_splits = effective_num_splits;
    params.pack_gqa = choose_pack_gqa(params);

    params.skip_scheduler_metadata_computation = false;
    params.varlen_sort_batches = true;
    params.head_swizzle = params.is_causal;
    params.prepare_varlen_pdl = params.b <= kPrepareVarlenMaxBatches1Cta;

    int const b_rounded = round_multiple(params.b, 4);
    int const num_prepare_batch_vectors = 2 + 1 + (params.head_swizzle ? 1 : 0);
    int const head_swizzle_offset = b_rounded * 3;
    int const tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors;
    int* metadata_ptr = scheduler_metadata.data_ptr<int>();
    params.num_splits_dynamic_ptr = metadata_ptr;
    params.num_m_blocks_ptr = metadata_ptr + b_rounded;
    params.varlen_batch_idx_ptr = metadata_ptr + b_rounded * 2;
    params.num_nheads_in_l2_ptr = params.head_swizzle ? metadata_ptr + head_swizzle_offset : nullptr;
    params.tile_count_semaphore = metadata_ptr + tile_count_semaphore_offset;
    params.tile_count_semaphore_offset = tile_count_semaphore_offset;

    if (params.num_splits > 1) {
        params.oaccum_ptr = out_accum.data_ptr();
        params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
        params.oaccum_split_stride = out_accum.stride(0);
        params.oaccum_batch_stride = 0;
        params.oaccum_head_stride = out_accum.stride(1);
        params.oaccum_row_stride = out_accum.stride(2);
        params.lseaccum_split_stride = softmax_lse_accum.stride(0);
        params.lseaccum_batch_stride = 0;
        params.lseaccum_head_stride = softmax_lse_accum.stride(1);
    }

    return params;
}

py::object forward_kvcache(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor cache_seqlens,
    int64_t num_splits,
    bool return_lse,
    py::object is_causal_obj) {
    check_bshd(q, "q");
    check_bshd(k_cache, "k_cache");
    check_bshd(v_cache, "v_cache");
    TORCH_CHECK(cache_seqlens.is_cuda(), "cache_seqlens must be a CUDA tensor");
    TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt32, "cache_seqlens must have dtype torch.int32");
    TORCH_CHECK(cache_seqlens.dim() == 1, "cache_seqlens must have shape [B]");
    TORCH_CHECK(cache_seqlens.is_contiguous(), "cache_seqlens must be contiguous");

    TORCH_CHECK(q.device() == k_cache.device() && q.device() == v_cache.device() && q.device() == cache_seqlens.device(),
                "q, k_cache, v_cache, and cache_seqlens must be on the same CUDA device");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0,
                "q requires positive B, Sq, and QH dimensions");
    TORCH_CHECK(k_cache.size(0) == q.size(0) && v_cache.size(0) == q.size(0),
                "q, k_cache, and v_cache must have the same batch size B");
    TORCH_CHECK(k_cache.size(1) > 0 && k_cache.size(1) == v_cache.size(1),
                "k_cache and v_cache must have the same positive cache capacity");
    TORCH_CHECK(k_cache.size(2) > 0 && k_cache.size(2) == v_cache.size(2),
                "k_cache and v_cache must have the same positive KV head count");
    TORCH_CHECK(q.size(2) % k_cache.size(2) == 0,
                "QH must be divisible by KVH. Got QH=", q.size(2), ", KVH=", k_cache.size(2));
    TORCH_CHECK(cache_seqlens.numel() == q.size(0), "cache_seqlens must have shape [B]");
    // None preserves the original KV-cache behavior: single-token decode uses
    // the equivalent noncausal 128x176 tile and chunks use bottom-right causal.
    bool const is_causal = is_causal_obj.is_none()
        ? q.size(1) != 1
        : is_causal_obj.cast<bool>();
    TORCH_CHECK(!is_causal || q.size(1) <= k_cache.size(1),
                "Causal Sq must not exceed the KV-cache capacity. Got Sq=", q.size(1),
                ", capacity=", k_cache.size(1));
    TORCH_CHECK(num_splits >= 0 && num_splits <= 128,
                "num_splits must be 0 (auto), 1 (NoSplit), or in [2, 128]. Got ", num_splits);

    for (auto value : {q.size(0), q.size(1), q.size(2), k_cache.size(1), k_cache.size(2)}) {
        TORCH_CHECK(value <= std::numeric_limits<int>::max(), "all tensor dimensions must fit in int32");
    }
    TORCH_CHECK(q.size(0) * q.size(1) <= std::numeric_limits<int>::max(), "B * Sq must fit in int32");
    TORCH_CHECK(k_cache.size(0) * k_cache.size(1) <= std::numeric_limits<int>::max(),
                "B * Sk_capacity must fit in int32");

    c10::cuda::CUDAGuard device_guard(q.device());
    auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 9 && properties->minor == 0,
                "forward_kvcache only supports Hopper SM90. Current device capability is ",
                properties->major, ".", properties->minor);

    auto out = torch::empty_like(q);
    auto softmax_lse = torch::empty(
        {q.size(0), q.size(2), q.size(1)},
        q.options().dtype(torch::kFloat));

    // Compute the requested/automatic upper bound before allocating Split workspace.
    Flash_fwd_params heuristic_params{};
    heuristic_params.b = static_cast<int>(q.size(0));
    heuristic_params.seqlen_q = static_cast<int>(q.size(1));
    heuristic_params.seqlen_k = static_cast<int>(k_cache.size(1));
    heuristic_params.h = static_cast<int>(q.size(2));
    heuristic_params.h_k = static_cast<int>(k_cache.size(2));
    heuristic_params.d = kHeadDim;
    heuristic_params.dv = kHeadDim;
    heuristic_params.is_causal = is_causal;
    heuristic_params.num_sm = properties->multiProcessorCount;
    int const effective_num_splits = num_splits == 0
        ? choose_num_splits(heuristic_params)
        : static_cast<int>(num_splits);

    int const b_rounded = round_multiple(static_cast<int>(q.size(0)), 4);
    int const num_prepare_batch_vectors = 3 + (is_causal ? 1 : 0);
    auto scheduler_metadata = torch::empty(
        {1 + b_rounded * num_prepare_batch_vectors},
        q.options().dtype(torch::kInt32));

    torch::Tensor out_accum;
    torch::Tensor softmax_lse_accum;
    if (effective_num_splits > 1) {
        out_accum = torch::empty(
            {effective_num_splits, q.size(0), q.size(2), q.size(1), kHeadDim},
            q.options().dtype(torch::kFloat));
        softmax_lse_accum = torch::empty(
            {effective_num_splits, q.size(0), q.size(2), q.size(1)},
            q.options().dtype(torch::kFloat));
    }

    auto params = make_kvcache_params(
        q,
        k_cache,
        v_cache,
        cache_seqlens,
        out,
        softmax_lse,
        scheduler_metadata,
        out_accum,
        softmax_lse_accum,
        effective_num_splits,
        is_causal);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
    min_fa3_varlen_demo::run_min_fa3_kvcache_fwd(params, stream);
    if (params.num_splits > 1) {
        min_fa3_varlen_demo::run_min_fa3_kvcache_combine(params, stream, true);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (return_lse) {
        return py::make_tuple(out, softmax_lse);
    }
    return py::cast(out);
}

py::object forward_kvcache_varlen(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor cu_seqlens_q,
    torch::Tensor cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    torch::Tensor cu_seqlens_q_host,
    torch::Tensor cu_seqlens_k_host,
    int64_t num_splits,
    bool return_lse,
    py::object is_causal_obj) {
    check_packed_qkv(q, "q");
    check_packed_qkv(k_cache, "k_cache");
    check_packed_qkv(v_cache, "v_cache");
    check_cu_seqlens(cu_seqlens_q, "cu_seqlens_q");
    check_cu_seqlens(cu_seqlens_k, "cu_seqlens_k");
    check_cu_seqlens_host(cu_seqlens_q_host, cu_seqlens_q, "cu_seqlens_q_host");
    check_cu_seqlens_host(cu_seqlens_k_host, cu_seqlens_k, "cu_seqlens_k_host");

    TORCH_CHECK(q.device() == k_cache.device() && q.device() == v_cache.device(),
                "q, k_cache, and v_cache must be on the same CUDA device");
    TORCH_CHECK(q.device() == cu_seqlens_q.device() && q.device() == cu_seqlens_k.device(),
                "q, k_cache, v_cache, cu_seqlens_q, and cu_seqlens_k must be on the same CUDA device");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0,
                "q requires positive total_q and QH dimensions");
    TORCH_CHECK(k_cache.size(0) > 0 && k_cache.size(1) > 0,
                "k_cache requires positive total_k and KVH dimensions");
    TORCH_CHECK(k_cache.size(0) == v_cache.size(0),
                "k_cache and v_cache must have the same total token count");
    TORCH_CHECK(k_cache.size(1) == v_cache.size(1),
                "k_cache and v_cache must have the same KV head count");
    TORCH_CHECK(q.size(1) % k_cache.size(1) == 0,
                "QH must be divisible by KVH. Got QH=", q.size(1), ", KVH=", k_cache.size(1));
    TORCH_CHECK(cu_seqlens_q.numel() == cu_seqlens_k.numel(),
                "cu_seqlens_q and cu_seqlens_k must have the same length");
    TORCH_CHECK(max_seqlen_q > 0 && max_seqlen_k > 0,
                "max_seqlen_q and max_seqlen_k must be positive");
    TORCH_CHECK(num_splits >= 0 && num_splits <= 128,
                "num_splits must be 0 (auto), 1 (NoSplit), or in [2, 128]. Got ", num_splits);

    constexpr int64_t kMaxRoundedSeqlen = std::numeric_limits<int>::max() - (kBlockM - 1);
    TORCH_CHECK(max_seqlen_q <= kMaxRoundedSeqlen && max_seqlen_k <= kMaxRoundedSeqlen,
                "max sequence lengths must fit in int32 after rounding to 128");
    TORCH_CHECK(q.size(0) <= std::numeric_limits<int>::max()
                    && k_cache.size(0) <= std::numeric_limits<int>::max(),
                "total_q and total_k must fit in int32");
    TORCH_CHECK(q.size(1) <= std::numeric_limits<int>::max()
                    && k_cache.size(1) <= std::numeric_limits<int>::max(),
                "QH and KVH must fit in int32");
    TORCH_CHECK(cu_seqlens_q.numel() - 1 <= std::numeric_limits<int>::max() / 4,
                "batch size is too large for scheduler metadata");

    int const batch_size = static_cast<int>(cu_seqlens_q.numel() - 1);
    int const actual_max_seqlen_q = validate_host_cu_seqlens(
        cu_seqlens_q_host, q.size(0), "cu_seqlens_q_host");
    int const actual_max_seqlen_k = validate_host_cu_seqlens(
        cu_seqlens_k_host, k_cache.size(0), "cu_seqlens_k_host");
    TORCH_CHECK(max_seqlen_q == actual_max_seqlen_q,
                "max_seqlen_q must equal the maximum length in cu_seqlens_q_host. Got ",
                max_seqlen_q, " vs ", actual_max_seqlen_q);
    TORCH_CHECK(max_seqlen_k == actual_max_seqlen_k,
                "max_seqlen_k must equal the maximum length in cu_seqlens_k_host. Got ",
                max_seqlen_k, " vs ", actual_max_seqlen_k);

    bool const is_causal = is_causal_obj.is_none()
        ? max_seqlen_q != 1
        : is_causal_obj.cast<bool>();
    if (is_causal) {
        int const* q_offsets = cu_seqlens_q_host.data_ptr<int>();
        int const* k_offsets = cu_seqlens_k_host.data_ptr<int>();
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            int const q_len = q_offsets[batch_idx + 1] - q_offsets[batch_idx];
            int const k_len = k_offsets[batch_idx + 1] - k_offsets[batch_idx];
            TORCH_CHECK(q_len <= k_len,
                        "Causal packed attention requires q_len <= k_len for every sequence. Sequence ",
                        batch_idx, " has q_len=", q_len, ", k_len=", k_len);
        }
    }

    c10::cuda::CUDAGuard device_guard(q.device());
    auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 9 && properties->minor == 0,
                "forward_kvcache_varlen only supports Hopper SM90. Current device capability is ",
                properties->major, ".", properties->minor);

    auto out = torch::empty_like(q);
    auto softmax_lse = torch::empty(
        {q.size(1), q.size(0)},
        q.options().dtype(torch::kFloat));

    Flash_fwd_params heuristic_params{};
    heuristic_params.b = batch_size;
    heuristic_params.seqlen_q = static_cast<int>(max_seqlen_q);
    heuristic_params.seqlen_k = static_cast<int>(max_seqlen_k);
    heuristic_params.h = static_cast<int>(q.size(1));
    heuristic_params.h_k = static_cast<int>(k_cache.size(1));
    heuristic_params.d = kHeadDim;
    heuristic_params.dv = kHeadDim;
    heuristic_params.is_causal = is_causal;
    heuristic_params.num_sm = properties->multiProcessorCount;
    int const effective_num_splits = num_splits == 0
        ? choose_num_splits(heuristic_params)
        : static_cast<int>(num_splits);

    int const b_rounded = round_multiple(batch_size, 4);
    int const num_prepare_batch_vectors = 3 + (is_causal ? 1 : 0);
    auto scheduler_metadata = torch::empty(
        {1 + b_rounded * num_prepare_batch_vectors},
        q.options().dtype(torch::kInt32));

    torch::Tensor out_accum;
    torch::Tensor softmax_lse_accum;
    if (effective_num_splits > 1) {
        out_accum = torch::empty(
            {effective_num_splits, q.size(1), q.size(0), kHeadDim},
            q.options().dtype(torch::kFloat));
        softmax_lse_accum = torch::empty(
            {effective_num_splits, q.size(1), q.size(0)},
            q.options().dtype(torch::kFloat));
    }

    auto params = make_kvcache_varlen_params(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        static_cast<int>(max_seqlen_q),
        static_cast<int>(max_seqlen_k),
        out,
        softmax_lse,
        scheduler_metadata,
        out_accum,
        softmax_lse_accum,
        effective_num_splits,
        is_causal);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
    min_fa3_varlen_demo::run_min_fa3_kvcache_fwd(params, stream);
    if (params.num_splits > 1) {
        min_fa3_varlen_demo::run_min_fa3_kvcache_combine(params, stream, true);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (return_lse) {
        return py::make_tuple(out, softmax_lse);
    }
    return py::cast(out);
}

}  // namespace

void bind_min_fa3_kvcache(py::module_& module) {
    module.def(
        "forward_kvcache",
        &forward_kvcache,
        py::arg("q"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("cache_seqlens"),
        py::kw_only(),
        py::arg("num_splits") = 0,
        py::arg("return_lse") = false,
        py::arg("is_causal") = py::none(),
        "Minimal Hopper FA3 dense KV-cache decode/chunk-prefill forward. "
        "is_causal=None preserves the Sq-dependent legacy behavior; explicit false "
        "allows noncausal context attention with Sq > Sk_capacity.");
    module.def(
        "forward_kvcache_varlen",
        &forward_kvcache_varlen,
        py::arg("q"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("cu_seqlens_q"),
        py::arg("cu_seqlens_k"),
        py::arg("max_seqlen_q"),
        py::arg("max_seqlen_k"),
        py::kw_only(),
        py::arg("cu_seqlens_q_host"),
        py::arg("cu_seqlens_k_host"),
        py::arg("num_splits") = 0,
        py::arg("return_lse") = false,
        py::arg("is_causal") = py::none(),
        "Minimal Hopper FA3 packed-varlen read-only KV-cache decode/chunk-prefill forward. "
        "The CPU cu_seqlens mirrors are required for synchronization-free validation.");
}
