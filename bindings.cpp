#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "min_fa3_launch_override.h"
#include "min_fa3_params.h"
#include "min_fa3_varlen_params.h"

namespace py = pybind11;

void bind_parallel_remote_load(py::module_& m);
void bind_varlen_ring(py::module_& m);
void bind_varlen_mega_ring(py::module_& m);
void bind_min_fa3_backward(py::module_& m);

namespace {

using BshdParams = ::Flash_fwd_params;
using VarlenParams = min_fa3_varlen_demo::Flash_fwd_params;

int round_multiple(int x, int m) {
    return (x + m - 1) / m * m;
}

void check_bshd(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must have dtype torch.bfloat16");
    TORCH_CHECK(t.dim() == 4, name, " must have shape [B, S, H, D]");
    TORCH_CHECK(t.size(3) == 128, name, " must have head_dim D=128");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous BSHD");
}

void check_varlen_qkv(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must have dtype torch.bfloat16");
    TORCH_CHECK(t.dim() == 3, name, " must have shape [total_tokens, H, D]");
    TORCH_CHECK(t.size(2) == 128, name, " must have head_dim D=128");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous [total_tokens, H, D]");
}

void check_cu_seqlens(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must have dtype torch.int32");
    TORCH_CHECK(t.dim() == 1, name, " must be 1D");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.numel() >= 2, name, " must have shape [B + 1] with B >= 1");
}

void check_cu_seqlens_host(const torch::Tensor& t, const torch::Tensor& device_t, const char* name) {
    TORCH_CHECK(!t.is_cuda(), name, " must be a CPU tensor");
    TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must have dtype torch.int32");
    TORCH_CHECK(t.dim() == 1, name, " must be 1D");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.numel() == device_t.numel(),
                name, " must have the same length as its CUDA cu_seqlens tensor");
}

std::optional<int> parse_manual_block_count(py::object manual_block_count_obj) {
    if (manual_block_count_obj.is_none()) {
        return std::nullopt;
    }
    return min_fa3_detail::validate_manual_block_count(
        manual_block_count_obj.cast<int64_t>(),
        "manual_block_count");
}

uint32_t check_grid_dim_component(int64_t value, const char* name) {
    TORCH_CHECK(
        value >= 0 && value <= std::numeric_limits<uint32_t>::max(),
        name,
        " must be between 0 and ",
        std::numeric_limits<uint32_t>::max(),
        ". Got ",
        value);
    return static_cast<uint32_t>(value);
}

BshdParams make_bshd_params(const torch::Tensor& q,
                            const torch::Tensor& k,
                            const torch::Tensor& v,
                            torch::Tensor& out,
                            torch::Tensor& softmax_lse,
                            torch::Tensor& tile_count_semaphore,
                            bool is_causal) {
    BshdParams params{};
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.v_dim_stride = v.stride(-1);
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);

    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);
    params.o_batch_stride = out.stride(0);

    params.softmax_lse_ptr = softmax_lse.data_ptr();

    params.b = q.size(0);
    params.seqlen_q = q.size(1);
    params.seqlen_k = k.size(1);
    params.h = q.size(2);
    params.h_k = k.size(2);
    params.d = q.size(3);
    params.dv = v.size(3);
    params.seqlen_q_rounded = params.seqlen_q;
    params.seqlen_k_rounded = params.seqlen_k;
    params.d_rounded = params.d;
    params.dv_rounded = params.dv;
    params.scale_softmax = 1.0f / std::sqrt(static_cast<float>(params.d));

    params.is_causal = is_causal;
    params.is_local = false;
    params.window_size_left = params.seqlen_k - 1;
    params.window_size_right = is_causal ? 0 : params.seqlen_q - 1;
    params.attention_chunk = 0;

    params.num_splits = 1;
    params.tile_count_semaphore = tile_count_semaphore.defined() ? tile_count_semaphore.data_ptr<int>() : nullptr;

    auto* props = at::cuda::getCurrentDeviceProperties();
    params.arch = props->major * 10 + props->minor;
    params.num_sm = props->multiProcessorCount;

    return params;
}

VarlenParams make_varlen_params(const torch::Tensor& q,
                                const torch::Tensor& k,
                                const torch::Tensor& v,
                                const torch::Tensor& cu_seqlens_q,
                                const torch::Tensor& cu_seqlens_k,
                                int max_seqlen_q,
                                int max_seqlen_k,
                                torch::Tensor& out,
                                torch::Tensor& softmax_lse,
                                torch::Tensor& scheduler_metadata,
                                bool is_causal) {
    VarlenParams params{};
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;
    params.is_fp32 = false;
    params.is_e4m3 = false;

    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.q_row_stride = q.stride(0);
    params.k_row_stride = k.stride(0);
    params.v_row_stride = v.stride(0);
    params.q_head_stride = q.stride(1);
    params.k_head_stride = k.stride(1);
    params.v_head_stride = v.stride(1);
    params.v_dim_stride = v.stride(2);

    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(0);
    params.o_head_stride = out.stride(1);
    params.softmax_lse_ptr = softmax_lse.data_ptr();

    params.b = cu_seqlens_q.size(0) - 1;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    params.seqlen_k_rounded = round_multiple(max_seqlen_k, 128);
    params.h = q.size(1);
    params.h_k = k.size(1);
    params.d = q.size(2);
    params.d_rounded = params.d;
    params.total_q = q.size(0);
    params.total_k = k.size(0);
    params.b_k = params.b;
    params.dv = v.size(2);
    params.dv_rounded = params.dv;
    params.scale_softmax = 1.0f / std::sqrt(static_cast<float>(params.d));

    params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q.data_ptr());
    params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k.data_ptr());
    params.leftpad_k = nullptr;
    params.seqused_q = nullptr;
    params.seqused_k = nullptr;

    params.is_causal = is_causal;
    params.is_local = false;
    params.window_size_left = max_seqlen_k - 1;
    params.window_size_right = is_causal ? 0 : max_seqlen_q - 1;
    params.attention_chunk = 0;

    params.num_splits = 1;
    params.skip_scheduler_metadata_computation = false;
    params.varlen_sort_batches = true;
    params.head_swizzle = is_causal;
    params.prepare_varlen_pdl = params.b <= 992;

    auto* props = at::cuda::getCurrentDeviceProperties();
    params.arch = props->major * 10 + props->minor;
    params.num_sm = props->multiProcessorCount;

    int b_rounded = round_multiple(params.b, 4);
    int num_prepare_batch_vectors = 2;
    if (params.varlen_sort_batches) {
        num_prepare_batch_vectors += 1;
    }
    if (params.head_swizzle) {
        num_prepare_batch_vectors += 1;
    }
    int head_swizzle_offset = b_rounded * (params.varlen_sort_batches ? 3 : 2);
    int tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors;
    int* metadata_ptr = scheduler_metadata.data_ptr<int>();
    params.num_splits_dynamic_ptr = metadata_ptr;
    params.num_m_blocks_ptr = metadata_ptr + b_rounded;
    params.varlen_batch_idx_ptr = params.varlen_sort_batches ? metadata_ptr + b_rounded * 2 : nullptr;
    params.num_nheads_in_l2_ptr = params.head_swizzle ? metadata_ptr + head_swizzle_offset : nullptr;
    params.tile_count_semaphore = metadata_ptr + tile_count_semaphore_offset;
    params.tile_count_semaphore_offset = tile_count_semaphore_offset;

    return params;
}

py::object forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal,
    py::object manual_block_count_obj,
    bool return_lse) {
    auto manual_block_count = parse_manual_block_count(manual_block_count_obj);
    check_bshd(q, "q");
    check_bshd(k, "k");
    check_bshd(v, "v");

    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, v must be on the same CUDA device");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "q, k, v must have the same batch size B");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v must have the same sequence length Sk");
    TORCH_CHECK(k.size(2) == v.size(2), "k and v must have the same KV head count");
    TORCH_CHECK(q.size(2) % k.size(2) == 0,
                "This demo requires qhead % kvhead == 0 for GQA/MQA. Got qhead=",
                q.size(2), ", kvhead=", k.size(2));
    TORCH_CHECK(v.size(3) == 128, "v must have head_dim D=128");

    c10::cuda::CUDAGuard device_guard(q.device());
    auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 9 && props->minor == 0,
                "min_fa3_demo only supports Hopper SM90. Current device capability is ",
                props->major, ".", props->minor);

    auto out = torch::empty_like(q);
    auto lse = torch::empty({q.size(0), q.size(2), q.size(1)}, q.options().dtype(torch::kFloat));
    auto semaphore = is_causal ? torch::zeros({1}, q.options().dtype(torch::kInt32)) : torch::Tensor();

    auto params = make_bshd_params(q, k, v, out, lse, semaphore, is_causal);
    run_min_fa3_fwd(params, at::cuda::getCurrentCUDAStream(q.get_device()), manual_block_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (return_lse) {
        return py::make_tuple(out, lse);
    }
    return py::cast(out);
}

py::object forward_varlen(torch::Tensor q,
                          torch::Tensor k,
                          torch::Tensor v,
                          torch::Tensor cu_seqlens_q,
                          torch::Tensor cu_seqlens_k,
                          torch::Tensor cu_seqlens_q_host,
                          torch::Tensor cu_seqlens_k_host,
                          int64_t max_seqlen_q,
                          int64_t max_seqlen_k,
                          bool is_causal,
                          py::object manual_block_count_obj,
                          bool return_lse) {
    auto manual_block_count = parse_manual_block_count(manual_block_count_obj);
    check_varlen_qkv(q, "q");
    check_varlen_qkv(k, "k");
    check_varlen_qkv(v, "v");
    check_cu_seqlens(cu_seqlens_q, "cu_seqlens_q");
    check_cu_seqlens(cu_seqlens_k, "cu_seqlens_k");
    check_cu_seqlens_host(cu_seqlens_q_host, cu_seqlens_q, "cu_seqlens_q_host");
    check_cu_seqlens_host(cu_seqlens_k_host, cu_seqlens_k, "cu_seqlens_k_host");

    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, v must be on the same CUDA device");
    TORCH_CHECK(q.device() == cu_seqlens_q.device() && q.device() == cu_seqlens_k.device(),
                "q, k, v, cu_seqlens_q, and cu_seqlens_k must be on the same CUDA device");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v must have the same KV head count");
    TORCH_CHECK(q.size(1) % k.size(1) == 0,
                "This demo requires qhead % kvhead == 0 for GQA/MQA. Got qhead=",
                q.size(1), ", kvhead=", k.size(1));
    TORCH_CHECK(v.size(2) == 128, "v must have head_dim D=128");
    TORCH_CHECK(cu_seqlens_q.size(0) == cu_seqlens_k.size(0), "cu_seqlens_q and cu_seqlens_k must have the same length");
    TORCH_CHECK(max_seqlen_q > 0 && max_seqlen_k > 0, "max_seqlen_q and max_seqlen_k must be positive");
    TORCH_CHECK(max_seqlen_q <= std::numeric_limits<int>::max() && max_seqlen_k <= std::numeric_limits<int>::max(),
                "max seqlens must fit in int32");

    c10::cuda::CUDAGuard device_guard(q.device());
    auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 9 && props->minor == 0,
                "min_fa3_demo only supports Hopper SM90. Current device capability is ",
                props->major, ".", props->minor);

    int batch_size = cu_seqlens_q_host.size(0) - 1;
    int const* cu_seqlens_q_host_ptr = cu_seqlens_q_host.data_ptr<int>();
    int const* cu_seqlens_k_host_ptr = cu_seqlens_k_host.data_ptr<int>();
    TORCH_CHECK(batch_size >= 1, "varlen demo requires batch size B >= 1");
    TORCH_CHECK(cu_seqlens_q_host_ptr[0] == 0, "cu_seqlens_q_host must start with 0");
    TORCH_CHECK(cu_seqlens_k_host_ptr[0] == 0, "cu_seqlens_k_host must start with 0");
    TORCH_CHECK(cu_seqlens_q_host_ptr[batch_size] == q.size(0),
                "cu_seqlens_q_host[-1] must equal q.size(0). Got ", cu_seqlens_q_host_ptr[batch_size],
                " vs ", q.size(0));
    TORCH_CHECK(cu_seqlens_k_host_ptr[batch_size] == k.size(0),
                "cu_seqlens_k_host[-1] must equal k.size(0). Got ", cu_seqlens_k_host_ptr[batch_size],
                " vs ", k.size(0));

    auto out = torch::empty_like(q);
    auto lse = torch::empty({q.size(1), q.size(0)}, q.options().dtype(torch::kFloat));

    int b_rounded = round_multiple(batch_size, 4);
    bool varlen_sort_batches = true;
    bool head_swizzle = is_causal;
    int num_prepare_batch_vectors = 2 + (varlen_sort_batches ? 1 : 0) + (head_swizzle ? 1 : 0);
    int metadata_size = 1 + b_rounded * num_prepare_batch_vectors;
    auto scheduler_metadata = torch::empty({metadata_size}, q.options().dtype(torch::kInt32));

    auto params = make_varlen_params(q,
                                     k,
                                     v,
                                     cu_seqlens_q,
                                     cu_seqlens_k,
                                     static_cast<int>(max_seqlen_q),
                                     static_cast<int>(max_seqlen_k),
                                     out,
                                     lse,
                                     scheduler_metadata,
                                     is_causal);
    min_fa3_varlen_demo::run_min_fa3_varlen_fwd(
        params,
        at::cuda::getCurrentCUDAStream(q.get_device()),
        manual_block_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (return_lse) {
        return py::make_tuple(out, lse);
    }
    return py::cast(out);
}

py::tuple debug_resolve_launch_grid_shape(
    int64_t auto_grid_x,
    int64_t auto_grid_y,
    int64_t auto_grid_z,
    py::object manual_block_count_obj) {
    auto manual_block_count = parse_manual_block_count(manual_block_count_obj);
    dim3 auto_grid_dims{
        check_grid_dim_component(auto_grid_x, "auto_grid_x"),
        check_grid_dim_component(auto_grid_y, "auto_grid_y"),
        check_grid_dim_component(auto_grid_z, "auto_grid_z"),
    };
    dim3 resolved_grid_dims = min_fa3_detail::resolve_launch_grid_shape(auto_grid_dims, manual_block_count);
    return py::make_tuple(resolved_grid_dims.x, resolved_grid_dims.y, resolved_grid_dims.z);
}

}  // namespace

PYBIND11_MODULE(_min_fa3_op, m) {
    m.def(
        "forward",
        &forward,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("is_causal"),
        py::kw_only(),
        py::arg("manual_block_count") = py::none(),
        py::arg("return_lse") = false,
        "Minimal Hopper FlashAttention forward demo.\n\n"
        "manual_block_count is an optional grid.x thread-block count override. "
        "When omitted, the launch grid is computed automatically by get_grid_shape(...).");
    m.def(
        "forward_varlen",
        &forward_varlen,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("cu_seqlens_q"),
        py::arg("cu_seqlens_k"),
        py::arg("cu_seqlens_q_host"),
        py::arg("cu_seqlens_k_host"),
        py::arg("max_seqlen_q"),
        py::arg("max_seqlen_k"),
        py::arg("is_causal"),
        py::kw_only(),
        py::arg("manual_block_count") = py::none(),
        py::arg("return_lse") = false,
        "Minimal Hopper varlen FlashAttention forward demo.\n\n"
        "manual_block_count is an optional grid.x thread-block count override. "
        "When omitted, the launch grid is computed automatically by get_grid_shape(...).");
    m.def(
        "_debug_resolve_launch_grid_shape",
        &debug_resolve_launch_grid_shape,
        py::arg("auto_grid_x"),
        py::arg("auto_grid_y") = 1,
        py::arg("auto_grid_z") = 1,
        py::kw_only(),
        py::arg("manual_block_count") = py::none(),
        "Internal CPU-side helper for testing launch-grid override resolution.");
    bind_parallel_remote_load(m);
    bind_min_fa3_backward(m);
    bind_varlen_ring(m);
    // MEGA_RING: explicit default-off multi-step fused ring-attention entry.
    bind_varlen_mega_ring(m);
}
