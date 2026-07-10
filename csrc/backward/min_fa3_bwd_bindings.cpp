// Host binding copied and trimmed from Hopper backward source:
// - hopper/flash_api.cpp

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <limits>
#include <optional>

#include "backward/min_fa3_bwd_params.h"

namespace py = pybind11;

namespace {

using min_fa3_backward::Flash_bwd_params;

int round_multiple(int x, int m) {
    return (x + m - 1) / m * m;
}

void check_sm90() {
    auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        props->major == 9 && props->minor == 0,
        "min_fa3 backward only supports Hopper SM90. Current device capability is ",
        props->major,
        ".",
        props->minor);
}

void check_bf16_cuda(const torch::Tensor& tensor, const char* name, int dim) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must have dtype torch.bfloat16");
    TORCH_CHECK(tensor.dim() == dim, name, " must have ", dim, " dimensions");
    TORCH_CHECK(tensor.size(-1) == 128, name, " must have head_dim D=128");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(const torch::Tensor& reference, const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same CUDA device as q");
}

void check_lse(const torch::Tensor& lse, const torch::Tensor& q) {
    TORCH_CHECK(lse.is_cuda(), "softmax_lse must be a CUDA tensor");
    TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "softmax_lse must have dtype torch.float32");
    TORCH_CHECK(lse.is_contiguous(), "softmax_lse must be contiguous");
    check_same_device(q, lse, "softmax_lse");
}

void check_cu_seqlens(const torch::Tensor& tensor, const torch::Tensor& q, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must have dtype torch.int32");
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() >= 2, name, " must have shape [B + 1]");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    check_same_device(q, tensor, name);
}

torch::Tensor get_grad_output(
    const std::optional<torch::Tensor>& provided,
    const torch::Tensor& like,
    const char* name) {
    if (!provided.has_value()) {
        return torch::empty_like(like);
    }
    const torch::Tensor& grad = provided.value();
    check_bf16_cuda(grad, name, like.dim());
    check_same_device(like, grad, name);
    TORCH_CHECK(grad.sizes() == like.sizes(), name, " must have shape ", like.sizes());
    return grad;
}

void fill_common_params(
    Flash_bwd_params& params,
    const torch::Tensor& dout,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& out,
    const torch::Tensor& softmax_lse,
    torch::Tensor& dq,
    torch::Tensor& dk,
    torch::Tensor& dv,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int total_q,
    int total_k,
    int q_row_dim,
    int q_head_dim,
    bool is_causal,
    bool deterministic) {
    params = {};
    params.is_bf16 = true;

    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.q_row_stride = q.stride(q_row_dim);
    params.k_row_stride = k.stride(q_row_dim);
    params.v_row_stride = v.stride(q_row_dim);
    params.q_head_stride = q.stride(q_head_dim);
    params.k_head_stride = k.stride(q_head_dim);
    params.v_head_stride = v.stride(q_head_dim);

    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(q_row_dim);
    params.o_head_stride = out.stride(q_head_dim);
    params.softmax_lse_ptr = softmax_lse.data_ptr();

    params.do_ptr = dout.data_ptr();
    params.do_row_stride = dout.stride(q_row_dim);
    params.do_head_stride = dout.stride(q_head_dim);
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dq_row_stride = dq.stride(q_row_dim);
    params.dk_row_stride = dk.stride(q_row_dim);
    params.dv_row_stride = dv.stride(q_row_dim);
    params.dq_head_stride = dq.stride(q_head_dim);
    params.dk_head_stride = dk.stride(q_head_dim);
    params.dv_head_stride = dv.stride(q_head_dim);

    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.total_q = total_q;
    params.total_k = total_k;
    params.h = q.size(-2);
    params.h_k = k.size(-2);
    params.d = 128;
    params.d_rounded = 128;
    params.dv = 128;
    params.dv_rounded = 128;
    params.scale_softmax = 1.0f / std::sqrt(128.0f);
    params.window_size_left = seqlen_k - 1;
    params.window_size_right = is_causal ? 0 : seqlen_q - 1;
    params.is_causal = is_causal;
    params.is_local = false;
    params.deterministic = deterministic;

    auto* props = at::cuda::getCurrentDeviceProperties();
    params.arch = props->major * 10 + props->minor;
    params.num_sm = props->multiProcessorCount;
}

py::tuple backward(
    torch::Tensor dout,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor softmax_lse,
    bool is_causal,
    bool deterministic,
    std::optional<torch::Tensor> dq_opt,
    std::optional<torch::Tensor> dk_opt,
    std::optional<torch::Tensor> dv_opt) {
    check_bf16_cuda(q, "q", 4);
    check_bf16_cuda(k, "k", 4);
    check_bf16_cuda(v, "v", 4);
    check_bf16_cuda(out, "out", 4);
    check_bf16_cuda(dout, "dout", 4);
    check_lse(softmax_lse, q);

    check_same_device(q, k, "k");
    check_same_device(q, v, "v");
    check_same_device(q, out, "out");
    check_same_device(q, dout, "dout");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && k.size(1) > 0, "B, Sq, and Sk must be positive");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "q, k, v must have the same B");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v must have the same Sk");
    TORCH_CHECK(k.size(2) == v.size(2), "k and v must have the same KV head count");
    TORCH_CHECK(q.size(2) % k.size(2) == 0, "qhead must be divisible by kvhead");
    TORCH_CHECK(out.sizes() == q.sizes(), "out must have the same shape as q");
    TORCH_CHECK(dout.sizes() == q.sizes(), "dout must have the same shape as q");
    TORCH_CHECK(
        softmax_lse.sizes() == torch::IntArrayRef({q.size(0), q.size(2), q.size(1)}),
        "softmax_lse must have shape [B, QH, Sq]");

    c10::cuda::CUDAGuard device_guard(q.device());
    check_sm90();

    auto dq = get_grad_output(dq_opt, q, "dq");
    auto dk = get_grad_output(dk_opt, k, "dk");
    auto dv = get_grad_output(dv_opt, v, "dv");

    int block_m = is_causal ? 64 : 80;
    int block_n = 128;
    int seqlen_q_rounded = round_multiple(q.size(1), block_m);
    int seqlen_k_rounded = round_multiple(k.size(1), block_n);
    auto float_options = q.options().dtype(torch::kFloat32);
    auto int_options = q.options().dtype(torch::kInt32);
    auto softmax_d = torch::empty({q.size(0), q.size(2), seqlen_q_rounded}, float_options);
    auto softmax_lse_log2 = torch::empty_like(softmax_d);
    auto dq_accum = torch::empty({q.size(0), q.size(2), seqlen_q_rounded * 128}, float_options);
    torch::Tensor dk_accum;
    torch::Tensor dv_accum;
    if (q.size(2) != k.size(2)) {
        dk_accum = torch::zeros({q.size(0), k.size(2), seqlen_k_rounded * 128}, float_options);
        dv_accum = torch::zeros({q.size(0), k.size(2), seqlen_k_rounded * 128}, float_options);
    }
    auto dq_semaphore = torch::empty(
        {(q.size(1) + block_m - 1) / block_m, q.size(0), q.size(2)}, int_options);
    torch::Tensor dk_semaphore;
    torch::Tensor dv_semaphore;
    if (q.size(2) != k.size(2) && deterministic) {
        dk_semaphore = torch::zeros(
            {(k.size(1) + block_n - 1) / block_n, q.size(0), k.size(2)}, int_options);
        dv_semaphore = torch::zeros_like(dk_semaphore);
    }

    Flash_bwd_params params{};
    fill_common_params(
        params, dout, q, k, v, out, softmax_lse, dq, dk, dv,
        q.size(0), q.size(1), k.size(1), q.size(0) * q.size(1), k.size(0) * k.size(1),
        1, 2, is_causal, deterministic);
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);
    params.do_batch_stride = dout.stride(0);
    params.dq_batch_stride = dq.stride(0);
    params.dk_batch_stride = dk.stride(0);
    params.dv_batch_stride = dv.stride(0);
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.dsoftmax_sum = softmax_d.data_ptr();
    params.softmax_lse_log2_ptr = softmax_lse_log2.data_ptr();
    params.dq_accum_ptr = dq_accum.data_ptr();
    params.dk_accum_ptr = dk_accum.defined() ? dk_accum.data_ptr() : nullptr;
    params.dv_accum_ptr = dv_accum.defined() ? dv_accum.data_ptr() : nullptr;
    params.dq_semaphore = dq_semaphore.data_ptr<int>();
    params.dk_semaphore = dk_semaphore.defined() ? dk_semaphore.data_ptr<int>() : nullptr;
    params.dv_semaphore = dv_semaphore.defined() ? dv_semaphore.data_ptr<int>() : nullptr;

    min_fa3_backward::run_min_fa3_bwd(
        params, at::cuda::getCurrentCUDAStream(q.get_device()).stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return py::make_tuple(dq, dk, dv);
}

py::tuple backward_varlen(
    torch::Tensor dout,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor softmax_lse,
    torch::Tensor cu_seqlens_q,
    torch::Tensor cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    bool is_causal,
    bool deterministic,
    std::optional<torch::Tensor> dq_opt,
    std::optional<torch::Tensor> dk_opt,
    std::optional<torch::Tensor> dv_opt) {
    check_bf16_cuda(q, "q", 3);
    check_bf16_cuda(k, "k", 3);
    check_bf16_cuda(v, "v", 3);
    check_bf16_cuda(out, "out", 3);
    check_bf16_cuda(dout, "dout", 3);
    check_lse(softmax_lse, q);
    check_cu_seqlens(cu_seqlens_q, q, "cu_seqlens_q");
    check_cu_seqlens(cu_seqlens_k, q, "cu_seqlens_k");

    check_same_device(q, k, "k");
    check_same_device(q, v, "v");
    check_same_device(q, out, "out");
    check_same_device(q, dout, "dout");
    TORCH_CHECK(q.size(0) > 0 && k.size(0) > 0, "total_q and total_k must be positive");
    TORCH_CHECK(k.size(0) == v.size(0), "k and v must have the same total_k");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v must have the same KV head count");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "qhead must be divisible by kvhead");
    TORCH_CHECK(out.sizes() == q.sizes(), "out must have the same shape as q");
    TORCH_CHECK(dout.sizes() == q.sizes(), "dout must have the same shape as q");
    TORCH_CHECK(cu_seqlens_q.numel() == cu_seqlens_k.numel(), "cu_seqlens_q and cu_seqlens_k must have the same B");
    TORCH_CHECK(max_seqlen_q > 0 && max_seqlen_k > 0, "max seqlens must be positive");
    TORCH_CHECK(
        max_seqlen_q <= std::numeric_limits<int>::max() && max_seqlen_k <= std::numeric_limits<int>::max(),
        "max seqlens must fit int32");
    TORCH_CHECK(
        softmax_lse.sizes() == torch::IntArrayRef({q.size(1), q.size(0)}),
        "softmax_lse must have shape [QH, total_q]");

    c10::cuda::CUDAGuard device_guard(q.device());
    check_sm90();

    auto dq = get_grad_output(dq_opt, q, "dq");
    auto dk = get_grad_output(dk_opt, k, "dk");
    auto dv = get_grad_output(dv_opt, v, "dv");

    int batch_size = cu_seqlens_q.numel() - 1;
    int block_m = is_causal ? 64 : 80;
    int block_n = 128;
    int seqlen_q_rounded = round_multiple(max_seqlen_q, block_m);
    int seqlen_k_rounded = round_multiple(max_seqlen_k, block_n);
    int total_q_padded = round_multiple(q.size(0) + batch_size * block_m, block_m);
    int total_k_padded = round_multiple(k.size(0) + batch_size * block_n, block_n);
    auto float_options = q.options().dtype(torch::kFloat32);
    auto int_options = q.options().dtype(torch::kInt32);
    auto softmax_d = torch::empty({q.size(1), total_q_padded}, float_options);
    auto softmax_lse_log2 = torch::empty_like(softmax_d);
    auto dq_accum = torch::empty({q.size(1), total_q_padded * 128}, float_options);
    torch::Tensor dk_accum;
    torch::Tensor dv_accum;
    if (q.size(1) != k.size(1)) {
        dk_accum = torch::zeros({k.size(1), total_k_padded, 128}, float_options);
        dv_accum = torch::zeros({k.size(1), total_k_padded, 128}, float_options);
    }
    auto dq_semaphore = torch::empty(
        {(max_seqlen_q + block_m - 1) / block_m, batch_size, q.size(1)}, int_options);
    torch::Tensor dk_semaphore;
    torch::Tensor dv_semaphore;
    if (q.size(1) != k.size(1) && deterministic) {
        dk_semaphore = torch::zeros(
            {(max_seqlen_k + block_n - 1) / block_n, batch_size, k.size(1)}, int_options);
        dv_semaphore = torch::zeros_like(dk_semaphore);
    }

    Flash_bwd_params params{};
    fill_common_params(
        params, dout, q, k, v, out, softmax_lse, dq, dk, dv,
        batch_size, max_seqlen_q, max_seqlen_k, q.size(0), k.size(0),
        0, 1, is_causal, deterministic);
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
    params.dsoftmax_sum = softmax_d.data_ptr();
    params.softmax_lse_log2_ptr = softmax_lse_log2.data_ptr();
    params.dq_accum_ptr = dq_accum.data_ptr();
    params.dk_accum_ptr = dk_accum.defined() ? dk_accum.data_ptr() : nullptr;
    params.dv_accum_ptr = dv_accum.defined() ? dv_accum.data_ptr() : nullptr;
    params.dq_semaphore = dq_semaphore.data_ptr<int>();
    params.dk_semaphore = dk_semaphore.defined() ? dk_semaphore.data_ptr<int>() : nullptr;
    params.dv_semaphore = dv_semaphore.defined() ? dv_semaphore.data_ptr<int>() : nullptr;

    min_fa3_backward::run_min_fa3_bwd(
        params, at::cuda::getCurrentCUDAStream(q.get_device()).stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return py::make_tuple(dq, dk, dv);
}

}  // namespace

void bind_min_fa3_backward(py::module_& module) {
    module.def(
        "backward",
        &backward,
        py::arg("dout"),
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("out"),
        py::arg("softmax_lse"),
        py::arg("is_causal"),
        py::kw_only(),
        py::arg("deterministic") = false,
        py::arg("dq") = py::none(),
        py::arg("dk") = py::none(),
        py::arg("dv") = py::none());
    module.def(
        "backward_varlen",
        &backward_varlen,
        py::arg("dout"),
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("out"),
        py::arg("softmax_lse"),
        py::arg("cu_seqlens_q"),
        py::arg("cu_seqlens_k"),
        py::arg("max_seqlen_q"),
        py::arg("max_seqlen_k"),
        py::arg("is_causal"),
        py::kw_only(),
        py::arg("deterministic") = false,
        py::arg("dq") = py::none(),
        py::arg("dk") = py::none(),
        py::arg("dv") = py::none());
}
