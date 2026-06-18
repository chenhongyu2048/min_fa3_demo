#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <limits>

#include "min_fa3_varlen_params.h"
#include "min_fa3_varlen_ring_launch.h"

namespace py = pybind11;

namespace {

using RingVarlenParams = min_fa3_varlen_demo::Ring_fwd_params;
using VarlenParams = min_fa3_varlen_demo::Flash_fwd_params;

int round_multiple(int x, int m) {
    return (x + m - 1) / m * m;
}

void check_varlen_qkv(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must have dtype torch.bfloat16");
    TORCH_CHECK(t.dim() == 3, name, " must have shape [total_tokens, H, D]");
    TORCH_CHECK(t.size(2) == 128, name, " must have head_dim D=128");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous [total_tokens, H, D]");
}

void check_parallel_varlen_base(const kittens::py::TKParallelTensor& t, const char* name) {
    TORCH_CHECK(t.data_.is_cuda(), name, " must wrap a CUDA tensor");
    TORCH_CHECK(t.data_.scalar_type() == torch::kBFloat16, name, " must wrap dtype torch.bfloat16");
    TORCH_CHECK(t.data_.dim() == 3, name, " must wrap a [rows, H, D] tensor");
    TORCH_CHECK(t.data_.size(2) == 128, name, " must wrap head_dim D=128");
    TORCH_CHECK(t.data_.is_contiguous(), name, " must wrap a contiguous tensor");
}

void check_prefetch_buffer(const torch::Tensor& t,
                           const torch::Tensor& ref,
                           const char* name) {
    check_varlen_qkv(t, name);
    TORCH_CHECK(t.device() == ref.device(), name, " must be on the same CUDA device as its reference tensor");
    TORCH_CHECK(t.sizes().vec() == ref.sizes().vec(), name, " must have the same shape as its reference tensor");
}

void check_out(const torch::Tensor& t, const torch::Tensor& q, const char* name) {
    check_varlen_qkv(t, name);
    TORCH_CHECK(t.device() == q.device(), name, " must be on the same CUDA device as q");
    TORCH_CHECK(t.sizes().vec() == q.sizes().vec(), name, " must have the same shape as q");
}

void check_lse(const torch::Tensor& t, const torch::Tensor& q, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat, name, " must have dtype torch.float32");
    TORCH_CHECK(t.dim() == 2, name, " must have shape [qhead, total_q]");
    TORCH_CHECK(t.size(0) == q.size(1) && t.size(1) == q.size(0),
                name, " must have shape [qhead, total_q]");
    TORCH_CHECK(t.device() == q.device(), name, " must be on the same CUDA device as q");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

torch::Tensor resolve_out(py::object out_obj, const torch::Tensor& q) {
    if (out_obj.is_none()) {
        return torch::zeros_like(q);
    }
    auto out = out_obj.cast<torch::Tensor>();
    check_out(out, q, "out");
    return out;
}

torch::Tensor resolve_lse(py::object lse_obj, const torch::Tensor& q) {
    if (lse_obj.is_none()) {
        return torch::full({q.size(1), q.size(0)}, -std::numeric_limits<float>::infinity(), q.options().dtype(torch::kFloat));
    }
    auto lse = lse_obj.cast<torch::Tensor>();
    check_lse(lse, q, "lse");
    return lse;
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

RingVarlenParams make_ring_varlen_params(const torch::Tensor& q,
                                         const torch::Tensor& k,
                                         const torch::Tensor& v,
                                         const torch::Tensor& cu_seqlens_q,
                                         const torch::Tensor& cu_seqlens_k,
                                         int max_seqlen_q,
                                         int max_seqlen_k,
                                         torch::Tensor& out,
                                         torch::Tensor& softmax_lse,
                                         torch::Tensor& scheduler_metadata,
                                         bool is_causal,
                                         int num_comp_sm,
                                         int num_comm_sm,
                                         int src_dev,
                                         int ring_rank,
                                         int ring_world_size,
                                         int ring_step) {
    RingVarlenParams params{};
    static_cast<VarlenParams&>(params) = make_varlen_params(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        out,
        softmax_lse,
        scheduler_metadata,
        is_causal);
    params.num_comp_sm = num_comp_sm;
    params.num_comm_sm = num_comm_sm;
    params.src_dev = src_dev;
    params.ring_rank = ring_rank;
    params.ring_world_size = ring_world_size;
    params.ring_step = ring_step;
    return params;
}

py::object forward_varlen_ring(torch::Tensor q,
                               torch::Tensor k,
                               torch::Tensor v,
                               kittens::py::TKParallelTensor& remote_k,
                               kittens::py::TKParallelTensor& remote_v,
                               torch::Tensor cu_seqlens_q,
                               torch::Tensor cu_seqlens_k,
                               torch::Tensor cu_seqlens_q_host,
                               torch::Tensor cu_seqlens_k_host,
                               int64_t max_seqlen_q,
                               int64_t max_seqlen_k,
                               bool is_causal,
                               int64_t src_rank,
                               int64_t num_comp_sm,
                               int64_t num_comm_sm,
                               int64_t ring_step,
                               torch::Tensor prefetch_k,
                               torch::Tensor prefetch_v,
                               py::object out_obj,
                               py::object lse_obj,
                               bool return_lse) {
    check_varlen_qkv(q, "q");
    check_varlen_qkv(k, "k");
    check_varlen_qkv(v, "v");
    check_parallel_varlen_base(remote_k, "remote_k");
    check_parallel_varlen_base(remote_v, "remote_v");
    check_cu_seqlens(cu_seqlens_q, "cu_seqlens_q");
    check_cu_seqlens(cu_seqlens_k, "cu_seqlens_k");
    check_cu_seqlens_host(cu_seqlens_q_host, cu_seqlens_q, "cu_seqlens_q_host");
    check_cu_seqlens_host(cu_seqlens_k_host, cu_seqlens_k, "cu_seqlens_k_host");

    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, v must be on the same CUDA device");
    TORCH_CHECK(q.device() == cu_seqlens_q.device() && q.device() == cu_seqlens_k.device(),
                "q, k, v, cu_seqlens_q, and cu_seqlens_k must be on the same CUDA device");
    TORCH_CHECK(remote_k.data_.device() == q.device(), "remote_k must be created on the same local CUDA device as q");
    TORCH_CHECK(remote_v.data_.device() == q.device(), "remote_v must be created on the same local CUDA device as q");
    TORCH_CHECK(remote_k.data_.sizes().vec() == k.sizes().vec(), "remote_k must have the same shape as local k");
    TORCH_CHECK(remote_v.data_.sizes().vec() == v.sizes().vec(), "remote_v must have the same shape as local v");
    TORCH_CHECK(remote_k.data_.data_ptr() != remote_v.data_.data_ptr(),
                "remote_k and remote_v must be separate TKParallelTensor allocations");
    TORCH_CHECK(remote_k.local_rank_ == q.get_device(), "remote_k local_rank must match q.device.index");
    TORCH_CHECK(remote_v.local_rank_ == q.get_device(), "remote_v local_rank must match q.device.index");
    TORCH_CHECK(remote_k.local_world_size_ == remote_v.local_world_size_,
                "remote_k and remote_v must have the same local_world_size");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v must have the same KV head count");
    TORCH_CHECK(q.size(1) % k.size(1) == 0,
                "This demo requires qhead % kvhead == 0 for GQA/MQA. Got qhead=",
                q.size(1), ", kvhead=", k.size(1));
    TORCH_CHECK(v.size(2) == 128, "v must have head_dim D=128");
    TORCH_CHECK(cu_seqlens_q.size(0) == cu_seqlens_k.size(0), "cu_seqlens_q and cu_seqlens_k must have the same length");
    TORCH_CHECK(max_seqlen_q > 0 && max_seqlen_k > 0, "max_seqlen_q and max_seqlen_k must be positive");
    TORCH_CHECK(max_seqlen_q <= std::numeric_limits<int>::max() && max_seqlen_k <= std::numeric_limits<int>::max(),
                "max seqlens must fit in int32");
    TORCH_CHECK(num_comp_sm > 0, "num_comp_sm must be positive. Got ", num_comp_sm);
    TORCH_CHECK(num_comm_sm >= 0, "num_comm_sm must be non-negative. Got ", num_comm_sm);
    TORCH_CHECK(src_rank >= 0 && src_rank < remote_k.local_world_size_,
                "src_rank must be in [0, local_world_size). Got src_rank=",
                src_rank,
                ", local_world_size=",
                remote_k.local_world_size_);
    TORCH_CHECK(ring_step >= 0, "ring_step must be non-negative. Got ", ring_step);
    TORCH_CHECK(num_comp_sm <= std::numeric_limits<int>::max() && num_comm_sm <= std::numeric_limits<int>::max(),
                "num_comp_sm and num_comm_sm must fit in int32");
    TORCH_CHECK(
        q.size(0) <= std::numeric_limits<int>::max() &&
            k.size(0) <= std::numeric_limits<int>::max() &&
            q.size(1) <= std::numeric_limits<int>::max() &&
            k.size(1) <= std::numeric_limits<int>::max(),
        "Ring varlen path requires q/k token and head counts to fit in int32");
    TORCH_CHECK(
        k.size(0) * k.size(1) <= std::numeric_limits<int>::max(),
        "Ring varlen path requires total_k * kv_heads to fit in int32. Got total_k=",
        k.size(0),
        ", kv_heads=",
        k.size(1));
    TORCH_CHECK(
        prefetch_k.defined() == prefetch_v.defined(),
        "prefetch_k and prefetch_v must either both be provided or both be omitted");
    if (prefetch_k.defined()) {
        check_prefetch_buffer(prefetch_k, k, "prefetch_k");
        check_prefetch_buffer(prefetch_v, v, "prefetch_v");
    }

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

    auto out = resolve_out(out_obj, q);
    auto lse = resolve_lse(lse_obj, q);

    int b_rounded = round_multiple(batch_size, 4);
    bool varlen_sort_batches = true;
    bool head_swizzle = is_causal;
    int num_prepare_batch_vectors = 2 + (varlen_sort_batches ? 1 : 0) + (head_swizzle ? 1 : 0);
    int metadata_size = 1 + b_rounded * num_prepare_batch_vectors;
    auto scheduler_metadata = torch::empty({metadata_size}, q.options().dtype(torch::kInt32));

    auto k_staging = torch::Tensor();
    auto v_staging = torch::Tensor();
    if (num_comm_sm > 0) {
        k_staging = prefetch_k.defined() ? prefetch_k : torch::empty_like(k);
        v_staging = prefetch_v.defined() ? prefetch_v : torch::empty_like(v);
    }

    auto params = make_ring_varlen_params(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        static_cast<int>(max_seqlen_q),
        static_cast<int>(max_seqlen_k),
        out,
        lse,
        scheduler_metadata,
        is_causal,
        static_cast<int>(num_comp_sm),
        static_cast<int>(num_comm_sm),
        static_cast<int>(src_rank),
        remote_k.local_rank_,
        remote_k.local_world_size_,
        static_cast<int>(ring_step));
    if (num_comm_sm > 0) {
        params.local_k_staging_ptr = k_staging.data_ptr();
        params.local_v_staging_ptr = v_staging.data_ptr();
    }

    auto stream = at::cuda::getDefaultCUDAStream(q.get_device());
    if (num_comm_sm > 0) {
        k_staging.record_stream(stream);
        v_staging.record_stream(stream);
    }
    min_fa3_varlen_demo::run_min_fa3_varlen_ring_fwd(params, remote_k, remote_v, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (return_lse) {
        return py::make_tuple(out, lse);
    }
    return py::cast(out);
}

}  // namespace

void bind_varlen_ring(py::module_& m) {
    m.def(
        "forward_varlen_ring",
        &forward_varlen_ring,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("remote_k"),
        py::arg("remote_v"),
        py::arg("cu_seqlens_q"),
        py::arg("cu_seqlens_k"),
        py::arg("cu_seqlens_q_host"),
        py::arg("cu_seqlens_k_host"),
        py::arg("max_seqlen_q"),
        py::arg("max_seqlen_k"),
        py::arg("is_causal"),
        py::arg("src_rank"),
        py::arg("num_comp_sm"),
        py::arg("num_comm_sm"),
        py::arg("ring_step"),
        py::arg("prefetch_k"),
        py::arg("prefetch_v"),
        py::arg("out") = py::none(),
        py::arg("lse") = py::none(),
        py::arg("return_lse") = false,
        "Minimal Hopper varlen ring-attention demo.\n\n"
        "Compute CTAs read the current local K/V buffer while communication CTAs optionally prefetch the next K/V "
        "buffer from src_rank into prefetch_k/prefetch_v.");
}
