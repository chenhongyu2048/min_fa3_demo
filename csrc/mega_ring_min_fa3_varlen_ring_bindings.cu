// Mega ring variant copied and trimmed from csrc/min_fa3_varlen_ring_bindings.cu.
// Changes are marked with MEGA_RING comments.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <limits>

// MEGA_RING: use the fused multi-step mega-ring launch instead of the
// single-step ring launch.
#include "mega_ring_min_fa3_varlen_ring_launch.h"
#include "min_fa3_varlen_params.h"

namespace py = pybind11;

namespace {

using RingVarlenParams = min_fa3_varlen_demo::Ring_fwd_params;
using VarlenParams = min_fa3_varlen_demo::Flash_fwd_params;

__global__ void set_mega_ring_local_kv_ready_count(int* counts, int rank, int value) {
    counts[rank] = value;
}

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

void check_parallel_varlen_qkv(const kittens::py::TKParallelTensor& t, const char* name) {
    // MEGA_RING: remote K/V are VMM-backed TKParallelTensor buffers, but the
    // attention path still consumes their local tensor view as regular varlen
    // [total_tokens, H, D] storage.
    check_varlen_qkv(t.data_, name);
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
    // MEGA_RING: shape_K describes the full local [world_size * local_total_k]
    // buffer, while cu_seqlens_k still describes one per-rank KV block.
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

// MEGA_RING: keep the original ring params type and populate the extra
// multi-step scheduler/readiness fields required by the fused launch.
RingVarlenParams make_mega_ring_varlen_params(const torch::Tensor& q,
                                              const torch::Tensor& k,
                                              const torch::Tensor& v,
                                              const torch::Tensor& cu_seqlens_q,
                                              const torch::Tensor& cu_seqlens_k,
                                              int max_seqlen_q,
                                              int max_seqlen_k,
                                              int local_total_k,
                                              torch::Tensor& out,
                                              torch::Tensor& softmax_lse,
                                              torch::Tensor& scheduler_metadata,
                                              bool is_causal,
                                              int num_comp_sm,
                                              int num_comm_sm,
                                              int ring_rank,
                                              int ring_world_size,
                                              int tiles_per_step,
                                              torch::Tensor& kv_ready_counts,
                                              torch::Tensor& step_ready) {
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
    // MEGA_RING: the launch covers all ring steps. Source rank selection is
    // derived in-kernel from ring_rank, ring_world_size, and ring_step.
    params.src_dev = 0;
    params.ring_rank = ring_rank;
    params.ring_world_size = ring_world_size;
    params.ring_step = 0;
    params.mega_ring_tiles_per_step = tiles_per_step;
    params.mega_ring_total_k_per_rank = local_total_k;
    params.mega_ring_kv_ready_counts = kv_ready_counts.data_ptr<int>();
    params.mega_ring_step_ready = step_ready.data_ptr<int>();
    return params;
}

// MEGA_RING: step counters are indexed by the original per-step varlen Q tile
// id. Each counter tracks how many ring steps have completed for that Q tile.
int compute_tiles_per_step(const int* cu_seqlens_q_host, int batch_size, int q_heads) {
    int tiles = 0;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        int const start = cu_seqlens_q_host[batch_idx];
        int const end = cu_seqlens_q_host[batch_idx + 1];
        TORCH_CHECK(end >= start, "cu_seqlens_q_host must be nondecreasing");
        tiles += round_multiple(end - start, 128) / 128 * q_heads;
    }
    return tiles;
}

// MEGA_RING: public binding launches all ring steps in one CUDA kernel. The
// caller provides full [world_size * local_total_k, KVH, D] K/V buffers instead
// of src_rank/ring_step plus temporary prefetch buffers.
torch::Tensor forward_varlen_mega_ring(torch::Tensor q,
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
                                       int64_t num_comp_sm,
                                       int64_t num_comm_sm) {
    check_varlen_qkv(q, "q");
    check_varlen_qkv(k, "k");
    check_varlen_qkv(v, "v");
    check_parallel_varlen_qkv(remote_k, "remote_k");
    check_parallel_varlen_qkv(remote_v, "remote_v");
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
    // MEGA_RING: K and V are separate VMM-backed tensors, but they must describe
    // the same local rank in the same local ring.
    TORCH_CHECK(remote_k.local_rank_ == q.get_device(), "remote_k local_rank must match q.device.index");
    TORCH_CHECK(remote_v.local_rank_ == q.get_device(), "remote_v local_rank must match q.device.index");
    TORCH_CHECK(remote_k.local_world_size_ == remote_v.local_world_size_,
                "remote_k and remote_v must have the same local_world_size");
    TORCH_CHECK(remote_k.local_rank_ == remote_v.local_rank_,
                "remote_k and remote_v must have the same local_rank");
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
    TORCH_CHECK(num_comp_sm <= std::numeric_limits<int>::max() && num_comm_sm <= std::numeric_limits<int>::max(),
                "num_comp_sm and num_comm_sm must fit in int32");

    c10::cuda::CUDAGuard device_guard(q.device());
    auto* props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 9 && props->minor == 0,
                "min_fa3_demo only supports Hopper SM90. Current device capability is ",
                props->major, ".", props->minor);

    int const batch_size = cu_seqlens_q_host.size(0) - 1;
    int const* cu_seqlens_q_host_ptr = cu_seqlens_q_host.data_ptr<int>();
    int const* cu_seqlens_k_host_ptr = cu_seqlens_k_host.data_ptr<int>();
    // MEGA_RING: ring identity is taken from the TKParallelTensor allocation.
    int const world_size = remote_k.local_world_size_;
    int const ring_rank = remote_k.local_rank_;
    TORCH_CHECK(batch_size >= 1, "varlen demo requires batch size B >= 1");
    TORCH_CHECK(cu_seqlens_q_host_ptr[0] == 0, "cu_seqlens_q_host must start with 0");
    TORCH_CHECK(cu_seqlens_k_host_ptr[0] == 0, "cu_seqlens_k_host must start with 0");
    TORCH_CHECK(cu_seqlens_q_host_ptr[batch_size] == q.size(0),
                "cu_seqlens_q_host[-1] must equal q.size(0). Got ", cu_seqlens_q_host_ptr[batch_size],
                " vs ", q.size(0));
    // MEGA_RING: cu_seqlens_k describes one rank-local KV block, while K/V
    // storage contains one such block for every rank in local rank order.
    int const local_total_k = cu_seqlens_k_host_ptr[batch_size];
    TORCH_CHECK(local_total_k > 0, "cu_seqlens_k_host[-1] must be positive");
    TORCH_CHECK(k.size(0) == int64_t(world_size) * local_total_k,
                "mega ring k must have shape [world_size * local_total_k, KVH, D]. Got k.size(0)=",
                k.size(0), ", world_size=", world_size, ", local_total_k=", local_total_k);
    TORCH_CHECK(v.size(0) == int64_t(world_size) * local_total_k,
                "mega ring v must have shape [world_size * local_total_k, KVH, D]. Got v.size(0)=",
                v.size(0), ", world_size=", world_size, ", local_total_k=", local_total_k);
    TORCH_CHECK(
        k.size(0) * k.size(1) <= std::numeric_limits<int>::max(),
        "Mega ring varlen path requires total_k * kv_heads to fit in int32. Got total_k=",
        k.size(0),
        ", kv_heads=",
        k.size(1));
    // MEGA_RING: the current remote-load vector path copies a full KV row whose
    // flattened KVH * D width is fixed to 1024 bf16 values.
    TORCH_CHECK(k.size(1) * k.size(2) == 1024,
                "Mega ring communication path currently requires kv_heads * head_dim == 1024. Got kv_heads=",
                k.size(1), ", head_dim=", k.size(2));
    TORCH_CHECK(world_size == 1 || num_comm_sm > 0,
                "mega ring requires num_comm_sm > 0 when world_size > 1");

    auto out = torch::zeros_like(q);
    auto lse = torch::full({q.size(1), q.size(0)}, -std::numeric_limits<float>::infinity(), q.options().dtype(torch::kFloat));

    int b_rounded = round_multiple(batch_size, 4);
    bool varlen_sort_batches = true;
    bool head_swizzle = is_causal;
    int num_prepare_batch_vectors = 2 + (varlen_sort_batches ? 1 : 0) + (head_swizzle ? 1 : 0);
    int metadata_size = 1 + b_rounded * num_prepare_batch_vectors;
    auto scheduler_metadata = torch::empty({metadata_size}, q.options().dtype(torch::kInt32));

    // MEGA_RING: device counters coordinate remote K/V availability and
    // per-Q-tile reduction ordering across ring steps.
    int const tiles_per_step = compute_tiles_per_step(cu_seqlens_q_host_ptr, batch_size, q.size(1));
    TORCH_CHECK(tiles_per_step > 0, "mega ring requires at least one Q tile per step");
    auto kv_ready_counts = torch::zeros({world_size}, q.options().dtype(torch::kInt32));
    auto step_ready = torch::zeros({tiles_per_step}, q.options().dtype(torch::kInt32));

    // MEGA_RING: initial local rank block is already resident in the local
    // concatenated K/V buffer.
    auto stream = at::cuda::getDefaultCUDAStream(q.get_device());
    set_mega_ring_local_kv_ready_count<<<1, 1, 0, stream>>>(
        kv_ready_counts.data_ptr<int>(), ring_rank, local_total_k * 2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto params = make_mega_ring_varlen_params(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        static_cast<int>(max_seqlen_q),
        static_cast<int>(max_seqlen_k),
        local_total_k,
        out,
        lse,
        scheduler_metadata,
        is_causal,
        static_cast<int>(num_comp_sm),
        static_cast<int>(num_comm_sm),
        ring_rank,
        world_size,
        tiles_per_step,
        kv_ready_counts,
        step_ready);

    // MEGA_RING: one fused launch runs communication CTAs and compute CTAs.
    min_fa3_varlen_demo::run_mega_ring_min_fa3_varlen_ring_fwd(params, remote_k, remote_v, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace

void bind_varlen_mega_ring(py::module_& m) {
    m.def(
        "forward_varlen_mega_ring",
        &forward_varlen_mega_ring,
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
        py::arg("num_comp_sm"),
        py::arg("num_comm_sm"),
        "MEGA_RING: explicit multi-step fused Hopper varlen ring-attention demo.\n\n"
        "K/V must be contiguous [world_size * local_total_k, kv_heads, 128] buffers. "
        "The kernel performs persistent compute and remote K/V TMA loads in one launch.");
}
