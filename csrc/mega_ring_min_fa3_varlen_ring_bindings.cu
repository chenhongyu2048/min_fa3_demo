// Mega ring variant copied and trimmed from csrc/min_fa3_varlen_ring_bindings.cu.
// Changes are marked with MEGA_RING comments.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// MEGA_RING: use the fused multi-step mega-ring launch instead of the
// single-step ring launch.
#include "mega_ring_min_fa3_varlen_ring_launch.h"
#include "min_fa3_varlen_params.h"

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

void check_parallel_varlen_qkv(const kittens::py::TKParallelTensor& t, const char* name) {
    // MEGA_RING: remote K/V are VMM-backed TKParallelTensor buffers, but the
    // attention path still consumes their local tensor view as regular varlen
    // [total_tokens, H, D] storage.
    check_varlen_qkv(t.data_, name);
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

void check_ring_metadata_host(const torch::Tensor& t, int64_t batch_size, const char* name) {
    TORCH_CHECK(!t.is_cuda(), name, " must be a CPU tensor");
    TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must have dtype torch.int32");
    TORCH_CHECK(t.dim() == 1, name, " must be 1D");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.numel() == batch_size, name, " must have shape [B]");
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
                                              int rank_kv_capacity,
                                              torch::Tensor& out,
                                              torch::Tensor& softmax_lse,
                                              torch::Tensor& scheduler_metadata,
                                              bool is_causal,
                                              int num_comp_sm,
                                              int num_comm_sm,
                                              int ring_rank,
                                              int ring_world_size,
                                              int* ring_sizes,
                                              const min_fa3_varlen_demo::MegaRingHierarchyDesc& hierarchy,
                                              int* half_cu_seqlens,
                                              torch::Tensor& kv_ready_counts,
                                              torch::Tensor& step_ready,
                                              torch::Tensor& scan_cursor,
                                              torch::Tensor& completed_tiles,
                                              void* q_descriptor_ptr,
                                              void* o_descriptor_ptr,
                                              void* lse_descriptor_ptr) {
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
    params.mega_ring_rank_kv_capacity = rank_kv_capacity;
    params.mega_ring_ring_sizes = ring_sizes;
    params.mega_ring_hierarchy = hierarchy;
    params.mega_ring_half_cu_seqlens = half_cu_seqlens;
    params.mega_ring_kv_ready_counts = kv_ready_counts.data_ptr<int>();
    params.mega_ring_step_ready = step_ready.data_ptr<int>();
    params.mega_ring_scan_cursor = scan_cursor.data_ptr<int>();
    params.mega_ring_completed_tiles = completed_tiles.data_ptr<int>();
    params.q_ptr = q_descriptor_ptr;
    params.o_ptr = o_descriptor_ptr;
    params.softmax_lse_ptr = lse_descriptor_ptr;
    return params;
}

// MEGA_RING: step counters are indexed by the original per-step varlen Q tile
// id. Each counter tracks how many ring steps have completed for that Q tile.
int64_t compute_tiles_per_step(const int* cu_seqlens_q_host, int batch_size, int q_heads) {
    int64_t tiles = 0;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        int const start = cu_seqlens_q_host[batch_idx];
        int const end = cu_seqlens_q_host[batch_idx + 1];
        TORCH_CHECK(end >= start, "cu_seqlens_q_host must be nondecreasing");
        tiles += ((int64_t(end - start) + 127) / 128) * q_heads;
    }
    return tiles;
}

int64_t compute_tiles_for_batch_range(const int* cu_seqlens_q_host,
                                      int batch_begin,
                                      int batch_end,
                                      int q_heads) {
    int64_t tiles = 0;
    for (int batch_idx = batch_begin; batch_idx < batch_end; ++batch_idx) {
        int const start = cu_seqlens_q_host[batch_idx];
        int const end = cu_seqlens_q_host[batch_idx + 1];
        TORCH_CHECK(end >= start, "cu_seqlens_q_host must be nondecreasing");
        tiles += ((int64_t(end - start) + 127) / 128) * q_heads;
    }
    return tiles;
}

// MEGA_RING: public binding launches all ring steps in one CUDA kernel. The
// caller provides full [world_size * local_total_k, KVH, D] K/V buffers instead
// of src_rank/ring_step plus temporary prefetch buffers.
py::object forward_varlen_mega_ring(torch::Tensor q,
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
                                    int64_t num_comm_sm,
                                    torch::Tensor global_seqlens_host,
                                    torch::Tensor ring_sizes_host,
                                    torch::Tensor ring_starts_host,
                                    py::object out_obj,
                                    py::object lse_obj,
                                    bool return_lse) {
    check_varlen_qkv(q, "q");
    check_varlen_qkv(k, "k");
    check_varlen_qkv(v, "v");
    check_parallel_varlen_qkv(remote_k, "remote_k");
    check_parallel_varlen_qkv(remote_v, "remote_v");
    check_cu_seqlens(cu_seqlens_q, "cu_seqlens_q");
    check_cu_seqlens(cu_seqlens_k, "cu_seqlens_k");
    check_cu_seqlens_host(cu_seqlens_q_host, cu_seqlens_q, "cu_seqlens_q_host");
    check_cu_seqlens_host(cu_seqlens_k_host, cu_seqlens_k, "cu_seqlens_k_host");
    int const batch_size = cu_seqlens_q_host.size(0) - 1;
    check_ring_metadata_host(global_seqlens_host, batch_size, "global_seqlens_host");
    check_ring_metadata_host(ring_sizes_host, batch_size, "ring_sizes_host");
    check_ring_metadata_host(ring_starts_host, batch_size, "ring_starts_host");

    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, v must be on the same CUDA device");
    TORCH_CHECK(q.device() == cu_seqlens_q.device() && q.device() == cu_seqlens_k.device(),
                "q, k, v, cu_seqlens_q, and cu_seqlens_k must be on the same CUDA device");
    TORCH_CHECK(remote_k.data_.device() == q.device(), "remote_k must be created on the same local CUDA device as q");
    TORCH_CHECK(remote_v.data_.device() == q.device(), "remote_v must be created on the same local CUDA device as q");
    TORCH_CHECK(remote_k.data_.sizes().vec() == k.sizes().vec(), "remote_k must have the same shape as local k");
    TORCH_CHECK(remote_v.data_.sizes().vec() == v.sizes().vec(), "remote_v must have the same shape as local v");
    TORCH_CHECK(remote_k.data_.data_ptr() == k.data_ptr(), "k must be remote_k.data_ so compute and IPC use the same arena");
    TORCH_CHECK(remote_v.data_.data_ptr() == v.data_ptr(), "v must be remote_v.data_ so compute and IPC use the same arena");
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
    TORCH_CHECK(remote_k.local_world_size_ == 2 || remote_k.local_world_size_ == 4 || remote_k.local_world_size_ == 8,
                "hierarchical mega ring forward requires world_size in {2, 4, 8}. Got ",
                remote_k.local_world_size_);
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
    TORCH_CHECK(num_comp_sm + num_comm_sm <= props->multiProcessorCount,
                "num_comp_sm + num_comm_sm must not exceed the device SM count (",
                props->multiProcessorCount, "). Got ", num_comp_sm + num_comm_sm);

    int const* cu_seqlens_q_host_ptr = cu_seqlens_q_host.data_ptr<int>();
    int const* cu_seqlens_k_host_ptr = cu_seqlens_k_host.data_ptr<int>();
    int const* global_seqlens_host_ptr = global_seqlens_host.data_ptr<int>();
    int const* ring_sizes_host_ptr = ring_sizes_host.data_ptr<int>();
    int const* ring_starts_host_ptr = ring_starts_host.data_ptr<int>();
    int const world_size = remote_k.local_world_size_;
    int const ring_rank = remote_k.local_rank_;
    TORCH_CHECK(batch_size >= 1, "varlen demo requires batch size B >= 1");
    TORCH_CHECK(cu_seqlens_q_host_ptr[0] == 0, "cu_seqlens_q_host must start with 0");
    TORCH_CHECK(cu_seqlens_k_host_ptr[0] == 0, "cu_seqlens_k_host must start with 0");
    TORCH_CHECK(cu_seqlens_q_host_ptr[batch_size] == q.size(0),
                "cu_seqlens_q_host[-1] must equal q.size(0). Got ", cu_seqlens_q_host_ptr[batch_size],
                " vs ", q.size(0));
    int const local_total_k = cu_seqlens_k_host_ptr[batch_size];
    TORCH_CHECK(cu_seqlens_k_host_ptr[batch_size] >= 0, "cu_seqlens_k_host[-1] must be non-negative");
    TORCH_CHECK(k.size(0) == v.size(0) && k.size(0) % world_size == 0,
                "k and v arena rows must match and be divisible by world_size");
    int64_t const rank_kv_capacity_i64 = k.size(0) / world_size;
    TORCH_CHECK(rank_kv_capacity_i64 > 0 && rank_kv_capacity_i64 <= std::numeric_limits<int>::max(),
                "rank_kv_capacity must be positive and fit in int32. Got ", rank_kv_capacity_i64);
    int const rank_kv_capacity = static_cast<int>(rank_kv_capacity_i64);
    TORCH_CHECK(rank_kv_capacity % 128 == 0,
                "mega ring requires rank_kv_capacity to be 128-row aligned. Got ", rank_kv_capacity);
    TORCH_CHECK(local_total_k <= rank_kv_capacity,
                "local_total_k exceeds rank_kv_capacity. Got ", local_total_k, " vs ", rank_kv_capacity);
    TORCH_CHECK(
        k.size(0) * k.size(1) <= std::numeric_limits<int>::max(),
        "Mega ring varlen path requires total_k * kv_heads to fit in int32. Got total_k=",
        k.size(0),
        ", kv_heads=",
        k.size(1));
    // MEGA_RING_TILE_COPY: every physical 16-row TMA subtile spans the full
    // flattened KVH * D width, which is fixed to 1024 bf16 values.
    TORCH_CHECK(k.size(1) * k.size(2) == 1024,
                "Mega ring communication path currently requires kv_heads * head_dim == 1024. Got kv_heads=",
                k.size(1), ", head_dim=", k.size(2));

    auto half_cu_seqlens_host = torch::zeros({batch_size + 1}, torch::TensorOptions().dtype(torch::kInt32));
    int* half_host_ptr = half_cu_seqlens_host.data_ptr<int>();
    int previous_ring_size = 8;
    int max_local_q = 0;
    int max_local_k = 0;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        int const q_len = cu_seqlens_q_host_ptr[batch_idx + 1] - cu_seqlens_q_host_ptr[batch_idx];
        int const k_len = cu_seqlens_k_host_ptr[batch_idx + 1] - cu_seqlens_k_host_ptr[batch_idx];
        TORCH_CHECK(q_len >= 0 && k_len >= 0, "cu_seqlens must be nondecreasing at batch=", batch_idx);
        TORCH_CHECK(q_len % 128 == 0 && k_len % 128 == 0,
                    "mega ring requires every local q/k length to be 128-row aligned at batch=",
                    batch_idx, ". q_len=", q_len, ", k_len=", k_len);
        int const global_len = global_seqlens_host_ptr[batch_idx];
        int const ring_size = ring_sizes_host_ptr[batch_idx];
        int const ring_start = ring_starts_host_ptr[batch_idx];
        TORCH_CHECK(global_len > 0, "global_seqlens_host must be positive at batch=", batch_idx);
        TORCH_CHECK(ring_size == 1 || ring_size == 2 || ring_size == 4 || ring_size == 8,
                    "ring_sizes_host must contain only 1, 2, 4, or 8. batch=", batch_idx, ", value=", ring_size);
        TORCH_CHECK(ring_size <= previous_ring_size,
                    "batch must be ordered by non-increasing ring size. batch=", batch_idx);
        previous_ring_size = ring_size;
        TORCH_CHECK(ring_start >= 0 && ring_start % ring_size == 0 && ring_start + ring_size <= world_size,
                    "invalid aligned ring range at batch=", batch_idx, ": start=", ring_start, ", size=", ring_size);
        TORCH_CHECK(global_len % ring_size == 0,
                    "global length must be divisible by ring size at batch=", batch_idx);
        bool const is_member = ring_rank >= ring_start && ring_rank < ring_start + ring_size;
        int const expected_local_len = is_member ? global_len / ring_size : 0;
        TORCH_CHECK(q_len == expected_local_len && k_len == expected_local_len,
                    "local q/k length does not match hierarchical ring metadata at batch=", batch_idx,
                    ". expected=", expected_local_len, ", q_len=", q_len, ", k_len=", k_len);
        int half_len = 0;
        if (is_causal && ring_size > 1 && is_member) {
            TORCH_CHECK(q_len % 2 == 0, "causal CP local length must be even at batch=", batch_idx);
            half_len = q_len / 2;
            TORCH_CHECK(half_len % 128 == 0,
                        "causal CP half length must be 128-aligned at batch=", batch_idx,
                        ". half_len=", half_len);
        }
        half_host_ptr[batch_idx + 1] = half_host_ptr[batch_idx] + half_len;
        max_local_q = std::max(max_local_q, q_len);
        max_local_k = std::max(max_local_k, k_len);
    }
    TORCH_CHECK(max_local_q <= max_seqlen_q && max_local_k <= max_seqlen_k,
                "max_seqlen_q/max_seqlen_k must cover every local sequence");

    auto ring_sizes = torch::empty({batch_size}, q.options().dtype(torch::kInt32));
    ring_sizes.copy_(ring_sizes_host, false);
    torch::Tensor half_cu_seqlens;
    if (is_causal) {
        half_cu_seqlens = torch::empty({batch_size + 1}, q.options().dtype(torch::kInt32));
        half_cu_seqlens.copy_(half_cu_seqlens_host, false);
    }

    min_fa3_varlen_demo::MegaRingHierarchyDesc hierarchy{};
    constexpr int kRingSizes[4] = {8, 4, 2, 1};
    constexpr int kKvReadyBases[4] = {0, 7, 10, 11};
    int batch_cursor = 0;
    int64_t reduction_tiles = 0;
    int64_t remote_tiles = 0;
    int64_t total_comm_tasks = 0;
    int64_t const base_work_tiles = compute_tiles_per_step(cu_seqlens_q_host_ptr, batch_size, q.size(1));
    int64_t total_work_tiles = base_work_tiles;
    for (int level_idx = 0; level_idx < 4; ++level_idx) {
        int const ring_size = kRingSizes[level_idx];
        int const batch_begin = batch_cursor;
        while (batch_cursor < batch_size && ring_sizes_host_ptr[batch_cursor] == ring_size) {
            ++batch_cursor;
        }
        int const batch_end = batch_cursor;
        int64_t const full_tiles = compute_tiles_for_batch_range(
            cu_seqlens_q_host_ptr, batch_begin, batch_end, q.size(1));
        int64_t const half_tiles = is_causal
            ? compute_tiles_for_batch_range(half_host_ptr, batch_begin, batch_end, q.size(1))
            : 0;
        TORCH_CHECK(full_tiles <= std::numeric_limits<int>::max() && half_tiles <= std::numeric_limits<int>::max(),
                    "per-level tile count must fit in int32");
        auto& level = hierarchy.levels[level_idx];
        level.ring_size = ring_size;
        level.batch_begin = batch_begin;
        level.batch_end = batch_end;
        level.row_begin = cu_seqlens_k_host_ptr[batch_begin];
        level.full_rows = cu_seqlens_k_host_ptr[batch_end] - level.row_begin;
        level.half_row_begin = half_host_ptr[batch_begin];
        level.half_rows = half_host_ptr[batch_end] - level.half_row_begin;
        level.full_tiles = static_cast<int>(full_tiles);
        level.half_tiles = static_cast<int>(half_tiles);
        level.reduction_base = static_cast<int>(reduction_tiles);
        level.kv_ready_base = kKvReadyBases[level_idx];
        if (ring_size > 1) {
            int const ring_base = (ring_rank / ring_size) * ring_size;
            int const ring_local_rank = ring_rank - ring_base;
            total_work_tiles += is_causal
                ? int64_t(ring_local_rank) * full_tiles + int64_t(ring_size - 1 - ring_local_rank) * half_tiles
                : int64_t(ring_size - 1) * full_tiles;
            if (is_causal) {
                // Back-half Q tiles consume every remote source. Front-half Q
                // tiles consume the lower-rank half-KV segments when rank > 0.
                remote_tiles += half_tiles * (ring_local_rank > 0 ? 2 : 1);
            }
            int const kv_block_n = is_causal ? 128 : 176;
            int64_t const full_kv_tiles = (int64_t(level.full_rows) + kv_block_n - 1) / kv_block_n;
            int64_t const half_kv_tiles = (int64_t(level.half_rows) + kv_block_n - 1) / kv_block_n;
            total_comm_tasks += is_causal
                ? 2 * (int64_t(ring_local_rank) * half_kv_tiles
                       + int64_t(ring_size - 1 - ring_local_rank) * full_kv_tiles)
                : 2 * int64_t(ring_size - 1) * full_kv_tiles;
            reduction_tiles += full_tiles;
        }
    }
    TORCH_CHECK(batch_cursor == batch_size, "failed to partition all batches into hierarchical ring levels");
    TORCH_CHECK(base_work_tiles <= std::numeric_limits<int>::max()
                    && total_work_tiles <= std::numeric_limits<int>::max()
                    && reduction_tiles <= std::numeric_limits<int>::max()
                    && remote_tiles <= std::numeric_limits<int>::max(),
                "mega-ring total work and reduction tile counts must fit in int32");
    TORCH_CHECK(total_comm_tasks <= std::numeric_limits<int>::max(),
                "mega-ring communication task count must fit in int32");
    hierarchy.base_work_tiles = static_cast<int>(base_work_tiles);
    hierarchy.total_work_tiles = static_cast<int>(total_work_tiles);
    hierarchy.reduction_tiles = static_cast<int>(reduction_tiles);
    hierarchy.remote_tiles = static_cast<int>(remote_tiles);
    TORCH_CHECK(num_comm_sm > 0 || reduction_tiles == 0,
                "num_comm_sm must be positive when this rank has G8/G4/G2 replay work");

    auto out = resolve_out(out_obj, q);
    auto lse = resolve_lse(lse_obj, q);
    // The reduction epilogue treats O/LSE as running state starting at
    // (zero, -inf), including step 0 and caller-provided output buffers.
    out.zero_();
    lse.fill_(-std::numeric_limits<float>::infinity());

    int b_rounded = round_multiple(batch_size, 4);
    bool varlen_sort_batches = true;
    bool head_swizzle = is_causal;
    int num_prepare_batch_vectors = 2 + (varlen_sort_batches ? 1 : 0) + (head_swizzle ? 1 : 0);
    int metadata_size = 1 + b_rounded * num_prepare_batch_vectors;
    auto scheduler_metadata = torch::empty({metadata_size}, q.options().dtype(torch::kInt32));

    auto kv_ready_counts = torch::zeros({11}, q.options().dtype(torch::kInt32));
    auto step_ready = torch::zeros({std::max<int64_t>(reduction_tiles, 1)}, q.options().dtype(torch::kInt32));
    auto scan_cursor = torch::zeros({1}, q.options().dtype(torch::kInt32));
    auto completed_tiles = torch::zeros({1}, q.options().dtype(torch::kInt32));

    torch::Tensor q_descriptor = q;
    torch::Tensor out_descriptor = out;
    torch::Tensor lse_descriptor = lse;
    if (q.size(0) == 0) {
        q_descriptor = torch::empty({1, q.size(1), q.size(2)}, q.options());
        out_descriptor = torch::empty_like(q_descriptor);
        lse_descriptor = torch::empty({q.size(1), 1}, q.options().dtype(torch::kFloat));
    }
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

    auto params = make_mega_ring_varlen_params(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        static_cast<int>(max_seqlen_q),
        static_cast<int>(max_seqlen_k),
        rank_kv_capacity,
        out,
        lse,
        scheduler_metadata,
        is_causal,
        static_cast<int>(num_comp_sm),
        static_cast<int>(num_comm_sm),
        ring_rank,
        world_size,
        ring_sizes.data_ptr<int>(),
        hierarchy,
        is_causal ? static_cast<int*>(half_cu_seqlens.data_ptr()) : nullptr,
        kv_ready_counts,
        step_ready,
        scan_cursor,
        completed_tiles,
        q_descriptor.data_ptr(),
        out_descriptor.data_ptr(),
        lse_descriptor.data_ptr());

    // MEGA_RING: one fused launch runs communication CTAs and compute CTAs.
    min_fa3_varlen_demo::run_mega_ring_min_fa3_varlen_ring_fwd(params, remote_k, remote_v, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (return_lse) {
        return py::make_tuple(out, lse);
    }
    return py::cast(out);
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
        py::arg("global_seqlens_host"),
        py::arg("ring_sizes_host"),
        py::arg("ring_starts_host"),
        py::arg("out") = py::none(),
        py::arg("lse") = py::none(),
        py::arg("return_lse") = false,
        "MEGA_RING: hierarchical fused Hopper varlen ring-attention forward.\n\n"
        "K/V must be contiguous [world_size * rank_kv_capacity, 8, 128] arenas. "
        "The kernel performs persistent compute and remote K/V TMA loads in one launch.");
}
