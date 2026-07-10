// Mega ring variant copied and trimmed from csrc/min_fa3_varlen_ring_bindings.cu.
// Changes are marked with MEGA_RING comments.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <algorithm>
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

__global__ void set_mega_ring_local_kv_ready_count(int* counts, int rank, int value) {
    counts[rank] = value;
}

__device__ int find_batch_for_row(const int* cu_seqlens, int batch_size, int row) {
    int lo = 0;
    int hi = batch_size;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (cu_seqlens[mid] <= row) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__device__ void add_ready_chunk(int* chunk_done,
                                int interval_id,
                                int row_in_interval,
                                int max_chunks,
                                int chunk_rows,
                                int value) {
    int const chunk_idx = row_in_interval / chunk_rows;
    atomicAdd(chunk_done + interval_id * max_chunks + chunk_idx, value);
}

__global__ void pack_ready_once_local_kv_kernel(const uint4* src_k,
                                                const uint4* src_v,
                                                uint4* dst_k,
                                                uint4* dst_v,
                                                const int* cu_src,
                                                const int* cu_compact,
                                                const int* cp_batch_mask,
                                                const int* half_cu,
                                                int batch_size,
                                                int local_total,
                                                int local_rank,
                                                int world_size,
                                                bool is_causal,
                                                int vecs_per_row,
                                                int* chunk_done,
                                                int ready_max_chunks,
                                                int ready_chunk_rows) {
    int const local_row_global = int(blockIdx.x);
    if (local_row_global >= local_total) {
        return;
    }
    int const batch_idx = find_batch_for_row(cu_src, batch_size, local_row_global);
    int const row_in_batch = local_row_global - cu_src[batch_idx];
    int const local_len = cu_src[batch_idx + 1] - cu_src[batch_idx];
    bool const is_cp = cp_batch_mask != nullptr && cp_batch_mask[batch_idx] != 0;
    int dst_row = cu_compact[batch_idx] + row_in_batch;
    int signal_interval_a = -1;
    int signal_row_a = 0;
    int signal_interval_b = -1;
    int signal_row_b = 0;

    if (is_cp) {
        if (is_causal) {
            int const half_len = half_cu[batch_idx + 1] - half_cu[batch_idx];
            if (row_in_batch < half_len) {
                dst_row = cu_compact[batch_idx] + local_rank * half_len + row_in_batch;
                signal_interval_a = batch_idx * 2;
                signal_row_a = dst_row - cu_compact[batch_idx];
                signal_interval_b = batch_idx * 2 + 1;
                signal_row_b = signal_row_a;
            } else {
                int const back_row = row_in_batch - half_len;
                dst_row = cu_compact[batch_idx] + world_size * half_len
                        + (world_size - 1 - local_rank) * half_len + back_row;
                signal_interval_a = batch_idx * 2 + 1;
                signal_row_a = dst_row - cu_compact[batch_idx];
            }
        } else {
            dst_row = cu_compact[batch_idx] + local_rank * local_len + row_in_batch;
            signal_interval_a = batch_idx;
            signal_row_a = dst_row - cu_compact[batch_idx];
        }
    }

    int const src_row = local_rank * local_total + local_row_global;
    int const lane = int(threadIdx.x);
    if (lane < vecs_per_row) {
        dst_k[dst_row * vecs_per_row + lane] = src_k[src_row * vecs_per_row + lane];
        dst_v[dst_row * vecs_per_row + lane] = src_v[src_row * vecs_per_row + lane];
    }
    __syncthreads();
    if (lane == 0 && is_cp) {
        add_ready_chunk(chunk_done, signal_interval_a, signal_row_a, ready_max_chunks, ready_chunk_rows, 2);
        if (signal_interval_b >= 0) {
            add_ready_chunk(chunk_done, signal_interval_b, signal_row_b, ready_max_chunks, ready_chunk_rows, 2);
        }
    }
}

__global__ void initialize_ready_end_from_chunks_kernel(const int* chunk_done,
                                                        int* ready_end,
                                                        const int* interval_rows,
                                                        int intervals,
                                                        int max_chunks,
                                                        int chunk_rows) {
    int const interval_id = int(blockIdx.x);
    if (interval_id >= intervals || threadIdx.x != 0) {
        return;
    }
    int const total_rows = interval_rows[interval_id];
    int cursor_rows = 0;
    for (int chunk_idx = 0; chunk_idx < max_chunks && cursor_rows < total_rows; ++chunk_idx) {
        int const rows = min(chunk_rows, total_rows - cursor_rows);
        int const target = 2 * rows;
        int const done = chunk_done[interval_id * max_chunks + chunk_idx];
        if (done < target) {
            break;
        }
        cursor_rows += rows;
    }
    ready_end[interval_id] = cursor_rows;
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

void check_global_seqlens_host(const torch::Tensor& t, int64_t batch_size, const char* name) {
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
                                              int tiles_per_half_step,
                                              int cp_total_k_per_rank,
                                              int cp_tiles_per_step,
                                              int cp_tiles_per_half_step,
                                              int* cp_batch_mask,
                                              int* half_cu_seqlens,
                                              torch::Tensor& kv_ready_counts,
                                              torch::Tensor& step_ready,
                                              bool ready_once,
                                              const torch::Tensor& source_cu_seqlens_k,
                                              torch::Tensor* ready_end,
                                              torch::Tensor* chunk_done,
                                              torch::Tensor* publish_lock,
                                              torch::Tensor* ready_interval_rows,
                                              int ready_intervals,
                                              int ready_max_chunks,
                                              int ready_chunk_rows) {
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
    params.mega_ring_tiles_per_half_step = tiles_per_half_step;
    params.mega_ring_total_k_per_rank = local_total_k;
    params.mega_ring_cp_total_k_per_rank = cp_total_k_per_rank;
    params.mega_ring_cp_tiles_per_step = cp_tiles_per_step;
    params.mega_ring_cp_tiles_per_half_step = cp_tiles_per_half_step;
    params.mega_ring_cp_batch_mask = cp_batch_mask;
    params.mega_ring_half_cu_seqlens = half_cu_seqlens;
    params.mega_ring_kv_ready_counts = kv_ready_counts.data_ptr<int>();
    params.mega_ring_step_ready = step_ready.data_ptr<int>();
    params.mega_ring_ready_once = ready_once;
    params.mega_ring_source_cu_seqlens_k = ready_once ? static_cast<int*>(source_cu_seqlens_k.data_ptr()) : nullptr;
    params.mega_ring_ready_end = ready_end != nullptr ? ready_end->data_ptr<int>() : nullptr;
    params.mega_ring_chunk_done = chunk_done != nullptr ? chunk_done->data_ptr<int>() : nullptr;
    params.mega_ring_publish_lock = publish_lock != nullptr ? publish_lock->data_ptr<int>() : nullptr;
    params.mega_ring_ready_interval_rows = ready_interval_rows != nullptr ? ready_interval_rows->data_ptr<int>() : nullptr;
    params.mega_ring_ready_intervals = ready_intervals;
    params.mega_ring_ready_max_chunks = ready_max_chunks;
    params.mega_ring_ready_chunk_rows = ready_chunk_rows;
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

int compute_tiles_per_step_masked(const int* cu_seqlens_q_host,
                                  const int* cp_batch_mask_host,
                                  int batch_size,
                                  int q_heads) {
    int tiles = 0;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        if (cp_batch_mask_host != nullptr && cp_batch_mask_host[batch_idx] == 0) {
            continue;
        }
        int const start = cu_seqlens_q_host[batch_idx];
        int const end = cu_seqlens_q_host[batch_idx + 1];
        TORCH_CHECK(end >= start, "cu_seqlens_q_host must be nondecreasing");
        tiles += round_multiple(end - start, 128) / 128 * q_heads;
    }
    return tiles;
}

int compute_total_k_masked(const int* cu_seqlens_k_host,
                           const int* cp_batch_mask_host,
                           int batch_size) {
    int total = 0;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        if (cp_batch_mask_host != nullptr && cp_batch_mask_host[batch_idx] == 0) {
            continue;
        }
        int const start = cu_seqlens_k_host[batch_idx];
        int const end = cu_seqlens_k_host[batch_idx + 1];
        TORCH_CHECK(end >= start, "cu_seqlens_k_host must be nondecreasing");
        total += end - start;
    }
    return total;
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
                                    py::object half_cu_seqlens_obj,
                                    py::object half_cu_seqlens_host_obj,
                                    py::object out_obj,
                                    py::object lse_obj,
                                    bool return_lse,
                                    py::object global_seqlens_host_obj,
                                    int64_t cp_threshold,
                                    bool ready_once) {
    check_varlen_qkv(q, "q");
    check_varlen_qkv(k, "k");
    check_varlen_qkv(v, "v");
    check_parallel_varlen_qkv(remote_k, "remote_k");
    check_parallel_varlen_qkv(remote_v, "remote_v");
    check_cu_seqlens(cu_seqlens_q, "cu_seqlens_q");
    check_cu_seqlens(cu_seqlens_k, "cu_seqlens_k");
    check_cu_seqlens_host(cu_seqlens_q_host, cu_seqlens_q, "cu_seqlens_q_host");
    check_cu_seqlens_host(cu_seqlens_k_host, cu_seqlens_k, "cu_seqlens_k_host");
    bool const hybrid_mode = !global_seqlens_host_obj.is_none();
    torch::Tensor half_cu_seqlens;
    torch::Tensor half_cu_seqlens_host;
    if (is_causal && !hybrid_mode) {
        TORCH_CHECK(!half_cu_seqlens_obj.is_none(), "causal mega ring zigzag requires half_cu_seqlens");
        TORCH_CHECK(!half_cu_seqlens_host_obj.is_none(), "causal mega ring zigzag requires half_cu_seqlens_host");
        half_cu_seqlens = half_cu_seqlens_obj.cast<torch::Tensor>();
        half_cu_seqlens_host = half_cu_seqlens_host_obj.cast<torch::Tensor>();
        check_cu_seqlens(half_cu_seqlens, "half_cu_seqlens");
        check_cu_seqlens_host(half_cu_seqlens_host, half_cu_seqlens, "half_cu_seqlens_host");
    }

    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, v must be on the same CUDA device");
    TORCH_CHECK(q.device() == cu_seqlens_q.device() && q.device() == cu_seqlens_k.device(),
                "q, k, v, cu_seqlens_q, and cu_seqlens_k must be on the same CUDA device");
    if (is_causal && !hybrid_mode) {
        TORCH_CHECK(half_cu_seqlens.device() == q.device(), "half_cu_seqlens must be on the same CUDA device as q");
    }
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
    TORCH_CHECK(cp_threshold >= 0 && cp_threshold <= std::numeric_limits<int>::max(),
                "cp_threshold must fit in non-negative int32. Got ", cp_threshold);
    torch::Tensor cp_batch_mask;
    std::vector<int> cp_batch_mask_host_vec;
    int const* cp_batch_mask_host_ptr = nullptr;
    if (hybrid_mode) {
        auto global_seqlens_host = global_seqlens_host_obj.cast<torch::Tensor>();
        check_global_seqlens_host(global_seqlens_host, batch_size, "global_seqlens_host");
        int const* global_seqlens_host_ptr = global_seqlens_host.data_ptr<int>();
        cp_batch_mask_host_vec.resize(batch_size);
        auto cp_batch_mask_host = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kInt32));
        int* cp_batch_mask_host_tensor_ptr = cp_batch_mask_host.data_ptr<int>();
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            int const is_cp = global_seqlens_host_ptr[batch_idx] > static_cast<int>(cp_threshold) ? 1 : 0;
            cp_batch_mask_host_vec[batch_idx] = is_cp;
            cp_batch_mask_host_tensor_ptr[batch_idx] = is_cp;
        }
        cp_batch_mask = torch::empty({batch_size}, q.options().dtype(torch::kInt32));
        cp_batch_mask.copy_(cp_batch_mask_host, false);
        cp_batch_mask_host_ptr = cp_batch_mask_host_vec.data();
    }
    int const* half_cu_seqlens_host_ptr = (!hybrid_mode && is_causal) ? half_cu_seqlens_host.data_ptr<int>() : nullptr;
    // MEGA_RING: ring identity is taken from the TKParallelTensor allocation.
    int const world_size = remote_k.local_world_size_;
    int const ring_rank = remote_k.local_rank_;
    TORCH_CHECK(batch_size >= 1, "varlen demo requires batch size B >= 1");
    TORCH_CHECK(cu_seqlens_q_host_ptr[0] == 0, "cu_seqlens_q_host must start with 0");
    TORCH_CHECK(cu_seqlens_k_host_ptr[0] == 0, "cu_seqlens_k_host must start with 0");
    if (is_causal && !hybrid_mode) {
        TORCH_CHECK(half_cu_seqlens_host.size(0) == cu_seqlens_q_host.size(0),
                    "half_cu_seqlens_host must have the same length as cu_seqlens_q_host");
        TORCH_CHECK(half_cu_seqlens_host_ptr[0] == 0, "half_cu_seqlens_host must start with 0");
    }
    TORCH_CHECK(cu_seqlens_q_host_ptr[batch_size] == q.size(0),
                "cu_seqlens_q_host[-1] must equal q.size(0). Got ", cu_seqlens_q_host_ptr[batch_size],
                " vs ", q.size(0));
    // MEGA_RING: cu_seqlens_k describes one rank-local KV block, while K/V
    // storage contains one such block for every rank in local rank order.
    int const local_total_k = cu_seqlens_k_host_ptr[batch_size];
    TORCH_CHECK(local_total_k > 0, "cu_seqlens_k_host[-1] must be positive");
    int tiles_per_half_step = 0;
    if (is_causal) {
        if (hybrid_mode) {
            half_cu_seqlens_host = torch::empty({batch_size + 1}, torch::TensorOptions().dtype(torch::kInt32));
            int* half_host_ptr = half_cu_seqlens_host.data_ptr<int>();
            half_host_ptr[0] = 0;
            for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                int const q_len = cu_seqlens_q_host_ptr[batch_idx + 1] - cu_seqlens_q_host_ptr[batch_idx];
                int const k_len = cu_seqlens_k_host_ptr[batch_idx + 1] - cu_seqlens_k_host_ptr[batch_idx];
                TORCH_CHECK(q_len >= 0 && k_len >= 0, "cu_seqlens must be nondecreasing");
                int half_len = 0;
                if (cp_batch_mask_host_ptr[batch_idx] != 0) {
                    TORCH_CHECK(q_len == k_len,
                                "causal hybrid mega ring requires q/k local lengths to match for CP batch=",
                                batch_idx, ". Got q_len=", q_len, ", k_len=", k_len);
                    TORCH_CHECK(q_len % 2 == 0,
                                "causal hybrid mega ring requires even local length for CP batch=",
                                batch_idx, ". Got q_len=", q_len);
                    half_len = q_len / 2;
                    TORCH_CHECK(half_len > 0, "causal hybrid mega ring requires positive half_len for CP batch=", batch_idx);
                    TORCH_CHECK(half_len % 128 == 0,
                                "causal hybrid mega ring requires CP half_len to be 128-aligned. batch=",
                                batch_idx, ", half_len=", half_len);
                }
                half_host_ptr[batch_idx + 1] = half_host_ptr[batch_idx] + half_len;
            }
            half_cu_seqlens = torch::empty({batch_size + 1}, q.options().dtype(torch::kInt32));
            half_cu_seqlens.copy_(half_cu_seqlens_host, false);
            half_cu_seqlens_host_ptr = half_cu_seqlens_host.data_ptr<int>();
            tiles_per_half_step = compute_tiles_per_step_masked(
                half_cu_seqlens_host_ptr,
                cp_batch_mask_host_ptr,
                batch_size,
                q.size(1));
        } else {
            for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                int const q_len = cu_seqlens_q_host_ptr[batch_idx + 1] - cu_seqlens_q_host_ptr[batch_idx];
                int const k_len = cu_seqlens_k_host_ptr[batch_idx + 1] - cu_seqlens_k_host_ptr[batch_idx];
                int const half_len = half_cu_seqlens_host_ptr[batch_idx + 1] - half_cu_seqlens_host_ptr[batch_idx];
                TORCH_CHECK(half_len > 0, "causal mega ring zigzag requires positive half_len for every batch");
                TORCH_CHECK(q_len == 2 * half_len,
                            "causal mega ring zigzag requires q seqlen == 2 * half_len. batch=", batch_idx,
                            ", q_len=", q_len, ", half_len=", half_len);
                TORCH_CHECK(k_len == 2 * half_len,
                            "causal mega ring zigzag requires k seqlen == 2 * half_len. batch=", batch_idx,
                            ", k_len=", k_len, ", half_len=", half_len);
                TORCH_CHECK(half_len % 128 == 0,
                            "causal mega ring zigzag requires half_len to be 128-aligned. batch=", batch_idx,
                            ", half_len=", half_len);
            }
            tiles_per_half_step = compute_tiles_per_step(half_cu_seqlens_host_ptr, batch_size, q.size(1));
            TORCH_CHECK(tiles_per_half_step > 0, "causal mega ring zigzag requires at least one half-Q tile per step");
        }
    }
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
    auto out = resolve_out(out_obj, q);
    auto lse = resolve_lse(lse_obj, q);

    int b_rounded = round_multiple(batch_size, 4);
    bool varlen_sort_batches = true;
    bool head_swizzle = is_causal;
    int num_prepare_batch_vectors = 2 + (varlen_sort_batches ? 1 : 0) + (head_swizzle ? 1 : 0);
    int metadata_size = 1 + b_rounded * num_prepare_batch_vectors;
    auto scheduler_metadata = torch::empty({metadata_size}, q.options().dtype(torch::kInt32));

    // MEGA_RING: device counters coordinate remote K/V availability and
    // per-Q-tile reduction ordering across ring steps.
    int const tiles_per_step = compute_tiles_per_step(cu_seqlens_q_host_ptr, batch_size, q.size(1));
    int const cp_tiles_per_step = hybrid_mode
        ? compute_tiles_per_step_masked(cu_seqlens_q_host_ptr, cp_batch_mask_host_ptr, batch_size, q.size(1))
        : tiles_per_step;
    int const cp_tiles_per_half_step = hybrid_mode ? tiles_per_half_step : tiles_per_half_step;
    int const cp_total_k_per_rank = hybrid_mode
        ? compute_total_k_masked(cu_seqlens_k_host_ptr, cp_batch_mask_host_ptr, batch_size)
        : local_total_k;
    TORCH_CHECK(tiles_per_step > 0, "mega ring requires at least one Q tile per step");
    if (is_causal && !hybrid_mode) {
        TORCH_CHECK(tiles_per_step == 2 * tiles_per_half_step,
                    "causal mega ring zigzag requires T_full == 2 * T_half. Got T_full=",
                    tiles_per_step, ", T_half=", tiles_per_half_step);
    }
    if (is_causal && hybrid_mode && cp_tiles_per_step > 0) {
        TORCH_CHECK(cp_tiles_per_step == 2 * cp_tiles_per_half_step,
                    "causal hybrid mega ring requires CP T_full == 2 * CP T_half. Got CP T_full=",
                    cp_tiles_per_step, ", CP T_half=", cp_tiles_per_half_step);
    }
    TORCH_CHECK(world_size == 1 || num_comm_sm > 0 || cp_total_k_per_rank == 0,
                "mega ring requires num_comm_sm > 0 when world_size > 1 and at least one CP sequence is present");
    auto kv_ready_counts = torch::zeros({world_size}, q.options().dtype(torch::kInt32));
    int const step_ready_size = hybrid_mode ? std::max(cp_tiles_per_step, 1) : tiles_per_step;
    auto step_ready = torch::zeros({step_ready_size}, q.options().dtype(torch::kInt32));
    bool const use_ready_once = ready_once && hybrid_mode;
    torch::Tensor ready_end;
    torch::Tensor chunk_done;
    torch::Tensor publish_lock;
    torch::Tensor ready_interval_rows;
    torch::Tensor attn_k = k;
    torch::Tensor attn_v = v;
    torch::Tensor attn_cu_seqlens_k = cu_seqlens_k;
    int attn_max_seqlen_k = static_cast<int>(max_seqlen_k);
    int ready_intervals = 0;
    int ready_max_chunks = 0;
    int const ready_chunk_rows = 128;
    auto stream = at::cuda::getDefaultCUDAStream(q.get_device());
    if (use_ready_once) {
        ready_intervals = batch_size * (is_causal ? 2 : 1);
        int max_compact_rows = 0;
        auto compact_cu_host = torch::empty({batch_size + 1}, torch::TensorOptions().dtype(torch::kInt32));
        int* compact_cu_host_ptr = compact_cu_host.data_ptr<int>();
        compact_cu_host_ptr[0] = 0;
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            int const start = cu_seqlens_k_host_ptr[batch_idx];
            int const end = cu_seqlens_k_host_ptr[batch_idx + 1];
            int const local_len = end - start;
            int compact_len = local_len;
            if (cp_batch_mask_host_ptr != nullptr && cp_batch_mask_host_ptr[batch_idx] != 0) {
                compact_len = local_len * world_size;
            }
            compact_cu_host_ptr[batch_idx + 1] = compact_cu_host_ptr[batch_idx] + compact_len;
            max_compact_rows = std::max(max_compact_rows, compact_len);
        }
        attn_max_seqlen_k = max_compact_rows;
        attn_cu_seqlens_k = torch::empty({batch_size + 1}, q.options().dtype(torch::kInt32));
        attn_cu_seqlens_k.copy_(compact_cu_host, false);
        int const compact_total = compact_cu_host_ptr[batch_size];
        attn_k = torch::empty({compact_total, k.size(1), k.size(2)}, k.options());
        attn_v = torch::empty({compact_total, v.size(1), v.size(2)}, v.options());
        ready_max_chunks = std::max(1, round_multiple(max_compact_rows, ready_chunk_rows) / ready_chunk_rows);
        ready_end = torch::zeros({ready_intervals}, q.options().dtype(torch::kInt32));
        chunk_done = torch::zeros({ready_intervals * ready_max_chunks}, q.options().dtype(torch::kInt32));
        publish_lock = torch::zeros({ready_intervals}, q.options().dtype(torch::kInt32));
        auto ready_interval_rows_host = torch::empty({ready_intervals}, torch::TensorOptions().dtype(torch::kInt32));
        int* interval_rows_ptr = ready_interval_rows_host.data_ptr<int>();
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            int const start = cu_seqlens_k_host_ptr[batch_idx];
            int const end = cu_seqlens_k_host_ptr[batch_idx + 1];
            int const local_len = end - start;
            bool const is_cp = cp_batch_mask_host_ptr != nullptr && cp_batch_mask_host_ptr[batch_idx] != 0;
            if (is_causal) {
                int const half_len = is_cp ? local_len / 2 : 0;
                interval_rows_ptr[batch_idx * 2] = is_cp ? (ring_rank + 1) * half_len : 0;
                interval_rows_ptr[batch_idx * 2 + 1] = is_cp ? (2 * world_size - ring_rank) * half_len : 0;
            } else {
                interval_rows_ptr[batch_idx] = is_cp ? world_size * local_len : 0;
            }
        }
        ready_interval_rows = torch::empty({ready_intervals}, q.options().dtype(torch::kInt32));
        ready_interval_rows.copy_(ready_interval_rows_host, false);

        int const vecs_per_row = static_cast<int>(k.size(1) * k.size(2) * sizeof(at::BFloat16) / sizeof(uint4));
        TORCH_CHECK(vecs_per_row * int(sizeof(uint4)) == k.size(1) * k.size(2) * int(sizeof(at::BFloat16)),
                    "ready_once compact pack requires row byte size to be a multiple of uint4");
        pack_ready_once_local_kv_kernel<<<local_total_k, 128, 0, stream>>>(
            reinterpret_cast<const uint4*>(k.data_ptr()),
            reinterpret_cast<const uint4*>(v.data_ptr()),
            reinterpret_cast<uint4*>(attn_k.data_ptr()),
            reinterpret_cast<uint4*>(attn_v.data_ptr()),
            cu_seqlens_k.data_ptr<int>(),
            attn_cu_seqlens_k.data_ptr<int>(),
            static_cast<int*>(cp_batch_mask.data_ptr()),
            is_causal ? static_cast<int*>(half_cu_seqlens.data_ptr()) : nullptr,
            batch_size,
            local_total_k,
            ring_rank,
            world_size,
            is_causal,
            vecs_per_row,
            chunk_done.data_ptr<int>(),
            ready_max_chunks,
            ready_chunk_rows);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        initialize_ready_end_from_chunks_kernel<<<ready_intervals, 1, 0, stream>>>(
            chunk_done.data_ptr<int>(),
            ready_end.data_ptr<int>(),
            ready_interval_rows.data_ptr<int>(),
            ready_intervals,
            ready_max_chunks,
            ready_chunk_rows);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // MEGA_RING: initial local rank block is already resident in the local
    // concatenated K/V buffer.
    set_mega_ring_local_kv_ready_count<<<1, 1, 0, stream>>>(
        kv_ready_counts.data_ptr<int>(), ring_rank, cp_total_k_per_rank * 2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto params = make_mega_ring_varlen_params(
        q,
        attn_k,
        attn_v,
        cu_seqlens_q,
        attn_cu_seqlens_k,
        static_cast<int>(max_seqlen_q),
        attn_max_seqlen_k,
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
        tiles_per_half_step,
        cp_total_k_per_rank,
        cp_tiles_per_step,
        cp_tiles_per_half_step,
        hybrid_mode ? static_cast<int*>(cp_batch_mask.data_ptr()) : nullptr,
        is_causal ? static_cast<int*>(half_cu_seqlens.data_ptr()) : nullptr,
        kv_ready_counts,
        step_ready,
        use_ready_once,
        cu_seqlens_k,
        use_ready_once ? &ready_end : nullptr,
        use_ready_once ? &chunk_done : nullptr,
        use_ready_once ? &publish_lock : nullptr,
        use_ready_once ? &ready_interval_rows : nullptr,
        ready_intervals,
        ready_max_chunks,
        ready_chunk_rows);

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
        py::arg("half_cu_seqlens") = py::none(),
        py::arg("half_cu_seqlens_host") = py::none(),
        py::arg("out") = py::none(),
        py::arg("lse") = py::none(),
        py::arg("return_lse") = false,
        py::arg("global_seqlens_host") = py::none(),
        py::arg("cp_threshold") = 2048,
        py::arg("ready_once") = true,
        "MEGA_RING: explicit multi-step fused Hopper varlen ring-attention demo.\n\n"
        "K/V must be contiguous [world_size * local_total_k, kv_heads, 128] buffers. "
        "The kernel performs persistent compute and remote K/V TMA loads in one launch.");
}
