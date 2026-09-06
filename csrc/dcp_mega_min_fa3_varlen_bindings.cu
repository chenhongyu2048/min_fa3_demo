// DCP mega binding copied and trimmed from csrc/min_fa3_kvcache_bindings.cu
// and csrc/mega_ring_min_fa3_varlen_ring_bindings.cu.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#ifdef CHECK_CUDA
#undef CHECK_CUDA
#endif
#ifdef CHECK_CONTIGUOUS
#undef CHECK_CONTIGUOUS
#endif
#ifdef CHECK_INPUT
#undef CHECK_INPUT
#endif

#include "dcp_mega_min_fa3_varlen_params.h"

namespace py = pybind11;

namespace {

using min_fa3_varlen_demo::Flash_fwd_params;
using min_fa3_varlen_demo::dcp_mega::DCPMega_fwd_params;
using min_fa3_varlen_demo::dcp_mega::MetadataHeader;
using min_fa3_varlen_demo::dcp_mega::kPhaseTimestampCount;

constexpr int kHeadDim = 128;

struct ReusableTimingEvents {
    explicit ReusableTimingEvents(int device_) : device(device_) {
        c10::cuda::CUDAGuard guard(device);
        C10_CUDA_CHECK(cudaEventCreate(&start));
        C10_CUDA_CHECK(cudaEventCreate(&end));
    }

    ~ReusableTimingEvents() {
        if (start == nullptr && end == nullptr) {
            return;
        }
        try {
            c10::cuda::CUDAGuard guard(device);
            if (start != nullptr) {
                cudaEventDestroy(start);
            }
            if (end != nullptr) {
                cudaEventDestroy(end);
            }
        } catch (...) {
        }
    }

    int device;
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
};

ReusableTimingEvents& reusable_timing_events(int device) {
    thread_local std::unordered_map<
        int, std::unique_ptr<ReusableTimingEvents>> events_by_device;
    auto [it, inserted] = events_by_device.try_emplace(device);
    if (inserted) {
        it->second = std::make_unique<ReusableTimingEvents>(device);
    }
    return *it->second;
}

void check_packed_bf16(torch::Tensor const& tensor, char const* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
                name, " must have dtype torch.bfloat16");
    TORCH_CHECK(tensor.dim() == 3 && tensor.size(2) == kHeadDim,
                name, " must have shape [total_tokens, heads, 128]");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_int32(torch::Tensor const& tensor, char const* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32,
                name, " must have dtype torch.int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(torch::Tensor const& q,
                       torch::Tensor const& tensor,
                       char const* name) {
    TORCH_CHECK(tensor.device() == q.device(), name, " must be on the same device as q");
}

void check_alignment(void const* pointer, uintptr_t alignment, char const* name) {
    TORCH_CHECK(pointer != nullptr, name, " must not be null");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(pointer) % alignment == 0,
                name, " must be ", alignment, "-byte aligned");
}

void check_parallel_base(
    kittens::py::TKParallelTensor const& arena,
    torch::Tensor const& q,
    int node_world_size,
    char const* name) {
    TORCH_CHECK(arena.data_.is_cuda(), name, ".data_ must be CUDA");
    TORCH_CHECK(arena.data_.is_contiguous(), name, ".data_ must be contiguous");
    TORCH_CHECK(arena.data_.device() == q.device(), name, " must be on q.device");
    TORCH_CHECK(arena.local_rank_ == q.get_device(),
                name, " local_rank must match q.device.index");
    TORCH_CHECK(arena.local_world_size_ == node_world_size,
                name, " local_world_size mismatch");
    TORCH_CHECK(static_cast<int>(arena.raw_ptrs_.size()) == node_world_size,
                name, " raw IPC pointer count mismatch");
    check_alignment(arena.data_.data_ptr(), 128, name);
}

void validate_metadata_header(
    MetadataHeader const& header,
    int64_t metadata_used,
    int64_t metadata_capacity,
    int64_t q_ready_capacity,
    int64_t attention_done_capacity,
    int64_t total_q,
    int64_t total_vectors,
    int64_t batch_size,
    int dcp_size) {
    TORCH_CHECK(header.version == 7,
                "unsupported DCP mega metadata version; expected version 7");
    TORCH_CHECK(header.used_ints == metadata_used,
                "metadata_used does not match the pinned header");
    TORCH_CHECK(metadata_used >= 40 && metadata_used <= metadata_capacity,
                "DCP mega metadata size is out of bounds");
    TORCH_CHECK(header.total_q == total_q && header.total_vectors == total_vectors,
                "DCP mega metadata shape does not match q");
    TORCH_CHECK(header.batch_size == batch_size,
                "DCP mega metadata batch size mismatch");
    TORCH_CHECK(header.q_task_count >= 0,
                "Q task count must be nonnegative");
    TORCH_CHECK(header.q_ready_count >= 0
                    && header.q_ready_count <= q_ready_capacity,
                "q_ready workspace is too small");
    TORCH_CHECK(header.attention_count >= 0
                    && header.attention_count <= attention_done_capacity,
                "attention_done workspace is too small");
    TORCH_CHECK(header.chunk_attention_count >= 0
                    && header.history_attention_count >= 0
                    && header.chunk_attention_count + header.history_attention_count
                        == header.attention_count,
                "invalid attention domain counts");
    TORCH_CHECK(header.effective_num_splits >= 1
                    && header.effective_num_splits <= 128,
                "invalid effective split count");
    TORCH_CHECK(header.chunk_num_splits >= 1
                    && header.chunk_num_splits <= header.effective_num_splits
                    && header.history_num_splits >= 1
                    && header.history_num_splits <= header.effective_num_splits,
                "invalid chunk/history split upper bounds");
    TORCH_CHECK(header.pack_gqa == 1,
                "DCP mega metadata must use PackGQA");
    TORCH_CHECK(header.split == 0 || header.split == 1,
                "split header must be boolean");
    TORCH_CHECK(header.block_n == 128 || header.block_n == 176,
                "BlockN must be 128 or 176");
    TORCH_CHECK(header.reserved_v2_communication_layout == 1,
                "DCP mega metadata uses an unsupported communication layout");
    TORCH_CHECK(header.dcp_size == dcp_size,
                "metadata DCP size mismatch");
    TORCH_CHECK(header.token_block_count == (total_q + 15) / 16,
                "invalid metadata token-block count");
    int64_t const hq_local = total_vectors / total_q;
    TORCH_CHECK(hq_local == 4 || hq_local == 8,
                "DCP mega requires Hq_local in {4, 8}");
    TORCH_CHECK(header.q_task_count == header.token_block_count * dcp_size
                    && header.q_ready_count == header.token_block_count,
                "invalid fixed-layout Q queue counts");
    TORCH_CHECK(header.publish_count == header.token_block_count * dcp_size,
                "invalid fixed-layout publish queue count");
    TORCH_CHECK(
        header.final_count == (total_q + 3) / 4
            || header.final_count == (total_q + 7) / 8
            || header.final_count == (total_q + 15) / 16,
        "invalid adaptive final queue count");
    TORCH_CHECK(header.history_combine_count >= header.publish_count
                    && header.history_combine_count
                        <= total_vectors * dcp_size,
                "invalid history combine descriptor count");
    TORCH_CHECK(header.receive_count
                        == header.token_block_count * (dcp_size - 1)
                    && header.tile_ready_count == header.receive_count,
                "invalid fixed-layout receive/tile-ready queue counts");

    auto check_range = [&](int offset, int64_t elements, char const* name) {
        TORCH_CHECK(offset >= 40 && elements >= 0
                        && int64_t(offset) + elements <= metadata_used,
                    name, " range is outside the metadata image");
    };
    check_range(header.attention_offset, int64_t(header.attention_count) * 8,
                "attention descriptors");
    check_range(header.q_tasks_offset, int64_t(header.q_task_count) * 4,
                "Q tasks");
    check_range(header.q_dependencies_offset, header.q_dependency_count,
                "Q dependencies");
    check_range(header.publish_offset, int64_t(header.publish_count) * 8,
                "publish descriptors");
    check_range(header.history_combine_offset,
                int64_t(header.history_combine_count) * 8,
                "history combine descriptors");
    check_range(header.publish_dependencies_offset,
                header.publish_dependency_count, "publish dependencies");
    check_range(header.final_offset, int64_t(header.final_count) * 8,
                "final descriptors");
    check_range(header.final_dependencies_offset,
                header.final_dependency_count, "final dependencies");
    check_range(header.chunk_splits_offset, batch_size, "chunk split array");
    check_range(header.history_splits_offset, batch_size, "history split array");
    TORCH_CHECK(
        header.attention_offset == 40
            && header.q_tasks_offset
                == header.attention_offset + header.attention_count * 8
            && header.q_dependencies_offset
                == header.q_tasks_offset + header.q_task_count * 4
            && header.publish_offset
                == header.q_dependencies_offset + header.q_dependency_count
            && header.history_combine_offset
                == header.publish_offset + header.publish_count * 8
            && header.publish_dependencies_offset
                == header.history_combine_offset
                    + header.history_combine_count * 8
            && header.final_offset
                == header.publish_dependencies_offset
                    + header.publish_dependency_count
            && header.final_dependencies_offset
                == header.final_offset + header.final_count * 8
            && header.chunk_splits_offset
                == header.final_dependencies_offset + header.final_dependency_count
            && header.history_splits_offset
                == header.chunk_splits_offset + batch_size
            && header.used_ints == header.history_splits_offset + batch_size,
        "DCP mega metadata v7 ranges must be contiguous and non-overlapping");
}

Flash_fwd_params make_attention_params(
    torch::Tensor const& q,
    torch::Tensor const& k,
    torch::Tensor const& v,
    torch::Tensor const& cu_seqlens_q,
    torch::Tensor const& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    torch::Tensor& out,
    torch::Tensor& lse,
    torch::Tensor& out_partial,
    torch::Tensor& lse_partial,
    int effective_num_splits,
    bool pack_gqa,
    bool is_causal,
    int num_sms) {
    Flash_fwd_params params{};
    params.is_bf16 = true;
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
    params.softmax_lse_ptr = lse.data_ptr();

    params.b = cu_seqlens_q.numel() - 1;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_q_rounded = (max_seqlen_q + 127) / 128 * 128;
    params.seqlen_k_rounded = (max_seqlen_k + 127) / 128 * 128;
    params.h = q.size(1);
    params.h_k = k.size(1);
    params.d = kHeadDim;
    params.d_rounded = kHeadDim;
    params.dv = kHeadDim;
    params.dv_rounded = kHeadDim;
    params.total_q = q.size(0);
    params.total_k = k.size(0);
    params.b_k = params.b;
    params.scale_softmax = 1.0f / std::sqrt(float(kHeadDim));
    params.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    params.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
    params.is_causal = is_causal;
    params.is_local = false;
    params.window_size_left = max_seqlen_k - 1;
    params.window_size_right = is_causal ? 0 : max_seqlen_q - 1;
    params.attention_chunk = 0;
    params.num_splits = effective_num_splits;
    params.pack_gqa = pack_gqa;
    params.skip_scheduler_metadata_computation = true;
    params.arch = 90;
    params.num_sm = num_sms;

    params.oaccum_ptr = out_partial.data_ptr();
    params.oaccum_split_stride = out_partial.stride(0);
    params.oaccum_batch_stride = 0;
    params.oaccum_head_stride = out_partial.stride(1);
    params.oaccum_row_stride = out_partial.stride(2);
    params.softmax_lseaccum_ptr = lse_partial.data_ptr();
    params.lseaccum_split_stride = lse_partial.stride(0);
    params.lseaccum_batch_stride = 0;
    params.lseaccum_head_stride = lse_partial.stride(1);
    return params;
}

double forward_chunk_prefill_varlen_dcp_mega(
    torch::Tensor q,
    torch::Tensor k_history,
    torch::Tensor v_history,
    torch::Tensor k_chunk,
    torch::Tensor v_chunk,
    torch::Tensor cu_seqlens_q,
    torch::Tensor cu_seqlens_history,
    int64_t max_seqlen_q,
    int64_t max_seqlen_history,
    kittens::py::TKParallelTensor& ipc_q,
    kittens::py::TKParallelTensor& ipc_history_send_o,
    kittens::py::TKParallelTensor& ipc_history_send_lse,
    kittens::py::TKParallelTensor& ipc_tile_ready,
    kittens::py::TKParallelTensor& ipc_barrier,
    torch::Tensor q_group,
    torch::Tensor chunk_o,
    torch::Tensor chunk_lse,
    torch::Tensor history_o,
    torch::Tensor history_lse,
    torch::Tensor history_receive_o,
    torch::Tensor history_receive_lse,
    torch::Tensor chunk_o_partial,
    torch::Tensor chunk_lse_partial,
    torch::Tensor history_o_partial,
    torch::Tensor history_lse_partial,
    torch::Tensor final_o,
    torch::Tensor final_lse,
    torch::Tensor metadata_host,
    torch::Tensor metadata_device,
    int64_t metadata_used,
    torch::Tensor q_ready,
    torch::Tensor attention_done,
    torch::Tensor publish_ready,
    torch::Tensor receive_ready,
    torch::Tensor queue_state,
    torch::Tensor phase_timestamps,
    torch::Tensor graph_post_phase,
    torch::Tensor cta_trace,
    int64_t cta_trace_iteration,
    bool record_phase_timestamps,
    std::vector<int64_t> dcp_node_ranks,
    int64_t dcp_rank,
    int64_t num_comm_sm,
    bool return_lse,
    bool metadata_prepared,
    int64_t pre_phase,
    bool run_post_barrier,
    bool run_pre_barrier,
    bool measure_kernel,
    bool graph_replay) {
    check_packed_bf16(q, "q");
    check_packed_bf16(k_history, "k_history");
    check_packed_bf16(v_history, "v_history");
    check_packed_bf16(k_chunk, "k_chunk");
    check_packed_bf16(v_chunk, "v_chunk");
    check_packed_bf16(q_group, "q_group");
    check_packed_bf16(chunk_o, "chunk_o");
    check_packed_bf16(history_o, "history_o");
    check_packed_bf16(final_o, "final_o");
    check_cuda_int32(cu_seqlens_q, "cu_seqlens_q");
    check_cuda_int32(cu_seqlens_history, "cu_seqlens_history");
    check_cuda_int32(metadata_device, "metadata_device");
    check_cuda_int32(q_ready, "q_ready");
    check_cuda_int32(attention_done, "attention_done");
    check_cuda_int32(publish_ready, "publish_ready");
    check_cuda_int32(receive_ready, "receive_ready");
    check_cuda_int32(queue_state, "queue_state");
    check_cuda_int32(graph_post_phase, "graph_post_phase");
    TORCH_CHECK(cta_trace.is_cuda() && cta_trace.scalar_type() == torch::kInt64
                    && cta_trace.is_contiguous() && cta_trace.dim() == 2
                    && cta_trace.size(1) == 8,
                "cta_trace must be a contiguous CUDA int64 [capacity, 8] tensor");
    TORCH_CHECK(phase_timestamps.is_cuda()
                    && phase_timestamps.scalar_type() == torch::kInt64
                    && phase_timestamps.is_contiguous()
                    && phase_timestamps.dim() == 1
                    && phase_timestamps.numel() >= kPhaseTimestampCount,
                "phase_timestamps must be a contiguous CUDA int64 buffer with at least ",
                kPhaseTimestampCount, " slots");
    TORCH_CHECK(graph_post_phase.numel() == 1,
                "graph_post_phase must contain exactly one int32 value");
    TORCH_CHECK(!graph_replay || metadata_prepared,
                "DCP mega CUDA Graph replay requires prepared metadata");
    TORCH_CHECK(!graph_replay || !measure_kernel,
                "DCP mega CUDA Graph replay uses external graph timing events");
    TORCH_CHECK(!metadata_host.is_cuda()
                    && metadata_host.scalar_type() == torch::kInt32
                    && metadata_host.is_contiguous()
                    && metadata_host.is_pinned(),
                "metadata_host must be a contiguous pinned CPU int32 tensor");
    TORCH_CHECK(metadata_device.dim() == 1 && metadata_host.dim() == 1,
                "metadata tensors must be one-dimensional");

    for (auto const& named : std::vector<std::pair<torch::Tensor const*, char const*>>{
             {&k_history, "k_history"}, {&v_history, "v_history"},
             {&k_chunk, "k_chunk"}, {&v_chunk, "v_chunk"},
             {&cu_seqlens_q, "cu_seqlens_q"},
             {&cu_seqlens_history, "cu_seqlens_history"},
             {&q_group, "q_group"}, {&chunk_o, "chunk_o"},
             {&chunk_lse, "chunk_lse"}, {&history_o, "history_o"},
             {&history_lse, "history_lse"},
             {&history_receive_o, "history_receive_o"},
             {&history_receive_lse, "history_receive_lse"},
             {&chunk_o_partial, "chunk_o_partial"},
             {&chunk_lse_partial, "chunk_lse_partial"},
             {&history_o_partial, "history_o_partial"},
             {&history_lse_partial, "history_lse_partial"},
             {&final_o, "final_o"}, {&final_lse, "final_lse"},
             {&metadata_device, "metadata_device"}, {&q_ready, "q_ready"},
             {&attention_done, "attention_done"},
             {&publish_ready, "publish_ready"},
             {&receive_ready, "receive_ready"}, {&queue_state, "queue_state"},
             {&phase_timestamps, "phase_timestamps"},
             {&graph_post_phase, "graph_post_phase"},
             {&cta_trace, "cta_trace"}}) {
        check_same_device(q, *named.first, named.second);
    }

    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0, "q dimensions must be positive");
    TORCH_CHECK(k_history.size(0) > 0 && k_history.size(1) == 1,
                "history requires positive tokens and Hkv_group == 1");
    TORCH_CHECK(k_history.sizes() == v_history.sizes(),
                "history K/V shapes must match");
    TORCH_CHECK(k_chunk.sizes() == v_chunk.sizes()
                    && k_chunk.size(0) == q.size(0) && k_chunk.size(1) == 1,
                "chunk K/V must have shape [total_q, 1, 128]");
    TORCH_CHECK(cu_seqlens_q.numel() == cu_seqlens_history.numel()
                    && cu_seqlens_q.numel() >= 2,
                "Q/history cu_seqlens must have the same [B + 1] shape");
    TORCH_CHECK(max_seqlen_q > 0 && max_seqlen_history > 0,
                "max sequence lengths must be positive");

    c10::cuda::CUDAGuard guard(q.device());
    auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 9 && properties->minor == 0,
                "DCP mega only supports Hopper SM90");
    TORCH_CHECK(num_comm_sm > 0
                    && num_comm_sm <= properties->multiProcessorCount - 1,
                "num_comm_sm must leave at least one compute SM");

    int const dcp_size = dcp_node_ranks.size();
    TORCH_CHECK(dcp_size == 2 || dcp_size == 4 || dcp_size == 8,
                "DCP size must be 2, 4, or 8");
    TORCH_CHECK(dcp_rank >= 0 && dcp_rank < dcp_size,
                "dcp_rank is out of range");
    int const node_world_size = ipc_q.local_world_size_;
    TORCH_CHECK(node_world_size == 2 || node_world_size == 4 || node_world_size == 8,
                "node IPC world size must be 2, 4, or 8");
    std::vector<bool> seen(node_world_size, false);
    for (int rank = 0; rank < dcp_size; ++rank) {
        int64_t const node_rank = dcp_node_ranks[rank];
        TORCH_CHECK(node_rank >= 0 && node_rank < node_world_size,
                    "DCP node rank is out of range");
        TORCH_CHECK(!seen[node_rank], "DCP node ranks must be unique");
        seen[node_rank] = true;
    }
    TORCH_CHECK(dcp_node_ranks[dcp_rank] == q.get_device(),
                "current DCP rank must map to q.device.index");

    check_parallel_base(ipc_q, q, node_world_size, "ipc_q");
    check_parallel_base(ipc_history_send_o, q, node_world_size,
                        "ipc_history_send_o");
    check_parallel_base(ipc_history_send_lse, q, node_world_size,
                        "ipc_history_send_lse");
    check_parallel_base(ipc_tile_ready, q, node_world_size, "ipc_tile_ready");
    check_parallel_base(ipc_barrier, q, node_world_size, "ipc_barrier");
    TORCH_CHECK(ipc_q.data_.scalar_type() == torch::kBFloat16
                    && ipc_q.data_.dim() == 3
                    && ipc_q.data_.size(1) == q.size(1)
                    && ipc_q.data_.size(2) == 128
                    && ipc_q.data_.size(0) >= q.size(0)
                    && ipc_q.data_.size(0) % 16 == 0,
                "ipc_q has incompatible shape/dtype");
    TORCH_CHECK(q.data_ptr() == ipc_q.data_.data_ptr(),
                "q must be the prefix of ipc_q.data_");
    int64_t const token_capacity = ipc_history_send_o.data_.size(1);
    int64_t const vector_capacity = token_capacity * q.size(1);
    int64_t const total_vectors = q.size(0) * q.size(1);
    TORCH_CHECK(ipc_history_send_o.data_.scalar_type() == torch::kBFloat16
                    && ipc_history_send_o.data_.dim() == 4
                    && ipc_history_send_o.data_.size(0) == dcp_size
                    && token_capacity >= q.size(0)
                    && token_capacity % 16 == 0
                    && ipc_history_send_o.data_.size(2) == q.size(1)
                    && ipc_history_send_o.data_.size(3) == 128,
                "ipc_history_send_o must be [DCP, capacity_q, Hq_local, 128] BF16");
    TORCH_CHECK(ipc_history_send_lse.data_.scalar_type() == torch::kFloat32
                    && ipc_history_send_lse.data_.dim() == 3
                    && ipc_history_send_lse.data_.size(0) == dcp_size
                    && ipc_history_send_lse.data_.size(1) == token_capacity
                    && ipc_history_send_lse.data_.size(2) == q.size(1),
                "ipc_history_send_lse must be [DCP, capacity_q, Hq_local] FP32");
    int64_t const token_block_capacity = token_capacity / 16;
    TORCH_CHECK(ipc_tile_ready.data_.scalar_type() == torch::kInt32
                    && ipc_tile_ready.data_.dim() == 2
                    && ipc_tile_ready.data_.size(0) == dcp_size
                    && ipc_tile_ready.data_.size(1) == token_block_capacity,
                "ipc_tile_ready must be [DCP, capacity_token_blocks] int32");
    TORCH_CHECK(ipc_barrier.data_.scalar_type() == torch::kInt32
                    && ipc_barrier.data_.numel() >= 1,
                "ipc_barrier must contain at least one int32 phase flag");
    check_alignment(ipc_history_send_lse.data_.data_ptr(), 4,
                    "ipc_history_send_lse");
    check_alignment(ipc_tile_ready.data_.data_ptr(), 4, "ipc_tile_ready");

    TORCH_CHECK(metadata_used >= 40
                    && metadata_used <= metadata_host.numel()
                    && metadata_used <= metadata_device.numel(),
                "metadata_used exceeds host/device storage");
    MetadataHeader header{};
    std::memcpy(&header, metadata_host.data_ptr<int>(), sizeof(header));
    validate_metadata_header(
        header,
        metadata_used,
        std::min(metadata_host.numel(), metadata_device.numel()),
        q_ready.numel(),
        attention_done.numel(),
        q.size(0),
        total_vectors,
        cu_seqlens_q.numel() - 1,
        dcp_size);
    TORCH_CHECK(header.effective_num_splits <= chunk_o_partial.size(0)
                    && header.effective_num_splits <= history_o_partial.size(0),
                "partial workspace split capacity is too small");
    TORCH_CHECK(publish_ready.numel() >= header.publish_count,
                "publish_ready capacity is too small");
    TORCH_CHECK(receive_ready.numel() >= header.receive_count,
                "receive_ready capacity is too small");
    TORCH_CHECK(ipc_tile_ready.data_.numel()
                    >= int64_t(dcp_size) * header.token_block_count,
                "tile-ready IPC arena is too small");

    int64_t const group_heads = dcp_size * q.size(1);
    TORCH_CHECK(q_group.size(0) == ipc_q.data_.size(0)
                    && q_group.size(1) == group_heads,
                "q_group must be [capacity_q, DCP * Hq_local, 128]");
    TORCH_CHECK(chunk_o.dim() == 3 && chunk_o.size(0) >= q.size(0)
                    && chunk_o.size(1) == q.size(1),
                "chunk_o capacity is invalid");
    TORCH_CHECK(history_o.dim() == 3 && history_o.size(0) >= q.size(0)
                    && history_o.size(1) == group_heads,
                "history_o capacity is invalid");
    TORCH_CHECK(history_receive_o.scalar_type() == torch::kBFloat16
                    && history_receive_o.dim() == 4
                    && history_receive_o.size(0) == dcp_size
                    && history_receive_o.size(1) == token_capacity
                    && history_receive_o.size(2) == q.size(1)
                    && history_receive_o.size(3) == 128,
                "history_receive_o must be [DCP, capacity_q, Hq_local, 128] BF16");
    TORCH_CHECK(history_receive_lse.scalar_type() == torch::kFloat32
                    && history_receive_lse.is_contiguous()
                    && history_receive_lse.dim() == 3
                    && history_receive_lse.size(0) == dcp_size
                    && history_receive_lse.size(1) == token_capacity
                    && history_receive_lse.size(2) == q.size(1),
                "history_receive_lse must be [DCP, capacity_q, Hq_local] FP32");
    TORCH_CHECK(final_o.sizes() == q.sizes(),
                "final_o must have the active q shape");
    TORCH_CHECK(chunk_lse.scalar_type() == torch::kFloat32
                    && chunk_lse.dim() == 2
                    && chunk_lse.size(0) == q.size(1)
                    && chunk_lse.size(1) >= q.size(0),
                "chunk_lse capacity is invalid");
    TORCH_CHECK(history_lse.scalar_type() == torch::kFloat32
                    && history_lse.dim() == 2
                    && history_lse.size(0) == group_heads
                    && history_lse.size(1) >= q.size(0),
                "history_lse capacity is invalid");
    TORCH_CHECK(final_lse.scalar_type() == torch::kFloat32
                    && final_lse.dim() == 2
                    && final_lse.size(0) == q.size(1)
                    && final_lse.size(1) == q.size(0),
                "final_lse must be [Hq_local, total_q]");
    for (auto const& partial : {chunk_o_partial, history_o_partial,
                                chunk_lse_partial, history_lse_partial}) {
        TORCH_CHECK(partial.scalar_type() == torch::kFloat32
                        && partial.is_contiguous(),
                    "all partial workspaces must be contiguous FP32 tensors");
    }
    TORCH_CHECK(chunk_o_partial.dim() == 4
                    && chunk_o_partial.size(1) == q.size(1)
                    && chunk_o_partial.size(2) >= q.size(0)
                    && chunk_o_partial.size(3) == 128,
                "chunk_o_partial shape is invalid");
    TORCH_CHECK(history_o_partial.dim() == 4
                    && history_o_partial.size(1) == group_heads
                    && history_o_partial.size(2) >= q.size(0)
                    && history_o_partial.size(3) == 128,
                "history_o_partial shape is invalid");
    TORCH_CHECK(chunk_lse_partial.dim() == 3
                    && chunk_lse_partial.size(1) == q.size(1)
                    && chunk_lse_partial.size(2) >= q.size(0),
                "chunk_lse_partial shape is invalid");
    TORCH_CHECK(history_lse_partial.dim() == 3
                    && history_lse_partial.size(1) == group_heads
                    && history_lse_partial.size(2) >= q.size(0),
                "history_lse_partial shape is invalid");
    TORCH_CHECK(queue_state.numel() >= 9,
                "queue_state requires at least 9 int32 slots");
    check_alignment(phase_timestamps.data_ptr(), 8, "phase_timestamps");
    check_alignment(q_group.data_ptr(), 128, "q_group");
    check_alignment(history_receive_o.data_ptr(), 128, "history_receive_o");

    DCPMega_fwd_params params{};
    params.chunk = make_attention_params(
        q, k_chunk, v_chunk, cu_seqlens_q, cu_seqlens_q,
        max_seqlen_q, max_seqlen_q,
        chunk_o, chunk_lse, chunk_o_partial, chunk_lse_partial,
        header.effective_num_splits, header.pack_gqa, true,
        properties->multiProcessorCount);
    params.history = make_attention_params(
        q_group, k_history, v_history, cu_seqlens_q, cu_seqlens_history,
        max_seqlen_q, max_seqlen_history,
        history_o, history_lse, history_o_partial, history_lse_partial,
        header.effective_num_splits, header.pack_gqa, false,
        properties->multiProcessorCount);
    params.q_group_ptr = q_group.data_ptr();
    params.history_receive_o_ptr = history_receive_o.data_ptr();
    params.history_receive_lse_ptr = history_receive_lse.data_ptr<float>();
    params.final_o_ptr = final_o.data_ptr();
    params.final_lse_ptr = final_lse.data_ptr<float>();
    params.chunk_lse_head_stride = chunk_lse.stride(0);
    params.history_lse_head_stride = history_lse.stride(0);
    params.final_lse_head_stride = final_lse.stride(0);
    params.metadata = metadata_device.data_ptr<int>();
    params.metadata_header = header;
    params.q_ready = q_ready.data_ptr<int>();
    params.attention_done = attention_done.data_ptr<int>();
    params.publish_ready = publish_ready.data_ptr<int>();
    params.receive_ready = receive_ready.data_ptr<int>();
    params.queue_state = queue_state.data_ptr<int>();
    params.phase_timestamps = record_phase_timestamps
        ? reinterpret_cast<uint64_t*>(phase_timestamps.data_ptr<int64_t>())
        : nullptr;
    params.graph_post_phase = graph_replay
        ? graph_post_phase.data_ptr<int32_t>() : nullptr;
    params.dcp_size = dcp_size;
    params.dcp_rank = dcp_rank;
    params.hq_local = q.size(1);
    params.ipc_q_token_capacity = ipc_q.data_.size(0);
    params.ipc_vector_capacity = vector_capacity;
    params.ipc_token_block_capacity = token_block_capacity;
    params.device = q.get_device();
    params.num_sms = properties->multiProcessorCount;
    params.num_comm_sm = num_comm_sm;
    params.cta_trace = cta_trace.numel() == 0 ? nullptr : cta_trace.data_ptr<int64_t>();
    params.cta_trace_capacity = static_cast<int>(cta_trace.size(0));
    params.cta_trace_iteration = static_cast<int>(cta_trace_iteration);
    params.return_lse = return_lse;
    for (int rank = 0; rank < dcp_size; ++rank) {
        int const node_rank = dcp_node_ranks[rank];
        params.ipc_q_ptrs[rank] = ipc_q.raw_ptrs_[node_rank];
        params.ipc_history_send_o_ptrs[rank]
            = ipc_history_send_o.raw_ptrs_[node_rank];
        params.ipc_history_send_lse_ptrs[rank]
            = static_cast<float*>(ipc_history_send_lse.raw_ptrs_[node_rank]);
        params.ipc_tile_ready_ptrs[rank]
            = static_cast<int32_t*>(ipc_tile_ready.raw_ptrs_[node_rank]);
        params.ipc_barrier_ptrs[rank]
            = static_cast<int32_t*>(ipc_barrier.raw_ptrs_[node_rank]);
        check_alignment(params.ipc_q_ptrs[rank], 128, "remote IPC Q base");
        check_alignment(params.ipc_history_send_o_ptrs[rank], 128,
                        "remote history send O base");
        check_alignment(params.ipc_history_send_lse_ptrs[rank], 4,
                        "remote history send LSE base");
        check_alignment(params.ipc_tile_ready_ptrs[rank], 4,
                        "remote tile-ready base");
        check_alignment(params.ipc_barrier_ptrs[rank], 4,
                        "remote barrier base");
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if (!metadata_prepared) {
        C10_CUDA_CHECK(cudaMemcpyAsync(
            metadata_device.data_ptr<int>(),
            metadata_host.data_ptr<int>(),
            metadata_used * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream));
    }
    if (!metadata_prepared || graph_replay) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            q_ready.data_ptr<int>(), 0,
            header.q_ready_count * sizeof(int32_t), stream));
        C10_CUDA_CHECK(cudaMemsetAsync(
            attention_done.data_ptr<int>(), 0,
            header.attention_count * sizeof(int32_t), stream));
        C10_CUDA_CHECK(cudaMemsetAsync(
            publish_ready.data_ptr<int>(), 0,
            header.publish_count * sizeof(int32_t), stream));
        C10_CUDA_CHECK(cudaMemsetAsync(
            receive_ready.data_ptr<int>(), 0,
            header.receive_count * sizeof(int32_t), stream));
        C10_CUDA_CHECK(cudaMemsetAsync(
            queue_state.data_ptr<int>(), 0, queue_state.nbytes(), stream));
        if (record_phase_timestamps) {
            C10_CUDA_CHECK(cudaMemsetAsync(
                phase_timestamps.data_ptr<int64_t>(), 0,
                kPhaseTimestampCount * sizeof(int64_t), stream));
        }
    }
    if (cta_trace.numel() > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            cta_trace.data_ptr<int64_t>(), 0, cta_trace.nbytes(), stream));
    }
    if (graph_replay) {
        min_fa3_varlen_demo::dcp_mega::advance_dcp_mega_graph_phase(
            graph_post_phase.data_ptr<int32_t>(), stream);
    } else {
        int64_t const tile_ready_phase
            = metadata_prepared ? pre_phase : header.pre_phase;
        TORCH_CHECK(tile_ready_phase > 0
                        && tile_ready_phase <= std::numeric_limits<int32_t>::max(),
                    "DCP mega requires a positive int32 tile-ready phase");
        params.tile_ready_phase = int(tile_ready_phase);
    }

    if (run_pre_barrier) {
        if (graph_replay) {
            min_fa3_varlen_demo::dcp_mega::run_dcp_mega_graph_barrier(
                params, -1, stream);
        } else {
            min_fa3_varlen_demo::dcp_mega::run_dcp_mega_barrier(
                params, metadata_prepared ? int(pre_phase) : header.pre_phase, stream);
        }
    }
    if (measure_kernel) {
        ReusableTimingEvents& events = reusable_timing_events(q.get_device());
        params.timing_start = events.start;
        params.timing_end = events.end;
    }
    min_fa3_varlen_demo::dcp_mega::run_dcp_mega_varlen_fwd(params, stream);
    if (run_post_barrier) {
        if (graph_replay) {
            min_fa3_varlen_demo::dcp_mega::run_dcp_mega_graph_barrier(
                params, 0, stream);
        } else {
            min_fa3_varlen_demo::dcp_mega::run_dcp_mega_barrier(
                params, header.post_phase, stream);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    double elapsed_ms = 0.0;
    if (measure_kernel) {
        C10_CUDA_CHECK(cudaEventSynchronize(params.timing_end));
        float elapsed_ms_float = 0.0f;
        C10_CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms_float, params.timing_start, params.timing_end));
        elapsed_ms = elapsed_ms_float;
    }
    return elapsed_ms;
}

void dcp_mega_varlen_barrier(
    kittens::py::TKParallelTensor& ipc_barrier,
    std::vector<int64_t> dcp_node_ranks,
    int64_t dcp_rank,
    int64_t phase) {
    TORCH_CHECK(ipc_barrier.data_.is_cuda()
                    && ipc_barrier.data_.scalar_type() == torch::kInt32
                    && ipc_barrier.data_.is_contiguous()
                    && ipc_barrier.data_.numel() >= 1,
                "ipc_barrier must be a contiguous CUDA int32 arena");
    c10::cuda::CUDAGuard guard(ipc_barrier.data_.device());
    int const node_world_size = ipc_barrier.local_world_size_;
    TORCH_CHECK(node_world_size == 2 || node_world_size == 4 || node_world_size == 8,
                "node IPC world size must be 2, 4, or 8");
    TORCH_CHECK(ipc_barrier.local_rank_ == ipc_barrier.data_.get_device(),
                "ipc_barrier local rank must match its CUDA device");
    TORCH_CHECK(static_cast<int>(ipc_barrier.raw_ptrs_.size()) == node_world_size,
                "ipc_barrier raw pointer count mismatch");
    int const dcp_size = dcp_node_ranks.size();
    TORCH_CHECK(dcp_size == 2 || dcp_size == 4 || dcp_size == 8,
                "DCP size must be 2, 4, or 8");
    TORCH_CHECK(dcp_rank >= 0 && dcp_rank < dcp_size,
                "dcp_rank is out of range");
    TORCH_CHECK(phase > 0 && phase <= std::numeric_limits<int32_t>::max(),
                "DCP mega barrier phase must be a positive int32 value");

    DCPMega_fwd_params params{};
    params.dcp_size = dcp_size;
    params.dcp_rank = dcp_rank;
    std::vector<bool> seen(node_world_size, false);
    for (int rank = 0; rank < dcp_size; ++rank) {
        int64_t const node_rank = dcp_node_ranks[rank];
        TORCH_CHECK(node_rank >= 0 && node_rank < node_world_size,
                    "DCP node rank is out of range");
        TORCH_CHECK(!seen[node_rank], "DCP node ranks must be unique");
        seen[node_rank] = true;
        params.ipc_barrier_ptrs[rank]
            = static_cast<int32_t*>(ipc_barrier.raw_ptrs_[node_rank]);
    }
    TORCH_CHECK(dcp_node_ranks[dcp_rank] == ipc_barrier.data_.get_device(),
                "current DCP rank must map to ipc_barrier.device.index");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(
        ipc_barrier.data_.get_device()).stream();
    min_fa3_varlen_demo::dcp_mega::run_dcp_mega_barrier(
        params, int(phase), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void bind_dcp_mega_varlen(py::module_& module) {
    module.def(
        "forward_chunk_prefill_varlen_dcp_mega",
        &forward_chunk_prefill_varlen_dcp_mega,
        py::arg("q"),
        py::arg("k_history"),
        py::arg("v_history"),
        py::arg("k_chunk"),
        py::arg("v_chunk"),
        py::arg("cu_seqlens_q"),
        py::arg("cu_seqlens_history"),
        py::arg("max_seqlen_q"),
        py::arg("max_seqlen_history"),
        py::arg("ipc_q"),
        py::arg("ipc_history_send_o"),
        py::arg("ipc_history_send_lse"),
        py::arg("ipc_tile_ready"),
        py::arg("ipc_barrier"),
        py::arg("q_group"),
        py::arg("chunk_o"),
        py::arg("chunk_lse"),
        py::arg("history_o"),
        py::arg("history_lse"),
        py::arg("history_receive_o"),
        py::arg("history_receive_lse"),
        py::arg("chunk_o_partial"),
        py::arg("chunk_lse_partial"),
        py::arg("history_o_partial"),
        py::arg("history_lse_partial"),
        py::arg("final_o"),
        py::arg("final_lse"),
        py::arg("metadata_host"),
        py::arg("metadata_device"),
        py::arg("metadata_used"),
        py::arg("q_ready"),
        py::arg("attention_done"),
        py::arg("publish_ready"),
        py::arg("receive_ready"),
        py::arg("queue_state"),
        py::arg("phase_timestamps"),
        py::arg("graph_post_phase"),
        py::arg("cta_trace"),
        py::arg("cta_trace_iteration"),
        py::arg("record_phase_timestamps"),
        py::arg("dcp_node_ranks"),
        py::arg("dcp_rank"),
        py::arg("num_comm_sm"),
        py::arg("return_lse") = false,
        py::arg("metadata_prepared") = false,
        py::arg("pre_phase") = 0,
        py::arg("run_post_barrier") = true,
        py::arg("run_pre_barrier") = true,
        py::arg("measure_kernel") = false,
        py::arg("graph_replay") = false,
        "Persistent single-node SM90 BF16 D=128 batched varlen DCP mega forward.");
    module.def(
        "_dcp_mega_varlen_barrier",
        &dcp_mega_varlen_barrier,
        py::arg("ipc_barrier"),
        py::arg("dcp_node_ranks"),
        py::arg("dcp_rank"),
        py::arg("phase"),
        "Internal phase barrier used by prepared DCP mega benchmark replay.");
}
