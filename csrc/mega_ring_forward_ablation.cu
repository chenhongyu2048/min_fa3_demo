// Causal forward ablation launcher copied and trimmed from
// csrc/mega_ring_min_fa3_varlen_ring_launch.cu.

#include "mega_ring_forward_ablation.h"

namespace min_fa3_varlen_demo {
namespace forward_ablation {

template <int WorldSize>
void run_profile(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    torch::Tensor& scratch_o,
    torch::Tensor& scratch_lse,
    Profile profile,
    int completed_storage_size,
    cudaStream_t stream,
    bool prepare_only,
    bool compute_only) {
    TORCH_CHECK(params.is_causal, "forward ablation supports causal mode only");
    TORCH_CHECK(params.ring_world_size == WorldSize,
                "forward ablation world-size dispatch mismatch");
    TORCH_CHECK(params.num_comp_sm > 0,
                "forward ablation requires num_comp_sm > 0");
    TORCH_CHECK(params.num_comm_sm >= 0,
                "forward ablation requires num_comm_sm >= 0");
    TORCH_CHECK(compute_only || params.num_comm_sm > 0
                    || params.mega_ring_hierarchy.reduction_tiles == 0,
                "num_comm_sm must be positive when this rank has "
                "hierarchical CP replay work");
    TORCH_CHECK(params.h_k * params.d == 1024,
                "forward ablation requires KVH * D == 1024");
    TORCH_CHECK(params.d == 128 && params.dv == 128,
                "forward ablation requires D=128");
    TORCH_CHECK(completed_storage_size >= (params.mega_ring_stats ? 4 : 1),
                "completed_tiles storage is too small for the selected profile");

    if (!params.skip_scheduler_metadata_computation) {
        prepare_varlen_num_blocks(
            params,
            stream,
            kPackGQA,
            128,
            128,
            false);
        CHECK_CUDA_KERNEL_LAUNCH();
    }
    if (prepare_only) { return; }

    CHECK_CUDA(cudaMemsetAsync(
        params.tile_count_semaphore, 0, sizeof(int), stream));
    CHECK_CUDA(cudaMemsetAsync(
        params.mega_ring_kv_ready_counts, 0,
        kMegaRingNumKvReadySections * sizeof(int), stream));
    if (compute_only) {
        // MOTIVATION_T2: K/V were materialized in every local IPC arena before
        // timing. A large ready value lets the unchanged copied mainloop
        // consume those rows without launching communication CTAs.
        CHECK_CUDA(cudaMemsetAsync(
            params.mega_ring_kv_ready_counts, 0x7f,
            kMegaRingNumKvReadySections * sizeof(int), stream));
    }
    CHECK_CUDA(cudaMemsetAsync(
        params.mega_ring_step_ready, 0,
        size_t(params.mega_ring_hierarchy.reduction_tiles) * sizeof(int), stream));
    CHECK_CUDA(cudaMemsetAsync(
        params.mega_ring_scan_cursor, 0, sizeof(int), stream));
    CHECK_CUDA(cudaMemsetAsync(
        params.mega_ring_completed_tiles, 0,
        size_t(completed_storage_size) * sizeof(int), stream));
    if (params.mega_ring_stats != nullptr) {
        CHECK_CUDA(cudaMemsetAsync(
            params.mega_ring_stats, 0,
            2 * sizeof(unsigned long long), stream));
    }

    bool const collect_stats = params.mega_ring_stats != nullptr;
    switch (profile) {
        case Profile::StepExternalReduce:
            if (collect_stats) {
                run_steps<WorldSize, false, true>(
                    params, remote_k, remote_v, scratch_o, scratch_lse, stream);
            } else {
                run_steps<WorldSize, false, false>(
                    params, remote_k, remote_v, scratch_o, scratch_lse, stream);
            }
            break;
        case Profile::StepFusedReduce:
            if (collect_stats) {
                run_steps<WorldSize, true, true>(
                    params, remote_k, remote_v, scratch_o, scratch_lse, stream);
            } else {
                run_steps<WorldSize, true, false>(
                    params, remote_k, remote_v, scratch_o, scratch_lse, stream);
            }
            break;
        case Profile::LinearQueueNoRecycle:
            if (collect_stats) {
                run_linear<WorldSize, false, true>(params, remote_k, remote_v, stream);
            } else {
                run_linear<WorldSize, false, false>(params, remote_k, remote_v, stream);
            }
            break;
        case Profile::LinearQueueRecycle:
            if (collect_stats) {
                run_linear<WorldSize, true, true>(params, remote_k, remote_v, stream);
            } else {
                run_linear<WorldSize, true, false>(params, remote_k, remote_v, stream);
            }
            break;
        case Profile::DynamicSegmentRecycle:
        case Profile::HybridBrPbs:
            params.ring_step = -1;
            run_mega_ring_min_fa3_varlen_ring_fwd(
                params, remote_k, remote_v, stream);
            break;
        default:
            TORCH_CHECK(false, "unknown forward ablation profile");
    }
}

void run(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    torch::Tensor& scratch_o,
    torch::Tensor& scratch_lse,
    Profile profile,
    int completed_storage_size,
    cudaStream_t stream,
    bool prepare_only,
    bool compute_only) {
    switch (params.ring_world_size) {
        case 2: run_profile<2>(params, remote_k, remote_v, scratch_o, scratch_lse, profile, completed_storage_size, stream, prepare_only, compute_only); break;
        case 4: run_profile<4>(params, remote_k, remote_v, scratch_o, scratch_lse, profile, completed_storage_size, stream, prepare_only, compute_only); break;
        case 8: run_profile<8>(params, remote_k, remote_v, scratch_o, scratch_lse, profile, completed_storage_size, stream, prepare_only, compute_only); break;
        default: TORCH_CHECK(false, "forward ablation requires CP world size 2, 4, or 8");
    }
}

}  // namespace forward_ablation
}  // namespace min_fa3_varlen_demo
