// Causal forward ablation launcher copied and trimmed from
// csrc/mega_ring_min_fa3_varlen_ring_launch.cu.

#include "mega_ring_forward_ablation.h"

namespace min_fa3_varlen_demo {
namespace forward_ablation {

void run(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    torch::Tensor& scratch_o,
    torch::Tensor& scratch_lse,
    Profile profile,
    int completed_storage_size,
    int map_storage_size,
    cudaStream_t stream,
    bool prepare_only) {
    TORCH_CHECK(params.is_causal, "forward ablation supports causal mode only");
    bool const dynamic = profile == Profile::DynamicSegmentRecycle || profile == Profile::HybridBrPbs;
    TORCH_CHECK(params.ring_world_size == 2 || params.ring_world_size == 4
                    || params.ring_world_size == 8,
                "forward ablation requires 2, 4, or 8 GPUs");
    TORCH_CHECK(params.num_comp_sm > 0,
                "forward ablation requires num_comp_sm > 0");
    TORCH_CHECK(params.num_comm_sm >= 0,
                "forward ablation requires num_comm_sm >= 0");
    TORCH_CHECK(params.num_comm_sm > 0
                    || params.mega_ring_hierarchy.reduction_tiles == 0,
                "num_comm_sm must be positive when this rank has "
                "G8/G4/G2 replay work");
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
    if (dynamic) {
        CHECK_CUDA(cudaMemsetAsync(params.mega_ring_kv_ready_prefix, 0, params.b * sizeof(int), stream));
        CHECK_CUDA(cudaMemsetAsync(params.mega_ring_q_assigned, 0,
            3 * size_t(params.mega_ring_hierarchy.reduction_tiles > 0
                ? params.mega_ring_hierarchy.reduction_tiles : 1) * sizeof(int), stream));
        if (map_storage_size > 0) {
            CHECK_CUDA(cudaMemsetAsync(params.mega_ring_kv_map, 0,
                size_t(map_storage_size) * sizeof(int), stream));
        }
        CHECK_CUDA(cudaMemsetAsync(params.mega_ring_comm_phase_barrier, 0, sizeof(int), stream));
    } else {
        CHECK_CUDA(cudaMemsetAsync(params.mega_ring_kv_ready_counts, 0,
            kMegaRingNumKvReadySections * sizeof(int), stream));
        CHECK_CUDA(cudaMemsetAsync(params.mega_ring_step_ready, 0,
            size_t(params.mega_ring_hierarchy.reduction_tiles) * sizeof(int), stream));
    }
    CHECK_CUDA(cudaMemsetAsync(params.mega_ring_scan_cursor, 0, sizeof(int), stream));
    CHECK_CUDA(cudaMemsetAsync(params.mega_ring_completed_tiles, 0,
        size_t(completed_storage_size) * sizeof(int), stream));
    if (params.mega_ring_stats != nullptr) {
        CHECK_CUDA(cudaMemsetAsync(
            params.mega_ring_stats, 0,
            2 * sizeof(unsigned long long), stream));
    }

    if (dynamic) {
        params.ring_step = -1;
        run_mega_ring_min_fa3_varlen_ring_fwd(params, remote_k, remote_v, stream);
        return;
    }
    bool const collect_stats = params.mega_ring_stats != nullptr;
    auto launch_legacy = [&]<int NumDevices>() {
        switch (profile) {
            case Profile::StepExternalReduce:
                if (collect_stats) {
                    run_steps<false, true, NumDevices>(
                        params, remote_k, remote_v, scratch_o, scratch_lse, stream);
                } else {
                    run_steps<false, false, NumDevices>(
                        params, remote_k, remote_v, scratch_o, scratch_lse, stream);
                }
                break;
            case Profile::StepFusedReduce:
                if (collect_stats) {
                    run_steps<true, true, NumDevices>(
                        params, remote_k, remote_v, scratch_o, scratch_lse, stream);
                } else {
                    run_steps<true, false, NumDevices>(
                        params, remote_k, remote_v, scratch_o, scratch_lse, stream);
                }
                break;
            case Profile::LinearQueueNoRecycle:
                if (collect_stats) {
                    run_linear<false, true, NumDevices>(params, remote_k, remote_v, stream);
                } else {
                    run_linear<false, false, NumDevices>(params, remote_k, remote_v, stream);
                }
                break;
            case Profile::LinearQueueRecycle:
                if (collect_stats) {
                    run_linear<true, true, NumDevices>(params, remote_k, remote_v, stream);
                } else {
                    run_linear<true, false, NumDevices>(params, remote_k, remote_v, stream);
                }
                break;
            default:
                TORCH_CHECK(false, "unknown forward ablation profile");
        }
    };
    switch (params.ring_world_size) {
        case 2: launch_legacy.template operator()<2>(); break;
        case 4: launch_legacy.template operator()<4>(); break;
        case 8: launch_legacy.template operator()<8>(); break;
    }
}

}  // namespace forward_ablation
}  // namespace min_fa3_varlen_demo
