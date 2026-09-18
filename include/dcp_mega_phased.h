// Included inside dcp_mega::detail after the production phase helpers.
// Motivation D1: retain the task image and FA3 implementation, but require
// all CTAs to execute one task class at a time. No per-task instrumentation.
#pragma once

template <typename Config, bool Trace>
CUTLASS_DEVICE void phased_rank_barrier(
    typename Config::KernelParams const& params, int slot, int64_t* row) {
    // The preceding grid barrier establishes local completion. Only CTA0
    // participates in the IPC rendezvous; the following grid barrier releases
    // every CTA. Slot 0 belongs to the existing pre/post-launch protocol.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if constexpr (Trace) { row[5] = read_globaltimer(); }
        int const epoch = params.graph_post_phase != nullptr
            ? *params.graph_post_phase - 1 : params.tile_ready_phase;
        store_release_system_s32(
            params.phase_barrier_remote[params.dcp_rank] + slot, epoch);
        for (int rank = 0; rank < Config::kDCPSize; ++rank) {
            while (load_acquire_system_s32(
                       params.phase_barrier_remote[rank] + slot) < epoch) {
                __nanosleep(64);
            }
        }
        if constexpr (Trace) { row[6] = read_globaltimer(); }
    }
}

template <typename Config, bool Trace>
CUTLASS_GLOBAL
__launch_bounds__(Config::MaxThreadsPerBlock, 1)
void dcp_mega_phased_kernel(
    CUTLASS_GRID_CONSTANT typename Config::KernelParams const params) {
    extern __shared__ char smem_buf[];
    auto& helper = *reinterpret_cast<typename Config::HelperSharedStorage*>(smem_buf);
    auto grid = cooperative_groups::this_grid();
    grid.sync();
    for (int phase = 0; phase < 6; ++phase) {
        int64_t* row = nullptr;
        if constexpr (Trace) {
            row = params.sm_trace + (phase * params.num_sms + int(blockIdx.x)) * 7;
            if (threadIdx.x == 0) {
                uint32_t sm;
                asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
                row[0] = sm;
                row[1] = read_globaltimer();
                row[5] = 0;
                row[6] = 0;
            }
        }
        if ((phase == 1 || phase == 2) && blockIdx.x == 0 && threadIdx.x == 0) {
            params.queue_state[kAttentionDynamicCounter] = 0;
        }
        if (phase == 0 || phase == 4) {
            phased_rank_barrier<Config, Trace>(params, phase == 0 ? 1 : 2, row);
        }
        grid.sync();
        if constexpr (Trace) {
            if (threadIdx.x == 0) { row[2] = read_globaltimer(); }
        }
        // Keep work in every warp after the leader's start marker. Present in
        // trace-off too, so the trace switch changes only the instrumentation.
        __syncthreads();
        if (phase == 0) {
            run_q_allgather<Config, true>(params, helper);
        } else if (phase == 1 || phase == 2) {
            typename Config::AttentionKernel attention;
            attention.template operator()<true>(params, smem_buf, phase - 1);
        } else if (phase == 3) {
            run_history_combine<Config>(params, helper);
        } else if (phase == 4) {
            run_communication_post_q<Config, true>(params, helper);
        } else {
            run_final_combine<Config, true>(params, helper);
        }
        // The phase helpers drain their TMA/WGMMA work before returning.
        // In particular the warp-queued history combine needs CTA convergence.
        __syncthreads();
        if constexpr (Trace) {
            if (threadIdx.x == 0) { row[3] = read_globaltimer(); }
        }
        grid.sync();
        if constexpr (Trace) {
            if (threadIdx.x == 0) { row[4] = read_globaltimer(); }
        }
    }
}

template <typename Config, bool Trace>
void launch_dcp_mega_phased(
    DCPMega_fwd_params const& params,
    typename Config::KernelParams const& kernel_params,
    cudaStream_t stream) {
    auto kernel = dcp_mega_phased_kernel<Config, Trace>;
    constexpr int kMaxCachedCUDADevices = 64;
    TORCH_CHECK(params.device >= 0 && params.device < kMaxCachedCUDADevices,
                "DCP phased device exceeds launch cache capacity");
    static std::once_flag configured[kMaxCachedCUDADevices];
    static int shared_bytes[kMaxCachedCUDADevices]{};
    std::call_once(configured[params.device], [&] {
        cudaStreamCaptureStatus capture_status;
        CHECK_CUDA(cudaStreamIsCapturing(stream, &capture_status));
        TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                    "warm up the phased specialization before graph capture");
        cudaDeviceProp properties;
        CHECK_CUDA(cudaGetDeviceProperties(&properties, params.device));
        TORCH_CHECK(properties.cooperativeLaunch,
                    "DCP phased execution requires cooperative launch");
        // Reserve enough shared memory to enforce at most one CTA per SM.
        // Both trace specializations reserve exactly the same resources.
        int const one_cta_bytes = (int(properties.sharedMemPerMultiprocessor) / 256 + 1) * 128;
        int const smem = Config::SharedStorageSize > one_cta_bytes
            ? Config::SharedStorageSize : one_cta_bytes;
        TORCH_CHECK(smem <= int(properties.sharedMemPerBlockOptin),
                    "DCP phased shared memory exceeds the opt-in limit");
        CHECK_CUDA(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        int active_blocks = 0;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, kernel, Config::MaxThreadsPerBlock, smem));
        TORCH_CHECK(active_blocks == 1 && params.num_sms == properties.multiProcessorCount,
                    "DCP phased execution requires one resident CTA per SM");
        shared_bytes[params.device] = smem;
    });
    void* arguments[] = {const_cast<typename Config::KernelParams*>(&kernel_params)};
    if (params.timing_start != nullptr) {
        CHECK_CUDA(cudaEventRecord(params.timing_start, stream));
    }
    CHECK_CUDA(cudaLaunchCooperativeKernel(
        reinterpret_cast<void const*>(kernel), dim3(params.num_sms),
        dim3(Config::MaxThreadsPerBlock), arguments, shared_bytes[params.device], stream));
    if (params.timing_end != nullptr) {
        CHECK_CUDA(cudaEventRecord(params.timing_end, stream));
    }
    CHECK_CUDA_KERNEL_LAUNCH();
}
