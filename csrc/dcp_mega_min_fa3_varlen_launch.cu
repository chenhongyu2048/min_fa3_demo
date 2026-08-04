// DCP mega launch dispatch copied and trimmed from csrc/min_fa3_kvcache_launch.cu.

#include "dcp_mega_min_fa3_varlen_launch.h"

#include <torch/extension.h>

namespace min_fa3_varlen_demo::dcp_mega {

namespace {

struct alignas(128) BarrierParams {
    int32_t* flags[8];
    int32_t phase;
    int32_t dcp_size;
    int32_t dcp_rank;
};

__device__ inline void store_release_system_s32(int32_t* address, int32_t value) {
    asm volatile("{st.release.sys.global.s32 [%0], %1;}"
                 :: "l"(address), "r"(value) : "memory");
}

__device__ inline int32_t load_acquire_system_s32(int32_t const* address) {
    int32_t value;
    asm volatile("{ld.acquire.sys.global.s32 %0, [%1];}"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__global__ void dcp_phase_barrier_kernel(BarrierParams params) {
    int const lane = int(threadIdx.x);
    if (lane == params.dcp_rank) {
        store_release_system_s32(params.flags[lane], params.phase);
    }
    __syncwarp();
    if (lane < params.dcp_size) {
        while (load_acquire_system_s32(params.flags[lane]) < params.phase) {
            __nanosleep(64);
        }
    }
    __syncwarp();
}

}  // namespace

void run_dcp_mega_barrier(
    DCPMega_fwd_params const& params,
    int phase,
    cudaStream_t stream) {
    TORCH_CHECK(phase > 0, "DCP mega barrier phase must be positive");
    BarrierParams barrier{};
    for (int rank = 0; rank < params.dcp_size; ++rank) {
        TORCH_CHECK(params.ipc_barrier_ptrs[rank] != nullptr,
                    "DCP mega barrier IPC pointer is null at rank ", rank);
        barrier.flags[rank] = params.ipc_barrier_ptrs[rank];
    }
    barrier.phase = phase;
    barrier.dcp_size = params.dcp_size;
    barrier.dcp_rank = params.dcp_rank;
    dcp_phase_barrier_kernel<<<1, 32, 0, stream>>>(barrier);
    CHECK_CUDA_KERNEL_LAUNCH();
}

void run_dcp_mega_varlen_fwd(
    DCPMega_fwd_params& params,
    cudaStream_t stream) {
    MetadataHeader const& header = params.metadata_header;
    TORCH_CHECK(header.version == 2, "unsupported DCP mega metadata version");
    TORCH_CHECK(header.effective_num_splits >= 1 && header.effective_num_splits <= 128,
                "invalid effective_num_splits in DCP mega metadata");
    TORCH_CHECK(header.block_n == 128 || header.block_n == 176,
                "DCP mega BlockN must be 128 or 176");
    TORCH_CHECK(header.pack_gqa
                    && header.reserved_v2_communication_layout == 1
                    && (params.hq_local == 4 || params.hq_local == 8),
                "DCP mega requires fixed PackGQA communication with Hq_local in {4, 8}");

    if (header.split) {
        if (header.block_n == 128) { run_pack_split_bn128(params, stream); }
        else { run_pack_split_bn176(params, stream); }
    } else {
        if (header.block_n == 128) { run_pack_nosplit_bn128(params, stream); }
        else { run_pack_nosplit_bn176(params, stream); }
    }
}

}  // namespace min_fa3_varlen_demo::dcp_mega
