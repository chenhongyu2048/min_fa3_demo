// Copied and trimmed from Hopper backward launch source:
// - hopper/flash_api.cpp

#include "backward/min_fa3_bwd_launch.h"

namespace min_fa3_backward {

void run_min_fa3_bwd(Flash_bwd_params& params, cudaStream_t stream) {
    if (params.is_causal) {
        run_min_fa3_bwd_sm90<true>(params, stream);
    } else {
        run_min_fa3_bwd_sm90<false>(params, stream);
    }
}

template <int NumDevices>
void run_min_fa3_bwd_mega_ring_sm90(
        Flash_bwd_params& params,
        kittens::py::TKParallelTensor& remote_k,
        kittens::py::TKParallelTensor& remote_v,
        kittens::py::TKParallelTensor& remote_dk_accum,
        kittens::py::TKParallelTensor& remote_dv_accum,
        cudaStream_t stream) {
    using Config = BwdConfig<true>;
    run_flash_bwd<
        90,
        kHeadDim,
        Config::kBlockM,
        kBlockN,
        Element,
        true,   // causal; non-local ring steps switch masks at runtime
        false,
        false,
        true,   // varlen
        false,  // deterministic
        true,   // always use the FP32 dKV accumulation epilogue
        kStagesdO,
        kStagesdS,
        kSdPSwapAB,
        kdKVSwapAB,
        Config::kdQSwapAB,
        kNumMmaWarpGroups,
        kAtomLayoutMSdP,
        kAtomLayoutNdKV,
        kAtomLayoutMdQ,
        kVInRegs,
        true,   // mega ring
        NumDevices
    >(params, stream, &remote_k, &remote_v, &remote_dk_accum, &remote_dv_accum);
}

void run_min_fa3_bwd_mega_ring(
        Flash_bwd_params& params,
        kittens::py::TKParallelTensor& remote_k,
        kittens::py::TKParallelTensor& remote_v,
        kittens::py::TKParallelTensor& remote_dk_accum,
        kittens::py::TKParallelTensor& remote_dv_accum,
        cudaStream_t stream) {
    switch (params.ring_world_size) {
        case 1: run_min_fa3_bwd_mega_ring_sm90<1>(params, remote_k, remote_v, remote_dk_accum, remote_dv_accum, stream); break;
        case 2: run_min_fa3_bwd_mega_ring_sm90<2>(params, remote_k, remote_v, remote_dk_accum, remote_dv_accum, stream); break;
        case 3: run_min_fa3_bwd_mega_ring_sm90<3>(params, remote_k, remote_v, remote_dk_accum, remote_dv_accum, stream); break;
        case 4: run_min_fa3_bwd_mega_ring_sm90<4>(params, remote_k, remote_v, remote_dk_accum, remote_dv_accum, stream); break;
        case 5: run_min_fa3_bwd_mega_ring_sm90<5>(params, remote_k, remote_v, remote_dk_accum, remote_dv_accum, stream); break;
        case 6: run_min_fa3_bwd_mega_ring_sm90<6>(params, remote_k, remote_v, remote_dk_accum, remote_dv_accum, stream); break;
        case 7: run_min_fa3_bwd_mega_ring_sm90<7>(params, remote_k, remote_v, remote_dk_accum, remote_dv_accum, stream); break;
        case 8: run_min_fa3_bwd_mega_ring_sm90<8>(params, remote_k, remote_v, remote_dk_accum, remote_dv_accum, stream); break;
        default: TORCH_CHECK(false, "Unsupported mega-ring backward world size: ", params.ring_world_size);
    }
}

}  // namespace min_fa3_backward
