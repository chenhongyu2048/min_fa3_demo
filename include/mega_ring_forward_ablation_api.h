// Public declaration for the causal CP4/CP8 forward ablation launcher. Kernel
// definitions remain private to csrc/mega_ring_forward_ablation.cu.

#pragma once

#include <torch/extension.h>

#include "kittens.cuh"
#include "pyutils/parallel_tensor.cuh"
#include "min_fa3_varlen_params.h"

namespace min_fa3_varlen_demo {
namespace forward_ablation {

enum class Profile : int {
    StepExternalReduce = 1,
    StepFusedReduce = 2,
    LinearQueueNoRecycle = 3,
    LinearQueueRecycle = 4,
    DynamicSegmentRecycle = 5,
    HybridBrPbs = 6,
};

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
    bool compute_only);

}  // namespace forward_ablation
}  // namespace min_fa3_varlen_demo
