// Copied and trimmed from Hopper forward sources:
// - hopper/flash_fwd_launch_template.h
// Ring-attention-specific wrapper around the minimal SM90 bf16 varlen forward path.

#include "min_fa3_varlen_ring_launch.h"

#include <limits>

namespace min_fa3_varlen_demo {

void run_min_fa3_varlen_ring_fwd(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    cudaStream_t stream) {
    TORCH_CHECK(params.num_comp_sm > 0, "Ring varlen kernel requires num_comp_sm > 0. Got ", params.num_comp_sm);
    TORCH_CHECK(params.num_comm_sm >= 0, "Ring varlen kernel requires num_comm_sm >= 0. Got ", params.num_comm_sm);
    TORCH_CHECK(
        params.num_comp_sm + params.num_comm_sm <= std::numeric_limits<int>::max(),
        "Ring varlen kernel total block count must fit in int32");
    TORCH_CHECK(remote_k.local_world_size_ == remote_v.local_world_size_, "remote_k and remote_v must share the same local_world_size");
    TORCH_CHECK(remote_k.local_rank_ == remote_v.local_rank_, "remote_k and remote_v must share the same local_rank");
    TORCH_CHECK(
        params.src_dev >= 0 && params.src_dev < remote_k.local_world_size_,
        "src_dev must be in [0, local_world_size). Got src_dev=",
        params.src_dev,
        ", local_world_size=",
        remote_k.local_world_size_);
    TORCH_CHECK(
        params.num_comm_sm == 0 || (params.local_k_staging_ptr != nullptr && params.local_v_staging_ptr != nullptr),
        "Ring varlen kernel requires local staging buffers when num_comm_sm > 0");

    if (params.is_causal) {
        ring_detail::dispatch_ring_world_size<true>(params, remote_k, remote_v, stream);
    } else {
        ring_detail::dispatch_ring_world_size<false>(params, remote_k, remote_v, stream);
    }
}

}  // namespace min_fa3_varlen_demo
