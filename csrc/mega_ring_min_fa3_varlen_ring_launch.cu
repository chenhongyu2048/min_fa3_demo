// Mega ring variant copied and trimmed from csrc/min_fa3_varlen_ring_launch.cu.
// Changes are marked with MEGA_RING comments.

#include "mega_ring_min_fa3_varlen_ring_launch.h"

#include <limits>

namespace min_fa3_varlen_demo {

void run_mega_ring_min_fa3_varlen_ring_fwd(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    cudaStream_t stream) {
    TORCH_CHECK(params.num_comp_sm > 0, "Mega ring varlen kernel requires num_comp_sm > 0. Got ", params.num_comp_sm);
    TORCH_CHECK(params.num_comm_sm >= 0, "Mega ring varlen kernel requires num_comm_sm >= 0. Got ", params.num_comm_sm);
    TORCH_CHECK(params.ring_world_size >= 1, "Mega ring varlen kernel requires ring_world_size >= 1. Got ", params.ring_world_size);
    TORCH_CHECK(params.num_comp_sm + params.num_comm_sm <= std::numeric_limits<int>::max(),
                "Mega ring varlen kernel total block count must fit in int32");
    TORCH_CHECK(remote_k.local_world_size_ == remote_v.local_world_size_, "remote_k and remote_v must share the same local_world_size");
    TORCH_CHECK(remote_k.local_rank_ == remote_v.local_rank_, "remote_k and remote_v must share the same local_rank");
    TORCH_CHECK(remote_k.local_world_size_ == params.ring_world_size,
                "remote_k local_world_size must match ring_world_size. Got local_world_size=",
                remote_k.local_world_size_, ", ring_world_size=", params.ring_world_size);
    TORCH_CHECK(remote_k.local_rank_ == params.ring_rank,
                "remote_k local_rank must match ring_rank. Got local_rank=",
                remote_k.local_rank_, ", ring_rank=", params.ring_rank);
    TORCH_CHECK(params.mega_ring_tiles_per_step > 0, "Mega ring varlen kernel requires mega_ring_tiles_per_step > 0. Got ", params.mega_ring_tiles_per_step);
    TORCH_CHECK(params.mega_ring_total_k_per_rank > 0, "Mega ring varlen kernel requires mega_ring_total_k_per_rank > 0. Got ", params.mega_ring_total_k_per_rank);
    TORCH_CHECK(params.mega_ring_kv_ready_counts != nullptr, "Mega ring varlen kernel requires mega_ring_kv_ready_counts storage");
    TORCH_CHECK(params.mega_ring_step_ready != nullptr, "Mega ring varlen kernel requires mega_ring_step_ready counter storage");
    TORCH_CHECK(params.num_comm_sm > 0 || params.ring_world_size == 1, "Mega ring varlen kernel requires num_comm_sm > 0 when ring_world_size > 1");

    if (params.is_causal) {
        mega_ring_detail::dispatch_mega_ring_world_size<true>(params, remote_k, remote_v, stream);
    } else {
        mega_ring_detail::dispatch_mega_ring_world_size<false>(params, remote_k, remote_v, stream);
    }
}

}  // namespace min_fa3_varlen_demo
