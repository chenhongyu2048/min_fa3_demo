// Copied and trimmed from Hopper forward sources:
// - hopper/flash_fwd_launch_template.h

#include "min_fa3_varlen_launch.h"

namespace min_fa3_varlen_demo {

void run_min_fa3_varlen_fwd(
    Flash_fwd_params& params,
    cudaStream_t stream,
    std::optional<int> manual_block_count) {
    if (params.is_causal) {
        run_min_fa3_varlen_sm90<true>(params, stream, manual_block_count);
    } else {
        run_min_fa3_varlen_sm90<false>(params, stream, manual_block_count);
    }
}

}  // namespace min_fa3_varlen_demo
