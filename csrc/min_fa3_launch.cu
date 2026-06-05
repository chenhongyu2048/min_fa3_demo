// Copied and trimmed from Hopper forward sources:
// - hopper/flash_fwd_launch_template.h

#include "min_fa3_launch.h"

void run_min_fa3_fwd(
    Flash_fwd_params& params,
    cudaStream_t stream,
    std::optional<int> manual_block_count) {
    if (params.is_causal) {
        min_fa3_demo::run_min_fa3_sm90<true>(params, stream, manual_block_count);
    } else {
        min_fa3_demo::run_min_fa3_sm90<false>(params, stream, manual_block_count);
    }
}
