// Copied and trimmed from Hopper forward sources:
// - hopper/flash_fwd_launch_template.h

#include "min_fa3_launch.h"

void run_min_fa3_fwd(Flash_fwd_params& params, cudaStream_t stream) {
    if (params.is_causal) {
        min_fa3_demo::run_min_fa3_sm90<true>(params, stream);
    } else {
        min_fa3_demo::run_min_fa3_sm90<false>(params, stream);
    }
}
