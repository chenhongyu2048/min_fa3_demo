// Copied and trimmed from Hopper forward sources:
// - hopper/flash_fwd_launch_template.h

#include "min_fa3_varlen_launch.h"

namespace min_fa3_varlen_demo {

void run_min_fa3_varlen_fwd(Flash_fwd_params& params, cudaStream_t stream) {
    if (params.is_causal) {
        run_min_fa3_varlen_sm90<true>(params, stream);
    } else {
        run_min_fa3_varlen_sm90<false>(params, stream);
    }
}

}  // namespace min_fa3_varlen_demo
