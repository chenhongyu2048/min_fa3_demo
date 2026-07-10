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

}  // namespace min_fa3_backward
