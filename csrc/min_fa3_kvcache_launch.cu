// Copied and trimmed from Hopper forward sources:
// - hopper/flash_fwd_launch_template.h

#include "min_fa3_kvcache_launch.h"
#include "min_fa3_varlen_launch.h"

namespace min_fa3_varlen_demo {

void run_min_fa3_kvcache_fwd(Flash_fwd_params& params, cudaStream_t stream) {
    if (params.num_splits > 1) {
        if (params.is_causal) {
            run_min_fa3_varlen_sm90<true, true, true>(params, stream, std::nullopt);
        } else {
            run_min_fa3_varlen_sm90<false, true, true>(params, stream, std::nullopt);
        }
    } else if (params.pack_gqa) {
        if (params.is_causal) {
            run_min_fa3_varlen_sm90<true, true, false>(params, stream, std::nullopt);
        } else {
            run_min_fa3_varlen_sm90<false, true, false>(params, stream, std::nullopt);
        }
    } else if (params.is_causal) {
        run_min_fa3_varlen_sm90<true, false, false>(params, stream, std::nullopt);
    } else {
        run_min_fa3_varlen_sm90<false, false, false>(params, stream, std::nullopt);
    }
}

}  // namespace min_fa3_varlen_demo
