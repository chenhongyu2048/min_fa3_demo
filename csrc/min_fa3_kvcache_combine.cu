// Copied and trimmed from Hopper forward sources:
// - hopper/flash_fwd_combine_launch_template.h

#include "min_fa3_kvcache_combine_launch.h"
#include "min_fa3_kvcache_launch.h"

namespace min_fa3_varlen_demo {

void run_min_fa3_kvcache_combine(
    Flash_fwd_params& params,
    cudaStream_t stream,
    bool enable_pdl) {
    bool const varlen_q = params.cu_seqlens_q != nullptr || params.seqused_q != nullptr;
    if (varlen_q) {
        if (params.num_splits <= 32) {
            run_min_fa3_kvcache_combine_sm90<5, true>(params, stream, enable_pdl);
        } else if (params.num_splits <= 64) {
            run_min_fa3_kvcache_combine_sm90<6, true>(params, stream, enable_pdl);
        } else {
            run_min_fa3_kvcache_combine_sm90<7, true>(params, stream, enable_pdl);
        }
    } else if (params.num_splits <= 32) {
        run_min_fa3_kvcache_combine_sm90<5, false>(params, stream, enable_pdl);
    } else if (params.num_splits <= 64) {
        run_min_fa3_kvcache_combine_sm90<6, false>(params, stream, enable_pdl);
    } else {
        run_min_fa3_kvcache_combine_sm90<7, false>(params, stream, enable_pdl);
    }
}

}  // namespace min_fa3_varlen_demo
