// Copied and trimmed from Hopper forward sources:
// - hopper/flash_fwd_launch_template.h
// - hopper/flash_fwd_combine_launch_template.h

#pragma once

#include <cuda_runtime.h>

#include "min_fa3_varlen_params.h"

namespace min_fa3_varlen_demo {

void run_min_fa3_kvcache_fwd(Flash_fwd_params& params, cudaStream_t stream);
void run_min_fa3_kvcache_combine(Flash_fwd_params& params, cudaStream_t stream, bool enable_pdl);

}  // namespace min_fa3_varlen_demo
