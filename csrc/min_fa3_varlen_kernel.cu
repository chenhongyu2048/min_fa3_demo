// Copied and trimmed from Hopper forward sources:
// - hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu

#include "min_fa3_varlen_launch.h"

template void min_fa3_varlen_demo::run_min_fa3_varlen_sm90<false>(
    min_fa3_varlen_demo::Flash_fwd_params& params,
    cudaStream_t stream);
template void min_fa3_varlen_demo::run_min_fa3_varlen_sm90<true>(
    min_fa3_varlen_demo::Flash_fwd_params& params,
    cudaStream_t stream);
