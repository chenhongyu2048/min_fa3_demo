// Copied and trimmed from Hopper forward sources:
// - hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu

#include "min_fa3_launch.h"

template void min_fa3_demo::run_min_fa3_sm90<false>(Flash_fwd_params& params, cudaStream_t stream);
template void min_fa3_demo::run_min_fa3_sm90<true>(Flash_fwd_params& params, cudaStream_t stream);
