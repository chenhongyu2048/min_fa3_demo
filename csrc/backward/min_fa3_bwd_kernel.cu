// Copied and trimmed from Hopper backward instantiation source:
// - hopper/instantiations/flash_bwd_hdim128_bf16_sm90.cu

#include "backward/min_fa3_bwd_launch.h"

template void min_fa3_backward::run_min_fa3_bwd_sm90<false>(
    min_fa3_backward::Flash_bwd_params& params,
    cudaStream_t stream);
template void min_fa3_backward::run_min_fa3_bwd_sm90<true>(
    min_fa3_backward::Flash_bwd_params& params,
    cudaStream_t stream);
