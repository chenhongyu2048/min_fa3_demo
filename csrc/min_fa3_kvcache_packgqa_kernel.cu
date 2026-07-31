// Copied and trimmed from:
// hopper/instantiations/flash_fwd_hdim128_bf16_packgqa_sm90.cu

#include "min_fa3_varlen_launch.h"

template void min_fa3_varlen_demo::run_min_fa3_varlen_sm90<false, true, false>(
    min_fa3_varlen_demo::Flash_fwd_params& params,
    cudaStream_t stream,
    std::optional<int> manual_block_count);
template void min_fa3_varlen_demo::run_min_fa3_varlen_sm90<true, true, false>(
    min_fa3_varlen_demo::Flash_fwd_params& params,
    cudaStream_t stream,
    std::optional<int> manual_block_count);
