// DCP mega instance copied and trimmed from the Hopper varlen launch template.
#include "dcp_mega_min_fa3_varlen_launch.h"

namespace min_fa3_varlen_demo::dcp_mega {
void run_pack_split_bn128(DCPMega_fwd_params& p, cudaStream_t s) {
    #define DCP_MEGA_CASE(DCP) \
        case DCP: \
            if (p.hq_local == 4) { \
                detail::launch_dcp_mega_instance<true, 128, DCP, 4>(p, s); \
            } else { \
                detail::launch_dcp_mega_instance<true, 128, DCP, 8>(p, s); \
            } \
            break
    switch (p.dcp_size) {
        DCP_MEGA_CASE(2);
        DCP_MEGA_CASE(4);
        DCP_MEGA_CASE(8);
        default: TORCH_CHECK(false, "unsupported DCP size");
    }
    #undef DCP_MEGA_CASE
}
}  // namespace min_fa3_varlen_demo::dcp_mega
