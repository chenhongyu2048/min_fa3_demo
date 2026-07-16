// Shared topology descriptors copied and trimmed from the local hierarchical
// Hopper mega-ring forward path.  Forward and backward keep separate params
// structs and share only this fixed G8/G4/G2/G1 topology metadata.

#pragma once

namespace min_fa3_varlen_demo {

constexpr int kMegaRingNumLevels = 4;
constexpr int kMegaRingNumKvReadySections = 11;
constexpr int kMegaRingNumDkvSections = 15;

struct MegaRingLevelDesc {
    int ring_size = 1;
    int batch_begin = 0;
    int batch_end = 0;
    int row_begin = 0;
    int full_rows = 0;
    int half_row_begin = 0;
    int half_rows = 0;
    int full_tiles = 0;
    int half_tiles = 0;
    int reduction_base = 0;
    int kv_ready_base = 0;
};

struct MegaRingHierarchyDesc {
    MegaRingLevelDesc levels[kMegaRingNumLevels]{};
    int base_work_tiles = 0;
    int total_work_tiles = 0;
    int reduction_tiles = 0;
    int remote_tiles = 0;
};

#if defined(__CUDACC__)
#define MIN_FA3_MEGARING_HD __host__ __device__
#else
#define MIN_FA3_MEGARING_HD
#endif

MIN_FA3_MEGARING_HD constexpr int mega_ring_size_for_level(int level_idx) {
    return 8 >> level_idx;
}

MIN_FA3_MEGARING_HD constexpr int mega_ring_kv_ready_base(int level_idx) {
    return level_idx == 0 ? 0 : level_idx == 1 ? 7 : level_idx == 2 ? 10 : 11;
}

MIN_FA3_MEGARING_HD constexpr int mega_ring_dkv_section_base(int level_idx) {
    return level_idx == 0 ? 0 : level_idx == 1 ? 8 : level_idx == 2 ? 12 : 14;
}

MIN_FA3_MEGARING_HD constexpr int mega_ring_dkv_section(int level_idx, int ring_step) {
    return mega_ring_dkv_section_base(level_idx) + ring_step;
}

MIN_FA3_MEGARING_HD constexpr int mega_ring_level_for_size(int ring_size) {
    return ring_size == 8 ? 0 : ring_size == 4 ? 1 : ring_size == 2 ? 2 : 3;
}

#undef MIN_FA3_MEGARING_HD

}  // namespace min_fa3_varlen_demo
