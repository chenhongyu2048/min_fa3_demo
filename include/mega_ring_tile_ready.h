#pragma once

#include "cutlass/cutlass.h"

namespace min_fa3_varlen_demo::mega_ring {

struct alignas(16) CommWindowDesc {
    int batch_idx;
    int ring_step;
    int kv_begin;
    int kv_end;
};
static_assert(sizeof(CommWindowDesc) == 4 * sizeof(int));

CUTLASS_HOST_DEVICE
int remote_tiles(int half_tiles, int ring_size, int local_rank) {
    return local_rank * half_tiles + 2 * (ring_size - 1 - local_rank) * half_tiles;
}

CUTLASS_HOST_DEVICE
int q_remote_tiles(int m_block, int half_tiles, int ring_size, int local_rank) {
    return m_block < half_tiles ? local_rank * half_tiles
        : remote_tiles(half_tiles, ring_size, local_rank);
}

CUTLASS_HOST_DEVICE
int step_begin(int half_tiles, int local_rank, int step) {
    return step <= local_rank ? (step - 1) * half_tiles
        : local_rank * half_tiles + (step - local_rank - 1) * 2 * half_tiles;
}

}  // namespace min_fa3_varlen_demo::mega_ring
