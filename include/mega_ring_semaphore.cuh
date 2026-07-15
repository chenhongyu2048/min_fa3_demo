// Mega ring helper copied and trimmed from:
// - third_party/ThunderKittens/kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu
// - include/min_fa3_epilogue.h online-reduction synchronization needs.
// Changes are marked with MEGA_RING comments.

#pragma once

#include <cutlass/cutlass.h>

namespace min_fa3_varlen_demo::mega_ring {

enum : int {
    kTileStateBusy = 1 << 30,
    // MEGA_RING_SEGMENTS: one long-lived int carries the claimed causal
    // segment. Ring sizes are limited to 2/4/8, so four bits also cover the
    // invalid world_size sentinel (8).
    kSegmentBeginMask = 0x0f,
    kSegmentEndShift = 4,
    kSegmentEndMask = 0x0f << kSegmentEndShift,
    kSegmentTerminalBit = 1 << 8,
};

CUTLASS_HOST_DEVICE
constexpr int pack_segment_meta(int begin_step, int end_step, bool terminal_chunk) {
    return (begin_step & kSegmentBeginMask)
        | ((end_step & kSegmentBeginMask) << kSegmentEndShift)
        | (terminal_chunk ? kSegmentTerminalBit : 0);
}

CUTLASS_HOST_DEVICE
constexpr int segment_begin_step(int segment_meta) {
    return segment_meta & kSegmentBeginMask;
}

CUTLASS_HOST_DEVICE
constexpr int segment_end_step(int segment_meta) {
    return (segment_meta & kSegmentEndMask) >> kSegmentEndShift;
}

CUTLASS_HOST_DEVICE
constexpr bool segment_is_terminal(int segment_meta) {
    return (segment_meta & kSegmentTerminalBit) != 0;
}

CUTLASS_DEVICE
int load_acquire(int const* address) {
    int value;
    asm volatile("{ld.acquire.gpu.global.s32 %0, [%1];}" : "=r"(value) : "l"(address) : "memory");
    return value;
}

CUTLASS_DEVICE
void store_release(int* address, int value) {
    asm volatile("{st.release.gpu.global.s32 [%0], %1;}" :: "l"(address), "r"(value) : "memory");
}

CUTLASS_DEVICE
int compare_exchange_acquire(int* address, int compare, int value) {
    int old;
    asm volatile("{atom.acquire.gpu.global.cas.b32 %0, [%1], %2, %3;}"
                 : "=r"(old) : "l"(address), "r"(compare), "r"(value) : "memory");
    return old;
}

// MEGA_RING: wait on a monotonically increasing device-local global counter.
// This is used for K/V readiness and per-Q-tile ring-step ordering where the
// consumer CTA can poll the count directly.
CUTLASS_DEVICE
void wait_until_at_least(int const* counter, int target) {
    if (counter == nullptr) {
        return;
    }
    int value = 0;
    do {
        asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(value) : "l"(counter) : "memory");
        if (value < target) {
            __nanosleep(64);
        }
    } while (value < target);
}

// MEGA_RING: acquire variant for counters that publish TMA stores. Observing
// the target count also makes the async-proxy writes visible to the consumer.
CUTLASS_DEVICE
void wait_until_at_least_acquire(int const* counter, int target) {
    if (counter == nullptr || target <= 0) {
        return;
    }
    int value = 0;
    do {
        asm volatile("{ld.acquire.gpu.global.s32 %0, [%1];}" : "=r"(value) : "l"(counter) : "memory");
        if (value < target) {
            __nanosleep(64);
        }
    } while (value < target);
}

// MEGA_RING: no-return release add for device-local readiness counters. The
// consumer CTA observes readiness by polling the counter value, so the producer
// CTA does not need an atomic return value.
CUTLASS_DEVICE
void signal_release(int* counter, int value) {
    if (counter == nullptr) {
        return;
    }
    asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(counter), "r"(value) : "memory");
}

// MEGA_RING: system-scope publication for a counter owned by a peer GPU.
// The matching acquire load orders the owner's postprocess after all remote
// TMA stores completed before this signal.
CUTLASS_DEVICE
void signal_release_system(int* counter, int value) {
    if (counter == nullptr) {
        return;
    }
    asm volatile("{red.release.sys.global.add.s32 [%0], %1;}" :: "l"(counter), "r"(value) : "memory");
}

CUTLASS_DEVICE
void wait_until_at_least_acquire_system(int const* counter, int target) {
    if (counter == nullptr || target <= 0) {
        return;
    }
    int value = 0;
    do {
        asm volatile("{ld.acquire.sys.global.s32 %0, [%1];}" : "=r"(value) : "l"(counter) : "memory");
        if (value < target) {
            __nanosleep(64);
        }
    } while (value < target);
}

}  // namespace min_fa3_varlen_demo::mega_ring
