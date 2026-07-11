// Mega ring helper copied and trimmed from:
// - third_party/ThunderKittens/kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu
// - include/min_fa3_epilogue.h online-reduction synchronization needs.
// Changes are marked with MEGA_RING comments.

#pragma once

#include <cutlass/cutlass.h>

namespace min_fa3_varlen_demo::mega_ring {

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
