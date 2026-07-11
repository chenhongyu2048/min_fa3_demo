// Mega ring helper copied and trimmed from:
// - third_party/ThunderKittens/kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu
// - include/min_fa3_epilogue.h online-reduction synchronization needs.
// Changes are marked with MEGA_RING comments.

#pragma once

#include <cutlass/cutlass.h>
#include <cstdint>

namespace min_fa3_varlen_demo::mega_ring {

// SM90 TMA writes global memory through the async proxy.  A producer must
// bridge that proxy before publishing the data to a generic-proxy reader in a
// different CTA.
CUTLASS_DEVICE
void fence_proxy_async_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

CUTLASS_DEVICE
uint32_t atomic_add_acq_rel_gpu(uint32_t* counter, uint32_t value) {
    uint32_t previous;
    asm volatile(
        "atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
        : "=r"(previous) : "l"(counter), "r"(value) : "memory");
    return previous;
}

CUTLASS_DEVICE
int atomic_add_acq_rel_gpu(int* counter, int value) {
    int previous;
    asm volatile(
        "atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
        : "=r"(previous) : "l"(counter), "r"(value) : "memory");
    return previous;
}

CUTLASS_DEVICE
int load_acquire_gpu(int const* ptr) {
    int value;
    asm volatile(
        "ld.acquire.gpu.global.s32 %0, [%1];"
        : "=r"(value) : "l"(ptr) : "memory");
    return value;
}

CUTLASS_DEVICE
void store_release_gpu(int* ptr, int value) {
    asm volatile(
        "st.release.gpu.global.s32 [%0], %1;"
        :: "l"(ptr), "r"(value) : "memory");
}

CUTLASS_DEVICE
int atomic_cas_acq_rel_gpu(int* ptr, int compare, int value) {
    int previous;
    asm volatile(
        "atom.acq_rel.gpu.global.cas.b32 %0, [%1], %2, %3;"
        : "=r"(previous)
        : "l"(ptr), "r"(compare), "r"(value)
        : "memory");
    return previous;
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

// MEGA_RING: acquire variant for device-local publication counters. TMA
// producers must execute fence_proxy_async_global() before the matching
// release; the acquire alone does not bridge proxy domains.
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
