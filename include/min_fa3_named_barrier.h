// Copied and trimmed from Hopper forward sources:
// - hopper/named_barrier.hpp

#pragma once

#include "cutlass/arch/barrier.h"

namespace flash {

CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, uint32_t barrier_id_) {
    static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
}

CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, cutlass::arch::ReservedNamedBarriers reserved_named_barriers) {
    uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
}

CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, uint32_t barrier_id_) {
    static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, cutlass::arch::ReservedNamedBarriers reserved_named_barriers) {
    uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

enum class FwdNamedBarriers {
    QueryEmpty = 0,
    WarpSchedulerWG1 = 1,
    WarpSchedulerWG2 = 2,
    WarpSchedulerWG3 = 3,
    AppendKV = 4,
    QueryRotated = 5,
    PFull = 6,
    PEmpty = 7,
};

}  // namespace flash
