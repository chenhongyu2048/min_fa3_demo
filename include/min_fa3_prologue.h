// Copied and trimmed from Hopper forward sources:
// - hopper/flash_fwd_kernel_sm90.h
// This file holds the minimal forward prologue used by the demo kernel:
// descriptor prefetch and barrier initialization before producer/consumer execution.

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>

namespace min_fa3_demo {

template <class CollectiveMainloop, class CollectiveEpilogue, class SharedStorage, class Params>
CUTLASS_DEVICE void run_prologue(Params const& params, SharedStorage& shared_storage, int warp_idx, int lane_predicate) {
    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
        CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    if (warp_idx == 0 && lane_predicate) {
        shared_storage.pipelines.barrier_Q.init(CollectiveMainloop::Use_TMA_Q ? 1 : CollectiveMainloop::NumProducerThreads);
        shared_storage.pipelines.barrier_O.init(cute::size(typename CollectiveMainloop::ClusterShape{})
                                                * (CollectiveEpilogue::Use_TMA_O ? 1 : CollectiveMainloop::NumMmaThreads));
    }
}

}  // namespace min_fa3_demo
