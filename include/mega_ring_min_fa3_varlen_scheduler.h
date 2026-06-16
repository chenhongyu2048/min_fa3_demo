// Mega ring variant copied and trimmed from include/min_fa3_varlen_scheduler.h.
// Changes are marked with MEGA_RING comments.

#pragma once

#include "min_fa3_varlen_scheduler.h"

namespace flash {

// MEGA_RING: persistent scheduler that maps the global tile stream to
// (ring_step, original_varlen_tile). The underlying varlen tile decoding stays
// copied from VarlenDynamicPersistentTileScheduler.
template<int kBlockM, int kBlockN, int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp,
         bool Split=false, bool PackGQA=false, bool WarpSpecialized=true, bool LPT = false, bool Sort = false, bool Prepared = true>
class MegaRingVarlenDynamicPersistentTileScheduler: public VarlenDynamicPersistentTileScheduler<kBlockM, kBlockN, NumMmaThreads, NumProducerThreads,
                                                                                                Split, PackGQA, WarpSpecialized, LPT, Sort, Prepared> {
    using Base = VarlenDynamicPersistentTileScheduler<kBlockM, kBlockN, NumMmaThreads, NumProducerThreads, Split, PackGQA, WarpSpecialized, LPT, Sort, Prepared>;

public:
    static constexpr bool EnableMegaRing = true;
    // MEGA_RING: one int4 for the original varlen work tile and one int4 for
    // ring metadata shared between producer and consumer warpgroups.
    using SharedStorage = cute::array<typename Base::SharedStorage, 2>;
    using Params = typename Base::Params;

    struct WorkTileInfo {
        int tile_idx, block, bidh, bidb;
        // MEGA_RING: ring_step selects the KV rank for this replay of the
        // original varlen tile; global_tile_idx is the expanded scheduler id
        // used later to recover the same Q tile across ring steps.
        int ring_step;
        int global_tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            // MEGA_RING: the expanded stream is valid only while the ring step
            // is in range and the underlying varlen tile decoded to a batch.
            return ring_step < params.mega_ring_world_size && bidb < params.num_batch;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            // MEGA_RING: block coordinates stay identical to the base varlen
            // scheduler. Ring metadata is consumed by mainloop/epilogue code,
            // not folded into the FA3 block coordinate.
            typename Base::WorkTileInfo base_work{tile_idx, block, bidh, bidb};
            return base_work.get_block_coord(params);
        }
    };

    CUTLASS_DEVICE
    MegaRingVarlenDynamicPersistentTileScheduler(SharedStorage* const smem_scheduler)
        // MEGA_RING: Base only needs a pointer to the first int4 slot; this
        // variant reserves the second slot for ring metadata.
        : Base(reinterpret_cast<typename Base::SharedStorage*>(smem_scheduler)) {}

    CUTLASS_DEVICE
    typename Base::SharedStorage* mega_ring_work_info_smem() const {
        // MEGA_RING: reinterpret the two-slot shared storage as base int4
        // entries so producer and consumer can exchange both base work and
        // ring metadata using the copied scheduler barrier protocol.
        return reinterpret_cast<typename Base::SharedStorage*>(this->work_info_smem);
    }

private:
    CUTLASS_DEVICE
    WorkTileInfo
    decode_mega_ring_tile(Params const& params,
                          int next_tile_idx, // expanded global tile id
                          typename Base::WorkTileInfo const& current_base_work) const {
        // MEGA_RING: tile ids are laid out as all original varlen tiles for
        // step 0, then all original varlen tiles for step 1, and so on.
        int const tiles_per_step = params.mega_ring_tiles_per_step;
        int ring_step = tiles_per_step > 0 ? next_tile_idx / tiles_per_step : 0;
        int step_tile_idx = tiles_per_step > 0 ? next_tile_idx - ring_step * tiles_per_step : next_tile_idx;
        if (ring_step >= params.mega_ring_world_size) {
            // MEGA_RING: mark the expanded tile as invalid using the same
            // bidb == num_batch sentinel as the base varlen scheduler while
            // preserving ring_step for validity checks.
            return {next_tile_idx, 0, 0, params.num_batch, ring_step, next_tile_idx};
        }
        // MEGA_RING: Base::tile_idx_to_work_tile assumes monotonic tile ids
        // within one varlen stream. Each new ring step restarts at tile 0, so
        // reset the decoder hint when step_tile_idx wraps backward.
        typename Base::WorkTileInfo decode_start =
            step_tile_idx < current_base_work.tile_idx || current_base_work.bidb >= params.num_batch
                ? typename Base::WorkTileInfo{0, 0, 0, 0}
                : current_base_work;
        typename Base::WorkTileInfo base_work = Base::tile_idx_to_work_tile(params, step_tile_idx, decode_start);
        return {base_work.tile_idx, base_work.block, base_work.bidh, base_work.bidb, ring_step, next_tile_idx};
    }

public:
    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            int const next_tile_idx = Base::virtual_block_idx(params);
            WorkTileInfo work_info = decode_mega_ring_tile(params, next_tile_idx, {0, 0, 0, 0});
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                typename Base::SharedStorage* smem = mega_ring_work_info_smem();
                smem[0] = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
                // MEGA_RING: the second int4 slot carries ring_step and the
                // original global tile id for the matching consumer warpgroup.
                smem[1] = make_int4(work_info.ring_step, next_tile_idx, 0, 0);
            }
            flash::named_barrier_arrive(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);   // TileCountSmemFull
            return work_info;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0, 0, 0});
        }
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            // MEGA_RING: tile_count_semaphore walks the expanded stream
            // [world_size][tiles_per_step], not just the base varlen stream.
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + Base::virtual_grid_dim_x(params);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            // thread 0 has the next tile_idx, just need to broadcast to the rest of warp 0
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
            // MEGA_RING: reconstruct the base decoder hint from the previous
            // per-step varlen tile, then decode the new expanded tile id.
            typename Base::WorkTileInfo current_base_work{__shfl_sync(0xffffffff, current_work.tile_idx, 1 /*lane*/),
                                                          current_work.block,
                                                          current_work.bidh,
                                                          current_work.bidb};
            WorkTileInfo work_info = decode_mega_ring_tile(params, new_tile_idx, current_base_work);
            flash::named_barrier_sync(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);  // TileCountSmemEmpty
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                typename Base::SharedStorage* smem = mega_ring_work_info_smem();
                smem[0] = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
                smem[1] = make_int4(work_info.ring_step, new_tile_idx, 0, 0);
            }
            flash::named_barrier_arrive(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);  // TileCountSmemFull
            return work_info;
        } else {
            flash::named_barrier_sync(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);  // TileCountSmemFull
            typename Base::SharedStorage* smem = const_cast<MegaRingVarlenDynamicPersistentTileScheduler*>(this)->mega_ring_work_info_smem();
            int4 work_info = smem[0];
            // MEGA_RING: the consumer needs both the original varlen tile and
            // the expanded global tile id; the latter determines ring-step
            // ordering for in-place O/LSE reduction.
            int4 mega_info = smem[1];
            flash::named_barrier_arrive(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);  // TileCountSmemEmpty
            return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w, mega_info.x, mega_info.y};
        }
    }
};

}  // namespace flash
