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
    static constexpr bool EnableQueuedInitialWork = true;
    // MEGA_RING_ZIGZAG: this scheduler is instantiated with LPT == IsCausal in
    // the mega-ring launch. Effectively that means:
    //   - causal instantiations use LPT=true and enable zigzag
    //   - non-causal instantiations use LPT=false and keep the old path
    // LPT still originates from the base varlen scheduler's tile-ordering
    // policy; we reuse it here as the compile-time causal/zigzag gate to avoid
    // adding another template boolean.
    static constexpr bool EnableZigzag = LPT;
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
        int step_tile_idx;
        int reduction_tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            // MEGA_RING: the expanded stream is valid only while the ring step
            // is in range and the underlying varlen tile decoded to a batch.
            return (params.mega_ring_ready_once || ring_step < params.mega_ring_world_size) && bidb < params.num_batch;
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
    int get_actual_batch(Params const& params, int bidb) const {
        if constexpr (Prepared && Sort) {
            return bidb < params.num_batch ? params.varlen_batch_idx_ptr[bidb] : bidb;
        } else {
            return bidb;
        }
    }

    CUTLASS_DEVICE
    bool is_cp_virtual_batch(Params const& params, int bidb) const {
        if (params.mega_ring_cp_batch_mask == nullptr) {
            return true;
        }
        int const actual_batch = get_actual_batch(params, bidb);
        return actual_batch < params.num_batch && params.mega_ring_cp_batch_mask[actual_batch] != 0;
    }

    template<bool HalfMBlocks=false, bool CpOnly=false, bool LocalOnly=false>
    CUTLASS_DEVICE
    typename Base::WorkTileInfo
    tile_idx_to_work_tile_linear(Params const& params, int next_tile_idx) const {
        static_assert(Prepared, "Mega-ring hybrid scheduler requires prepared varlen metadata");
        static_assert(!Split && !PackGQA, "Mega-ring hybrid scheduler only supports the minimal non-split non-PackGQA path");
        int group_start_tile = 0;
        for (int bidb = 0; bidb < params.num_batch; ++bidb) {
            int num_m_blocks = params.num_m_blocks_ptr[bidb];
            if constexpr (HalfMBlocks) {
                num_m_blocks /= 2;
            }
            if constexpr (CpOnly) {
                if (!is_cp_virtual_batch(params, bidb)) {
                    num_m_blocks = 0;
                }
            }
            if constexpr (LocalOnly) {
                if (is_cp_virtual_batch(params, bidb)) {
                    num_m_blocks = 0;
                }
            }
            int const batch_tiles = num_m_blocks * params.num_head;
            if (num_m_blocks > 0 && next_tile_idx < group_start_tile + batch_tiles) {
                int const mh_block = next_tile_idx - group_start_tile;
                int block, bidh;
                if constexpr (LPT) {
                    int const nheads_in_l2 = params.num_nheads_in_l2_ptr[bidb];
                    int const mh_in_l2 = nheads_in_l2 * num_m_blocks;
                    int const section_idx = mh_block / mh_in_l2;
                    int const l2_mod = mh_block - section_idx * mh_in_l2;
                    int const nheads_remainder = params.num_head - section_idx * nheads_in_l2;
                    int const nheads_in_this_section = nheads_in_l2 <= nheads_remainder ? nheads_in_l2 : nheads_remainder;
                    block = l2_mod / nheads_in_this_section;
                    int const bidh_residual = l2_mod - block * nheads_in_this_section;
                    bidh = section_idx * nheads_in_l2 + bidh_residual;
                    block = num_m_blocks - 1 - block;
                } else {
                    bidh = mh_block / num_m_blocks;
                    block = mh_block - bidh * num_m_blocks;
                }
                return {group_start_tile, block, bidh, bidb};
            }
            group_start_tile += batch_tiles;
        }
        return {next_tile_idx, 0, 0, params.num_batch};
    }

    CUTLASS_DEVICE
    WorkTileInfo
    decode_ready_once_tile(Params const& params, int next_tile_idx) const {
        static constexpr int kReadyLocal = 0;
        static constexpr int kReadyCpFull = 1;
        static constexpr int kReadyCpFront = 2;
        static constexpr int kReadyCpBack = 3;
        int const tiles_per_step = params.mega_ring_tiles_per_step;
        int const cp_tiles_per_step = params.mega_ring_cp_tiles_per_step;
        int const local_tiles = tiles_per_step - cp_tiles_per_step;
        if (next_tile_idx < local_tiles) {
            typename Base::WorkTileInfo base_work =
                tile_idx_to_work_tile_linear<false, false, true>(params, next_tile_idx);
            return {base_work.tile_idx, base_work.block, base_work.bidh, base_work.bidb,
                    kReadyLocal, next_tile_idx, next_tile_idx, next_tile_idx};
        }
        int rem = next_tile_idx - local_tiles;
        if constexpr (EnableZigzag) {
            int const cp_tiles_per_half_step = params.mega_ring_cp_tiles_per_half_step;
            if (rem < cp_tiles_per_half_step) {
                typename Base::WorkTileInfo base_work =
                    tile_idx_to_work_tile_linear<true, true, false>(params, rem);
                return {base_work.tile_idx, base_work.block, base_work.bidh, base_work.bidb,
                        kReadyCpFront, next_tile_idx, rem, rem};
            }
            rem -= cp_tiles_per_half_step;
            if (rem < cp_tiles_per_half_step) {
                typename Base::WorkTileInfo base_work =
                    tile_idx_to_work_tile_linear<true, true, false>(params, rem);
                return {base_work.tile_idx, base_work.block, base_work.bidh, base_work.bidb,
                        kReadyCpBack, next_tile_idx, rem, rem};
            }
        } else {
            if (rem < cp_tiles_per_step) {
                typename Base::WorkTileInfo base_work =
                    tile_idx_to_work_tile_linear<false, true, false>(params, rem);
                return {base_work.tile_idx, base_work.block, base_work.bidh, base_work.bidb,
                        kReadyCpFull, next_tile_idx, rem, rem};
            }
        }
        return {next_tile_idx, 0, 0, params.num_batch, -1, next_tile_idx, 0, 0};
    }

    CUTLASS_DEVICE
    int cp_full_tile_idx_from_work(Params const& params, typename Base::WorkTileInfo const& work) const {
        if (work.bidb >= params.num_batch || !is_cp_virtual_batch(params, work.bidb)) {
            return 0;
        }
        int group_start_tile = 0;
        for (int bidb = 0; bidb < work.bidb; ++bidb) {
            if (is_cp_virtual_batch(params, bidb)) {
                group_start_tile += params.num_m_blocks_ptr[bidb] * params.num_head;
            }
        }
        int const num_m_blocks = params.num_m_blocks_ptr[work.bidb];
        int mh_block;
        if constexpr (LPT) {
            int const nheads_in_l2 = params.num_nheads_in_l2_ptr[work.bidb];
            int const section_idx = work.bidh / nheads_in_l2;
            int const bidh_residual = work.bidh - section_idx * nheads_in_l2;
            int const nheads_remainder = params.num_head - section_idx * nheads_in_l2;
            int const nheads_in_this_section = nheads_in_l2 <= nheads_remainder ? nheads_in_l2 : nheads_remainder;
            int const block_in_l2_order = num_m_blocks - 1 - work.block;
            mh_block = section_idx * nheads_in_l2 * num_m_blocks
                     + block_in_l2_order * nheads_in_this_section
                     + bidh_residual;
        } else {
            mh_block = work.bidh * num_m_blocks + work.block;
        }
        return group_start_tile + mh_block;
    }

    CUTLASS_DEVICE
    WorkTileInfo
    decode_mega_ring_tile(Params const& params,
                          int next_tile_idx, // expanded global tile id
                          int current_ring_step,
                          typename Base::WorkTileInfo const& current_base_work) const {
        // MEGA_RING_ZIGZAG: non-causal keeps the original equal-sized
        // [world_size][T_full] stream. Causal uses two constant-width sections:
        // steps 0..r decode full-Q tiles, steps r+1..N-1 decode half-Q tiles.
        bool const hybrid_mode = params.mega_ring_cp_batch_mask != nullptr;
        int const tiles_per_step = params.mega_ring_tiles_per_step;
        int const tiles_per_half_step = hybrid_mode ? params.mega_ring_cp_tiles_per_half_step : params.mega_ring_tiles_per_half_step;
        int const cp_tiles_per_step = hybrid_mode ? params.mega_ring_cp_tiles_per_step : tiles_per_step;
        if (params.mega_ring_ready_once) {
            return decode_ready_once_tile(params, next_tile_idx);
        }
        int ring_step, step_tile_idx;
        bool use_cp_stream = false;
        bool q_use_half = false;
        if (!hybrid_mode) {
            if constexpr (EnableZigzag) {
                int const full_section_tiles = (params.mega_ring_rank + 1) * tiles_per_step;
                if (next_tile_idx < full_section_tiles) {
                    ring_step = tiles_per_step > 0 ? next_tile_idx / tiles_per_step : 0;
                    step_tile_idx = tiles_per_step > 0 ? next_tile_idx - ring_step * tiles_per_step : next_tile_idx;
                } else {
                    int const rem = next_tile_idx - full_section_tiles;
                    int const q = tiles_per_half_step > 0 ? rem / tiles_per_half_step : 0;
                    ring_step = params.mega_ring_rank + 1 + q;
                    step_tile_idx = tiles_per_half_step > 0 ? rem - q * tiles_per_half_step : rem;
                    q_use_half = true;
                }
            } else {
                ring_step = tiles_per_step > 0 ? next_tile_idx / tiles_per_step : 0;
                step_tile_idx = tiles_per_step > 0 ? next_tile_idx - ring_step * tiles_per_step : next_tile_idx;
            }
        } else if constexpr (EnableZigzag) {
            if (next_tile_idx < tiles_per_step) {
                ring_step = 0;
                step_tile_idx = next_tile_idx;
            }
            else {
                int const rem = next_tile_idx - tiles_per_step;
                int const cp_full_section_tiles = params.mega_ring_rank * cp_tiles_per_step;
                if (rem < cp_full_section_tiles) {
                    if (cp_tiles_per_step <= 0) {
                        return {next_tile_idx, 0, 0, params.num_batch, params.mega_ring_world_size, next_tile_idx, 0, 0};
                    }
                    ring_step = 1 + rem / cp_tiles_per_step;
                    step_tile_idx = rem - (ring_step - 1) * cp_tiles_per_step;
                    use_cp_stream = true;
                } else {
                    int const rem_half = rem - cp_full_section_tiles;
                    if (tiles_per_half_step <= 0) {
                        return {next_tile_idx, 0, 0, params.num_batch, params.mega_ring_world_size, next_tile_idx, 0, 0};
                    }
                    int const q = rem_half / tiles_per_half_step;
                    ring_step = params.mega_ring_rank + 1 + q;
                    step_tile_idx = rem_half - q * tiles_per_half_step;
                    use_cp_stream = true;
                    q_use_half = true;
                }
            }
        } else {
            if (next_tile_idx < tiles_per_step) {
                ring_step = 0;
                step_tile_idx = next_tile_idx;
            } else {
                if (cp_tiles_per_step <= 0) {
                    return {next_tile_idx, 0, 0, params.num_batch, params.mega_ring_world_size, next_tile_idx, 0, 0};
                }
                int const rem = next_tile_idx - tiles_per_step;
                ring_step = 1 + rem / cp_tiles_per_step;
                step_tile_idx = rem - (ring_step - 1) * cp_tiles_per_step;
                use_cp_stream = true;
            }
        }
        if (ring_step >= params.mega_ring_world_size) {
            // MEGA_RING: mark the expanded tile as invalid using the same
            // bidb == num_batch sentinel as the base varlen scheduler while
            // preserving ring_step for validity checks.
            return {next_tile_idx, 0, 0, params.num_batch, ring_step, next_tile_idx, step_tile_idx, step_tile_idx};
        }
        // MEGA_RING: Base::tile_idx_to_work_tile assumes monotonic tile ids
        // within one varlen stream. Each new ring step restarts at tile 0, so
        // reset the decoder hint when step_tile_idx wraps backward.
        typename Base::WorkTileInfo decode_start =
            ring_step != current_ring_step || step_tile_idx < current_base_work.tile_idx || current_base_work.bidb >= params.num_batch
                ? typename Base::WorkTileInfo{0, 0, 0, 0}
                : current_base_work;
        typename Base::WorkTileInfo base_work = use_cp_stream
            ? (q_use_half
                ? tile_idx_to_work_tile_linear<true, true>(params, step_tile_idx)
                : tile_idx_to_work_tile_linear<false, true>(params, step_tile_idx))
            : (q_use_half
                ? Base::template tile_idx_to_work_tile_impl<true>(params, step_tile_idx, decode_start)
                : Base::template tile_idx_to_work_tile_impl<false>(params, step_tile_idx, decode_start));
        int reduction_tile_idx = use_cp_stream ? step_tile_idx : (hybrid_mode ? cp_full_tile_idx_from_work(params, base_work) : step_tile_idx);
        if constexpr (EnableZigzag) {
            if (q_use_half && base_work.bidb < params.num_batch) {
                // MEGA_RING_ZIGZAG: step_ready is indexed in the full-Q tile
                // stream. The half-Q scheduler decodes the back half using
                // num_m_blocks_ptr[b] / 2, then maps the tile back to its full
                // stream position. This relies on seqlen/2 being 128-aligned.
                int const full_num_m_blocks = params.num_m_blocks_ptr[base_work.bidb];
                int const half_num_m_blocks = full_num_m_blocks / 2;
                int const full_block = base_work.block + half_num_m_blocks;
                int const full_group_start_tile = base_work.tile_idx * 2;
                if constexpr (LPT) {
                    int const nheads_in_l2 = params.num_nheads_in_l2_ptr[base_work.bidb];
                    int const section_idx = base_work.bidh / nheads_in_l2;
                    int const bidh_residual = base_work.bidh - section_idx * nheads_in_l2;
                    int const nheads_remainder = params.num_head - section_idx * nheads_in_l2;
                    int const nheads_in_this_section = nheads_in_l2 <= nheads_remainder ? nheads_in_l2 : nheads_remainder;
                    int const block_in_l2_order = full_num_m_blocks - 1 - full_block;
                    int const mh_block = section_idx * nheads_in_l2 * full_num_m_blocks
                                       + block_in_l2_order * nheads_in_this_section
                                       + bidh_residual;
                    reduction_tile_idx = full_group_start_tile + mh_block;
                } else {
                    reduction_tile_idx = full_group_start_tile + base_work.bidh * full_num_m_blocks + full_block;
                }
            }
        }
        return {base_work.tile_idx, base_work.block, base_work.bidh, base_work.bidb,
                ring_step, next_tile_idx, step_tile_idx, reduction_tile_idx};
    }

public:
    CUTLASS_DEVICE
    void
    publish_work_to_smem(WorkTileInfo const& work_info, int global_tile_idx) const {
        if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
            typename Base::SharedStorage* smem = mega_ring_work_info_smem();
            smem[0] = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            // MEGA_RING: the second int4 slot carries ring_step and the
            // original global tile id for the matching consumer warpgroup.
            smem[1] = make_int4(work_info.ring_step, global_tile_idx, work_info.step_tile_idx, work_info.reduction_tile_idx);
        }
        flash::named_barrier_arrive(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);   // TileCountSmemFull
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            int const next_tile_idx = Base::virtual_block_idx(params);
            WorkTileInfo work_info = decode_mega_ring_tile(params, next_tile_idx, -1, {0, 0, 0, 0});
            publish_work_to_smem(work_info, next_tile_idx);
            return work_info;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0, 0, 0, 0, 0});
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work_from_queue(Params const& params) const {
        if constexpr (IsProducerWarp) {
            int next_tile_idx = 0;
            if (threadIdx.x % NumProducerThreads == 0) {
                next_tile_idx = atomicAdd(params.tile_count_semaphore, 1) + Base::virtual_grid_dim_x(params);
            }
            next_tile_idx = __shfl_sync(0xffffffff, next_tile_idx, 0 /*lane*/);
            WorkTileInfo work_info = decode_mega_ring_tile(params, next_tile_idx, -1, {0, 0, 0, 0});
            publish_work_to_smem(work_info, next_tile_idx);
            return work_info;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0, 0, 0, 0, 0});
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
            WorkTileInfo work_info = decode_mega_ring_tile(params, new_tile_idx, current_work.ring_step, current_base_work);
            flash::named_barrier_sync(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);  // TileCountSmemEmpty
            publish_work_to_smem(work_info, new_tile_idx);
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
            return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w,
                                mega_info.x, mega_info.y, mega_info.z, mega_info.w};
        }
    }
};

}  // namespace flash
