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
    int get_actual_batch(Params const& params, int bidb) const {
        if constexpr (Prepared && Sort) {
            return bidb < params.num_batch ? params.varlen_batch_idx_ptr[bidb] : bidb;
        } else {
            return bidb;
        }
    }

    CUTLASS_DEVICE
    int virtual_batch_ring_size(Params const& params, int bidb) const {
        int const actual_batch = get_actual_batch(params, bidb);
        return actual_batch < params.num_batch ? params.mega_ring_ring_sizes[actual_batch] : 0;
    }

    template<bool HalfMBlocks=false>
    CUTLASS_DEVICE
    typename Base::WorkTileInfo
    tile_idx_to_level_work_linear(Params const& params, int next_tile_idx, int target_ring_size) const {
        static_assert(Prepared, "Mega-ring hybrid scheduler requires prepared varlen metadata");
        static_assert(!Split && !PackGQA, "Mega-ring hybrid scheduler only supports the minimal non-split non-PackGQA path");
        int group_start_tile = 0;
        for (int bidb = 0; bidb < params.num_batch; ++bidb) {
            int num_m_blocks = params.num_m_blocks_ptr[bidb];
            if constexpr (HalfMBlocks) {
                num_m_blocks /= 2;
            }
            if (virtual_batch_ring_size(params, bidb) != target_ring_size) {
                num_m_blocks = 0;
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
    int level_full_tile_idx_from_work(Params const& params,
                                      typename Base::WorkTileInfo const& work,
                                      int target_ring_size,
                                      bool q_use_half) const {
        if (work.bidb >= params.num_batch || virtual_batch_ring_size(params, work.bidb) != target_ring_size) {
            return 0;
        }
        int group_start_tile = 0;
        for (int bidb = 0; bidb < work.bidb; ++bidb) {
            if (virtual_batch_ring_size(params, bidb) == target_ring_size) {
                group_start_tile += params.num_m_blocks_ptr[bidb] * params.num_head;
            }
        }
        int const num_m_blocks = params.num_m_blocks_ptr[work.bidb];
        int const full_block = q_use_half ? work.block + num_m_blocks / 2 : work.block;
        int mh_block;
        if constexpr (LPT) {
            int const nheads_in_l2 = params.num_nheads_in_l2_ptr[work.bidb];
            int const section_idx = work.bidh / nheads_in_l2;
            int const bidh_residual = work.bidh - section_idx * nheads_in_l2;
            int const nheads_remainder = params.num_head - section_idx * nheads_in_l2;
            int const nheads_in_this_section = nheads_in_l2 <= nheads_remainder ? nheads_in_l2 : nheads_remainder;
            int const block_in_l2_order = num_m_blocks - 1 - full_block;
            mh_block = section_idx * nheads_in_l2 * num_m_blocks
                     + block_in_l2_order * nheads_in_this_section
                     + bidh_residual;
        } else {
            mh_block = work.bidh * num_m_blocks + full_block;
        }
        return group_start_tile + mh_block;
    }

    CUTLASS_DEVICE
    WorkTileInfo
    decode_mega_ring_tile(Params const& params,
                          int next_tile_idx, // expanded global tile id
                          int current_ring_step,
                          typename Base::WorkTileInfo const& current_base_work) const {
        if (next_tile_idx >= params.mega_ring_hierarchy.total_work_tiles) {
            return {next_tile_idx, 0, 0, params.num_batch, params.mega_ring_world_size,
                    next_tile_idx, 0, 0};
        }
        int tiles_all = 0;
        #pragma unroll
        for (int level_idx = 0; level_idx < 4; ++level_idx) {
            tiles_all += params.mega_ring_hierarchy.levels[level_idx].full_tiles;
        }
        int ring_step = 0;
        int step_tile_idx = next_tile_idx;
        int target_ring_size = 0;
        int target_level_idx = -1;
        bool q_use_half = false;
        if (next_tile_idx >= tiles_all) {
            int rem = next_tile_idx - tiles_all;
            #pragma unroll
            for (int level_idx = 0; level_idx < 3; ++level_idx) {
                auto const& level = params.mega_ring_hierarchy.levels[level_idx];
                int const ring_base = (params.mega_ring_rank / level.ring_size) * level.ring_size;
                int const ring_local_rank = params.mega_ring_rank - ring_base;
                int const full_steps = EnableZigzag ? ring_local_rank : level.ring_size - 1;
                int const full_section_tiles = full_steps * level.full_tiles;
                int const half_steps = EnableZigzag ? level.ring_size - 1 - ring_local_rank : 0;
                int const half_section_tiles = half_steps * level.half_tiles;
                if (rem < full_section_tiles) {
                    target_ring_size = level.ring_size;
                    target_level_idx = level_idx;
                    ring_step = 1 + rem / level.full_tiles;
                    step_tile_idx = rem - (ring_step - 1) * level.full_tiles;
                    break;
                }
                rem -= full_section_tiles;
                if (rem < half_section_tiles) {
                    target_ring_size = level.ring_size;
                    target_level_idx = level_idx;
                    int const half_step_idx = rem / level.half_tiles;
                    ring_step = ring_local_rank + 1 + half_step_idx;
                    step_tile_idx = rem - half_step_idx * level.half_tiles;
                    q_use_half = true;
                    break;
                }
                rem -= half_section_tiles;
            }
        }
        typename Base::WorkTileInfo base_work;
        if (target_ring_size == 0) {
            typename Base::WorkTileInfo decode_start = current_ring_step == 0
                && step_tile_idx >= current_base_work.tile_idx
                && current_base_work.bidb < params.num_batch
                    ? current_base_work
                    : typename Base::WorkTileInfo{0, 0, 0, 0};
            base_work = Base::template tile_idx_to_work_tile_impl<false>(params, step_tile_idx, decode_start);
            if (base_work.bidb < params.num_batch) {
                target_ring_size = virtual_batch_ring_size(params, base_work.bidb);
                #pragma unroll
                for (int level_idx = 0; level_idx < 4; ++level_idx) {
                    if (params.mega_ring_hierarchy.levels[level_idx].ring_size == target_ring_size) {
                        target_level_idx = level_idx;
                    }
                }
            }
        } else {
            base_work = q_use_half
                ? tile_idx_to_level_work_linear<true>(params, step_tile_idx, target_ring_size)
                : tile_idx_to_level_work_linear<false>(params, step_tile_idx, target_ring_size);
        }
        int reduction_tile_idx = 0;
        if (target_level_idx >= 0 && target_ring_size > 1 && base_work.bidb < params.num_batch) {
            reduction_tile_idx = params.mega_ring_hierarchy.levels[target_level_idx].reduction_base
                + level_full_tile_idx_from_work(params, base_work, target_ring_size, q_use_half);
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
