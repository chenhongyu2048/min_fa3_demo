// Forward-ablation schedulers copied and trimmed from
// include/mega_ring_min_fa3_varlen_scheduler.h. The linear and step variants
// support the motivation CP4/CP8 instances and do not replace the
// production dynamic scheduler.

#pragma once

#include "min_fa3_varlen_scheduler.h"
#include "mega_ring_semaphore.cuh"

namespace flash {

template<int kBlockM, int kBlockN,
         int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup,
         int NumProducerThreads=cutlass::NumThreadsPerWarp,
         bool StepOnly=false, bool CollectStats=false>
class MegaRingVarlenLinearPersistentTileScheduler
    : public VarlenDynamicPersistentTileScheduler<
          kBlockM, kBlockN, NumMmaThreads, NumProducerThreads,
          false, false, true, true, true, true> {
    using Base = VarlenDynamicPersistentTileScheduler<
        kBlockM, kBlockN, NumMmaThreads, NumProducerThreads,
        false, false, true, true, true, true>;

public:
    static constexpr bool EnableMegaRing = true;
    static constexpr bool EnableQueuedInitialWork = true;
    static constexpr bool CollectMegaRingStats = CollectStats;
    static constexpr bool EnableZigzag = true;
    static constexpr bool EnableChunkedSegments = false;
    using SharedStorage = cute::array<typename Base::SharedStorage, 2>;
    using Params = typename Base::Params;

    struct WorkTileInfo {
        int tile_idx, block, bidh, bidb;
        int segment_meta;
        int reduction_tile_idx;

        CUTLASS_DEVICE bool is_valid(Params const& params) const {
            return min_fa3_varlen_demo::mega_ring::segment_begin_step(segment_meta)
                    < params.mega_ring_world_size
                && bidb < params.num_batch;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            typename Base::WorkTileInfo base_work{tile_idx, block, bidh, bidb};
            return base_work.get_block_coord(params);
        }
    };

private:
    bool recycled_cta_ = false;

    CUTLASS_DEVICE typename Base::SharedStorage* scheduler_smem() const {
        return reinterpret_cast<typename Base::SharedStorage*>(this->work_info_smem);
    }

    CUTLASS_DEVICE int actual_batch(Params const& params, int bidb) const {
        return bidb < params.num_batch ? params.varlen_batch_idx_ptr[bidb] : bidb;
    }

    CUTLASS_DEVICE int batch_ring_size(Params const& params, int bidb) const {
        int const batch = actual_batch(params, bidb);
        return batch < params.num_batch ? params.mega_ring_ring_sizes[batch] : 0;
    }

    template<bool HalfMBlocks=false>
    CUTLASS_DEVICE typename Base::WorkTileInfo
    tile_idx_to_all_cp_work(Params const& params, int tile_idx) const {
        int group_start_tile = 0;
        for (int bidb = 0; bidb < params.num_batch; ++bidb) {
            int num_m_blocks = params.num_m_blocks_ptr[bidb];
            if constexpr (HalfMBlocks) { num_m_blocks /= 2; }
            if (batch_ring_size(params, bidb) != params.mega_ring_world_size) { num_m_blocks = 0; }
            int const batch_tiles = num_m_blocks * params.num_head;
            if (num_m_blocks > 0 && tile_idx < group_start_tile + batch_tiles) {
                int const mh_block = tile_idx - group_start_tile;
                int const nheads_in_l2 = params.num_nheads_in_l2_ptr[bidb];
                int const mh_in_l2 = nheads_in_l2 * num_m_blocks;
                int const section_idx = mh_block / mh_in_l2;
                int const l2_mod = mh_block - section_idx * mh_in_l2;
                int const nheads_remainder = params.num_head - section_idx * nheads_in_l2;
                int const nheads_in_section = nheads_in_l2 <= nheads_remainder
                    ? nheads_in_l2 : nheads_remainder;
                int block = l2_mod / nheads_in_section;
                int const bidh = section_idx * nheads_in_l2
                    + l2_mod - block * nheads_in_section;
                block = num_m_blocks - 1 - block;
                return {group_start_tile, block, bidh, bidb};
            }
            group_start_tile += batch_tiles;
        }
        return {tile_idx, 0, 0, params.num_batch};
    }

    CUTLASS_DEVICE int full_tile_idx(
        Params const& params,
        typename Base::WorkTileInfo const& work,
        bool q_use_half) const {
        if (work.bidb >= params.num_batch || batch_ring_size(params, work.bidb) != params.mega_ring_world_size) {
            return 0;
        }
        int group_start_tile = 0;
        for (int bidb = 0; bidb < work.bidb; ++bidb) {
            if (batch_ring_size(params, bidb) == params.mega_ring_world_size) {
                group_start_tile += params.num_m_blocks_ptr[bidb] * params.num_head;
            }
        }
        int const num_m_blocks = params.num_m_blocks_ptr[work.bidb];
        int const full_block = q_use_half ? work.block + num_m_blocks / 2 : work.block;
        int const nheads_in_l2 = params.num_nheads_in_l2_ptr[work.bidb];
        int const section_idx = work.bidh / nheads_in_l2;
        int const bidh_residual = work.bidh - section_idx * nheads_in_l2;
        int const nheads_remainder = params.num_head - section_idx * nheads_in_l2;
        int const nheads_in_section = nheads_in_l2 <= nheads_remainder
            ? nheads_in_l2 : nheads_remainder;
        int const block_in_l2_order = num_m_blocks - 1 - full_block;
        int const mh_block = section_idx * nheads_in_l2 * num_m_blocks
            + block_in_l2_order * nheads_in_section + bidh_residual;
        return group_start_tile + mh_block;
    }

    CUTLASS_DEVICE WorkTileInfo invalid_work(Params const& params, int ticket) const {
        return {ticket, 0, 0, params.num_batch,
                min_fa3_varlen_demo::mega_ring::pack_segment_meta(params.mega_ring_world_size, params.mega_ring_world_size, false), 0};
    }

    CUTLASS_DEVICE WorkTileInfo decode_step_ticket(
        Params const& params, int ticket) const {
        int const step = params.mega_ring_ablation_step;
        auto const& level = params.mega_ring_hierarchy.levels[0];
        int const ring_local_rank = params.mega_ring_rank;
        bool const q_use_half = step > ring_local_rank;
        int const step_tiles = q_use_half ? level.half_tiles : level.full_tiles;
        if (step < 0 || step >= params.mega_ring_world_size || ticket >= step_tiles) {
            return invalid_work(params, ticket);
        }
        typename Base::WorkTileInfo work = q_use_half
            ? tile_idx_to_all_cp_work<true>(params, ticket)
            : tile_idx_to_all_cp_work<false>(params, ticket);
        int const reduction_idx = level.reduction_base
            + full_tile_idx(params, work, q_use_half);
        return {work.tile_idx, work.block, work.bidh, work.bidb,
                min_fa3_varlen_demo::mega_ring::pack_segment_meta(step, step, false),
                reduction_idx};
    }

    CUTLASS_DEVICE WorkTileInfo decode_linear_ticket(
        Params const& params, int ticket) const {
        auto const& level = params.mega_ring_hierarchy.levels[0];
        if (ticket >= params.mega_ring_hierarchy.total_work_tiles) {
            return invalid_work(params, ticket);
        }
        int step = 0;
        int step_tile_idx = ticket;
        bool q_use_half = false;
        if (ticket >= level.full_tiles) {
            int rem = ticket - level.full_tiles;
            int const ring_local_rank = params.mega_ring_rank;
            int const full_section_tiles = ring_local_rank * level.full_tiles;
            if (rem < full_section_tiles) {
                step = 1 + rem / level.full_tiles;
                step_tile_idx = rem - (step - 1) * level.full_tiles;
            } else {
                rem -= full_section_tiles;
                int const half_step_idx = rem / level.half_tiles;
                step = ring_local_rank + 1 + half_step_idx;
                step_tile_idx = rem - half_step_idx * level.half_tiles;
                q_use_half = true;
            }
        }
        typename Base::WorkTileInfo work = q_use_half
            ? tile_idx_to_all_cp_work<true>(params, step_tile_idx)
            : tile_idx_to_all_cp_work<false>(params, step_tile_idx);
        int const reduction_idx = level.reduction_base
            + full_tile_idx(params, work, q_use_half);
        return {work.tile_idx, work.block, work.bidh, work.bidb,
                min_fa3_varlen_demo::mega_ring::pack_segment_meta(step, step, false),
                reduction_idx};
    }

    CUTLASS_DEVICE WorkTileInfo decode_ticket(Params const& params, int ticket) const {
        WorkTileInfo work;
        if constexpr (StepOnly) {
            work = decode_step_ticket(params, ticket);
        } else {
            work = decode_linear_ticket(params, ticket);
        }
        if constexpr (CollectStats) {
            if (recycled_cta_ && work.bidb < params.num_batch
                && threadIdx.x % NumProducerThreads == 0) {
                atomicAdd(params.mega_ring_completed_tiles + 1, 1);
            }
        }
        return work;
    }

public:
    CUTLASS_DEVICE
    MegaRingVarlenLinearPersistentTileScheduler(SharedStorage* smem_scheduler)
        : Base(reinterpret_cast<typename Base::SharedStorage*>(smem_scheduler)) {}

    CUTLASS_DEVICE void publish_work_to_smem(WorkTileInfo const& work) const {
        if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
            typename Base::SharedStorage* smem = scheduler_smem();
            smem[0] = make_int4(work.tile_idx, work.block, work.bidh, work.bidb);
            smem[1] = make_int4(work.segment_meta, work.reduction_tile_idx, 0, 0);
        }
        flash::named_barrier_arrive(
            Base::kNumThreads,
            cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) {
        if constexpr (IsProducerWarp) {
            recycled_cta_ = false;
            WorkTileInfo work = decode_ticket(params, Base::virtual_block_idx(params));
            publish_work_to_smem(work);
            return work;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0, 0, 0});
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE WorkTileInfo get_initial_work_from_queue(Params const& params) {
        if constexpr (IsProducerWarp) {
            recycled_cta_ = true;
            int ticket = 0;
            if (threadIdx.x % NumProducerThreads == 0) {
                ticket = atomicAdd(params.tile_count_semaphore, 1)
                    + Base::virtual_grid_dim_x(params);
            }
            ticket = __shfl_sync(0xffffffff, ticket, 0);
            WorkTileInfo work = decode_ticket(params, ticket);
            publish_work_to_smem(work);
            return work;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0, 0, 0});
        }
    }

    CUTLASS_DEVICE void prefetch_next_work(
        Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1)
                + Base::virtual_grid_dim_x(params);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE WorkTileInfo get_next_work(
        Params const& params, WorkTileInfo const& current_work) {
        if constexpr (IsProducerWarp) {
            int const ticket = __shfl_sync(0xffffffff, current_work.tile_idx, 0);
            WorkTileInfo work = decode_ticket(params, ticket);
            flash::named_barrier_sync(
                Base::kNumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            publish_work_to_smem(work);
            return work;
        } else {
            flash::named_barrier_sync(
                Base::kNumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            typename Base::SharedStorage* smem = scheduler_smem();
            int4 const base = smem[0];
            int4 const ring = smem[1];
            flash::named_barrier_arrive(
                Base::kNumThreads,
                cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            return {base.x, base.y, base.z, base.w, ring.x, ring.y};
        }
    }
};

}  // namespace flash
