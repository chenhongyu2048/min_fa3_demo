// Mega ring variant copied and trimmed from include/min_fa3_varlen_scheduler.h.
// Changes are marked with MEGA_RING comments.

#pragma once

#include "min_fa3_varlen_scheduler.h"
#include "mega_ring_semaphore.cuh"
#include "mega_ring_tile_ready.h"
#include <ratio>

namespace flash {

// MEGA_RING: persistent scheduler that maps the global tile stream to
// (Q tile, immutable KV interval) in causal mode and ring steps otherwise.
// The underlying varlen tile decoding stays
// copied from VarlenDynamicPersistentTileScheduler.
template<int kBlockM, int kBlockN, int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp,
         bool Split=false, bool PackGQA=false, bool WarpSpecialized=true, bool LPT = false, bool Sort = false, bool Prepared = true,
         bool CollectStats = false, typename MinKvRatio = std::ratio<1, 2>,
         typename MaxKvRatio = std::ratio<4, 1>>
class MegaRingVarlenDynamicPersistentTileScheduler: public VarlenDynamicPersistentTileScheduler<kBlockM, kBlockN, NumMmaThreads, NumProducerThreads,
                                                                                                Split, PackGQA, WarpSpecialized, LPT, Sort, Prepared> {
    using Base = VarlenDynamicPersistentTileScheduler<kBlockM, kBlockN, NumMmaThreads, NumProducerThreads, Split, PackGQA, WarpSpecialized, LPT, Sort, Prepared>;

public:
    static constexpr bool EnableMegaRing = true;
    static constexpr bool EnableQueuedInitialWork = true;
    static constexpr bool CollectMegaRingStats = CollectStats;
    // MEGA_RING_ZIGZAG: this scheduler is instantiated with LPT == IsCausal in
    // the mega-ring launch. Effectively that means:
    //   - causal instantiations use LPT=true and enable zigzag
    //   - non-causal instantiations use LPT=false and keep the old path
    // LPT still originates from the base varlen scheduler's tile-ordering
    // policy; we reuse it here as the compile-time causal/zigzag gate to avoid
    // adding another template boolean.
    static constexpr bool EnableZigzag = LPT;
    static constexpr bool EnableChunkedSegments = EnableZigzag;
    static_assert(!EnableChunkedSegments || (kBlockM == 128 && kBlockN == 128));
    // Two descriptor slots; causal consumers alone own the third slot for
    // broadcasting output initialization while the producer scans ahead.
    static_assert(MinKvRatio::num > 0 && MaxKvRatio::num > 0);
    static_assert(std::ratio_greater_equal<MaxKvRatio, MinKvRatio>::value);
    using SharedStorage = cute::array<typename Base::SharedStorage, EnableChunkedSegments ? 3 : 2>;
    using Params = typename Base::Params;

    struct WorkTileInfo {
        int tile_idx, block, bidh, bidb;
        // Legacy ring-step metadata. Causal tasks use the full-width interval below.
        int segment_meta;
        int reduction_tile_idx;
        int kv_begin = 0, kv_end = 0, q_state_idx = -1;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            if constexpr (EnableChunkedSegments) { return bidb < params.num_batch; }
            // MEGA_RING: work is valid only while the segment begin is in
            // range and the underlying varlen tile decoded to a batch.
            return min_fa3_varlen_demo::mega_ring::segment_begin_step(segment_meta)
                < params.mega_ring_world_size && bidb < params.num_batch;
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
        // variant reserves additional slots for interval metadata and epilogue state.
        : Base(reinterpret_cast<typename Base::SharedStorage*>(smem_scheduler)) {}

    CUTLASS_DEVICE
    typename Base::SharedStorage* mega_ring_work_info_smem() const {
        // The producer writes only slots 0/1; causal consumers own slot 2.
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

    CUTLASS_DEVICE
    int level_idx_for_reduction_tile(Params const& params, int reduction_tile_idx) const {
        #pragma unroll
        for (int level_idx = 0; level_idx < 3; ++level_idx) {
            auto const& level = params.mega_ring_hierarchy.levels[level_idx];
            if (reduction_tile_idx >= level.reduction_base
                && reduction_tile_idx < level.reduction_base + level.full_tiles) {
                return level_idx;
            }
        }
        return 3;
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
                          int next_tile_idx, // base tile id or dynamic-work ticket
                          int current_ring_step,
                          typename Base::WorkTileInfo const& current_base_work) const {
        if constexpr (EnableChunkedSegments) {
            int const g1_tiles = params.mega_ring_hierarchy.levels[3].full_tiles;
            if (next_tile_idx < g1_tiles) {
                auto const work = tile_idx_to_level_work_linear<false>(params, next_tile_idx, 1);
                return {work.tile_idx, work.block, work.bidh, work.bidb, 0, 0, 0, 0, -1};
            }
            int const total = params.mega_ring_hierarchy.reduction_tiles;
            int const lane = threadIdx.x % cutlass::NumThreadsPerWarp;
            int const groups = cute::ceil_div(total, 32);
            while (total > 0) {
                int completed = 0;
                if (lane == 0) {
                    completed = min_fa3_varlen_demo::mega_ring::load_acquire(params.mega_ring_completed_tiles);
                }
                // Every lane must take the same exit before full-warp collectives.
                if (__shfl_sync(0xffffffff, completed, 0) == total) { break; }
                unsigned group = 0;
                if (lane == 0) {
                    group = atomicInc(reinterpret_cast<unsigned*>(params.mega_ring_scan_cursor), groups - 1);
                }
                group = __shfl_sync(0xffffffff, group, 0);
                int const q_id = int(group) * 32 + lane;
                typename Base::WorkTileInfo work{0, 0, 0, params.num_batch};
                int a = 0, e = 0;
                bool candidate = false;
                if (q_id < total) {
                    int const level_idx = level_idx_for_reduction_tile(params, q_id);
                    auto const& level = params.mega_ring_hierarchy.levels[level_idx];
                    work = tile_idx_to_level_work_linear<false>(
                        params, q_id - level.reduction_base, level.ring_size);
                    int const b = get_actual_batch(params, work.bidb);
                    int const h = params.num_m_blocks_ptr[work.bidb] / 2;
                    int const r = params.mega_ring_rank % level.ring_size;
                    int const local = work.block + 1;
                    int const remote = min_fa3_varlen_demo::mega_ring::q_remote_tiles(
                        work.block, h, level.ring_size, r);
                    int const size = local + remote;
                    int const ready = local + min(remote, min_fa3_varlen_demo::mega_ring::load_acquire(
                        params.mega_ring_kv_ready_prefix + b));
                    a = min_fa3_varlen_demo::mega_ring::load_acquire(params.mega_ring_q_assigned + q_id);
                    int const tau_min = max(1, int((int64_t(h) * MinKvRatio::num
                        + MinKvRatio::den - 1) / MinKvRatio::den));
                    int const tau_max = max(1, int((int64_t(h) * MaxKvRatio::num
                        + MaxKvRatio::den - 1) / MaxKvRatio::den));
                    candidate = a < size && ready > a && (ready - a >= tau_min || ready == size);
                    e = min(ready, a + tau_max);
                }
                unsigned candidates = __ballot_sync(0xffffffff, candidate);
                while (candidates) {
                    int const selected = __ffs(candidates) - 1;
                    bool claimed = false;
                    if (lane == selected) {
                        claimed = min_fa3_varlen_demo::mega_ring::compare_exchange_acquire(
                            params.mega_ring_q_assigned + q_id, a, e) == a;
                        if constexpr (CollectStats) {
                            if (claimed) {
                                atomicAdd(params.mega_ring_completed_tiles + 1, e - a);
                                atomicMax(params.mega_ring_completed_tiles + 2, e - a);
                                atomicAdd(params.mega_ring_completed_tiles + 3, 1);
                            }
                        }
                    }
                    if (__shfl_sync(0xffffffff, int(claimed), selected)) {
                        return {__shfl_sync(0xffffffff, work.tile_idx, selected),
                                __shfl_sync(0xffffffff, work.block, selected),
                                __shfl_sync(0xffffffff, work.bidh, selected),
                                __shfl_sync(0xffffffff, work.bidb, selected), 0, 0,
                                __shfl_sync(0xffffffff, a, selected),
                                __shfl_sync(0xffffffff, e, selected),
                                __shfl_sync(0xffffffff, q_id, selected)};
                    }
                    candidates &= ~(1u << selected);
                }
                __nanosleep(64);
            }
            return {0, 0, 0, params.num_batch, 0, 0};
        }

        if (next_tile_idx >= params.mega_ring_hierarchy.total_work_tiles) {
            return {next_tile_idx, 0, 0, params.num_batch,
                    min_fa3_varlen_demo::mega_ring::pack_segment_meta(
                        params.mega_ring_world_size, params.mega_ring_world_size, false),
                    0};
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
                min_fa3_varlen_demo::mega_ring::pack_segment_meta(
                    ring_step, ring_step, false),
                reduction_tile_idx};
    }

public:
    CUTLASS_DEVICE
    void
    publish_work_to_smem(WorkTileInfo const& work_info) const {
        if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
            typename Base::SharedStorage* smem = mega_ring_work_info_smem();
            smem[0] = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            // Use full-width KV endpoints in the causal descriptor.
            if constexpr (EnableChunkedSegments) {
                smem[1] = make_int4(work_info.kv_begin, work_info.kv_end, work_info.q_state_idx, 0);
            } else {
                smem[1] = make_int4(work_info.segment_meta, work_info.reduction_tile_idx, 0, 0);
            }
        }
        if constexpr (EnableChunkedSegments) {
            cutlass::arch::NamedBarrier::arrive(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
        } else {
            flash::named_barrier_arrive(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            int const next_tile_idx = Base::virtual_block_idx(params);
            WorkTileInfo work_info = decode_mega_ring_tile(params, next_tile_idx, -1, {0, 0, 0, 0});
            // Transfer the winning scan lane's acquire to the elected TMA issuer.
            if constexpr (EnableChunkedSegments) { __syncwarp(); }
            publish_work_to_smem(work_info);
            return work_info;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0, 0, 0});
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work_from_queue(Params const& params) const {
        if constexpr (IsProducerWarp) {
            int next_tile_idx = 0;
            if (threadIdx.x % NumProducerThreads == 0) {
                if constexpr (EnableChunkedSegments) {
                    if (params.mega_ring_hierarchy.levels[3].full_tiles > 0) {
                        next_tile_idx = atomicAdd(params.tile_count_semaphore, 1) + Base::virtual_grid_dim_x(params);
                    }
                } else {
                    next_tile_idx = atomicAdd(params.tile_count_semaphore, 1) + Base::virtual_grid_dim_x(params);
                }
            }
            next_tile_idx = __shfl_sync(0xffffffff, next_tile_idx, 0 /*lane*/);
            WorkTileInfo work_info = decode_mega_ring_tile(params, next_tile_idx, -1, {0, 0, 0, 0});
            // Transfer the winning scan lane's acquire to the elected TMA issuer.
            if constexpr (EnableChunkedSegments) { __syncwarp(); }
            publish_work_to_smem(work_info);
            return work_info;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0, 0, 0});
        }
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if constexpr (EnableChunkedSegments) {
            if (current_work.q_state_idx >= 0) { return; }
        }
        if (threadIdx.x % NumProducerThreads == 0) {
            // Nonblocking ticket fetch only: load() still has its final V
            // transfer after this callback. Scanning belongs in get_next_work().
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
            if constexpr (EnableChunkedSegments) {
                if (current_work.q_state_idx >= 0) {
                    new_tile_idx = params.mega_ring_hierarchy.levels[3].full_tiles;
                }
            }
            // MEGA_RING: reconstruct the base decoder hint from the previous
            // per-step varlen tile, then decode the new expanded tile id.
            typename Base::WorkTileInfo current_base_work{__shfl_sync(0xffffffff, current_work.tile_idx, 1 /*lane*/),
                                                          current_work.block,
                                                          current_work.bidh,
                                                          current_work.bidb};
            WorkTileInfo work_info = decode_mega_ring_tile(
                params, new_tile_idx,
                min_fa3_varlen_demo::mega_ring::segment_begin_step(current_work.segment_meta),
                current_base_work);
            if constexpr (EnableChunkedSegments) {
                cutlass::arch::NamedBarrier::sync(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            } else {
                flash::named_barrier_sync(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            }
            publish_work_to_smem(work_info);
            return work_info;
        } else {
            if constexpr (EnableChunkedSegments) {
                cutlass::arch::NamedBarrier::sync(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            } else {
                flash::named_barrier_sync(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            }
            typename Base::SharedStorage* smem = const_cast<MegaRingVarlenDynamicPersistentTileScheduler*>(this)->mega_ring_work_info_smem();
            int4 work_info = smem[0];
            // Consumers receive the exact interval claimed by the producer.
            int4 mega_info = smem[1];
            if constexpr (EnableChunkedSegments) {
                cutlass::arch::NamedBarrier::arrive(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            } else {
                flash::named_barrier_arrive(Base::kNumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            }
            if constexpr (EnableChunkedSegments) {
                return {work_info.x, work_info.y, work_info.z, work_info.w, 0, 0,
                        mega_info.x, mega_info.y, mega_info.z};
            } else {
                return {work_info.x, work_info.y, work_info.z, work_info.w, mega_info.x, mega_info.y};
            }
        }
    }
};

}  // namespace flash
