// Copied and trimmed from Hopper forward sources:
// - hopper/tile_scheduler.hpp
// Trimmed down to the scheduler variant used by the minimal SM90 varlen forward demo:
// VarlenDynamicPersistentTileScheduler with prepared scheduler metadata.

#pragma once

#include <cassert>

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

#include "min_fa3_named_barrier.h"
#include "min_fa3_varlen_params.h"
#include "utils.h"

namespace flash {

struct TileSchedulerArguments {
    int const num_blocks, num_head, num_batch, num_splits;
    int const qhead_per_khead;
    int const seqlen;
    int const seqlen_k, headdim, headdim_v, element_size;
    int* const tile_count_semaphore = nullptr;
    int const* const cu_seqlens = nullptr;
    int const* const seqused = nullptr;
    int const* const num_splits_dynamic_ptr = nullptr;
    int const* const num_m_blocks_ptr = nullptr;
    int const* const varlen_batch_idx_ptr = nullptr;
    int const* const num_nheads_in_l2_ptr = nullptr;
    int const virtual_grid_blocks = 0;
    int const compute_block_offset = 0;
    bool const use_virtual_grid = false;
    // MEGA_RING: optional scheduler bounds for the explicit mega-ring path.
    // Default values keep existing varlen and single-step ring paths unchanged.
    int const mega_ring_world_size = 1;
    int const mega_ring_rank = 0;
    int const* const mega_ring_ring_sizes = nullptr;
    min_fa3_varlen_demo::MegaRingHierarchyDesc const mega_ring_hierarchy{};
    int const* const mega_ring_kv_ready_counts = nullptr;
    int* const mega_ring_tile_states = nullptr;
    int* const mega_ring_scan_cursor = nullptr;
    int* const mega_ring_completed_tiles = nullptr;
};

template<int kBlockM, int kBlockN, int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp,
         bool Split=false, bool PackGQA=false, bool WarpSpecialized=true, bool LPT = false, bool Sort = false, bool Prepared = true>
class VarlenDynamicPersistentTileScheduler {

    static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
    static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;

public:
    using SharedStorage = int4;

protected:
    SharedStorage* const work_info_smem;

public:
    // MEGA_RING: default scheduler does not expose ring-step metadata.
    static constexpr bool EnableMegaRing = false;
    static constexpr bool EnableChunkedSegments = false;
    static constexpr bool CollectMegaRingStats = false;
    static constexpr bool EnableQueuedInitialWork = false;
    // MEGA_RING: expose the existing scheduler barrier participant count to
    // the copied mega-ring scheduler variant.
    static constexpr int kNumThreads = NumThreads;

    // Device side kernel params
    struct Params {
        int num_head, num_batch;
        int const qhead_per_khead;
        int const seqlen;
        // int const max_kvblocks_in_l2;
        cutlass::FastDivmod head_divmod;
        cutlass::FastDivmod nsplits_divmod;
        int* const tile_count_semaphore;
        int const* const cu_seqlens;
        int const* const seqused;
        int const* const num_splits_dynamic_ptr;
        int const* const num_m_blocks_ptr;
        int const* const varlen_batch_idx_ptr;
        // int const* const num_n_blocks_ptr;
        int const* const num_nheads_in_l2_ptr;
        int const virtual_grid_blocks;
        int const compute_block_offset;
        bool const use_virtual_grid;
        // MEGA_RING: default-off scheduler extension used only by
        // MegaRingVarlenDynamicPersistentTileScheduler.
        int const mega_ring_world_size;
        int const mega_ring_rank;
        int const* const mega_ring_ring_sizes;
        min_fa3_varlen_demo::MegaRingHierarchyDesc const mega_ring_hierarchy;
        int const* const mega_ring_kv_ready_counts;
        int* const mega_ring_tile_states;
        int* const mega_ring_scan_cursor;
        int* const mega_ring_completed_tiles;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        // If Split, for the purpose of scheduling, we pretend that instead there are
        // (args.num_splits * args.num_head) number of heads.
        assert(args.tile_count_semaphore != nullptr);
        assert(args.num_head < (1 << 16));  // We use the top 16 bits to store num_splits & split_idx
        assert(!Split || args.num_splits < (1 << 8)); // We use the top 8 bits to store num_splits
        // int const size_l2 = 50 * 1024 * 1024; // 50 MB
        // int const size_one_kvblock = kBlockN * (args.headdim + args.headdim_v) * args.element_size;
        // int max_kvblocks_in_l2 = size_l2 / size_one_kvblock;
        return {args.num_head, args.num_batch,
                args.qhead_per_khead, args.seqlen,
                // max_kvblocks_in_l2,
                cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(!Split ? 1 : args.num_splits),
                args.tile_count_semaphore, args.cu_seqlens, args.seqused,
                args.num_splits_dynamic_ptr,
                args.num_m_blocks_ptr,
                args.varlen_batch_idx_ptr,
                // aras.num_n_blocks_ptr,
                args.num_nheads_in_l2_ptr,
                args.virtual_grid_blocks,
                args.compute_block_offset,
                args.use_virtual_grid,
                args.mega_ring_world_size,
                args.mega_ring_rank,
                args.mega_ring_ring_sizes,
                args.mega_ring_hierarchy,
                args.mega_ring_kv_ready_counts,
                args.mega_ring_tile_states,
                args.mega_ring_scan_cursor,
                args.mega_ring_completed_tiles};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx, block, bidh, bidb;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            // if (blockIdx.x >= 0 && (threadIdx.x == 128 || threadIdx.x == 0)) { printf("blockIdx.x = %d, threadIdx.x = %d, checking valid, bidb = %d, params.num_batch = %d\n", blockIdx.x, threadIdx.x, bidb, params.num_batch); }
            return bidb < params.num_batch;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            auto get_actual_batch = [&](int virtual_batch) {
                if constexpr(Prepared && Sort) {
                    return params.varlen_batch_idx_ptr[virtual_batch];
                } else {
                    return virtual_batch;
                }
            };
            if constexpr (!Split) {
                return {block, bidh, get_actual_batch(bidb), 0 /*split_idx*/};
            } else {
                // the top 8 bits of bidh store num_splits and the next 8 bits store split_idx
                // reinterpret_cast to uint32_t to make sure we're not doing sign extension when we shift
                uint32_t bidh_packed = reinterpret_cast<uint32_t const&>(bidh);
                uint32_t bidh_actual_u = bidh_packed & 0x0000FFFF;
                int bidh_actual = reinterpret_cast<int&>(bidh_actual_u);
                // Use the top 16 bits of split_idx to store num_splits and the next 16 bits to store split_idx
                uint32_t split_idx_u = ((bidh_packed & 0x00FF0000) >> 16) + ((bidh_packed & 0xFF000000) >> 8);
                int split_idx = reinterpret_cast<int&>(split_idx_u);
                // int bidh_actual = params.nsplits_divmod.divmod(split_idx, bidh);
                // if (threadIdx.x == 128) {
                //     printf("blockIdx.x = %d, bidb = %d, bidh = %d, bidh_actual = %d, split_idx = %d\n", blockIdx.x, bidb, bidh, bidh_actual, split_idx);
                // }
                return {block, bidh_actual, get_actual_batch(bidb), split_idx};
            }
        }
    };

    CUTLASS_DEVICE
    VarlenDynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : work_info_smem(smem_scheduler) {};

    CUTLASS_DEVICE
    static int
    virtual_block_idx(Params const& params) {
        return params.use_virtual_grid ? int(blockIdx.x) - params.compute_block_offset : int(blockIdx.x);
    }

    CUTLASS_DEVICE
    static int
    virtual_grid_dim_x(Params const& params) {
        return params.use_virtual_grid ? params.virtual_grid_blocks : int(gridDim.x);
    }

    CUTLASS_DEVICE
    WorkTileInfo
    tile_idx_to_work_tile(Params const& params, int next_tile_idx, WorkTileInfo const& current_work) const {
        return tile_idx_to_work_tile_impl<false>(params, next_tile_idx, current_work);
    }

    template<bool HalfMBlocks=false>
    CUTLASS_DEVICE
    WorkTileInfo
    tile_idx_to_work_tile_impl(Params const& params, int next_tile_idx, WorkTileInfo const& current_work) const {
        if constexpr (HalfMBlocks) {
            static_assert(Prepared, "HalfMBlocks decode relies on prepared full-sequence scheduler metadata");
        }
        int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
        auto get_num_m_blocks = [&] (int bidb_start) {
            int batch_idx = lane + bidb_start;
            if constexpr (Prepared) {
                return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
                    ? (!HalfMBlocks ? params.num_m_blocks_ptr[batch_idx] : params.num_m_blocks_ptr[batch_idx] / 2) : 0;
            } else {
                int seqlen = params.seqlen * (!PackGQA ? 1 : params.qhead_per_khead);
                if (seqlen > kBlockM) {
                    if (params.seqused) {
                        seqlen = batch_idx < params.num_batch ? params.seqused[batch_idx] : 0;
                    } else if (params.cu_seqlens) {
                        int cur_cu_seqlen = batch_idx <= params.num_batch ? params.cu_seqlens[batch_idx] : 0;
                        int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
                        seqlen = next_cu_seqlen - cur_cu_seqlen;
                    } else {
                        seqlen = params.seqlen;
                    }
                    if constexpr (PackGQA) { seqlen *= params.qhead_per_khead; }
                }
                int num_m_blocks = cute::ceil_div(seqlen, kBlockM);
                return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
                    ? (!HalfMBlocks ? num_m_blocks : num_m_blocks / 2) : 0;
                    // ? params.num_m_blocks_ptr[batch_idx] : 0;
            }
        };

        auto get_num_splits = [&] (int bidb_start) {
            int batch_idx = lane + bidb_start;
            bool is_valid = batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1;
            if constexpr (!Split) {
                return is_valid ? 1 : 0;
            } else if constexpr(Prepared) {
                return is_valid ? params.num_splits_dynamic_ptr[batch_idx] : 0;
            } else {
                return is_valid ? params.nsplits_divmod.divisor : 0;
            }
        };

        int num_m_blocks = get_num_m_blocks(current_work.bidb);  // Different for each lane
        int num_splits = get_num_splits(current_work.bidb);
        int num_split_m_blocks = !Split ? num_m_blocks : num_m_blocks * num_splits;
        // Cumulative number of blocks for the next 31 batches
        int num_m_blocks_cumulative = warp_prefix_sum(num_split_m_blocks);
        // Total number of blocks for the next 31 batches
        int m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
        // Only the lower 16 bits are the actual bidh
        // int current_bidh = !Split ? current_work.bidh : (current_work.bidh & 0x0000FFFF);
        // int group_end_tile = current_work.tile_idx - current_work.block - current_bidh * __shfl_sync(0xffffffff, num_split_m_blocks, 0 /*lane*/) + m_blocks_in_group * params.num_head;  // Same for all lanes
        // if constexpr (Split) {
        //     int current_split_idx = (current_work.bidh & 0x00FF0000) >> 16;
        //     group_end_tile -= current_split_idx * __shfl_sync(0xffffffff, num_m_blocks, 0 /*lane*/);
        // }
        // NEW: current_work.tile_idx holds group_start_tile for starting batch
        int group_end_tile = current_work.tile_idx + m_blocks_in_group * params.num_head;  // Same for all lanes
        int bidb = current_work.bidb;
        // if (blockIdx.x <= 9 && threadIdx.x == 0) {
        //     printf("Before while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, cur tile_idx = %d, cur block = %d, cur bidh = %d, num_split_m_blocks = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, current_work.bidb, num_m_blocks, next_tile_idx, current_work.tile_idx, current_work.block, current_bidh, num_split_m_blocks, group_end_tile, m_blocks_in_group);
        // }
        // if (threadIdx.x == 0 && blockIdx.x == 0) { printf("tile_idx = %d, group_end_tile = %d, num_m_blocks_cumulative = %d, m_blocks_in_group = %d\n", current_work.tile_idx, group_end_tile, num_m_blocks_cumulative, m_blocks_in_group); }
        while (group_end_tile <= next_tile_idx) {
            bidb += cutlass::NumThreadsPerWarp - 1;
            if (bidb >= params.num_batch) {
                // if (blockIdx.x <= 9 && threadIdx.x == 0) {
                //     printf("Returning early, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
                // }
                return {next_tile_idx, 0, 0, params.num_batch};
            }
            num_m_blocks = get_num_m_blocks(bidb);
            num_splits = get_num_splits(bidb);
            num_split_m_blocks = !Split ? num_m_blocks : num_m_blocks * num_splits;
            num_m_blocks_cumulative = warp_prefix_sum(num_split_m_blocks);
            m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
            group_end_tile += m_blocks_in_group * params.num_head;
            // if (blockIdx.x <= 9 && threadIdx.x == 0) {
            //     printf("Bottom of while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
            // }
        }
        int group_start_tile = group_end_tile - m_blocks_in_group * params.num_head;
        // The next problem to process is the first one that does not have ending tile position
        // that is greater than or equal to tile index.
        int batch_idx_in_group = __popc(__ballot_sync(0xffffffff, group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx));
        // if (threadIdx.x == 31 || threadIdx.x == 0) { printf("blockIdx.x = %d, tidx %d, group_start_tile = %d, num_m_blocks_cumulative = %d, num_head = %d, next_tile_idx = %d, ballot = %x, batch_idx_in_group = %d\n", blockIdx.x, threadIdx.x, group_start_tile, num_m_blocks_cumulative, params.num_head, next_tile_idx, tmp, batch_idx_in_group); }
        bidb += batch_idx_in_group;
        num_m_blocks = __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);
        if constexpr (Split) { num_splits = __shfl_sync(0xffffffff, num_splits, batch_idx_in_group); }
        group_start_tile += (batch_idx_in_group == 0 ? 0 : __shfl_sync(0xffffffff, num_m_blocks_cumulative, batch_idx_in_group - 1)) * params.num_head;
        int mh_block = next_tile_idx - group_start_tile;
        int block, bidh;
        if constexpr (LPT) {
            if (!Split || num_splits == 1) {
                // NOTE: code for computing nheads_in_l2 directly left as reference
                // int num_n_blocks = params.num_n_blocks_ptr ? params.num_n_blocks_ptr[bidb] : num_m_blocks;
                // auto find_log2_floor = [&](int n) { return 31 - cutlass::clz(n); };
                // int nheads_in_l2 = params.max_kvblocks_in_l2 < num_n_blocks
                //     ? 1 : 1 << find_log2_floor(params.max_kvblocks_in_l2 / num_n_blocks);
                // if constexpr (!PackGQA) { nheads_in_l2 *= params.qhead_per_khead; }
                // nheads_in_l2 = min(nheads_in_l2, params.num_head);
                auto get_nheads_in_l2 = [&](int batch_idx) {
                    if constexpr(Prepared) {
                        return params.num_nheads_in_l2_ptr[batch_idx];
                    } else {
                        return !PackGQA ? params.qhead_per_khead : 1;
                    }
                };
                int nheads_in_l2 = get_nheads_in_l2(bidb);
                int mh_in_l2 = nheads_in_l2 * num_m_blocks;
                int section_idx = mh_block / mh_in_l2;
                int l2_mod = mh_block - section_idx * mh_in_l2;
                // tail section
                int nheads_remainder = params.num_head - section_idx * nheads_in_l2;
                int nheads_in_this_section = nheads_in_l2 <= nheads_remainder ? nheads_in_l2 : nheads_remainder;
                block = l2_mod / nheads_in_this_section;
                int bidh_residual = l2_mod - block * nheads_in_this_section;
                bidh = section_idx * nheads_in_l2 + bidh_residual;
                if constexpr(Split) {
                    // remember to set num_splits = 1 in work tile
                    uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
                    bidh = reinterpret_cast<int&>(bidh_packed);
                }
            } else {
                // NOTE: leave traverse heads first version for reference
                // block = params.head_divmod.divmod(bidh, mh_block);
                // if constexpr (Split) {
                //     int split_idx = block / num_m_blocks;
                //     block = block - split_idx * num_m_blocks;
                //     uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh) + (reinterpret_cast<uint32_t&>(split_idx) << 16) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
                //     bidh = reinterpret_cast<int&>(bidh_packed);
                // }
                bidh = mh_block / num_m_blocks;
                block = mh_block - bidh * num_m_blocks;
                if constexpr (Split) {
                    int bidh_actual = bidh / num_splits;
                    int split_idx = bidh - bidh_actual * num_splits;
                    uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh_actual) + (reinterpret_cast<uint32_t&>(split_idx) << 16) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
                    bidh = reinterpret_cast<int&>(bidh_packed);
                }
            }
            block = num_m_blocks - 1 - block;
        } else {
            bidh = mh_block / num_m_blocks;
            block = mh_block - bidh * num_m_blocks;
            if constexpr (Split) {
                int bidh_actual = bidh / num_splits;
                int split_idx = bidh - bidh_actual * num_splits;
                // TODO: idk why this gives wrong answer nondeterministically
                // int bidh_actual, split_idx;
                // split_idx = params.head_divmod.divmod(bidh_actual, bidh);
                // Use the top 8 bits to store num_splits and the next 8 bits to store split_idx
                // reinterpret_cast to uint32_t to make sure we're not doing sign extension when we shift
                uint32_t bidh_packed = reinterpret_cast<uint32_t&>(bidh_actual) + (reinterpret_cast<uint32_t&>(split_idx) << 16) + (reinterpret_cast<uint32_t&>(num_splits) << 24);
                // if (threadIdx.x == 0) {
                //     printf("blockIdx.x = %d, group_start_tiled = %d, bidb = %d, batch_idx_in_group = %d, mh_block = %d, num_m_blocks = %d, bidh = %d, bidh_actual = %d, split_idx = %d, num_splits = %d, bidh_packed = %d\n", blockIdx.x, group_start_tile, bidb, batch_idx_in_group, mh_block, num_m_blocks, bidh, bidh_actual, split_idx, num_splits, bidh_packed);
                // }
                bidh = reinterpret_cast<int&>(bidh_packed);
            }
            // if (blockIdx.x <= 9 && threadIdx.x == 0) {
            //     printf("Before returning, blockIdx.x = %d, threadIdx.x = %d, group_start_tile = %d, batch_idx_in_group = %d, bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d, m_blocks_in_group = %d, mh_block = %d, bidh = %d, block = %d\n", blockIdx.x, threadIdx.x, group_start_tile, batch_idx_in_group, bidb, num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group, mh_block, bidh, block);
            // }
        }
        return {group_start_tile, block, bidh, bidb};
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            WorkTileInfo work_info = tile_idx_to_work_tile(params, virtual_block_idx(params), {0, 0, 0, 0});
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return work_info;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0});
        }
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {
        // Don't arrive at the TileCountSmemEmpty barrier here, because get_initial_work will do that
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + virtual_grid_dim_x(params);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            // thread 0 has the next tile_idx, just need to broadcast to the rest of warp 0
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
            WorkTileInfo work_info = {__shfl_sync(0xffffffff, current_work.tile_idx, 1 /*lane*/), current_work.block, current_work.bidh, current_work.bidb};
            work_info = tile_idx_to_work_tile(params, new_tile_idx, work_info);
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return work_info;
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            int4 work_info = *work_info_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w};
        }
    }

};

}  // namespace flash
