// Mega ring variant copied and trimmed from include/min_fa3_varlen_ring_launch.h.
// Changes are marked with MEGA_RING comments.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <torch/extension.h>

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

// MEGA_RING: same TK include hygiene as the single-step ring launch.
#ifdef CHECK_CUDA
#undef CHECK_CUDA
#endif
#ifdef CHECK_CONTIGUOUS
#undef CHECK_CONTIGUOUS
#endif
#ifdef CHECK_INPUT
#undef CHECK_INPUT
#endif

#include "min_fa3_varlen_launch.h"
// MEGA_RING: scheduler exposes ring_step/global_tile_idx to the shared
// FlashAttnFwdSm90 kernel; semaphore helpers provide release/acquire ordering
// between communication CTAs and compute CTAs.
#include "mega_ring_min_fa3_varlen_scheduler.h"
#include "mega_ring_semaphore.cuh"

namespace min_fa3_varlen_demo {

using namespace cute;

namespace mega_ring_detail {

using namespace kittens;

template <bool IsCausal, int NumDevices, bool ReadyOnce=false>
struct MegaRingKernelConfig {
    // MEGA_RING: keep the same copied FA3 varlen mainloop/epilogue stack as
    // the single-step ring path, changing only the scheduler and kernel wrapper.
    using Config = FwdConfig<IsCausal>;
    using ArchTag = cutlass::arch::Sm90;
    using TileShape_MNK = Shape<Int<Config::kBlockM>, Int<Config::kBlockN>, Int<kHeadDim>>;
    using TileShape_MNK_PV = Shape<Int<Config::kBlockM>, Int<kHeadDimV>, Int<Config::kBlockN>>;
    using ClusterShape = Shape<Int<kClusterM>, _1, _1>;
    using CollectiveMainloop = flash::CollectiveMainloopFwdSm90<
        kStages,
        ClusterShape,
        TileShape_MNK,
        kHeadDimV,
        Element,
        float,
        ArchTag,
        IsCausal,
        kIsLocal,
        kHasSoftcap,
        kVarlen,
        kPagedKVNonTMA,
        kAppendKV,
        kHasQv,
        Config::MmaPV_is_RS,
        Config::IntraWGOverlap,
        kPackGQA,
        kSplit,
        kVColMajor>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<
        TileShape_MNK_PV,
        ClusterShape,
        ElementOut,
        ArchTag,
        CollectiveMainloop::NumMmaThreads,
        kVarlen,
        kPackGQA,
        kSplit,
        false,
        !ReadyOnce>;
    // MEGA_RING: scheduler expands the varlen tile stream across all ring
    // steps and carries per-tile ring metadata through producer/consumer WGs.
    using Scheduler = flash::MegaRingVarlenDynamicPersistentTileScheduler<
        Config::kBlockM,
        Config::kBlockN,
        CollectiveMainloop::NumMmaThreads,
        CollectiveMainloop::NumProducerThreads,
        false,
        false,
        true,
        IsCausal,
        true,
        true>;
    using AttnKernel = flash::enable_sm90<flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    static constexpr int kVecLength = 1024;
    static_assert(AttnKernel::MaxThreadsPerBlock % cutlass::NumThreadsPerWarp == 0);
    static constexpr int kNumWarpsPerBlock = AttnKernel::MaxThreadsPerBlock / cutlass::NumThreadsPerWarp;
    static constexpr int kNumCommChunks = kNumWarpsPerBlock / 2;
    static_assert(kNumCommChunks > 0);
    using shared_vec = sv_bf<kVecLength>;
    using staging_gl = gl<bf16, 1, 1, -1, kVecLength, shared_vec>;
    using remote_pgl = pgl<staging_gl, NumDevices, false>;
    static constexpr bool kIsCausal = IsCausal;

    // MEGA_RING: kernel params carry full concatenated local K/V storage and a
    // readiness counter instead of a single src_dev/ring_step pair.
    struct KernelParams {
        typename AttnKernel::Params compute{};
        remote_pgl remote_k;
        remote_pgl remote_v;
        staging_gl local_k;
        staging_gl local_v;
        int num_comp_sm;
        int num_comm_sm;
        int rows_per_rank;
        int half_rows_per_rank;
        int cp_rows_per_rank;
        int num_batch;
        int ring_rank;
        int ring_world_size;
        int const* cu_seqlens_k;
        int const* half_cu_seqlens;
        int const* cp_batch_mask;
        int* kv_ready_counts;
        bool ready_once;
        int* ready_end;
        int* chunk_done;
        int* publish_lock;
        int const* ready_interval_rows;
        int ready_intervals;
        int ready_max_chunks;
        int ready_chunk_rows;
    };
};

template <typename RingConfig>
constexpr void check_mega_ring_kernel_param_layout() {
    static_assert(alignof(CUtensorMap) >= 64);
    static_assert(alignof(typename RingConfig::AttnKernel::Params) >= 64);
    static_assert(alignof(typename RingConfig::remote_pgl) >= 64);
    static_assert(alignof(typename RingConfig::staging_gl) >= 64);
    static_assert(alignof(typename RingConfig::KernelParams) >= 64);

    static_assert(offsetof(typename RingConfig::KernelParams, compute) % 64 == 0);
    static_assert(offsetof(typename RingConfig::KernelParams, remote_k) % 64 == 0);
    static_assert(offsetof(typename RingConfig::KernelParams, remote_v) % 64 == 0);
    static_assert(offsetof(typename RingConfig::KernelParams, local_k) % 64 == 0);
    static_assert(offsetof(typename RingConfig::KernelParams, local_v) % 64 == 0);
}

template <typename RingConfig>
struct alignas(128) MegaRingRemoteLoadBarriers {
    semaphore arrived[RingConfig::kNumCommChunks];
    semaphore finished[RingConfig::kNumCommChunks];
};

template <typename RingConfig>
CUTLASS_DEVICE
void try_advance_ready_end(const typename RingConfig::KernelParams& params, int interval_id) {
    if (!params.ready_once || interval_id < 0 || interval_id >= params.ready_intervals) {
        return;
    }
    if (atomicCAS(params.publish_lock + interval_id, 0, 1) != 0) {
        return;
    }
    int const total_rows = params.ready_interval_rows[interval_id];
    int cursor_rows = min_fa3_varlen_demo::mega_ring::load_acquire(params.ready_end + interval_id);
    int cursor_chunk = params.ready_chunk_rows > 0 ? cursor_rows / params.ready_chunk_rows : 0;
    while (cursor_chunk < params.ready_max_chunks && cursor_rows < total_rows) {
        int const rows = cute::min(params.ready_chunk_rows, total_rows - cursor_rows);
        int const target = rows * 2;
        int const done = min_fa3_varlen_demo::mega_ring::load_acquire(
            params.chunk_done + interval_id * params.ready_max_chunks + cursor_chunk);
        if (done < target) {
            break;
        }
        cursor_rows += rows;
        ++cursor_chunk;
    }
    min_fa3_varlen_demo::mega_ring::store_release(params.ready_end + interval_id, cursor_rows);
    atomicExch(params.publish_lock + interval_id, 0);
}

template <typename RingConfig>
CUTLASS_DEVICE
void signal_ready_once_row(const typename RingConfig::KernelParams& params,
                           int interval_id,
                           int row_in_interval) {
    if (!params.ready_once || interval_id < 0 || row_in_interval < 0) {
        return;
    }
    int const chunk_idx = row_in_interval / params.ready_chunk_rows;
    int const old_done = min_fa3_varlen_demo::mega_ring::signal_release_return_old(
        params.chunk_done + interval_id * params.ready_max_chunks + chunk_idx,
        1);
    int const chunk_start = chunk_idx * params.ready_chunk_rows;
    int const total_rows = params.ready_interval_rows[interval_id];
    int const rows = cute::min(params.ready_chunk_rows, total_rows - chunk_start);
    if (rows > 0 && old_done + 1 >= rows * 2) {
        try_advance_ready_end<RingConfig>(params, interval_id);
    }
}

template <typename RingConfig>
CUTLASS_DEVICE
void run_mega_ring_remote_load(
    const typename RingConfig::KernelParams& params,
    int comm_bid,
    char* smem_buf) {
    // MEGA_RING: local-only runs keep the same launch shape but skip the
    // communication CTA body.
    if (comm_bid >= params.num_comm_sm || params.ring_world_size <= 1) {
        return;
    }

    tma_swizzle_allocator allocator(reinterpret_cast<int*>(smem_buf));
    typename RingConfig::shared_vec (&vec)[RingConfig::kNumCommChunks] = allocator.allocate<typename RingConfig::shared_vec, RingConfig::kNumCommChunks>();
    __shared__ MegaRingRemoteLoadBarriers<RingConfig> barriers;

    static_assert(sizeof(MegaRingRemoteLoadBarriers<RingConfig>) % 128 == 0);

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < RingConfig::kNumCommChunks; ++i) {
            init_semaphore(barriers.arrived[i], 0, 1);
            init_semaphore(barriers.finished[i], 0, 1);
        }
    }
    __syncthreads();

    // MEGA_RING: communication tile id maps directly to
    // (step, K/V selector, token row, destination/source rank block).
    bool const hybrid_mode = params.cp_batch_mask != nullptr;
    int const full_loads_per_step = params.rows_per_rank * 2;
    int const half_loads_per_step = (hybrid_mode ? params.rows_per_rank : params.half_rows_per_rank) * 2;
    int const half_section_tasks = RingConfig::kIsCausal ? params.ring_rank * half_loads_per_step : 0;
    int const total_tasks = RingConfig::kIsCausal
        ? half_section_tasks + (params.ring_world_size - 1 - params.ring_rank) * full_loads_per_step
        : (params.ring_world_size - 1) * full_loads_per_step;
    int const warp_id = warp::groupid();
    uint32_t phasebits = 0xFFFF0000;

    auto half_row_to_full_row = [&] (int half_row) {
        int lo = 0;
        int hi = params.num_batch;
        while (lo + 1 < hi) {
            int const mid = (lo + hi) / 2;
            if (params.half_cu_seqlens[mid] <= half_row) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return params.cu_seqlens_k[lo] + (half_row - params.half_cu_seqlens[lo]);
    };

    auto row_to_batch = [&] (int row_idx) {
        int lo = 0;
        int hi = params.num_batch;
        while (lo + 1 < hi) {
            int const mid = (lo + hi) / 2;
            if (params.cu_seqlens_k[mid] <= row_idx) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    };

    auto should_copy_row = [&] (int row_idx, bool kv_use_half) {
        if (params.cp_batch_mask == nullptr) {
            return true;
        }
        int const batch_idx = row_to_batch(row_idx);
        if (batch_idx >= params.num_batch || params.cp_batch_mask[batch_idx] == 0) {
            return false;
        }
        if (kv_use_half) {
            int const start = params.cu_seqlens_k[batch_idx];
            int const end = params.cu_seqlens_k[batch_idx + 1];
            int const half_len = (end - start) / 2;
            return row_idx - start < half_len;
        }
        return true;
    };

    auto compact_row_for = [&] (int load_kv_rank, int row_idx) {
        int const batch_idx = row_to_batch(row_idx);
        int const row_in_batch = row_idx - params.cu_seqlens_k[batch_idx];
        int const compact_base = params.compute.mainloop.cu_seqlens_k[batch_idx];
        if constexpr (RingConfig::kIsCausal) {
            int const half_len = params.half_cu_seqlens[batch_idx + 1] - params.half_cu_seqlens[batch_idx];
            if (half_len > 0 && row_in_batch < half_len) {
                return compact_base + load_kv_rank * half_len + row_in_batch;
            }
            if (half_len > 0) {
                return compact_base + params.ring_world_size * half_len
                    + (params.ring_world_size - 1 - load_kv_rank) * half_len
                    + (row_in_batch - half_len);
            }
        }
        int const local_len = params.cu_seqlens_k[batch_idx + 1] - params.cu_seqlens_k[batch_idx];
        return compact_base + load_kv_rank * local_len + row_in_batch;
    };

    auto signal_ready_once_for = [&] (int load_kv_rank, int row_idx) {
        int const batch_idx = row_to_batch(row_idx);
        if (params.cp_batch_mask != nullptr && params.cp_batch_mask[batch_idx] == 0) {
            return;
        }
        int const row_in_batch = row_idx - params.cu_seqlens_k[batch_idx];
        int const compact_base = params.compute.mainloop.cu_seqlens_k[batch_idx];
        int const compact_row = compact_row_for(load_kv_rank, row_idx);
        int const row_in_interval = compact_row - compact_base;
        if constexpr (RingConfig::kIsCausal) {
            int const half_len = params.half_cu_seqlens[batch_idx + 1] - params.half_cu_seqlens[batch_idx];
            if (half_len > 0 && row_in_batch < half_len) {
                if (load_kv_rank <= params.ring_rank) {
                    signal_ready_once_row<RingConfig>(params, batch_idx * 2, row_in_interval);
                }
                signal_ready_once_row<RingConfig>(params, batch_idx * 2 + 1, row_in_interval);
            } else if (half_len > 0 && load_kv_rank >= params.ring_rank) {
                signal_ready_once_row<RingConfig>(params, batch_idx * 2 + 1, row_in_interval);
            }
        } else {
            signal_ready_once_row<RingConfig>(params, batch_idx, row_in_interval);
        }
    };

    auto decode_task = [&] (int task_id) {
        int step, task_in_step, rows_this_step;
        bool kv_use_half = false;
        if constexpr (RingConfig::kIsCausal) {
            if (task_id < half_section_tasks) {
                int const half_step_idx = task_id / half_loads_per_step;
                step = half_step_idx + 1;
                task_in_step = task_id - half_step_idx * half_loads_per_step;
                rows_this_step = hybrid_mode ? params.rows_per_rank : params.half_rows_per_rank;
                kv_use_half = true;
            } else {
                int const rem = task_id - half_section_tasks;
                int const full_step_idx = rem / full_loads_per_step;
                step = params.ring_rank + 1 + full_step_idx;
                task_in_step = rem - full_step_idx * full_loads_per_step;
                rows_this_step = params.rows_per_rank;
            }
        } else {
            int const step_minus_one = task_id / full_loads_per_step;
            step = step_minus_one + 1;
            task_in_step = task_id - step_minus_one * full_loads_per_step;
            rows_this_step = params.rows_per_rank;
        }
        bool const is_v = task_in_step >= rows_this_step;
        int const logical_row = is_v ? task_in_step - rows_this_step : task_in_step;
        int row_idx = logical_row;
        if constexpr (RingConfig::kIsCausal) {
            if (kv_use_half && !hybrid_mode) {
                row_idx = half_row_to_full_row(logical_row);
            }
        }
        int const load_kv_rank = (params.ring_rank - step + params.ring_world_size) % params.ring_world_size;
        int const row_with_rank = load_kv_rank * params.rows_per_rank + row_idx;
        return cute::make_tuple(is_v, row_with_rank, load_kv_rank, row_idx, kv_use_half);
    };

    if (warp_id < RingConfig::kNumCommChunks && laneid() == 0) {
        int const chunk_id = warp_id;
        for (int task_id = RingConfig::kNumCommChunks * comm_bid + chunk_id;
             task_id < total_tasks;
             task_id += RingConfig::kNumCommChunks * params.num_comm_sm) {
            auto [is_v, row_with_rank, load_kv_rank, row_idx, kv_use_half] = decode_task(task_id);
            if (!should_copy_row(row_idx, kv_use_half)) {
                continue;
            }

            wait(barriers.finished[chunk_id], get_phasebit<1>(phasebits, 0));
            update_phasebit<1>(phasebits, 0);

            tma::expect_bytes(barriers.arrived[chunk_id], sizeof(typename RingConfig::shared_vec));
            if (!is_v) {
                tma::load_async(vec[chunk_id], params.remote_k[load_kv_rank], {row_with_rank, 0}, barriers.arrived[chunk_id]);
            } else {
                tma::load_async(vec[chunk_id], params.remote_v[load_kv_rank], {row_with_rank, 0}, barriers.arrived[chunk_id]);
            }
        }
    } else if (warp_id < 2 * RingConfig::kNumCommChunks && laneid() == 0) {
        int const chunk_id = warp_id - RingConfig::kNumCommChunks;
        for (int task_id = RingConfig::kNumCommChunks * comm_bid + chunk_id;
             task_id < total_tasks;
             task_id += RingConfig::kNumCommChunks * params.num_comm_sm) {
            auto [is_v, row_with_rank, load_kv_rank, row_idx, kv_use_half] = decode_task(task_id);
            if (!should_copy_row(row_idx, kv_use_half)) {
                continue;
            }

            wait(barriers.arrived[chunk_id], get_phasebit<0>(phasebits, 0));
            update_phasebit<0>(phasebits, 0);

            int const dst_row = params.ready_once ? compact_row_for(load_kv_rank, row_idx) : row_with_rank;
            if (!is_v) {
                tma::store_async(params.local_k, vec[chunk_id], {dst_row, 0});
            } else {
                tma::store_async(params.local_v, vec[chunk_id], {dst_row, 0});
            }
            tma::store_async_read_wait(); // wait for the store to finish reading from shared memory
            arrive(barriers.finished[chunk_id]);
            // MEGA_RING: the readiness counter is consumed from global memory,
            // so wait for the TMA store itself, not just its shared-memory read.
            tma::store_async_wait();

            // MEGA_RING: each completed K/V row contributes one count.
            // Compute CTAs wait until the rank block reaches rows_per_rank * 2.
            // Causal zigzag half-steps only signal the copied front-half rows.
            if (params.ready_once) {
                signal_ready_once_for(load_kv_rank, row_idx);
            } else {
                min_fa3_varlen_demo::mega_ring::signal_release(params.kv_ready_counts + load_kv_rank, 1);
            }
        }
    }
}

template <typename RingConfig>
CUTLASS_GLOBAL
#ifdef __CUDACC__
__launch_bounds__(
    RingConfig::AttnKernel::MaxThreadsPerBlock,
    RingConfig::AttnKernel::MinBlocksPerMultiprocessor)
#endif
void mega_ring_flash_attn_varlen_kernel(CUTLASS_GRID_CONSTANT typename RingConfig::KernelParams const params) {
    extern __shared__ char smem_buf[];

    // MEGA_RING: one grid contains both persistent attention CTAs and remote K/V copy CTAs. Compute CTAs occupy [0, num_comp_sm).
    if (int(blockIdx.x) >= params.num_comp_sm) {
        run_mega_ring_remote_load<RingConfig>(params, int(blockIdx.x) - params.num_comp_sm, smem_buf);
        __syncthreads();
        typename RingConfig::AttnKernel attn_kernel;
        attn_kernel(params.compute, smem_buf, true);
    } else {
        typename RingConfig::AttnKernel attn_kernel;
        attn_kernel(params.compute, smem_buf, false);
    }
}

template <bool IsCausal, int NumDevices, bool ReadyOnce=false>
void run_mega_ring_min_fa3_varlen_ring_sm90(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    cudaStream_t stream) {
    using RingConfig = MegaRingKernelConfig<IsCausal, NumDevices, ReadyOnce>;
    using AttnKernel = typename RingConfig::AttnKernel;
    check_mega_ring_kernel_param_layout<RingConfig>();

    // MEGA_RING: params.total_k is the full local concatenated KV extent;
    // cu_seqlens_k and mega_ring_total_k_per_rank describe one rank-local block.
    int const seqlen_q = params.total_q;
    int const batch_q = 1;
    int const batch_k = 1;
    int const local_total_k = params.mega_ring_total_k_per_rank;
    int const compact_rows = params.total_k;
    int const remote_rows = static_cast<int>(remote_k.data_.size(0));
    using Index = typename Flash_fwd_params::index_t;

    TORCH_CHECK(local_total_k > 0, "mega ring requires mega_ring_total_k_per_rank > 0");
    TORCH_CHECK(params.mega_ring_ready_once || compact_rows == local_total_k * params.ring_world_size,
                "mega ring expects total_k to describe the full [world_size * local_total_k] buffer. Got total_k=",
                compact_rows, ", local_total_k=", local_total_k, ", world_size=", params.ring_world_size);
    TORCH_CHECK(remote_rows == local_total_k * params.ring_world_size,
                "mega ring remote_k must describe the full source [world_size * local_total_k] buffer. Got remote rows=",
                remote_rows, ", local_total_k=", local_total_k, ", world_size=", params.ring_world_size);
    TORCH_CHECK(params.h_k * params.d == RingConfig::kVecLength,
                "Mega ring communication path currently requires kv_heads * head_dim == ", RingConfig::kVecLength,
                ". Got kv_heads=", params.h_k,", head_dim=", params.d);

    typename RingConfig::CollectiveMainloop::StrideV v_strides = make_stride(params.v_row_stride, _1{}, params.v_head_stride, Index{0});
    // MEGA_RING: mainloop sees a full [world_size * local_total_k] K/V tensor
    // and receives semaphore/rank metadata for waiting on remote blocks.
    typename RingConfig::CollectiveMainloop::Arguments mainloop_args{
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},
        {params.q_row_stride, _1{}, params.q_head_stride, Index{0}},
        static_cast<Element*>(params.k_ptr),
        {compact_rows, params.d, params.h_k, batch_k},
        {params.k_row_stride, _1{}, params.k_head_stride, Index{0}},
        static_cast<Element*>(params.v_ptr),
        params.dv,
        v_strides,
        static_cast<Element const*>(nullptr),
        {0, params.d, params.h_k, 0},
        {Index{0}, _1{}, Index{0}, Index{0}},
        static_cast<Element const*>(nullptr),
        {Index{0}, _1{}, Index{0}, Index{0}},
        static_cast<Element const*>(nullptr),
        {Index{0}, _1{}, Index{0}, Index{0}},
        static_cast<Element const*>(nullptr),
        {0, 0},
        {Index{0}, _1{}},
        static_cast<Element const*>(nullptr),
        {Index{0}, _1{}},
        false,
        static_cast<int const*>(nullptr),
        {0, 0},
        {Index{0}, _1{}},
        params.scale_softmax,
        nullptr, nullptr, nullptr,
        {Index{0}, Index{0}},
        {Index{0}, Index{0}},
        {Index{0}, Index{0}},
        params.window_size_left,
        params.window_size_right,
        params.attention_chunk,
        0.0f,
        params.num_splits,
        nullptr,
        params.cu_seqlens_q,
        params.cu_seqlens_k,
        nullptr,
        params.seqused_q,
        params.seqused_k,
        params.leftpad_k,
        nullptr,
        params.mega_ring_kv_ready_counts,
        params.mega_ring_step_ready,
        params.mega_ring_half_cu_seqlens,
        params.ring_rank,
        params.ring_world_size,
        local_total_k,
        params.mega_ring_cp_total_k_per_rank,
        params.mega_ring_cp_batch_mask,
        params.mega_ring_ready_once,
        params.mega_ring_ready_end,
        params.mega_ring_ready_intervals,
        params.mega_ring_ready_max_chunks,
        params.mega_ring_ready_chunk_rows};

    typename RingConfig::CollectiveEpilogue::Arguments epilogue_args{
        static_cast<ElementOut*>(params.o_ptr),
        {seqlen_q, params.dv, params.h, batch_q, 1},
        {params.o_row_stride, _1{}, params.o_head_stride, Index{0}, Index{0}},
        static_cast<float*>(nullptr),
        {Index{0}, _1{}, Index{0}, Index{0}, Index{0}},
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen_q, Index{0}, Index{0}},
        static_cast<float*>(nullptr),
        {_1{}, Index{0}, Index{0}, Index{0}},
        params.h_k,
        params.cu_seqlens_q,
        params.seqused_q};

    int num_blocks_m = cutlass::ceil_div(params.seqlen_q, RingConfig::Config::kBlockM);
    // MEGA_RING: scheduler args add the ring world size and per-step tile count
    // so tile ids can be decoded as (ring_step, original_varlen_tile).
    typename flash::TileSchedulerArguments scheduler_args{
        num_blocks_m,
        params.h,
        params.b,
        params.num_splits,
        params.h / params.h_k,
        params.seqlen_q,
        params.seqlen_k,
        params.d,
        params.dv,
        int(sizeof(Element)),
        params.tile_count_semaphore,
        params.cu_seqlens_q,
        params.seqused_q,
        params.num_splits_dynamic_ptr,
        params.num_m_blocks_ptr,
        params.varlen_batch_idx_ptr,
        params.num_nheads_in_l2_ptr,
        params.num_comp_sm,
        0,
        true,
        params.ring_world_size,
        params.ring_rank,
        params.mega_ring_tiles_per_step,
        params.mega_ring_tiles_per_half_step,
        params.mega_ring_cp_batch_mask,
        params.mega_ring_cp_tiles_per_step,
        params.mega_ring_cp_tiles_per_half_step,
        params.mega_ring_ready_once};

    if (!params.skip_scheduler_metadata_computation) {
        // MEGA_RING: still uses the copied varlen scheduler metadata prep; the
        // mega scheduler repeats that prepared per-step tile stream.
        prepare_varlen_num_blocks(
            params,
            stream,
            kPackGQA,
            RingConfig::Config::kBlockM,
            RingConfig::Config::kBlockN,
            params.prepare_varlen_pdl);
        CHECK_CUDA_KERNEL_LAUNCH();
    }

    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    typename AttnKernel::Params compute_params = AttnKernel::to_underlying_arguments({
        mainloop_args,
        epilogue_args,
        {device, params.num_comp_sm},
        scheduler_args});

    auto kernel = mega_ring_flash_attn_varlen_kernel<RingConfig>;
    dim3 grid_dims(uint32_t(params.num_comp_sm + params.num_comm_sm), 1, 1);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;

    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    // MEGA_RING: remote loads write directly into the full K/V buffers, not
    // into single-step prefetch staging tensors.
    uint64_t local_k_dst = reinterpret_cast<uint64_t>(params.k_ptr);
    uint64_t local_v_dst = reinterpret_cast<uint64_t>(params.v_ptr);

    auto check_64b_alignment = [](uint64_t ptr, const char* name) {
        TORCH_CHECK(ptr % 64 == 0, name, " must be 64-byte aligned for mega ring launch diagnostics");
    };
    check_64b_alignment(reinterpret_cast<uint64_t>(params.q_ptr), "q_ptr");
    check_64b_alignment(reinterpret_cast<uint64_t>(params.k_ptr), "k_ptr");
    check_64b_alignment(reinterpret_cast<uint64_t>(params.v_ptr), "v_ptr");
    check_64b_alignment(reinterpret_cast<uint64_t>(params.o_ptr), "o_ptr");
    check_64b_alignment(reinterpret_cast<uint64_t>(remote_k.data_.data_ptr()), "remote_k.data_ptr");
    check_64b_alignment(reinterpret_cast<uint64_t>(remote_v.data_.data_ptr()), "remote_v.data_ptr");

    // MEGA_RING: both remote and local GL/PGL views span the full concatenated
    // K/V buffer; rows_per_rank tells communication CTAs where each rank block
    // starts inside that flat row space.
    typename RingConfig::KernelParams kernel_params{
        compute_params,
        kittens::py::parallel_tensor_to_pgl<typename RingConfig::remote_pgl>(remote_k, 1, 1, remote_rows, RingConfig::kVecLength),
        kittens::py::parallel_tensor_to_pgl<typename RingConfig::remote_pgl>(remote_v, 1, 1, remote_rows, RingConfig::kVecLength),
        kittens::make_gl<typename RingConfig::staging_gl>(local_k_dst, 1, 1, compact_rows, RingConfig::kVecLength),
        kittens::make_gl<typename RingConfig::staging_gl>(local_v_dst, 1, 1, compact_rows, RingConfig::kVecLength),
        params.num_comp_sm,
        params.num_comm_sm,
        local_total_k,
        IsCausal ? local_total_k / 2 : 0,
        params.mega_ring_cp_total_k_per_rank > 0 || params.mega_ring_cp_batch_mask != nullptr
            ? params.mega_ring_cp_total_k_per_rank
            : local_total_k,
        params.b,
        params.ring_rank,
        params.ring_world_size,
        params.mega_ring_ready_once && params.mega_ring_source_cu_seqlens_k != nullptr
            ? params.mega_ring_source_cu_seqlens_k
            : params.cu_seqlens_k,
        params.mega_ring_half_cu_seqlens,
        params.mega_ring_cp_batch_mask,
        params.mega_ring_kv_ready_counts,
        params.mega_ring_ready_once,
        params.mega_ring_ready_end,
        params.mega_ring_chunk_done,
        params.mega_ring_publish_lock,
        params.mega_ring_ready_interval_rows,
        params.mega_ring_ready_intervals,
        params.mega_ring_ready_max_chunks,
        params.mega_ring_ready_chunk_rows};

    bool const launch_with_pdl = params.prepare_varlen_pdl && !params.skip_scheduler_metadata_computation;
    if (!launch_with_pdl) {
        kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
        CHECK_CUDA_KERNEL_LAUNCH();
    } else {
#if ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8)))
        cudaLaunchConfig_t config{};
        cudaLaunchAttribute attrs[1]{};

        config.gridDim = grid_dims;
        config.blockDim = block_dims;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
        config.attrs = attrs;
        config.numAttrs = 1;

        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;

        CHECK_CUDA(cudaLaunchKernelEx(&config, kernel, kernel_params));
        CHECK_CUDA_KERNEL_LAUNCH();
#else
        TORCH_CHECK(false, "Mega ring varlen PDL launch requires CUDA >= 11.8");
#endif
    }
}

template <bool IsCausal, bool ReadyOnce>
void dispatch_mega_ring_world_size(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    cudaStream_t stream) {
    // MEGA_RING: NumDevices remains a compile-time TK pgl parameter, so keep
    // the same explicit local_world_size dispatch style as the ring path.
    switch (remote_k.local_world_size_) {
        case 1:
            run_mega_ring_min_fa3_varlen_ring_sm90<IsCausal, 1, ReadyOnce>(params, remote_k, remote_v, stream);
            break;
        case 2:
            run_mega_ring_min_fa3_varlen_ring_sm90<IsCausal, 2, ReadyOnce>(params, remote_k, remote_v, stream);
            break;
        case 3:
            run_mega_ring_min_fa3_varlen_ring_sm90<IsCausal, 3, ReadyOnce>(params, remote_k, remote_v, stream);
            break;
        case 4:
            run_mega_ring_min_fa3_varlen_ring_sm90<IsCausal, 4, ReadyOnce>(params, remote_k, remote_v, stream);
            break;
        case 5:
            run_mega_ring_min_fa3_varlen_ring_sm90<IsCausal, 5, ReadyOnce>(params, remote_k, remote_v, stream);
            break;
        case 6:
            run_mega_ring_min_fa3_varlen_ring_sm90<IsCausal, 6, ReadyOnce>(params, remote_k, remote_v, stream);
            break;
        case 7:
            run_mega_ring_min_fa3_varlen_ring_sm90<IsCausal, 7, ReadyOnce>(params, remote_k, remote_v, stream);
            break;
        case 8:
            run_mega_ring_min_fa3_varlen_ring_sm90<IsCausal, 8, ReadyOnce>(params, remote_k, remote_v, stream);
            break;
        default:
            TORCH_CHECK(false, "Unsupported local_world_size for mega ring varlen path: ", remote_k.local_world_size_);
    }
}

}  // namespace mega_ring_detail

// MEGA_RING: public C++ entry for the Python binding; implemented in
// csrc/mega_ring_min_fa3_varlen_ring_launch.cu.
void run_mega_ring_min_fa3_varlen_ring_fwd(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    cudaStream_t stream);

}  // namespace min_fa3_varlen_demo
