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

template <bool IsCausal, int NumDevices, bool CollectStats = false>
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
        true>;
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
        true,
        CollectStats>;
    using AttnKernel = flash::enable_sm90<flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    static constexpr int kVecLength = 1024;
    // MEGA_RING_TILE_COPY: communication is scheduled in attention-sized KV
    // tiles, while each physical TMA transaction moves a 16-row subtile.  All
    // mega-ring row ranges and arena offsets are at least 128-row aligned.
    enum : int {
        kRowsPerTask = Config::kBlockN,
        kRowsPerTransfer = 16,
    };
    static_assert(kRowsPerTask % kRowsPerTransfer == 0);
    static_assert(128 % kRowsPerTransfer == 0);
    static_assert(AttnKernel::MaxThreadsPerBlock % cutlass::NumThreadsPerWarp == 0);
    static constexpr int kNumWarpsPerBlock = AttnKernel::MaxThreadsPerBlock / cutlass::NumThreadsPerWarp;
    static constexpr int kNumCommChunks = kNumWarpsPerBlock / 2;
    static_assert(kNumCommChunks > 0);
    using shared_tile = st_bf<kRowsPerTransfer, kVecLength>;
    using staging_gl = gl<bf16, 1, 1, -1, kVecLength, shared_tile>;
    using remote_pgl = pgl<staging_gl, NumDevices, false>;
    static constexpr int kCommSmemSize =
        int(sizeof(shared_tile)) * kNumCommChunks + 1024;
    static_assert(kCommSmemSize <= MAX_SHARED_MEMORY,
                  "mega-ring forward communication staging exceeds Hopper shared memory");
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
        int rank_kv_capacity;
        int num_batch;
        int ring_rank;
        int ring_world_size;
        int const* cu_seqlens_k;
        int const* half_cu_seqlens;
        min_fa3_varlen_demo::MegaRingHierarchyDesc hierarchy;
        int* kv_ready_counts;
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
    typename RingConfig::shared_tile (&tile)[RingConfig::kNumCommChunks] =
        allocator.allocate<typename RingConfig::shared_tile, RingConfig::kNumCommChunks>();
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

    int total_tasks = 0;
    #pragma unroll
    for (int level_idx = 0; level_idx < 3; ++level_idx) {
        auto const& level = params.hierarchy.levels[level_idx];
        int const ring_base = (params.ring_rank / level.ring_size) * level.ring_size;
        int const ring_local_rank = params.ring_rank - ring_base;
        for (int step = 1; step < level.ring_size; ++step) {
            bool const use_half = RingConfig::kIsCausal && step <= ring_local_rank;
            int const rows = use_half ? level.half_rows : level.full_rows;
            total_tasks += 2 * ((rows + RingConfig::kRowsPerTask - 1) / RingConfig::kRowsPerTask);
        }
    }
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

    auto decode_task = [&] (int task_id) {
        int rem = task_id;
        int decoded_level_idx = 0;
        int decoded_step = 1;
        int rows_this_section = 0;
        bool kv_use_half = false;
        bool found = false;
        #pragma unroll
        for (int level_idx = 0; level_idx < 3 && !found; ++level_idx) {
            auto const& level = params.hierarchy.levels[level_idx];
            int const ring_base = (params.ring_rank / level.ring_size) * level.ring_size;
            int const ring_local_rank = params.ring_rank - ring_base;
            for (int step = 1; step < level.ring_size; ++step) {
                bool const use_half = RingConfig::kIsCausal && step <= ring_local_rank;
                int const rows = use_half ? level.half_rows : level.full_rows;
                int const tiles = (rows + RingConfig::kRowsPerTask - 1) / RingConfig::kRowsPerTask;
                int const section_tasks = 2 * tiles;
                if (rem < section_tasks) {
                    decoded_level_idx = level_idx;
                    decoded_step = step;
                    rows_this_section = rows;
                    kv_use_half = use_half;
                    found = true;
                    break;
                }
                rem -= section_tasks;
            }
        }
        auto const& level = params.hierarchy.levels[decoded_level_idx];
        int const tiles_this_section =
            (rows_this_section + RingConfig::kRowsPerTask - 1) / RingConfig::kRowsPerTask;
        bool const is_v = rem >= tiles_this_section;
        int const tile_idx = is_v ? rem - tiles_this_section : rem;
        int const logical_row = tile_idx * RingConfig::kRowsPerTask;
        int const rows_remaining = rows_this_section - logical_row;
        // Host validation makes every section 128-row aligned.  Together with
        // the assertions above, the last logical task is therefore still an
        // integral number of 16-row transfers; there is no short-tail path.
        int const valid_rows = rows_remaining < RingConfig::kRowsPerTask
            ? rows_remaining : RingConfig::kRowsPerTask;
        int const row_idx = kv_use_half
            ? half_row_to_full_row(level.half_row_begin + logical_row)
            : level.row_begin + logical_row;
        int const ring_base = (params.ring_rank / level.ring_size) * level.ring_size;
        int const ring_local_rank = params.ring_rank - ring_base;
        int const load_kv_rank = ring_base
            + (ring_local_rank - decoded_step + level.ring_size) % level.ring_size;
        int const row_with_rank = load_kv_rank * params.rank_kv_capacity + row_idx;
        int const ready_idx = level.kv_ready_base + decoded_step - 1;
        return cute::make_tuple(is_v, row_with_rank, load_kv_rank, ready_idx, valid_rows);
    };

    if (warp_id < RingConfig::kNumCommChunks && laneid() == 0) {
        int const chunk_id = warp_id;
        for (int task_id = RingConfig::kNumCommChunks * comm_bid + chunk_id;
             task_id < total_tasks;
             task_id += RingConfig::kNumCommChunks * params.num_comm_sm) {
            auto [is_v, row_with_rank, source_rank, ready_idx, valid_rows] = decode_task(task_id);
            (void)ready_idx;
            int const num_transfers = valid_rows / RingConfig::kRowsPerTransfer;
            for (int transfer_idx = 0; transfer_idx < num_transfers; ++transfer_idx) {
                int const transfer_row =
                    row_with_rank + transfer_idx * RingConfig::kRowsPerTransfer;
                wait(barriers.finished[chunk_id], get_phasebit<1>(phasebits, 0));
                update_phasebit<1>(phasebits, 0);
                tma::expect_bytes(barriers.arrived[chunk_id], sizeof(typename RingConfig::shared_tile));
                if (!is_v) {
                    tma::load_async(tile[chunk_id], params.remote_k[source_rank],
                                    {transfer_row / RingConfig::kRowsPerTransfer, 0}, barriers.arrived[chunk_id]);
                } else {
                    tma::load_async(tile[chunk_id], params.remote_v[source_rank],
                                    {transfer_row / RingConfig::kRowsPerTransfer, 0}, barriers.arrived[chunk_id]);
                }
            }
        }
    } else if (warp_id < 2 * RingConfig::kNumCommChunks && laneid() == 0) {
        int const chunk_id = warp_id - RingConfig::kNumCommChunks;
        for (int task_id = RingConfig::kNumCommChunks * comm_bid + chunk_id;
             task_id < total_tasks;
             task_id += RingConfig::kNumCommChunks * params.num_comm_sm) {
            auto [is_v, row_with_rank, source_rank, ready_idx, valid_rows] = decode_task(task_id);
            (void)source_rank;
            int const num_transfers = valid_rows / RingConfig::kRowsPerTransfer;
            for (int transfer_idx = 0; transfer_idx < num_transfers; ++transfer_idx) {
                int const transfer_row =
                    row_with_rank + transfer_idx * RingConfig::kRowsPerTransfer;
                wait(barriers.arrived[chunk_id], get_phasebit<0>(phasebits, 0));
                update_phasebit<0>(phasebits, 0);
                if (!is_v) {
                    tma::store_async(params.local_k, tile[chunk_id],
                                     {transfer_row / RingConfig::kRowsPerTransfer, 0});
                } else {
                    tma::store_async(params.local_v, tile[chunk_id],
                                     {transfer_row / RingConfig::kRowsPerTransfer, 0});
                }
                tma::store_async_read_wait(); // wait for the store to finish reading from shared memory
                arrive(barriers.finished[chunk_id]);
                // The readiness counter is consumed from global memory, so
                // wait for the TMA store itself, not just its shared read.
                tma::store_async_wait();
            }

            // One count represents a complete logical K or V tile.
            min_fa3_varlen_demo::mega_ring::signal_release(params.kv_ready_counts + ready_idx, 1);
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

template <bool IsCausal, int NumDevices, bool CollectStats>
void run_mega_ring_min_fa3_varlen_ring_sm90(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    cudaStream_t stream) {
    using RingConfig = MegaRingKernelConfig<IsCausal, NumDevices, CollectStats>;
    using AttnKernel = typename RingConfig::AttnKernel;
    check_mega_ring_kernel_param_layout<RingConfig>();

    // Empty ranks still launch the persistent kernel. A one-row dummy backing
    // gives TMA descriptors a non-zero extent while the scheduler exposes no work.
    int const seqlen_q = params.total_q > 0 ? params.total_q : 1;
    int const batch_q = 1;
    int const batch_k = 1;
    int const rank_kv_capacity = params.mega_ring_rank_kv_capacity;
    int const remote_rows = params.total_k;
    using Index = typename Flash_fwd_params::index_t;

    TORCH_CHECK(rank_kv_capacity > 0, "mega ring requires mega_ring_rank_kv_capacity > 0");
    TORCH_CHECK(remote_rows == rank_kv_capacity * params.ring_world_size,
                "mega ring expects total_k to describe the full [world_size * rank_kv_capacity] arena. Got total_k=",
                remote_rows, ", rank_kv_capacity=", rank_kv_capacity, ", world_size=", params.ring_world_size);
    TORCH_CHECK(params.h_k * params.d == RingConfig::kVecLength,
                "Mega ring communication path currently requires kv_heads * head_dim == ", RingConfig::kVecLength,
                ". Got kv_heads=", params.h_k,", head_dim=", params.d);

    typename RingConfig::CollectiveMainloop::StrideV v_strides = make_stride(params.v_row_stride, _1{}, params.v_head_stride, Index{0});
    // MEGA_RING: mainloop sees a full [world_size * rank_kv_capacity] K/V tensor
    // and receives semaphore/rank metadata for waiting on remote blocks.
    typename RingConfig::CollectiveMainloop::Arguments mainloop_args{
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},
        {params.q_row_stride, _1{}, params.q_head_stride, Index{0}},
        static_cast<Element*>(params.k_ptr),
        {remote_rows, params.d, params.h_k, batch_k},
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
        rank_kv_capacity,
        params.mega_ring_ring_sizes,
        params.mega_ring_hierarchy,
        params.mega_ring_stats};

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
        params.mega_ring_ring_sizes,
        params.mega_ring_hierarchy,
        params.mega_ring_kv_ready_counts,
        params.mega_ring_step_ready,
        params.mega_ring_scan_cursor,
        params.mega_ring_completed_tiles};

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
    int smem_size = AttnKernel::SharedStorageSize > RingConfig::kCommSmemSize
        ? AttnKernel::SharedStorageSize : RingConfig::kCommSmemSize;

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
    // K/V buffer; rank_kv_capacity tells communication CTAs where each rank block
    // starts inside that flat row space.
    typename RingConfig::KernelParams kernel_params{
        compute_params,
        kittens::py::parallel_tensor_to_pgl<typename RingConfig::remote_pgl>(remote_k, 1, 1, remote_rows, RingConfig::kVecLength),
        kittens::py::parallel_tensor_to_pgl<typename RingConfig::remote_pgl>(remote_v, 1, 1, remote_rows, RingConfig::kVecLength),
        kittens::make_gl<typename RingConfig::staging_gl>(local_k_dst, 1, 1, remote_rows, RingConfig::kVecLength),
        kittens::make_gl<typename RingConfig::staging_gl>(local_v_dst, 1, 1, remote_rows, RingConfig::kVecLength),
        params.num_comp_sm,
        params.num_comm_sm,
        rank_kv_capacity,
        params.b,
        params.ring_rank,
        params.ring_world_size,
        params.cu_seqlens_k,
        params.mega_ring_half_cu_seqlens,
        params.mega_ring_hierarchy,
        params.mega_ring_kv_ready_counts};

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

template <bool IsCausal, bool CollectStats>
void dispatch_mega_ring_world_size(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    cudaStream_t stream) {
    // MEGA_RING: NumDevices remains a compile-time TK pgl parameter, so keep
    // the same explicit local_world_size dispatch style as the ring path.
    switch (remote_k.local_world_size_) {
        case 2:
            run_mega_ring_min_fa3_varlen_ring_sm90<IsCausal, 2, CollectStats>(params, remote_k, remote_v, stream);
            break;
        case 4:
            run_mega_ring_min_fa3_varlen_ring_sm90<IsCausal, 4, CollectStats>(params, remote_k, remote_v, stream);
            break;
        case 8:
            run_mega_ring_min_fa3_varlen_ring_sm90<IsCausal, 8, CollectStats>(params, remote_k, remote_v, stream);
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
