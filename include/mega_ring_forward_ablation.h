// Causal forward ablation copied and trimmed from
// include/mega_ring_min_fa3_varlen_ring_launch.h. The production dynamic
// mega-ring launcher remains the L5/L6 implementation.

#pragma once

#include <torch/extension.h>

#include "mega_ring_forward_ablation_api.h"
#include "mega_ring_min_fa3_varlen_ring_launch.h"
#include "mega_ring_forward_ablation_scheduler.h"

namespace min_fa3_varlen_demo {
namespace forward_ablation {

template<int WorldSize, bool StepOnly, bool EnableReduction, bool CollectStats>
struct KernelConfig {
    using Production = mega_ring_detail::MegaRingKernelConfig<true, WorldSize, 8, CollectStats>;
    using Config = typename Production::Config;
    using ArchTag = typename Production::ArchTag;
    using TileShape_MNK = typename Production::TileShape_MNK;
    using TileShape_MNK_PV = typename Production::TileShape_MNK_PV;
    using ClusterShape = typename Production::ClusterShape;
    using CollectiveMainloop = typename Production::CollectiveMainloop;
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
        EnableReduction>;
    using Scheduler = flash::MegaRingVarlenLinearPersistentTileScheduler<
        Config::kBlockM,
        Config::kBlockN,
        CollectiveMainloop::NumMmaThreads,
        CollectiveMainloop::NumProducerThreads,
        StepOnly,
        CollectStats>;
    using AttnKernel = flash::enable_sm90<flash::FlashAttnFwdSm90<
        CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    static constexpr int kVecLength = Production::kVecLength;
    enum : int {
        kRowsPerTask = Production::kRowsPerTask,
        kRowsPerTransfer = Production::kRowsPerTransfer,
    };
    static constexpr int kNumWarpsPerBlock =
        AttnKernel::MaxThreadsPerBlock / cutlass::NumThreadsPerWarp;
    static constexpr int kNumCommChunks = kNumWarpsPerBlock / 2;
    using shared_tile = typename Production::shared_tile;
    using staging_gl = typename Production::staging_gl;
    using remote_pgl = typename Production::remote_pgl;
    static constexpr int kCommSmemSize = Production::kCommSmemSize;
    static constexpr bool kIsCausal = true;

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
        MegaRingHierarchyDesc hierarchy;
        int* kv_ready_counts;
        int ring_step;
    };
};

template<typename RingConfig>
CUTLASS_DEVICE void run_step_remote_load(
    typename RingConfig::KernelParams const& params,
    int comm_bid,
    char* smem_buf) {
    if (comm_bid >= params.num_comm_sm || params.ring_step == 0) { return; }

    using namespace kittens;
    tma_swizzle_allocator allocator(reinterpret_cast<int*>(smem_buf));
    typename RingConfig::shared_tile (&tile)[RingConfig::kNumCommChunks] =
        allocator.allocate<typename RingConfig::shared_tile, RingConfig::kNumCommChunks>();
    __shared__ mega_ring_detail::MegaRingRemoteLoadBarriers<RingConfig> barriers;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < RingConfig::kNumCommChunks; ++i) {
            init_semaphore(barriers.arrived[i], 0, 1);
            init_semaphore(barriers.finished[i], 0, 1);
        }
    }
    __syncthreads();

    auto const& level = params.hierarchy.levels[0];
    bool const use_half = params.ring_step <= params.ring_rank;
    int const rows = use_half ? level.half_rows : level.full_rows;
    int const tiles = cute::ceil_div(rows, RingConfig::kRowsPerTask);
    int const total_tasks = 2 * tiles;
    int const source_rank =
        (params.ring_rank - params.ring_step + params.ring_world_size)
        % params.ring_world_size;
    int const ready_idx = params.ring_step - 1;
    int const warp_id = warp::groupid();
    uint32_t phasebits = 0xFFFF0000;

    auto half_row_to_full_row = [&] (int half_row) {
        int lo = 0;
        int hi = params.num_batch;
        while (lo + 1 < hi) {
            int const mid = (lo + hi) / 2;
            if (params.half_cu_seqlens[mid] <= half_row) { lo = mid; }
            else { hi = mid; }
        }
        return params.cu_seqlens_k[lo]
            + half_row - params.half_cu_seqlens[lo];
    };

    auto decode_task = [&] (int task_id) {
        bool const is_v = task_id >= tiles;
        int const tile_idx = is_v ? task_id - tiles : task_id;
        int const logical_row = tile_idx * RingConfig::kRowsPerTask;
        int const rows_remaining = rows - logical_row;
        int const valid_rows = rows_remaining < RingConfig::kRowsPerTask
            ? rows_remaining : RingConfig::kRowsPerTask;
        int const row_idx = use_half
            ? half_row_to_full_row(level.half_row_begin + logical_row)
            : level.row_begin + logical_row;
        int const row_with_rank = source_rank * params.rank_kv_capacity + row_idx;
        return cute::make_tuple(is_v, row_with_rank, valid_rows);
    };

    if (warp_id < RingConfig::kNumCommChunks && laneid() == 0) {
        int const chunk_id = warp_id;
        for (int task_id = RingConfig::kNumCommChunks * comm_bid + chunk_id;
             task_id < total_tasks;
             task_id += RingConfig::kNumCommChunks * params.num_comm_sm) {
            auto [is_v, row_with_rank, valid_rows] = decode_task(task_id);
            int const num_transfers = valid_rows / RingConfig::kRowsPerTransfer;
            for (int transfer_idx = 0; transfer_idx < num_transfers; ++transfer_idx) {
                int const transfer_row = row_with_rank
                    + transfer_idx * RingConfig::kRowsPerTransfer;
                wait(barriers.finished[chunk_id], get_phasebit<1>(phasebits, 0));
                update_phasebit<1>(phasebits, 0);
                tma::expect_bytes(barriers.arrived[chunk_id], sizeof(typename RingConfig::shared_tile));
                if (!is_v) {
                    tma::load_async(tile[chunk_id], params.remote_k[source_rank],
                                    {transfer_row / RingConfig::kRowsPerTransfer, 0},
                                    barriers.arrived[chunk_id]);
                } else {
                    tma::load_async(tile[chunk_id], params.remote_v[source_rank],
                                    {transfer_row / RingConfig::kRowsPerTransfer, 0},
                                    barriers.arrived[chunk_id]);
                }
            }
        }
    } else if (warp_id < 2 * RingConfig::kNumCommChunks && laneid() == 0) {
        int const chunk_id = warp_id - RingConfig::kNumCommChunks;
        for (int task_id = RingConfig::kNumCommChunks * comm_bid + chunk_id;
             task_id < total_tasks;
             task_id += RingConfig::kNumCommChunks * params.num_comm_sm) {
            auto [is_v, row_with_rank, valid_rows] = decode_task(task_id);
            int const num_transfers = valid_rows / RingConfig::kRowsPerTransfer;
            for (int transfer_idx = 0; transfer_idx < num_transfers; ++transfer_idx) {
                int const transfer_row = row_with_rank
                    + transfer_idx * RingConfig::kRowsPerTransfer;
                wait(barriers.arrived[chunk_id], get_phasebit<0>(phasebits, 0));
                update_phasebit<0>(phasebits, 0);
                if (!is_v) {
                    tma::store_async(params.local_k, tile[chunk_id],
                                     {transfer_row / RingConfig::kRowsPerTransfer, 0});
                } else {
                    tma::store_async(params.local_v, tile[chunk_id],
                                     {transfer_row / RingConfig::kRowsPerTransfer, 0});
                }
                tma::store_async_read_wait();
                arrive(barriers.finished[chunk_id]);
                tma::store_async_wait();
            }
            mega_ring::signal_release(params.kv_ready_counts + ready_idx, 1);
        }
    }
}

template<typename RingConfig>
CUTLASS_GLOBAL
#ifdef __CUDACC__
__launch_bounds__(
    RingConfig::AttnKernel::MaxThreadsPerBlock,
    RingConfig::AttnKernel::MinBlocksPerMultiprocessor)
#endif
void step_flash_attn_varlen_kernel(
    CUTLASS_GRID_CONSTANT typename RingConfig::KernelParams const params) {
    extern __shared__ char smem_buf[];
    if (int(blockIdx.x) >= params.num_comp_sm) {
        run_step_remote_load<RingConfig>(
            params, int(blockIdx.x) - params.num_comp_sm, smem_buf);
    } else {
        typename RingConfig::AttnKernel kernel;
        kernel(params.compute, smem_buf, false);
    }
}

template<bool RecycleComm, typename RingConfig>
CUTLASS_GLOBAL
#ifdef __CUDACC__
__launch_bounds__(
    RingConfig::AttnKernel::MaxThreadsPerBlock,
    RingConfig::AttnKernel::MinBlocksPerMultiprocessor)
#endif
void linear_flash_attn_varlen_kernel(
    CUTLASS_GRID_CONSTANT typename RingConfig::KernelParams const params) {
    extern __shared__ char smem_buf[];
    if (int(blockIdx.x) >= params.num_comp_sm) {
        mega_ring_detail::run_mega_ring_remote_load<RingConfig>(
            params, int(blockIdx.x) - params.num_comp_sm, smem_buf);
        if constexpr (RecycleComm) {
            __syncthreads();
            typename RingConfig::AttnKernel kernel;
            kernel(params.compute, smem_buf, true);
        }
    } else {
        typename RingConfig::AttnKernel kernel;
        kernel(params.compute, smem_buf, false);
    }
}

CUTLASS_GLOBAL void external_reduce_kernel(
    ElementOut const* partial_o,
    float const* partial_lse,
    ElementOut* running_o,
    float* running_lse,
    int const* cu_seqlens_q,
    int total_q,
    int q_heads,
    int num_batch,
    int ring_rank,
    int ring_step) {
    int const row = int(blockIdx.x);
    int const head = int(blockIdx.y);
    if (row >= total_q || head >= q_heads) { return; }

    int lo = 0;
    int hi = num_batch;
    while (lo + 1 < hi) {
        int const mid = (lo + hi) / 2;
        if (cu_seqlens_q[mid] <= row) { lo = mid; }
        else { hi = mid; }
    }
    int const row_in_batch = row - cu_seqlens_q[lo];
    int const batch_rows = cu_seqlens_q[lo + 1] - cu_seqlens_q[lo];
    if (ring_step > ring_rank && row_in_batch < batch_rows / 2) { return; }

    __shared__ float block_scale;
    int const lse_idx = head * total_q + row;
    if (threadIdx.x == 0) {
        float const block_lse = partial_lse[lse_idx];
        if (ring_step == 0) {
            running_lse[lse_idx] = block_lse;
            block_scale = 1.0f;
        } else {
            float const prev_lse = running_lse[lse_idx];
            if (prev_lse == -INFINITY && block_lse == -INFINITY) {
                running_lse[lse_idx] = -INFINITY;
                block_scale = -1.0f;
            } else {
                float const delta = block_lse - prev_lse;
                float const delta_exp = expf(-fabsf(delta));
                float const inv_one_plus_delta = 1.0f / (1.0f + delta_exp);
                running_lse[lse_idx] = fmaxf(prev_lse, block_lse)
                    + log1pf(delta_exp);
                block_scale = delta >= 0.0f
                    ? inv_one_plus_delta
                    : delta_exp * inv_one_plus_delta;
            }
        }
    }
    __syncthreads();

    int const idx = (row * q_heads + head) * 128 + int(threadIdx.x);
    float const block_out = static_cast<float>(partial_o[idx]);
    if (ring_step == 0) {
        running_o[idx] = partial_o[idx];
    } else if (block_scale < 0.0f) {
        running_o[idx] = ElementOut(0.0f);
    } else {
        float const prev_out = static_cast<float>(running_o[idx]);
        running_o[idx] = ElementOut(
            prev_out + block_scale * (block_out - prev_out));
    }
}

template<typename RingConfig>
typename RingConfig::KernelParams make_kernel_params(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v) {
    using Index = typename Flash_fwd_params::index_t;
    using AttnKernel = typename RingConfig::AttnKernel;
    int const seqlen_q = params.total_q > 0 ? params.total_q : 1;
    int const remote_rows = params.total_k;
    int const rank_kv_capacity = params.mega_ring_rank_kv_capacity;

    typename RingConfig::CollectiveMainloop::StrideV v_strides =
        make_stride(params.v_row_stride, _1{}, params.v_head_stride, Index{0});
    typename RingConfig::CollectiveMainloop::Arguments mainloop_args{
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, 1},
        {params.q_row_stride, _1{}, params.q_head_stride, Index{0}},
        static_cast<Element*>(params.k_ptr),
        {remote_rows, params.d, params.h_k, 1},
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
        {seqlen_q, params.dv, params.h, 1, 1},
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

    typename flash::TileSchedulerArguments scheduler_args{
        cutlass::ceil_div(params.seqlen_q, RingConfig::Config::kBlockM),
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
        params.mega_ring_completed_tiles,
        params.ring_step};

    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    typename AttnKernel::Params compute = AttnKernel::to_underlying_arguments({
        mainloop_args,
        epilogue_args,
        {device, params.num_comp_sm},
        scheduler_args});

    uint64_t const local_k = reinterpret_cast<uint64_t>(params.k_ptr);
    uint64_t const local_v = reinterpret_cast<uint64_t>(params.v_ptr);
    return {
        compute,
        kittens::py::parallel_tensor_to_pgl<typename RingConfig::remote_pgl>(
            remote_k, 1, 1, remote_rows, RingConfig::kVecLength),
        kittens::py::parallel_tensor_to_pgl<typename RingConfig::remote_pgl>(
            remote_v, 1, 1, remote_rows, RingConfig::kVecLength),
        kittens::make_gl<typename RingConfig::staging_gl>(
            local_k, 1, 1, remote_rows, RingConfig::kVecLength),
        kittens::make_gl<typename RingConfig::staging_gl>(
            local_v, 1, 1, remote_rows, RingConfig::kVecLength),
        params.num_comp_sm,
        params.num_comm_sm,
        rank_kv_capacity,
        params.b,
        params.ring_rank,
        params.ring_world_size,
        params.cu_seqlens_k,
        params.mega_ring_half_cu_seqlens,
        params.mega_ring_hierarchy,
        params.mega_ring_kv_ready_counts,
        params.ring_step};
}

template<typename RingConfig, typename Kernel>
void launch_kernel(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    cudaStream_t stream,
    Kernel kernel) {
    using AttnKernel = typename RingConfig::AttnKernel;
    auto kernel_params = make_kernel_params<RingConfig>(params, remote_k, remote_v);
    int const smem_size = AttnKernel::SharedStorageSize > RingConfig::kCommSmemSize
        ? AttnKernel::SharedStorageSize : RingConfig::kCommSmemSize;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<dim3(params.num_comp_sm + params.num_comm_sm),
             AttnKernel::get_block_shape(), smem_size, stream>>>(kernel_params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<int WorldSize, bool EnableReduction, bool CollectStats>
void run_steps(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    torch::Tensor& scratch_o,
    torch::Tensor& scratch_lse,
    cudaStream_t stream) {
    using RingConfig = KernelConfig<WorldSize, true, EnableReduction, CollectStats>;
    void* const running_o = params.o_ptr;
    void* const running_lse = params.softmax_lse_ptr;
    if constexpr (!EnableReduction) {
        params.o_ptr = scratch_o.data_ptr();
        params.softmax_lse_ptr = scratch_lse.data_ptr();
    }
    for (int step = 0; step < WorldSize; ++step) {
        params.ring_step = step;
        CHECK_CUDA(cudaMemsetAsync(params.tile_count_semaphore, 0, sizeof(int), stream));
        launch_kernel<RingConfig>(
            params, remote_k, remote_v, stream,
            step_flash_attn_varlen_kernel<RingConfig>);
        if constexpr (!EnableReduction) {
            dim3 const grid(params.total_q, params.h, 1);
            external_reduce_kernel<<<grid, 128, 0, stream>>>(
                static_cast<ElementOut const*>(params.o_ptr),
                static_cast<float const*>(params.softmax_lse_ptr),
                static_cast<ElementOut*>(running_o),
                static_cast<float*>(running_lse),
                params.cu_seqlens_q,
                params.total_q,
                params.h,
                params.b,
                params.ring_rank,
                step);
            CHECK_CUDA_KERNEL_LAUNCH();
        }
    }
    params.o_ptr = running_o;
    params.softmax_lse_ptr = running_lse;
}

template<int WorldSize, bool RecycleComm, bool CollectStats>
void run_linear(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    cudaStream_t stream) {
    using RingConfig = KernelConfig<WorldSize, false, true, CollectStats>;
    params.ring_step = -1;
    launch_kernel<RingConfig>(
        params, remote_k, remote_v, stream,
        linear_flash_attn_varlen_kernel<RecycleComm, RingConfig>);
}

}  // namespace forward_ablation
}  // namespace min_fa3_varlen_demo
