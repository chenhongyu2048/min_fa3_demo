// Copied and trimmed from Hopper backward launch source:
// - hopper/flash_bwd_launch_template.h

#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#ifdef CHECK_CUDA
#undef CHECK_CUDA
#endif
#ifdef CHECK_CONTIGUOUS
#undef CHECK_CONTIGUOUS
#endif
#ifdef CHECK_INPUT
#undef CHECK_INPUT
#endif

#include "cute/tensor.hpp"

#include "cutlass/device_kernel.h"  // For device_kernel
#include "cutlass/kernel_launch.h"  // For kernel_launch
#include "cutlass/cluster_launch.hpp"

#include "hopper_compat/cuda_check.h"
#include "hopper_compat/static_switch.h"
#include "backward/min_fa3_bwd_params.h"
#include "backward/min_fa3_bwd_traits.h"
#include "backward/min_fa3_bwd_preprocess.h"
#include "backward/min_fa3_bwd_postprocess.h"
#include "backward/min_fa3_bwd_scheduler.h"
#include "backward/min_fa3_bwd_mainloop.h"
#include "backward/min_fa3_bwd_epilogue.h"
#include "backward/min_fa3_bwd_kernel.h"

namespace min_fa3_backward {

using namespace cute;
using namespace kittens;

template <typename AttnKernel, int NumDevices>
struct MegaRingBwdCommConfig {
    static constexpr int kVecLength = 1024;
    static constexpr int kNumWarps = AttnKernel::MaxThreadsPerBlock / cutlass::NumThreadsPerWarp;
    static constexpr int kNumChunks = kNumWarps / 2;
    static_assert(kNumChunks > 0);

    using KShared = sv_bf<kVecLength>;
    using KGlobal = gl<bf16, 1, 1, -1, kVecLength, KShared>;
    using KRemote = pgl<KGlobal, NumDevices, false>;
    using DShared = sv_fl<kVecLength>;
    using DGlobal = gl<float, 1, 1, -1, kVecLength, DShared>;
    using DRemote = pgl<DGlobal, NumDevices, false>;
};

template <typename AttnKernel, int NumDevices>
struct alignas(128) MegaRingBwdKernelParams {
    using Comm = MegaRingBwdCommConfig<AttnKernel, NumDevices>;
    typename AttnKernel::Params compute;
    typename Comm::KRemote remote_k;
    typename Comm::KRemote remote_v;
    typename Comm::KGlobal local_k;
    typename Comm::KGlobal local_v;
    typename Comm::DGlobal local_dk_steps;
    typename Comm::DGlobal local_dv_steps;
    typename Comm::DRemote remote_dk_accum;
    typename Comm::DRemote remote_dv_accum;
    int rows_per_rank;
    int half_rows_per_rank;
    int dkv_rows_per_step;
    int num_batch;
    int const* cu_seqlens_k;
    int const* half_cu_seqlens;
    int* kv_ready;
    int* local_ready;
    int* expected_ready;
    int* dkv_comm_done;
    int ring_rank;
    int ring_world_size;
    int num_comp_sm;
    int num_comm_sm;
    int* remote_completion[8];
};

template <typename Comm>
struct alignas(128) MegaRingBwdTmaBarriers {
    semaphore arrived[Comm::kNumChunks];
    semaphore finished[Comm::kNumChunks];
};

template <typename Comm>
constexpr void check_mega_ring_bwd_tma_layout() {
    static_assert(sizeof(typename Comm::KShared) == Comm::kVecLength * sizeof(bf16));
    static_assert(sizeof(typename Comm::DShared) == Comm::kVecLength * sizeof(float));
    static_assert(sizeof(MegaRingBwdTmaBarriers<Comm>) % 128 == 0);
}

template <typename AttnKernel, int NumDevices>
CUTLASS_DEVICE void run_mega_ring_bwd_kv_load(
        MegaRingBwdKernelParams<AttnKernel, NumDevices> const& params,
        int comm_bid,
        char* smem_buf) {
    using Comm = MegaRingBwdCommConfig<AttnKernel, NumDevices>;
    if (params.ring_world_size <= 1) { return; }

    tma_swizzle_allocator allocator(reinterpret_cast<int*>(smem_buf));
    typename Comm::KShared (&vec)[Comm::kNumChunks] =
        allocator.allocate<typename Comm::KShared, Comm::kNumChunks>();
    __shared__ MegaRingBwdTmaBarriers<Comm> barriers;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < Comm::kNumChunks; ++i) {
            init_semaphore(barriers.arrived[i], 0, 1);
            init_semaphore(barriers.finished[i], 0, 1);
        }
    }
    __syncthreads();

    int const half_tasks = params.ring_rank * params.half_rows_per_rank * 2;
    int const full_tasks_per_step = params.rows_per_rank * 2;
    int const total_tasks = half_tasks
        + (params.ring_world_size - 1 - params.ring_rank) * full_tasks_per_step;
    int const warp_id = warp::groupid();
    uint32_t phasebits = 0xFFFF0000;

    auto half_row_to_full_row = [&] (int half_row) {
        int lo = 0, hi = params.num_batch;
        while (lo + 1 < hi) {
            int const mid = (lo + hi) / 2;
            if (params.half_cu_seqlens[mid] <= half_row) { lo = mid; }
            else { hi = mid; }
        }
        return params.cu_seqlens_k[lo] + half_row - params.half_cu_seqlens[lo];
    };

    auto decode_task = [&] (int task_id) {
        int step, task_in_step, rows_this_step;
        bool use_half;
        if (task_id < half_tasks) {
            int const step_idx = task_id / (params.half_rows_per_rank * 2);
            step = step_idx + 1;
            task_in_step = task_id - step_idx * params.half_rows_per_rank * 2;
            rows_this_step = params.half_rows_per_rank;
            use_half = true;
        } else {
            int const remaining = task_id - half_tasks;
            int const step_idx = remaining / full_tasks_per_step;
            step = params.ring_rank + 1 + step_idx;
            task_in_step = remaining - step_idx * full_tasks_per_step;
            rows_this_step = params.rows_per_rank;
            use_half = false;
        }
        bool const is_v = task_in_step >= rows_this_step;
        int const logical_row = is_v ? task_in_step - rows_this_step : task_in_step;
        int const row = use_half ? half_row_to_full_row(logical_row) : logical_row;
        int owner = params.ring_rank - step;
        if (owner < 0) { owner += params.ring_world_size; }
        return cute::make_tuple(step, is_v, owner * params.rows_per_rank + row, owner);
    };

    if (warp_id < Comm::kNumChunks && laneid() == 0) {
        int const chunk = warp_id;
        for (int task_id = Comm::kNumChunks * comm_bid + chunk;
             task_id < total_tasks;
             task_id += Comm::kNumChunks * params.num_comm_sm) {
            auto [step, is_v, row, owner] = decode_task(task_id);
            wait(barriers.finished[chunk], get_phasebit<1>(phasebits, 0));
            update_phasebit<1>(phasebits, 0);
            tma::expect_bytes(barriers.arrived[chunk], sizeof(typename Comm::KShared));
            if (is_v) { tma::load_async(vec[chunk], params.remote_v[owner], {row, 0}, barriers.arrived[chunk]); }
            else { tma::load_async(vec[chunk], params.remote_k[owner], {row, 0}, barriers.arrived[chunk]); }
        }
    } else if (warp_id < 2 * Comm::kNumChunks && laneid() == 0) {
        int const chunk = warp_id - Comm::kNumChunks;
        for (int task_id = Comm::kNumChunks * comm_bid + chunk;
             task_id < total_tasks;
             task_id += Comm::kNumChunks * params.num_comm_sm) {
            auto [step, is_v, row, owner] = decode_task(task_id);
            wait(barriers.arrived[chunk], get_phasebit<0>(phasebits, 0));
            update_phasebit<0>(phasebits, 0);
            if (is_v) { tma::store_async(params.local_v, vec[chunk], {row, 0}); }
            else { tma::store_async(params.local_k, vec[chunk], {row, 0}); }
            tma::store_async_read_wait();
            arrive(barriers.finished[chunk]);
            tma::store_async_wait();
            min_fa3_varlen_demo::mega_ring::signal_release(params.kv_ready + step, 1);
        }
    }
}

template <typename AttnKernel, int NumDevices>
CUTLASS_DEVICE void run_mega_ring_bwd_dkv_store(
        MegaRingBwdKernelParams<AttnKernel, NumDevices> const& params,
        int comm_bid,
        char* smem_buf) {
    using Comm = MegaRingBwdCommConfig<AttnKernel, NumDevices>;
    tma_swizzle_allocator allocator(reinterpret_cast<int*>(smem_buf));
    typename Comm::DShared (&vec)[Comm::kNumChunks] =
        allocator.allocate<typename Comm::DShared, Comm::kNumChunks>();
    __shared__ MegaRingBwdTmaBarriers<Comm> barriers;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < Comm::kNumChunks; ++i) {
            init_semaphore(barriers.arrived[i], 0, 1);
            init_semaphore(barriers.finished[i], 0, 1);
        }
    }
    __syncthreads();

    int const warp_id = warp::groupid();
    uint32_t phasebits = 0xFFFF0000;
    for (int step = 0; step < params.ring_world_size; ++step) {
        if (warp_id < 2 * Comm::kNumChunks && laneid() == 0) {
            min_fa3_varlen_demo::mega_ring::wait_until_at_least_acquire(
                params.local_ready + step, params.expected_ready[step]);
        }

        int owner = params.ring_rank - step;
        if (owner < 0) { owner += params.ring_world_size; }
        int const tasks_per_step = params.dkv_rows_per_step * 2;
        if (warp_id < Comm::kNumChunks && laneid() == 0) {
            int const chunk = warp_id;
            for (int task = Comm::kNumChunks * comm_bid + chunk;
                 task < tasks_per_step;
                 task += Comm::kNumChunks * params.num_comm_sm) {
                bool const is_v = task >= params.dkv_rows_per_step;
                int const row = is_v ? task - params.dkv_rows_per_step : task;
                int const local_row = step * params.dkv_rows_per_step + row;
                wait(barriers.finished[chunk], get_phasebit<1>(phasebits, 0));
                update_phasebit<1>(phasebits, 0);
                tma::expect_bytes(barriers.arrived[chunk], sizeof(typename Comm::DShared));
                if (is_v) { tma::load_async(vec[chunk], params.local_dv_steps, {local_row, 0}, barriers.arrived[chunk]); }
                else { tma::load_async(vec[chunk], params.local_dk_steps, {local_row, 0}, barriers.arrived[chunk]); }
            }
        } else if (warp_id < 2 * Comm::kNumChunks && laneid() == 0) {
            int const chunk = warp_id - Comm::kNumChunks;
            for (int task = Comm::kNumChunks * comm_bid + chunk;
                 task < tasks_per_step;
                 task += Comm::kNumChunks * params.num_comm_sm) {
                bool const is_v = task >= params.dkv_rows_per_step;
                int const row = is_v ? task - params.dkv_rows_per_step : task;
                wait(barriers.arrived[chunk], get_phasebit<0>(phasebits, 0));
                update_phasebit<0>(phasebits, 0);
                if (is_v) { tma::store_add_async(params.remote_dv_accum[owner], vec[chunk], {row, 0}); }
                else { tma::store_add_async(params.remote_dk_accum[owner], vec[chunk], {row, 0}); }
                tma::store_async_read_wait();
                arrive(barriers.finished[chunk]);
            }
            tma::store_async_wait();
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int const previous = atomicAdd(params.dkv_comm_done + step, 1);
            if (previous + 1 == params.num_comm_sm) {
                min_fa3_varlen_demo::mega_ring::signal_release_system(
                    params.remote_completion[owner], 1);
            }
        }
    }
}

template <typename AttnKernel, int NumDevices>
CUTLASS_GLOBAL
#ifdef __CUDACC__
__launch_bounds__(AttnKernel::MaxThreadsPerBlock, AttnKernel::MinBlocksPerMultiprocessor)
#endif
void mega_ring_flash_attn_bwd_kernel(
        CUTLASS_GRID_CONSTANT MegaRingBwdKernelParams<AttnKernel, NumDevices> const params) {
    extern __shared__ char smem_buf[];
    if (int(blockIdx.x) < params.num_comp_sm) {
        AttnKernel kernel;
        kernel(params.compute, smem_buf);
        return;
    }

    int const comm_bid = int(blockIdx.x) - params.num_comp_sm;
    run_mega_ring_bwd_kv_load<AttnKernel, NumDevices>(params, comm_bid, smem_buf);
    run_mega_ring_bwd_dkv_store<AttnKernel, NumDevices>(params, comm_bid, smem_buf);
}

static __global__ void mega_ring_bwd_wait_for_completion(int const* completion, int expected) {
    if (threadIdx.x == 0) {
        min_fa3_varlen_demo::mega_ring::wait_until_at_least_acquire_system(
            completion, expected);
    }
}

template <int Arch, int kHeadDim, int kBlockM, int kBlockN, typename Element,
          bool Is_causal, bool Is_local, bool Has_softcap, bool Varlen, bool Deterministic, bool GQA,
          int Stages_dO=2, int Stages_dS_or_QSm80=2,
          bool SdP_swapAB=true, bool dKV_swapAB=false, bool dQ_swapAB=false,
          int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
          bool V_in_regs=false, bool MegaRing=false, int NumDevices=1>
void run_flash_bwd(
        Flash_bwd_params &params,
        cudaStream_t stream,
        kittens::py::TKParallelTensor* remote_k_tensor = nullptr,
        kittens::py::TKParallelTensor* remote_v_tensor = nullptr,
        kittens::py::TKParallelTensor* remote_dk_tensor = nullptr,
        kittens::py::TKParallelTensor* remote_dv_tensor = nullptr) {
    static_assert(!(Is_causal && Is_local), "Is_causal and Is_local cannot be true at the same time.");
    using ElementAccum = float;
    static_assert(Arch == 90, "The minimal backward demo only supports SM90");
    using ArchTag = cutlass::arch::Sm90;

    int const total_q_padded_rounded = cute::round_up(params.total_q + params.b * kBlockM, kBlockM);
    int const total_k_logical = MegaRing ? params.local_total_k : params.total_k;
    int const total_k_padded_rounded = cute::round_up(total_k_logical + params.b * kBlockN, kBlockN);
    bool const is_varlen_q = params.cu_seqlens_q;
    bool const is_varlen_k = params.cu_seqlens_k;
    int seqlen_q = !is_varlen_q ? params.seqlen_q : params.total_q;
    int seqlen_k = !is_varlen_k ? params.seqlen_k : total_k_logical;
    int seqlen_k_storage = MegaRing ? params.total_k : seqlen_k;
    int seqlen_q_rounded = !is_varlen_q ? params.seqlen_q_rounded : total_q_padded_rounded;
    int seqlen_k_rounded = !is_varlen_k ? params.seqlen_k_rounded : total_k_padded_rounded;
    int batch_q = !is_varlen_q ? params.b : 1;
    int batch_k = !is_varlen_k ? params.b : 1;

    using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
    using PreprocessKernel = flash::FlashAttnBwdPreprocess<TileShape_MK, Element, ElementAccum, ArchTag, /*Clear_dQaccum=*/true, Varlen>;
    typename PreprocessKernel::Arguments preprocess_args {
        static_cast<Element const*>(params.o_ptr),
        {seqlen_q, params.dv, params.h, batch_q},  // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride, !is_varlen_q ? params.o_batch_stride : 0},  // stride_O
        static_cast<Element const*>(params.do_ptr),
        {params.do_row_stride, _1{}, params.do_head_stride, !is_varlen_q ? params.do_batch_stride : 0},  // stride_dO
        static_cast<float*>(params.dsoftmax_sum),
        {seqlen_q_rounded, params.h, batch_q},  // shape_dPsum
        {_1{}, seqlen_q_rounded, !is_varlen_q ? params.h * params.seqlen_q_rounded : 0},  // stride_dPsum
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen_q, !is_varlen_q ? params.h * params.seqlen_q : 0},  // stride_LSE
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {_1{}, seqlen_q_rounded, !is_varlen_q ? params.h * params.seqlen_q_rounded : 0},  // stride_LSE_log2
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, batch_q},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, !is_varlen_q ? params.d_rounded * seqlen_q_rounded * params.h : 0},  // stride_dQaccum
        params.b,
        params.dq_semaphore,
        params.cu_seqlens_q,
        params.seqused_q
    };
    typename PreprocessKernel::Params preprocess_params = PreprocessKernel::to_underlying_arguments(preprocess_args);
    int num_m_block = cute::ceil_div(params.seqlen_q, kBlockM);
    dim3 grid_m(num_m_block, params.h, params.b);
    CHECK_CUTLASS(cutlass::kernel_launch<PreprocessKernel>(grid_m, PreprocessKernel::MaxThreadsPerBlock, PreprocessKernel::SharedStorageSize, stream, preprocess_params, false /*launch_with_pdl*/));

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<_1, Int<1>, _1>;  // Currently doesn't not support cluster
    static constexpr int Stages = 2;
    static constexpr int Stages_dS = Stages_dS_or_QSm80;
    using CollectiveMainloop =
        flash::CollectiveMainloopBwdSm90<Stages, Stages_dO, Stages_dS, ClusterShape, TileShape_MNK, Element, ElementAccum, cutlass::arch::Sm90,
            Is_causal, Is_local, Has_softcap, Varlen, Deterministic,
            SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs, MegaRing>;
    using CollectiveEpilogue = std::conditional_t<
        !GQA && !MegaRing,
        flash::CollectiveEpilogueBwd<TileShape_MNK, Element, ArchTag, CollectiveMainloop::NumMmaThreads, Varlen, dKV_swapAB, NumMmaWarpGroups / AtomLayoutNdKV>,
        flash::CollectiveEpilogueBwdGQA<TileShape_MNK, ElementAccum, ArchTag, CollectiveMainloop::NumMmaThreads, Varlen, Deterministic>
    >;
    using Scheduler = std::conditional_t<MegaRing,
        flash::MegaRingSingleTileBwdLPTScheduler<Varlen, kBlockN, CollectiveMainloop::NumMmaThreads>,
        std::conditional_t<
            Is_causal,
            flash::SingleTileBwdLPTScheduler<Varlen, kBlockN, Is_causal && Deterministic /*SPT*/, !Deterministic /*Persistent*/, CollectiveMainloop::NumMmaThreads>,
            flash::SingleTileBwdScheduler<Varlen, kBlockN, !Deterministic /*Persistent*/>
        >>;
    using AttnKernel = flash::enable_sm90<
        flash::FlashAttnBwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, !is_varlen_q ? params.q_batch_stride : 0},  // stride_Q
        static_cast<Element const*>(params.k_ptr),
        {seqlen_k_storage, params.d, params.h_k, batch_k},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen_k ? params.k_batch_stride : 0},  // stride_K
        static_cast<Element const*>(params.v_ptr),
        {seqlen_k_storage, params.dv, params.h_k, batch_k},  // shape_V
        {params.v_row_stride, _1{}, params.v_head_stride, !is_varlen_k ? params.v_batch_stride : 0},  // stride_V
        static_cast<Element const*>(params.do_ptr),
        {seqlen_q, params.dv, params.h, batch_q},  // shape_dO
        {params.do_row_stride, _1{}, params.do_head_stride, !is_varlen_q ? params.do_batch_stride : 0},  // stride_dO
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, batch_q},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, !is_varlen_q ? params.d_rounded * params.seqlen_q_rounded * params.h : 0}, // stride_dQaccum
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {seqlen_q_rounded, params.h, batch_q},  // shape_LSE
        {_1{}, seqlen_q_rounded, !is_varlen_q ? params.h * params.seqlen_q_rounded : 0},  // stride_LSE_log2
        static_cast<float*>(params.dsoftmax_sum),
        {_1{}, seqlen_q_rounded, !is_varlen_q ? params.h * params.seqlen_q_rounded : 0},  // stride_dPsum
        params.scale_softmax,
        params.window_size_left, params.window_size_right, 0 /*attention_chunk*/,
        0.f,
        params.b,
        params.dq_semaphore,
        params.cu_seqlens_q, params.cu_seqlens_k,
        params.seqused_q, params.seqused_k,
        params.ring_rank, params.ring_world_size, params.local_total_k,
        params.half_cu_seqlens, MegaRing ? params.ring_kv_ready : nullptr,
        MegaRing ? params.ring_kv_expected_ready : nullptr
    };
    // The case work with GQA is ugly but idk how to fix it.
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<typename CollectiveEpilogue::Element*>(!GQA && !MegaRing ? params.dk_ptr : params.dk_accum_ptr),
        [&] {
            if constexpr (!GQA && !MegaRing) {
                return typename CollectiveEpilogue::ShapedKV {seqlen_k, params.d, params.h, batch_k};  // shape_dK
            } else {
                return typename CollectiveEpilogue::ShapedKV {seqlen_k_rounded * params.d_rounded, params.h_k, batch_k};  // shape_dKaccum
            }
        }(),
        [&] {
            if constexpr (!GQA && !MegaRing) {
                return typename CollectiveEpilogue::StridedKV {params.dk_row_stride, _1{}, params.dk_head_stride, !is_varlen_k ? params.dk_batch_stride : 0};  // stride_dK
            } else {
                return typename CollectiveEpilogue::StridedKV {_1{}, params.d_rounded * seqlen_k_rounded, !is_varlen_k ? params.h_k * params.d_rounded * params.seqlen_k_rounded : 0};  // stride_dKaccum
            }
        }(),
        static_cast<typename CollectiveEpilogue::Element*>(!GQA && !MegaRing ? params.dv_ptr : params.dv_accum_ptr),
        [&] {
            if constexpr (!GQA && !MegaRing) {
                return typename CollectiveEpilogue::ShapedKV {seqlen_k, params.dv, params.h, batch_k};  // shape_dV
            } else {
                return typename CollectiveEpilogue::ShapedKV {seqlen_k_rounded * params.dv_rounded, params.h_k, batch_k};  // shape_dVaccum
            }
        }(),
        [&] {
            if constexpr (!GQA && !MegaRing) {
                return typename CollectiveEpilogue::StridedKV {params.dv_row_stride, _1{}, params.dv_head_stride, !is_varlen_k ? params.dv_batch_stride : 0};  // stride_dV
            } else {
                return typename CollectiveEpilogue::StridedKV {_1{}, params.dv_rounded * seqlen_k_rounded, !is_varlen_k ? params.h_k * params.dv_rounded * params.seqlen_k_rounded : 0};  // stride_dVaccum
            }
        }(),
        params.b,
        params.h,
        params.dk_semaphore,
        params.dv_semaphore,
        params.cu_seqlens_k,
        params.seqused_k,
        MegaRing ? params.dkv_step_stride : 0,
        MegaRing ? params.ring_local_ready : nullptr,
    };

    int num_blocks_n = cutlass::ceil_div(params.seqlen_k, get<1>(TileShape_MNK{}));
    num_blocks_n = cutlass::round_up(num_blocks_n, size<1>(ClusterShape{}));
    flash::TileSchedulerArguments scheduler_args {
        num_blocks_n, params.h, params.b,
        params.h / params.h_k,
        params.seqlen_k,
        params.seqlen_q, params.d, params.dv, sizeof(Element),
        params.cu_seqlens_k, params.seqused_k,
        params.tile_count_semaphore,
        params.ring_world_size,
        params.ring_rank,
        params.num_comp_sm,
        params.half_cu_seqlens
    };

    int device;
    cudaGetDevice(&device);
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    // int smem_size_q = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_q));
    // int smem_size_do = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_do));
    // int smem_size_ds = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_ds));
    // int smem_size_dqacc = [&] {
    //     if constexpr (Arch >= 90) {
    //         return sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_dqacc));
    //     } else {
    //         return 0;
    //     }
    // }();
    // int smem_size_k = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_k));
    // int smem_size_v = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v));
    // int smem_size_lse = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_lse));
    // int smem_size_dpsum = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_dpsum));
    // printf("smem_size = %d, q = %d, k = %d, v = %d, do = %d, ds = %d, dqacc = %d, lse = %d, dpsum = %d\n", smem_size, smem_size_q, smem_size_k, smem_size_v, smem_size_do, smem_size_ds, smem_size_dqacc, smem_size_lse, smem_size_dpsum);
    if constexpr (MegaRing) {
        TORCH_CHECK(remote_k_tensor != nullptr && remote_v_tensor != nullptr &&
                    remote_dk_tensor != nullptr && remote_dv_tensor != nullptr,
                    "mega-ring backward TMA launch requires all remote tensors");
        using Comm = MegaRingBwdCommConfig<AttnKernel, NumDevices>;
        using KernelParams = MegaRingBwdKernelParams<AttnKernel, NumDevices>;
        check_mega_ring_bwd_tma_layout<Comm>();
        auto kernel = mega_ring_flash_attn_bwd_kernel<AttnKernel, NumDevices>;
        int const remote_rows = params.total_k;
        int const dkv_rows_per_step = int(params.dkv_step_stride / Comm::kVecLength);
        TORCH_CHECK(params.h_k * params.d == Comm::kVecLength,
                    "mega-ring backward K/V TMA row must contain KVH * D == ", Comm::kVecLength);
        TORCH_CHECK(params.dkv_step_stride % Comm::kVecLength == 0,
                    "mega-ring backward dKV accumulator must be divisible by the TMA row width");
        int const comm_smem_size = int(sizeof(typename Comm::DShared)) * Comm::kNumChunks + 1024;
        smem_size = smem_size > comm_smem_size ? smem_size : comm_smem_size;
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        auto check_64b_alignment = [] (void const* ptr, char const* name) {
            TORCH_CHECK(reinterpret_cast<uintptr_t>(ptr) % 64 == 0,
                        name, " must be 64-byte aligned for TMA");
        };
        check_64b_alignment(params.k_ptr, "k_ptr");
        check_64b_alignment(params.v_ptr, "v_ptr");
        check_64b_alignment(params.dk_accum_ptr, "dk_steps");
        check_64b_alignment(params.dv_accum_ptr, "dv_steps");
        check_64b_alignment(remote_k_tensor->data_.data_ptr(), "remote_k");
        check_64b_alignment(remote_v_tensor->data_.data_ptr(), "remote_v");
        check_64b_alignment(remote_dk_tensor->data_.data_ptr(), "remote_dk_accum");
        check_64b_alignment(remote_dv_tensor->data_.data_ptr(), "remote_dv_accum");

        KernelParams mega_params{
            kernel_params,
            kittens::py::parallel_tensor_to_pgl<typename Comm::KRemote>(
                *remote_k_tensor, 1, 1, remote_rows, Comm::kVecLength),
            kittens::py::parallel_tensor_to_pgl<typename Comm::KRemote>(
                *remote_v_tensor, 1, 1, remote_rows, Comm::kVecLength),
            kittens::make_gl<typename Comm::KGlobal>(
                reinterpret_cast<uint64_t>(params.k_ptr), 1, 1, remote_rows, Comm::kVecLength),
            kittens::make_gl<typename Comm::KGlobal>(
                reinterpret_cast<uint64_t>(params.v_ptr), 1, 1, remote_rows, Comm::kVecLength),
            kittens::make_gl<typename Comm::DGlobal>(
                reinterpret_cast<uint64_t>(params.dk_accum_ptr), 1, 1,
                params.ring_world_size * dkv_rows_per_step, Comm::kVecLength),
            kittens::make_gl<typename Comm::DGlobal>(
                reinterpret_cast<uint64_t>(params.dv_accum_ptr), 1, 1,
                params.ring_world_size * dkv_rows_per_step, Comm::kVecLength),
            kittens::py::parallel_tensor_to_pgl<typename Comm::DRemote>(
                *remote_dk_tensor, 1, 1, dkv_rows_per_step, Comm::kVecLength),
            kittens::py::parallel_tensor_to_pgl<typename Comm::DRemote>(
                *remote_dv_tensor, 1, 1, dkv_rows_per_step, Comm::kVecLength),
            params.local_total_k,
            params.local_total_k / 2,
            dkv_rows_per_step,
            params.b,
            params.cu_seqlens_k,
            params.half_cu_seqlens,
            params.ring_kv_ready,
            params.ring_local_ready,
            params.ring_expected_ready,
            params.ring_dkv_comm_done,
            params.ring_rank,
            params.ring_world_size,
            params.num_comp_sm,
            params.num_comm_sm,
            {params.remote_dkv_completion[0], params.remote_dkv_completion[1],
             params.remote_dkv_completion[2], params.remote_dkv_completion[3],
             params.remote_dkv_completion[4], params.remote_dkv_completion[5],
             params.remote_dkv_completion[6], params.remote_dkv_completion[7]}
        };
        dim3 mega_grid(params.num_comp_sm + params.num_comm_sm, 1, 1);
        kernel<<<mega_grid, block_dims, smem_size, stream>>>(mega_params);
        CHECK_CUDA_KERNEL_LAUNCH();
        mega_ring_bwd_wait_for_completion<<<1, 1, 0, stream>>>(
            params.ring_completion, params.ring_world_size);
        CHECK_CUDA_KERNEL_LAUNCH();
    } else if constexpr (size(ClusterShape{}) > 1) {
        void const* kernel = (void const*) cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
        CHECK_CUTLASS(cutlass::ClusterLauncher::launch(
            grid_dims, cluster_dims, block_dims, smem_size, stream, kernel, kernel_params, false /*launch_with_pdl*/));
    } else {
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<AttnKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        CHECK_CUTLASS(cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false /*launch_with_pdl*/));
    }

    using PostprocessKernel = flash::FlashAttnBwdPostprocessConvertdQ<TileShape_MK, Element, ElementAccum, ArchTag,
        AttnKernel::CollectiveMainloop::NumMmaThreads,
        typename AttnKernel::CollectiveMainloop::TiledMmadQ,
        AttnKernel::CollectiveMainloop::dQ_swapAB
        >;
    typename PostprocessKernel::Arguments postprocess_args {
        static_cast<ElementAccum const*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, batch_q},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, !is_varlen_q ? params.d_rounded * params.seqlen_q_rounded * params.h : 0}, // stride_dQaccum
        static_cast<Element*>(params.dq_ptr),
        {seqlen_q, params.d, params.h, batch_q},  // shape_dQ
        {params.dq_row_stride, _1{}, params.dq_head_stride, params.dq_batch_stride},  // stride_dQ
        params.scale_softmax,
        params.cu_seqlens_q,
        params.seqused_q
    };
    typename PostprocessKernel::Params postprocess_params = PostprocessKernel::to_underlying_arguments(postprocess_args);
    int num_m_block_postprocess = cute::ceil_div(params.seqlen_q, get<0>(TileShape_MK{}));
    dim3 grid_m_postprocess(num_m_block_postprocess, params.h, params.b);
    int smem_size_postprocess = PostprocessKernel::SharedStorageSize;
    if (smem_size_postprocess >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<PostprocessKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_postprocess));
    }
    CHECK_CUTLASS(cutlass::kernel_launch<PostprocessKernel>(grid_m_postprocess, PostprocessKernel::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_params, false /*launch_with_pdl*/));

    if constexpr (GQA || MegaRing) {
        using TileShape_NK = cute::Shape<Int<kBlockN>, Int<kHeadDim>>;
        using PostprocessKerneldKV = flash::FlashAttnBwdPostprocessConvertdQ<TileShape_NK, Element, ElementAccum, ArchTag,
            AttnKernel::CollectiveEpilogue::NumEpilogueThreads,
            typename AttnKernel::CollectiveMainloop::TiledMmadKV,
            AttnKernel::CollectiveMainloop::dKV_swapAB
            >;
        typename PostprocessKerneldKV::Arguments postprocess_dK_args {
            static_cast<ElementAccum const*>(MegaRing ? params.remote_dk_accum[params.ring_rank] : params.dk_accum_ptr),
            {seqlen_k_rounded * params.d_rounded, params.h_k, batch_k},  // shape_dKaccum
            {_1{}, seqlen_k_rounded * params.d_rounded, !is_varlen_k ? params.d_rounded * params.seqlen_k_rounded * params.h_k : 0},  // stride_dKaccum
            static_cast<Element*>(params.dk_ptr),
            {seqlen_k, params.d, params.h_k, batch_k},  // shape_dK
            {params.dk_row_stride, _1{}, params.dk_head_stride, params.dk_batch_stride},  // stride_dK
            1.f,
            params.cu_seqlens_k,
            params.seqused_k
        };
        typename PostprocessKerneldKV::Params postprocess_dK_params = PostprocessKerneldKV::to_underlying_arguments(postprocess_dK_args);
        typename PostprocessKerneldKV::Arguments postprocess_dV_args {
            static_cast<ElementAccum const*>(MegaRing ? params.remote_dv_accum[params.ring_rank] : params.dv_accum_ptr),
            {seqlen_k_rounded * params.dv_rounded, params.h_k, batch_k},  // shape_dVaccum
            {_1{}, seqlen_k_rounded * params.dv_rounded, !is_varlen_k ? params.dv_rounded * params.seqlen_k_rounded * params.h_k : 0},  // stride_dVaccum
            static_cast<Element*>(params.dv_ptr),
            {seqlen_k, params.dv, params.h_k, batch_k},  // shape_dV
            {params.dv_row_stride, _1{}, params.dv_head_stride, params.dv_batch_stride},  // stride_dV
            1.f,
            params.cu_seqlens_k,
            params.seqused_k
        };
        typename PostprocessKerneldKV::Params postprocess_dV_params = PostprocessKerneldKV::to_underlying_arguments(postprocess_dV_args);
        int num_n_block_postprocess = cute::ceil_div(params.seqlen_k, get<0>(TileShape_NK{}));
        dim3 grid_n_postprocess(num_n_block_postprocess, params.h_k, params.b);
        int smem_size_postprocess = PostprocessKerneldKV::SharedStorageSize;
        if (smem_size_postprocess >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<PostprocessKerneldKV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_postprocess));
        }
        CHECK_CUTLASS(cutlass::kernel_launch<PostprocessKerneldKV>(grid_n_postprocess, PostprocessKerneldKV::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_dK_params, false /*launch_with_pdl*/));
        CHECK_CUTLASS(cutlass::kernel_launch<PostprocessKerneldKV>(grid_n_postprocess, PostprocessKerneldKV::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_dV_params, false /*launch_with_pdl*/));
    }

}

template <bool IsCausal>
void run_min_fa3_bwd_sm90(Flash_bwd_params& params, cudaStream_t stream) {
    using Config = BwdConfig<IsCausal>;
    BOOL_SWITCH(params.cu_seqlens_q != nullptr || params.cu_seqlens_k != nullptr, Varlen, [&] {
        BOOL_SWITCH(params.h != params.h_k, GQA, [&] {
            BOOL_SWITCH(params.deterministic, Deterministic, [&] {
                run_flash_bwd<
                    90,
                    kHeadDim,
                    Config::kBlockM,
                    kBlockN,
                    Element,
                    IsCausal,
                    false,
                    false,
                    Varlen,
                    Deterministic,
                    GQA,
                    kStagesdO,
                    kStagesdS,
                    kSdPSwapAB,
                    kdKVSwapAB,
                    Config::kdQSwapAB,
                    kNumMmaWarpGroups,
                    kAtomLayoutMSdP,
                    kAtomLayoutNdKV,
                    kAtomLayoutMdQ,
                    kVInRegs>(params, stream);
            });
        });
    });
}

extern template void run_min_fa3_bwd_sm90<false>(
    Flash_bwd_params& params,
    cudaStream_t stream);
extern template void run_min_fa3_bwd_sm90<true>(
    Flash_bwd_params& params,
    cudaStream_t stream);

}  // namespace min_fa3_backward
