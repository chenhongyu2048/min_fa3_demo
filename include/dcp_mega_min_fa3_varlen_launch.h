// DCP mega wrapper copied and trimmed from include/min_fa3_varlen_launch.h and
// include/mega_ring_min_fa3_varlen_ring_launch.h. DCP_MEGA keeps the copied
// FA3 mainloop/epilogue and adds IPC Q work plus state publication/combination.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <torch/extension.h>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "kittens.cuh"

#include "dcp_mega_min_fa3_kernel.h"
#include "dcp_mega_min_fa3_varlen_params.h"
#include "dcp_mega_min_fa3_varlen_scheduler.h"
#include "hopper_compat/cuda_check.h"
#include "mega_ring_semaphore.cuh"
#include "min_fa3_epilogue.h"
#include "min_fa3_kernel.h"
#include "min_fa3_mainloop.h"
#include "min_fa3_varlen_traits.h"

namespace min_fa3_varlen_demo::dcp_mega::detail {

using namespace cute;

enum QueueStateIndex : int {
    kAttentionDynamicCounter = 0,
    kHistoryCombineCounter = 1,
    kFinalCounter = 2,
    kQAllGatherPhaseCounter = 3,
    kAttentionPhaseCounter = 4,
    kHistoryCombinePhaseCounter = 5,
    kReceivePhaseCounter = 6,
    kFinalCombinePhaseCounter = 7,
    kKernelPhaseCounter = 8,
    kQueueStateCount = 9,
};

CUTLASS_DEVICE uint64_t read_globaltimer() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

CUTLASS_DEVICE void record_phase_completion(
    int32_t* queue_state,
    uint64_t* phase_timestamps,
    int counter_index,
    int timestamp_index,
    int expected_ctas) {
    if (phase_timestamps != nullptr && threadIdx.x == 0) {
        int const previous = atomicAdd(queue_state + counter_index, 1);
        if (previous == expected_ctas - 1) {
            phase_timestamps[timestamp_index] = read_globaltimer();
        }
    }
}

CUTLASS_DEVICE void record_fused_history_publish_completion(
    int32_t* queue_state,
    uint64_t* phase_timestamps,
    int expected_ctas) {
    if (phase_timestamps != nullptr && threadIdx.x == 0) {
        int const previous = atomicAdd(
            queue_state + kHistoryCombinePhaseCounter, 1);
        if (previous == expected_ctas - 1) {
            uint64_t const timestamp = read_globaltimer();
            phase_timestamps[kHistoryCombineDoneTimestamp] = timestamp;
            phase_timestamps[kPublishDoneTimestamp] = timestamp;
        }
    }
}

CUTLASS_DEVICE void store_release_system_s32(
    int32_t* address,
    int32_t value) {
    asm volatile("{st.release.sys.global.s32 [%0], %1;}"
                 :: "l"(address), "r"(value) : "memory");
}

CUTLASS_DEVICE int32_t load_acquire_system_s32(int32_t const* address) {
    int32_t value;
    asm volatile("{ld.acquire.sys.global.s32 %0, [%1];}"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

CUTLASS_DEVICE int32_t atomic_add_acq_rel_device_s32(
    int32_t* address,
    int32_t value) {
    int32_t old;
    asm volatile("{atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;}"
                 : "=r"(old) : "l"(address), "r"(value) : "memory");
    return old;
}

CUTLASS_DEVICE void fence_acq_rel_system() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

CUTLASS_DEVICE void cp_async_float4(float* smem, float const* gmem) {
    uint32_t const smem_address
        = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(smem_address), "l"(gmem) : "memory");
}

CUTLASS_DEVICE void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int PendingGroups>
CUTLASS_DEVICE void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;"
                 :: "n"(PendingGroups) : "memory");
}

template <bool Split, int BlockN, int DCPSize, int CommHeads,
          int CombineMaxSplits>
struct DCPMegaKernelConfig {
    static_assert(BlockN == 128 || BlockN == 176);
    static_assert(DCPSize == 2 || DCPSize == 4 || DCPSize == 8);
    static_assert(CommHeads == 4 || CommHeads == 8);
    static_assert(
        (!Split && CombineMaxSplits == 1)
        || (Split && (CombineMaxSplits == 32
                      || CombineMaxSplits == 64
                      || CombineMaxSplits == 128)));
    static constexpr int kDCPSize = DCPSize;
    static constexpr int kCommHeads = CommHeads;
    static constexpr int kCombineMaxSplits = CombineMaxSplits;
    static constexpr int kCombineStages = 4;
    static constexpr int kHistoryVectorsPerTask = Split ? 1 : CommHeads;
    static constexpr int kReceiveScanWindow = 8;
    static_assert(kHistoryVectorsPerTask >= 1);
    static_assert(kHistoryVectorsPerTask <= CommHeads);
    static_assert(CommHeads % kHistoryVectorsPerTask == 0);
    static_assert(kReceiveScanWindow > 0);

    using ArchTag = cutlass::arch::Sm90;
    using TileShapeMNK = Shape<_128, Int<BlockN>, _128>;
    using TileShapeMNKPV = Shape<_128, _128, Int<BlockN>>;
    using ClusterShape = Shape<_1, _1, _1>;

    template <bool IsCausal>
    using Mainloop = flash::CollectiveMainloopFwdSm90<
        kStages,
        ClusterShape,
        TileShapeMNK,
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
        true,
        true,
        true,
        Split,
        kVColMajor,
        !IsCausal>;

    template <bool IsCausal>
    using Epilogue = flash::CollectiveEpilogueFwd<
        TileShapeMNKPV,
        ClusterShape,
        ElementOut,
        ArchTag,
        Mainloop<IsCausal>::NumMmaThreads,
        true,
        true,
        Split,
        false,
        false>;

    using Scheduler = flash::DCPMegaVarlenTileScheduler<
        128,
        BlockN,
        Mainloop<true>::NumMmaThreads,
        Mainloop<true>::NumProducerThreads,
        Split,
        DCPSize,
        CommHeads>;

    template <bool IsCausal>
    using Kernel = flash::enable_sm90<flash::FlashAttnFwdSm90<
        Mainloop<IsCausal>, Epilogue<IsCausal>, Scheduler>>;

    using ChunkKernel = Kernel<true>;
    using HistoryKernel = Kernel<false>;
    using AttentionKernel = flash::DCPMegaFlashAttnFwdSm90<
        Mainloop<true>, Mainloop<false>, Epilogue<true>, Scheduler>;
    static_assert(std::is_same_v<Epilogue<true>, Epilogue<false>>);
    static_assert(ChunkKernel::MaxThreadsPerBlock
                  == HistoryKernel::MaxThreadsPerBlock);
    static_assert(AttentionKernel::MaxThreadsPerBlock
                  == ChunkKernel::MaxThreadsPerBlock);
    static constexpr int MaxThreadsPerBlock
        = AttentionKernel::MaxThreadsPerBlock;
    static constexpr int kNumWarps = MaxThreadsPerBlock / cutlass::NumThreadsPerWarp;
    static constexpr int kNumCommChunks = kNumWarps / 2;
    using CommTile = kittens::st_bf<16, CommHeads * 128>;
    using QGlobal = kittens::gl<
        kittens::bf16, 1, -1, -1, CommHeads * 128,
        kittens::tma::descriptor<CommTile, kittens::dim::DEPTH>>;
    using QRemote = kittens::pgl<QGlobal, DCPSize, false>;
    using HistoryGlobal = kittens::gl<
        kittens::bf16, 1, DCPSize, -1, CommHeads * 128, CommTile>;
    using HistoryRemote = kittens::pgl<HistoryGlobal, DCPSize, false>;

    struct alignas(128) HelperSharedStorage {
        int work_id;
        int receive_task_ids[kNumCommChunks];
        float source_lse[kNumCommChunks][16 * CommHeads];
        alignas(128) CommTile comm_tiles[kNumCommChunks];
        alignas(16) kittens::semaphore arrived[kNumCommChunks];
        alignas(16) kittens::semaphore finished[kNumCommChunks];
        alignas(16) kittens::semaphore work_ready[kNumCommChunks];
    };

    static constexpr int SharedStorageSize =
        AttentionKernel::SharedStorageSize > int(sizeof(HelperSharedStorage))
        ? AttentionKernel::SharedStorageSize : int(sizeof(HelperSharedStorage));

    static_assert(MaxThreadsPerBlock % cutlass::NumThreadsPerWarp == 0);
    static_assert(kNumCommChunks >= 2);
    static_assert(kNumCommChunks == 6);
    static_assert(sizeof(CommTile)
                  == 16 * CommHeads * 128 * sizeof(kittens::bf16));
    static_assert(sizeof(CommTile) % 128 == 0);
    static_assert(
        sizeof(CommTile) * kNumCommChunks
        >= kNumWarps * kCombineStages * 128 * sizeof(float));
    static_assert(alignof(HelperSharedStorage) == 128);
    static_assert(
        alignof(typename ChunkKernel::SharedStorage)
        == alignof(typename HistoryKernel::SharedStorage));
    static_assert(
        alignof(typename ChunkKernel::SharedStorage)
        >= alignof(HelperSharedStorage));
    static_assert(AttentionKernel::SharedStorageSize
                  == ChunkKernel::SharedStorageSize);
    static_assert(AttentionKernel::SharedStorageSize
                  == HistoryKernel::SharedStorageSize);
    static_assert(SharedStorageSize <= 227 * 1024,
                  "DCP mega shared storage exceeds the Hopper opt-in limit");

    struct alignas(128) KernelParams {
        using ChunkParams = typename ChunkKernel::Params;
        using HistoryParams = typename HistoryKernel::Params;

        ChunkParams chunk;
        HistoryParams history;
        QRemote q_remote;
        QGlobal q_group;
        HistoryRemote history_send_remote;
        HistoryGlobal history_send_local;
        HistoryGlobal history_receive_local;
        Element* history_receive_o = nullptr;
        float* history_receive_lse = nullptr;
        Element* final_o = nullptr;
        float* final_lse = nullptr;
        int64_t final_lse_head_stride = 0;
        AttentionWorkDesc const* attention = nullptr;
        QTaskDesc const* q_tasks = nullptr;
        PublishWorkDesc const* publish = nullptr;
        HistoryCombineWorkDesc const* history_combine = nullptr;
        FinalWorkDesc const* final = nullptr;
        int32_t const* q_dependencies = nullptr;
        int32_t const* publish_dependencies = nullptr;
        int32_t const* final_dependencies = nullptr;
        int32_t const* chunk_sequence_splits = nullptr;
        int32_t const* history_sequence_splits = nullptr;
        int32_t* q_ready = nullptr;
        int32_t* attention_done = nullptr;
        int32_t* publish_ready = nullptr;
        int32_t* receive_ready = nullptr;
        int32_t* queue_state = nullptr;
        uint64_t* phase_timestamps = nullptr;
        int32_t const* graph_post_phase = nullptr;
        int const* cu_seqlens_q = nullptr;
        int total_q = 0;
        int total_vectors = 0;
        int batch_size = 0;
        int hq_local = 0;
        int q_row_stride = 0;
        int q_head_stride = 0;
        int q_group_row_stride = 0;
        int history_send_rank_stride = 0;
        int signal_rank_stride = 0;
        int num_q_tasks = 0;
        int chunk_attention_count = 0;
        int history_attention_count = 0;
        int publish_count = 0;
        int history_combine_count = 0;
        int final_count = 0;
        int effective_num_splits = 0;
        int dcp_rank = 0;
        int tile_ready_phase = 0;
        int num_sms = 0;
        int num_comm_sm = 0;
        bool return_lse = false;

        float* history_send_lse_remote[DCPSize]{};
        int32_t* tile_ready_remote[DCPSize]{};
        Element* history_send_o = nullptr;
        float* history_send_lse = nullptr;
        int token_block_capacity = 0;

        KernelParams(
            ChunkParams const& chunk_,
            HistoryParams const& history_,
            QRemote const& q_remote_,
            QGlobal const& q_group_,
            HistoryRemote const& history_send_remote_,
            HistoryGlobal const& history_send_local_,
            HistoryGlobal const& history_receive_local_)
            : chunk(chunk_), history(history_),
              q_remote(q_remote_), q_group(q_group_),
              history_send_remote(history_send_remote_),
              history_send_local(history_send_local_),
              history_receive_local(history_receive_local_) {}
    };

    static_assert(alignof(QRemote) >= 64);
    static_assert(alignof(QGlobal) >= 64);
    static_assert(alignof(HistoryRemote) >= 64);
    static_assert(alignof(HistoryGlobal) >= 64);
    static_assert(alignof(KernelParams) >= 128);
    static_assert(offsetof(KernelParams, q_remote) % 64 == 0);
    static_assert(offsetof(KernelParams, q_group) % 64 == 0);
    static_assert(offsetof(KernelParams, history_send_remote) % 64 == 0);
    static_assert(offsetof(KernelParams, history_send_local) % 64 == 0);
    static_assert(offsetof(KernelParams, history_receive_local) % 64 == 0);
};

template <typename Config>
CUTLASS_DEVICE void run_q_allgather(
    typename Config::KernelParams const& params,
    typename Config::HelperSharedStorage& shared) {
    static_assert(sizeof(typename Config::CommTile)
                  == 16 * Config::kCommHeads * 128 * sizeof(kittens::bf16));
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int chunk = 0; chunk < Config::kNumCommChunks; ++chunk) {
            kittens::init_semaphore(shared.arrived[chunk], 0, 1);
            kittens::init_semaphore(shared.finished[chunk], 0, 1);
        }
    }
    __syncthreads();

    int const warp_id = int(threadIdx.x) / cutlass::NumThreadsPerWarp;
    uint32_t phasebits = 0xFFFF0000;
    if (warp_id < Config::kNumCommChunks && kittens::laneid() == 0) {
        int const chunk = warp_id;
        for (int task_id = Config::kNumCommChunks * int(blockIdx.x) + chunk;
             task_id < params.num_q_tasks;
             task_id += Config::kNumCommChunks * params.num_comm_sm) {
            QTaskDesc const task = params.q_tasks[task_id];
            kittens::wait(
                shared.finished[chunk], kittens::get_phasebit<1>(phasebits, 0));
            kittens::update_phasebit<1>(phasebits, 0);
            kittens::tma::expect_bytes(
                shared.arrived[chunk], sizeof(typename Config::CommTile));
            kittens::tma::load_async<
                kittens::dim::DEPTH, kittens::cache_policy::NORMAL>(
                shared.comm_tiles[chunk], params.q_remote[task.src_rank],
                {0, task.token_begin / 16, 0, 0},
                shared.arrived[chunk]);
        }
    } else if (warp_id < 2 * Config::kNumCommChunks
               && kittens::laneid() == 0) {
        int const chunk = warp_id - Config::kNumCommChunks;
        for (int task_id = Config::kNumCommChunks * int(blockIdx.x) + chunk;
             task_id < params.num_q_tasks;
             task_id += Config::kNumCommChunks * params.num_comm_sm) {
            QTaskDesc const task = params.q_tasks[task_id];
            kittens::wait(
                shared.arrived[chunk], kittens::get_phasebit<0>(phasebits, 0));
            kittens::update_phasebit<0>(phasebits, 0);
            kittens::tma::store_async<
                kittens::dim::DEPTH, kittens::cache_policy::NORMAL>(
                params.q_group, shared.comm_tiles[chunk],
                {0, task.token_begin / 16, task.src_rank, 0});
            kittens::tma::store_async_read_wait();
            kittens::arrive(shared.finished[chunk]);
            kittens::tma::store_async_wait();
            asm volatile("fence.proxy.async.global;" ::: "memory");
            min_fa3_varlen_demo::mega_ring::signal_release(
                params.q_ready + task.token_begin / 16, 1);
        }
    }
    __syncthreads();
}

CUTLASS_DEVICE int batch_for_token(
    int token,
    int const* cu_seqlens,
    int batch_size) {
    int lo = 0;
    int hi = batch_size;
    while (lo + 1 < hi) {
        int const mid = (lo + hi) / 2;
        if (cu_seqlens[mid] <= token) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

template <typename Config>
CUTLASS_DEVICE float warp_allreduce_max(float value) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
    }
    return value;
}

template <typename Config>
CUTLASS_DEVICE float warp_allreduce_sum(float value) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        value += __shfl_xor_sync(0xffffffffu, value, mask);
    }
    return value;
}

template <typename Config>
CUTLASS_DEVICE void complete_history_combine_task(
    typename Config::KernelParams const& params,
    HistoryCombineWorkDesc const& work) {
    int const lane = kittens::laneid();
    __syncwarp();
    if (lane == 0) {
        PublishWorkDesc const publish = params.publish[work.publish_id];
        int const previous = atomic_add_acq_rel_device_s32(
            params.publish_ready + work.publish_id, 1);
        if (previous == publish.combine_task_count - 1
            && publish.dst_rank != params.dcp_rank) {
            // The acq_rel RMW observes every prior warp's release in this tile.
            // Promote the generic stores before publishing to the peer GPU.
            fence_acq_rel_system();
            int const token_block
                = publish.vector_begin / (16 * Config::kCommHeads);
            store_release_system_s32(
                params.tile_ready_remote[publish.dst_rank]
                    + params.dcp_rank * params.token_block_capacity
                    + token_block,
                params.graph_post_phase != nullptr
                    ? *params.graph_post_phase - 1
                    : params.tile_ready_phase);
        }
    }
}

template <typename Config>
CUTLASS_DEVICE void copy_single_history_split(
    typename Config::KernelParams const& params,
    PublishWorkDesc const& publish,
    int vector,
    int history_head) {
    int const lane = kittens::laneid();
    int const token = vector / params.hq_local;
    int const dim_begin = lane * 4;
    Element const* source = params.history.epilogue.ptr_O
        + token * params.q_group_row_stride
        + history_head * 128 + dim_begin;
    Element* destination = params.history_send_o
        + publish.dst_rank * params.history_send_rank_stride
        + vector * 128 + dim_begin;
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        destination[item] = source[item];
    }
    if (lane == 0) {
        params.history_send_lse[
            publish.dst_rank * params.signal_rank_stride + vector]
            = params.history.epilogue.ptr_LSE[
                history_head
                    * get<1>(params.history.epilogue.stride_LSE)
                + token];
    }
}

template <typename Config>
CUTLASS_DEVICE void combine_history_splits(
    typename Config::KernelParams const& params,
    typename Config::HelperSharedStorage& shared,
    HistoryCombineWorkDesc const& work,
    PublishWorkDesc const& publish,
    int vector,
    int history_head) {
    static_assert(Config::HistoryKernel::Split);
    constexpr int kWeightsPerLane = Config::kCombineMaxSplits / 32;
    int const lane = kittens::laneid();
    int const warp_id = int(threadIdx.x) / cutlass::NumThreadsPerWarp;
    int const token = vector / params.hq_local;
    int const splits = work.actual_splits;
    float weights[kWeightsPerLane];
    float max_lse = -INFINITY;

    #pragma unroll
    for (int group = 0; group < kWeightsPerLane; ++group) {
        int const split = lane + group * cutlass::NumThreadsPerWarp;
        float const lse = split < splits
            ? params.history.epilogue.ptr_LSE_partial[
                split * get<3>(params.history.epilogue.stride_LSE_partial)
                + history_head
                    * get<1>(params.history.epilogue.stride_LSE_partial)
                + token]
            : -INFINITY;
        weights[group] = lse;
        max_lse = fmaxf(max_lse, lse);
    }
    max_lse = warp_allreduce_max<Config>(max_lse);
    float const finite_max = isfinite(max_lse) ? max_lse : 0.0f;
    float denominator = 0.0f;
    #pragma unroll
    for (int group = 0; group < kWeightsPerLane; ++group) {
        float const lse = weights[group];
        weights[group] = isfinite(lse) ? expf(lse - finite_max) : 0.0f;
        denominator += weights[group];
    }
    denominator = warp_allreduce_sum<Config>(denominator);
    float const inv_denominator
        = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    #pragma unroll
    for (int group = 0; group < kWeightsPerLane; ++group) {
        weights[group] *= inv_denominator;
    }

    float* warp_stages = reinterpret_cast<float*>(&shared.comm_tiles[0])
        + warp_id * Config::kCombineStages * 128;
    int const dim_begin = lane * 4;
    auto partial_o = [&](int split) {
        return params.history.epilogue.ptr_O_partial
            + split * get<4>(params.history.epilogue.stride_O_partial)
            + history_head
                * get<2>(params.history.epilogue.stride_O_partial)
            + token * get<0>(params.history.epilogue.stride_O_partial)
            + dim_begin;
    };
    auto issue_load = [&](int split, int stage) {
        cp_async_float4(
            warp_stages + stage * 128 + dim_begin,
            partial_o(split));
    };

    #pragma unroll
    for (int split = 0; split < Config::kCombineStages - 1; ++split) {
        if (split < splits) {
            issue_load(split, split);
        }
        cp_async_commit_group();
    }

    float accum[4]{};
    int stage_load = Config::kCombineStages - 1;
    int stage_compute = 0;
    #pragma unroll 4
    for (int split = 0; split < splits; ++split) {
        if (split + Config::kCombineStages - 1 < splits) {
            issue_load(split + Config::kCombineStages - 1, stage_load);
        }
        cp_async_commit_group();
        stage_load = stage_load < Config::kCombineStages - 1
            ? stage_load + 1 : 0;
        cp_async_wait_group<Config::kCombineStages - 1>();

        float source_weight;
        int const weight_group = split / cutlass::NumThreadsPerWarp;
        switch (weight_group) {
            case 0: source_weight = weights[0]; break;
            case 1:
                if constexpr (kWeightsPerLane >= 2) {
                    source_weight = weights[1];
                } else {
                    source_weight = 0.0f;
                }
                break;
            case 2:
                if constexpr (kWeightsPerLane >= 3) {
                    source_weight = weights[2];
                } else {
                    source_weight = 0.0f;
                }
                break;
            default:
                if constexpr (kWeightsPerLane >= 4) {
                    source_weight = weights[3];
                } else {
                    source_weight = 0.0f;
                }
                break;
        }
        float const weight = __shfl_sync(
            0xffffffffu, source_weight,
            split % cutlass::NumThreadsPerWarp);
        float const* values
            = warp_stages + stage_compute * 128 + dim_begin;
        // FA3 leaves O_partial unspecified for invalid (-inf LSE) splits.
        // The broadcast weight is warp-uniform, so this branch is lightweight.
        if (weight > 0.0f) {
            #pragma unroll
            for (int item = 0; item < 4; ++item) {
                accum[item] += weight * values[item];
            }
        }
        stage_compute = stage_compute < Config::kCombineStages - 1
            ? stage_compute + 1 : 0;
    }

    Element* destination = params.history_send_o
        + publish.dst_rank * params.history_send_rank_stride
        + vector * 128 + dim_begin;
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
        destination[item] = Element(accum[item]);
    }
    if (lane == 0) {
        params.history_send_lse[
            publish.dst_rank * params.signal_rank_stride + vector]
            = denominator > 0.0f
            ? max_lse + logf(denominator) : -INFINITY;
    }
}

template <typename Config>
CUTLASS_DEVICE void run_history_combine(
    typename Config::KernelParams const& params,
    typename Config::HelperSharedStorage& shared) {
    int const lane = kittens::laneid();
    while (true) {
        int ticket = -1;
        if (lane == 0) {
            ticket = atomicAdd(
                params.queue_state + kHistoryCombineCounter, 1);
        }
        ticket = __shfl_sync(0xffffffffu, ticket, 0);
        if (ticket >= params.history_combine_count) {
            break;
        }
        HistoryCombineWorkDesc const work = params.history_combine[ticket];
        if (lane == 0) {
            for (int dep = 0; dep < work.dependency_count; ++dep) {
                int const completion_id = params.publish_dependencies[
                    work.dependency_begin + dep];
                min_fa3_varlen_demo::mega_ring::wait_until_at_least_acquire(
                    params.attention_done + completion_id, 1);
            }
        }
        __syncwarp();
        PublishWorkDesc const publish = params.publish[work.publish_id];
        #pragma unroll
        for (int vector_in_task = 0;
             vector_in_task < Config::kHistoryVectorsPerTask;
             ++vector_in_task) {
            if (vector_in_task < work.valid_vectors) {
                int const vector = work.vector_begin + vector_in_task;
                int const local_head = vector % params.hq_local;
                int const history_head
                    = publish.dst_rank * params.hq_local + local_head;
                if constexpr (Config::HistoryKernel::Split) {
                    if (work.actual_splits > 1) {
                        combine_history_splits<Config>(
                            params, shared, work, publish,
                            vector, history_head);
                    } else {
                        copy_single_history_split<Config>(
                            params, publish, vector, history_head);
                    }
                } else {
                    copy_single_history_split<Config>(
                        params, publish, vector, history_head);
                }
            }
        }
        complete_history_combine_task<Config>(params, work);
    }
}

template <typename Config>
CUTLASS_DEVICE void run_communication_post_q(
    typename Config::KernelParams const& params,
    typename Config::HelperSharedStorage& shared) {
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int chunk = 0; chunk < Config::kNumCommChunks; ++chunk) {
            kittens::init_semaphore(shared.arrived[chunk], 0, 1);
            kittens::init_semaphore(shared.finished[chunk], 0, 1);
            kittens::init_semaphore(shared.work_ready[chunk], 0, 1);
            shared.receive_task_ids[chunk] = -1;
        }
    }
    __syncthreads();

    int const warp_id = int(threadIdx.x) / cutlass::NumThreadsPerWarp;
    int const lane = kittens::laneid();
    constexpr int kReceiveSources = Config::kDCPSize - 1;
    int const total_receive_tasks = params.final_count * kReceiveSources;
    int const receive_stride = Config::kNumCommChunks * params.num_comm_sm;
    int const expected_phase = params.graph_post_phase != nullptr
        ? *params.graph_post_phase - 1 : params.tile_ready_phase;
    int const chunk = warp_id < Config::kNumCommChunks
        ? warp_id : warp_id - Config::kNumCommChunks;
    int const base_task
        = Config::kNumCommChunks * int(blockIdx.x) + chunk;
    int const task_count = base_task < total_receive_tasks
        ? 1 + (total_receive_tasks - 1 - base_task) / receive_stride
        : 0;

    if (warp_id < Config::kNumCommChunks) {
        int cursor = 0;
        int finished_phase = 1;
        for (int issued = 0; issued < task_count; ++issued) {
            if (lane == 0) {
                kittens::wait(shared.finished[chunk], finished_phase);
                finished_phase ^= 1;
            }
            __syncwarp();

            int selected_task = -1;
            if (lane == 0) {
                if (task_count == 1) {
                    selected_task = base_task;
                    int const final_id = selected_task / kReceiveSources;
                    int const source_ordinal
                        = selected_task - final_id * kReceiveSources;
                    int const source = source_ordinal
                        + (source_ordinal >= params.dcp_rank);
                    FinalWorkDesc const work = params.final[final_id];
                    int const token_block
                        = work.vector_begin / (16 * Config::kCommHeads);
                    while (load_acquire_system_s32(
                               params.tile_ready_remote[params.dcp_rank]
                                   + source * params.token_block_capacity
                                   + token_block) < expected_phase) {
                        __nanosleep(64);
                    }
                } else {
                    while (selected_task < 0) {
                        int ordinal = cursor;
                        #pragma unroll
                        for (int probe = 0;
                             probe < Config::kReceiveScanWindow;
                             ++probe) {
                            if (probe >= task_count) {
                                break;
                            }
                            int const task_id
                                = base_task + ordinal * receive_stride;
                            int next_ordinal = ordinal + 1;
                            if (next_ordinal == task_count) {
                                next_ordinal = 0;
                            }
                            ordinal = next_ordinal;
                            if (min_fa3_varlen_demo::mega_ring::load_acquire(
                                    params.receive_ready + task_id) >= 1) {
                                continue;
                            }
                            int const final_id = task_id / kReceiveSources;
                            int const source_ordinal
                                = task_id - final_id * kReceiveSources;
                            int const source = source_ordinal
                                + (source_ordinal >= params.dcp_rank);
                            FinalWorkDesc const work = params.final[final_id];
                            int const token_block = work.vector_begin
                                / (16 * Config::kCommHeads);
                            if (load_acquire_system_s32(
                                    params.tile_ready_remote[params.dcp_rank]
                                        + source * params.token_block_capacity
                                        + token_block) >= expected_phase) {
                                selected_task = task_id;
                                cursor = next_ordinal;
                                break;
                            }
                        }
                        if (selected_task < 0) {
                            cursor = ordinal;
                            __nanosleep(64);
                        }
                    }
                }
            }
            selected_task = __shfl_sync(0xffffffffu, selected_task, 0);
            __syncwarp();

            int const final_id = selected_task / kReceiveSources;
            int const source_ordinal
                = selected_task - final_id * kReceiveSources;
            int const source
                = source_ordinal + (source_ordinal >= params.dcp_rank);
            FinalWorkDesc const work = params.final[final_id];
            if (lane == 0) {
                shared.receive_task_ids[chunk] = selected_task;
            }
            for (int vector_in_task = lane;
                 vector_in_task < work.valid_vectors;
                 vector_in_task += cutlass::NumThreadsPerWarp) {
                int const vector = work.vector_begin + vector_in_task;
                shared.source_lse[chunk][vector_in_task]
                    = params.history_send_lse_remote[source][
                        params.dcp_rank * params.signal_rank_stride + vector];
            }
            __syncwarp();
            if (lane == 0) {
                kittens::arrive(shared.work_ready[chunk]);
                asm volatile("fence.proxy.async.global;" ::: "memory");
                kittens::tma::expect_bytes(
                    shared.arrived[chunk], sizeof(typename Config::CommTile));
                kittens::tma::load_async(
                    shared.comm_tiles[chunk],
                    params.history_send_remote[source],
                    {0, params.dcp_rank,
                     work.vector_begin / (16 * Config::kCommHeads), 0},
                    shared.arrived[chunk]);
            }
        }
    } else {
        int work_ready_phase = 0;
        int arrived_phase = 0;
        for (int received = 0; received < task_count; ++received) {
            int task_id = -1;
            if (lane == 0) {
                kittens::wait(shared.work_ready[chunk], work_ready_phase);
                work_ready_phase ^= 1;
                kittens::wait(shared.arrived[chunk], arrived_phase);
                arrived_phase ^= 1;
                asm volatile("" ::: "memory");
                task_id = shared.receive_task_ids[chunk];
            }
            task_id = __shfl_sync(0xffffffffu, task_id, 0);
            __syncwarp();

            int const final_id = task_id / kReceiveSources;
            int const source_ordinal = task_id - final_id * kReceiveSources;
            int const source
                = source_ordinal + (source_ordinal >= params.dcp_rank);
            FinalWorkDesc const work = params.final[final_id];
            for (int vector_in_task = lane;
                 vector_in_task < work.valid_vectors;
                 vector_in_task += cutlass::NumThreadsPerWarp) {
                int const vector = work.vector_begin + vector_in_task;
                params.history_receive_lse[
                    source * params.signal_rank_stride + vector]
                    = shared.source_lse[chunk][vector_in_task];
            }
            __syncwarp();
            if (lane == 0) {
                kittens::tma::store_async(
                    params.history_receive_local,
                    shared.comm_tiles[chunk],
                    {0, source,
                     work.vector_begin / (16 * Config::kCommHeads), 0});
                kittens::tma::store_async_read_wait();
                kittens::tma::store_async_wait();
                asm volatile("fence.proxy.async.global;" ::: "memory");
                min_fa3_varlen_demo::mega_ring::store_release(
                    params.receive_ready + task_id, 1);
                kittens::arrive(shared.finished[chunk]);
            }
        }
    }
    __syncthreads();
    record_phase_completion(
        params.queue_state, params.phase_timestamps,
        kReceivePhaseCounter, kReceiveDoneTimestamp,
        params.num_comm_sm);
}

template <typename Config>
CUTLASS_DEVICE void run_final_combine(
    typename Config::KernelParams const& params,
    typename Config::HelperSharedStorage& shared) {
    while (true) {
        if (threadIdx.x == 0) {
            shared.work_id = atomicAdd(params.queue_state + kFinalCounter, 1);
        }
        __syncthreads();
        if (shared.work_id >= params.final_count) {
            break;
        }
        FinalWorkDesc const work = params.final[shared.work_id];
        if (threadIdx.x == 0) {
            for (int dep = 0; dep < work.dependency_count; ++dep) {
                int const completion_id = params.final_dependencies[
                    work.dependency_begin + dep];
                min_fa3_varlen_demo::mega_ring::wait_until_at_least_acquire(
                    params.attention_done + completion_id, 1);
            }
            int const publish_id
                = params.dcp_rank * params.final_count + shared.work_id;
            PublishWorkDesc const publish = params.publish[publish_id];
            min_fa3_varlen_demo::mega_ring::wait_until_at_least_acquire(
                params.publish_ready + publish_id,
                publish.combine_task_count);
            #pragma unroll
            for (int source_ordinal = 0;
                 source_ordinal < Config::kDCPSize - 1;
                 ++source_ordinal) {
                int const receive_id
                    = shared.work_id * (Config::kDCPSize - 1)
                    + source_ordinal;
                min_fa3_varlen_demo::mega_ring::wait_until_at_least_acquire(
                    params.receive_ready + receive_id, 1);
            }
        }
        __syncthreads();
        if (threadIdx.x < 256) {
            constexpr int kVectorsPerWave = 16;
            constexpr int kWaves = Config::kCommHeads;
            int const vector_in_wave = int(threadIdx.x) / 16;
            int const lane = int(threadIdx.x) % 16;
            unsigned const subgroup = (int(threadIdx.x) % 32) / 16;
            unsigned const mask = 0xffffu << (subgroup * 16);
            #pragma unroll
            for (int wave = 0; wave < kWaves; ++wave) {
                int const vector_in_task
                    = wave * kVectorsPerWave + vector_in_wave;
                if (vector_in_task < work.valid_vectors) {
                    int const vector = work.vector_begin + vector_in_task;
                    int const token = vector / params.hq_local;
                    int const local_head = vector - token * params.hq_local;
                    int const batch = batch_for_token(
                        token, params.cu_seqlens_q, params.batch_size);
                    float max_lse = -INFINITY;
                    float denominator = 0.0f;
                    float accum[8]{};

                    #pragma unroll
                    for (int source = 0; source < Config::kDCPSize; ++source) {
                        bool const self_source = source == params.dcp_rank;
                        if (lane == 0) {
                            shared.source_lse[0][vector_in_task] = self_source
                                ? params.history_send_lse[
                                    params.dcp_rank * params.signal_rank_stride
                                    + vector]
                                : params.history_receive_lse[
                                    source * params.signal_rank_stride + vector];
                        }
                        __syncwarp(mask);
                        float const state_lse
                            = shared.source_lse[0][vector_in_task];
                        float const next_max
                            = max_lse > state_lse ? max_lse : state_lse;
                        float const previous_scale
                            = isfinite(max_lse) && isfinite(state_lse)
                            ? expf(max_lse - next_max)
                            : (isfinite(max_lse) ? 1.0f : 0.0f);
                        float const state_scale = isfinite(state_lse)
                            ? expf(state_lse - next_max) : 0.0f;
                        #pragma unroll
                        for (int item = 0; item < 8; ++item) {
                            int const dim = lane * 8 + item;
                            Element const* source_o = self_source
                                ? params.history_send_o
                                    + params.dcp_rank
                                        * params.history_send_rank_stride
                                : params.history_receive_o
                                    + source * params.history_send_rank_stride;
                            float const value = static_cast<float>(
                                source_o[vector * 128 + dim]);
                            accum[item]
                                = accum[item] * previous_scale + value * state_scale;
                        }
                        denominator = denominator * previous_scale + state_scale;
                        if (isfinite(state_lse)) { max_lse = next_max; }
                    }

                    int const chunk_splits = Config::ChunkKernel::Split
                        ? params.chunk_sequence_splits[batch] : 1;
                    for (int split = 0; split < chunk_splits; ++split) {
                        float state_lse;
                        if constexpr (Config::ChunkKernel::Split) {
                            state_lse = chunk_splits > 1
                                ? params.chunk.epilogue.ptr_LSE_partial[
                                    split * get<3>(params.chunk.epilogue.stride_LSE_partial)
                                    + local_head * get<1>(params.chunk.epilogue.stride_LSE_partial)
                                    + token]
                                : params.chunk.epilogue.ptr_LSE[
                                    local_head * get<1>(params.chunk.epilogue.stride_LSE)
                                    + token];
                        } else {
                            state_lse = params.chunk.epilogue.ptr_LSE[
                                local_head * get<1>(params.chunk.epilogue.stride_LSE)
                                + token];
                        }
                        float const next_max
                            = max_lse > state_lse ? max_lse : state_lse;
                        float const previous_scale
                            = isfinite(max_lse) && isfinite(state_lse)
                            ? expf(max_lse - next_max)
                            : (isfinite(max_lse) ? 1.0f : 0.0f);
                        float const state_scale = isfinite(state_lse)
                            ? expf(state_lse - next_max) : 0.0f;
                        #pragma unroll
                        for (int item = 0; item < 8; ++item) {
                            int const dim = lane * 8 + item;
                            float value;
                            if constexpr (Config::ChunkKernel::Split) {
                                value = chunk_splits > 1
                                    ? params.chunk.epilogue.ptr_O_partial[
                                        split * get<4>(params.chunk.epilogue.stride_O_partial)
                                        + local_head * get<2>(params.chunk.epilogue.stride_O_partial)
                                        + token * get<0>(params.chunk.epilogue.stride_O_partial)
                                        + dim]
                                    : static_cast<float>(params.chunk.epilogue.ptr_O[
                                        token * params.hq_local * 128
                                        + local_head * 128 + dim]);
                            } else {
                                value = static_cast<float>(
                                    params.chunk.epilogue.ptr_O[
                                        token * params.hq_local * 128
                                        + local_head * 128 + dim]);
                            }
                            accum[item]
                                = accum[item] * previous_scale + value * state_scale;
                        }
                        denominator = denominator * previous_scale + state_scale;
                        if (isfinite(state_lse)) { max_lse = next_max; }
                    }
                    #pragma unroll
                    for (int item = 0; item < 8; ++item) {
                        int const dim = lane * 8 + item;
                        float const value = denominator > 0.0f
                            ? accum[item] / denominator : 0.0f;
                        params.final_o[vector * 128 + dim] = Element(value);
                    }
                    if (lane == 0 && params.return_lse) {
                        params.final_lse[
                            local_head * params.final_lse_head_stride + token]
                            = denominator > 0.0f
                            ? max_lse + logf(denominator) : -INFINITY;
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <typename Config>
CUTLASS_GLOBAL
__launch_bounds__(Config::MaxThreadsPerBlock, 1)
void dcp_mega_varlen_kernel(
    CUTLASS_GRID_CONSTANT typename Config::KernelParams const params) {
    extern __shared__ char smem_buf[];
    // DCP_MEGA: communication/reduction phases reuse the copied FA dynamic
    // shared region only while the FA pipeline is inactive.
    typename Config::HelperSharedStorage& helper_shared
        = *reinterpret_cast<typename Config::HelperSharedStorage*>(smem_buf);
    if (params.phase_timestamps != nullptr
        && blockIdx.x == 0 && threadIdx.x == 0) {
        params.phase_timestamps[kKernelStartTimestamp] = read_globaltimer();
    }
    bool const communication_cta = int(blockIdx.x) < params.num_comm_sm;
    if (communication_cta) {
        run_q_allgather<Config>(params, helper_shared);
        record_phase_completion(
            params.queue_state, params.phase_timestamps,
            kQAllGatherPhaseCounter, kQAllGatherDoneTimestamp,
            params.num_comm_sm);
        run_communication_post_q<Config>(params, helper_shared);
        record_phase_completion(
            params.queue_state, params.phase_timestamps,
            kKernelPhaseCounter, kKernelDoneTimestamp,
            params.num_sms);
        return;
    }
    // DCP_MEGA: communication CTAs never initialize the FA pipeline. Every
    // compute CTA runs one unified chunk/history attention queue, then all CTA
    // threads rendezvous before the shared region is reused by local combines.
    typename Config::AttentionKernel attention_kernel;
    attention_kernel(params, smem_buf);
    __syncthreads();
    record_phase_completion(
        params.queue_state, params.phase_timestamps,
        kAttentionPhaseCounter, kAttentionDoneTimestamp,
        params.num_sms - params.num_comm_sm);
    run_history_combine<Config>(params, helper_shared);
    // History combine uses independent per-warp queues. This is the only CTA
    // convergence point before the existing CTA-oriented final combine.
    __syncthreads();
    record_fused_history_publish_completion(
        params.queue_state, params.phase_timestamps,
        params.num_sms - params.num_comm_sm);
    run_final_combine<Config>(params, helper_shared);
    record_phase_completion(
        params.queue_state, params.phase_timestamps,
        kFinalCombinePhaseCounter, kFinalCombineDoneTimestamp,
        params.num_sms - params.num_comm_sm);
    record_phase_completion(
        params.queue_state, params.phase_timestamps,
        kKernelPhaseCounter, kKernelDoneTimestamp,
        params.num_sms);
}

template <typename Kernel, typename Mainloop, typename Epilogue>
typename Kernel::Params make_attention_kernel_params(
    Flash_fwd_params const& params,
    int32_t const* metadata,
    MetadataHeader const& header,
    int32_t* attention_dynamic_counter,
    int32_t const* chunk_sequence_splits,
    int32_t const* history_sequence_splits,
    int32_t const* q_ready,
    int32_t* attention_done,
    int num_compute_ctas,
    int compute_block_offset,
    int num_sms,
    int64_t lse_head_stride) {
    using Index = Flash_fwd_params::index_t;
    typename Mainloop::StrideV v_strides = make_stride(
        params.v_row_stride, _1{}, params.v_head_stride, Index{0});
    typename Mainloop::Arguments mainloop_args{
        static_cast<Element const*>(params.q_ptr),
        {params.total_q, params.d, params.h, 1},
        {params.q_row_stride, _1{}, params.q_head_stride, Index{0}},
        static_cast<Element*>(params.k_ptr),
        {params.total_k, params.d, params.h_k, 1},
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
        nullptr,
        nullptr,
        nullptr,
        nullptr};
    typename Epilogue::Arguments epilogue_args{
        static_cast<ElementOut*>(params.o_ptr),
        {params.total_q, params.dv, params.h, 1, params.num_splits},
        {params.o_row_stride, _1{}, params.o_head_stride, Index{0}, Index{0}},
        static_cast<float*>(params.oaccum_ptr),
        {params.oaccum_row_stride, _1{}, params.oaccum_head_stride,
         Index{0}, params.oaccum_split_stride},
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, lse_head_stride, Index{0}, Index{0}},
        static_cast<float*>(params.softmax_lseaccum_ptr),
        {_1{}, params.lseaccum_head_stride, Index{0}, params.lseaccum_split_stride},
        params.h_k,
        params.cu_seqlens_q,
        nullptr};
    int const* attention_ints = metadata + header.attention_offset;
    int const* q_dependencies = metadata + header.q_dependencies_offset;
    typename flash::TileSchedulerArguments scheduler_args{
        num_compute_ctas,
        params.h,
        params.b,
        params.num_splits,
        params.h / params.h_k,
        params.seqlen_q,
        params.seqlen_k,
        params.d,
        params.dv,
        int(sizeof(Element)),
        attention_dynamic_counter,
        params.cu_seqlens_q,
        nullptr,
        chunk_sequence_splits,
        q_dependencies,
        nullptr,
        history_sequence_splits,
        header.attention_count,
        compute_block_offset,
        true,
        1,
        0,
        attention_ints,
        {},
        q_ready,
        nullptr,
        nullptr,
        attention_done};
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    return Kernel::to_underlying_arguments({
        mainloop_args,
        epilogue_args,
        {device, num_sms},
        scheduler_args});
}

template <bool Split, int BlockN, int DCPSize, int CommHeads,
          int CombineMaxSplits>
void launch_dcp_mega_instance(
    DCPMega_fwd_params& params,
    cudaStream_t stream) {
    using Config = DCPMegaKernelConfig<
        Split, BlockN, DCPSize, CommHeads, CombineMaxSplits>;
    using ChunkMainloop = typename Config::template Mainloop<true>;
    using ChunkEpilogue = typename Config::template Epilogue<true>;
    using HistoryMainloop = typename Config::template Mainloop<false>;
    using HistoryEpilogue = typename Config::template Epilogue<false>;

    int32_t const* metadata = params.metadata;
    MetadataHeader const& header = params.metadata_header;
    TORCH_CHECK(params.hq_local == CommHeads,
                "DCP mega communication specialization does not match Hq_local");
    if constexpr (Split) {
        TORCH_CHECK(
            header.history_num_splits <= CombineMaxSplits,
            "DCP mega history split upper bound exceeds combine bucket");
    } else {
        TORCH_CHECK(
            header.history_num_splits == 1,
            "DCP mega nonsplit kernel requires one history split");
    }
    int const num_compute_ctas = params.num_sms - params.num_comm_sm;
    int32_t* attention_dynamic_counter
        = params.queue_state + kAttentionDynamicCounter;
    int32_t const* chunk_sequence_splits
        = metadata + header.chunk_splits_offset;
    int32_t const* history_sequence_splits
        = metadata + header.history_splits_offset;
    auto chunk_kernel_params = make_attention_kernel_params<
        typename Config::ChunkKernel, ChunkMainloop, ChunkEpilogue>(
            params.chunk,
            metadata,
            header,
            attention_dynamic_counter,
            chunk_sequence_splits,
            history_sequence_splits,
            params.q_ready,
            params.attention_done,
            num_compute_ctas,
            params.num_comm_sm,
            params.num_sms,
            params.chunk_lse_head_stride);
    auto history_kernel_params = make_attention_kernel_params<
        typename Config::HistoryKernel, HistoryMainloop, HistoryEpilogue>(
            params.history,
            metadata,
            header,
            attention_dynamic_counter,
            chunk_sequence_splits,
            history_sequence_splits,
            params.q_ready,
            params.attention_done,
            num_compute_ctas,
            params.num_comm_sm,
            params.num_sms,
            params.history_lse_head_stride);

    uint64_t q_remote_ptrs[DCPSize]{};
    uint64_t history_remote_ptrs[DCPSize]{};
    #pragma unroll
    for (int rank = 0; rank < DCPSize; ++rank) {
        q_remote_ptrs[rank]
            = reinterpret_cast<uint64_t>(params.ipc_q_ptrs[rank]);
        history_remote_ptrs[rank]
            = reinterpret_cast<uint64_t>(params.ipc_history_send_o_ptrs[rank]);
    }
    int const comm_width = CommHeads * 128;
    auto q_remote = kittens::make_pgl<typename Config::QRemote>(
        q_remote_ptrs, 1, params.ipc_q_token_capacity,
        1, comm_width);
    auto q_group = kittens::make_gl<typename Config::QGlobal>(
        reinterpret_cast<uint64_t>(params.q_group_ptr),
        1, params.ipc_q_token_capacity, DCPSize, comm_width);
    int const history_rows = params.ipc_q_token_capacity;
    auto history_remote = kittens::make_pgl<typename Config::HistoryRemote>(
        history_remote_ptrs, 1, DCPSize, history_rows, comm_width);
    auto history_send_local = kittens::make_gl<typename Config::HistoryGlobal>(
        reinterpret_cast<uint64_t>(params.ipc_history_send_o_ptrs[params.dcp_rank]),
        1, DCPSize, history_rows, comm_width);
    auto history_receive_local = kittens::make_gl<typename Config::HistoryGlobal>(
        reinterpret_cast<uint64_t>(params.history_receive_o_ptr),
        1, DCPSize, history_rows, comm_width);
    typename Config::KernelParams kernel_params(
        chunk_kernel_params,
        history_kernel_params,
        q_remote,
        q_group,
        history_remote,
        history_send_local,
        history_receive_local);
    #pragma unroll
    for (int rank = 0; rank < DCPSize; ++rank) {
        kernel_params.history_send_lse_remote[rank]
            = params.ipc_history_send_lse_ptrs[rank];
        kernel_params.tile_ready_remote[rank] = params.ipc_tile_ready_ptrs[rank];
    }
    kernel_params.history_receive_o
        = static_cast<Element*>(params.history_receive_o_ptr);
    kernel_params.history_receive_lse = params.history_receive_lse_ptr;
    kernel_params.final_o = static_cast<Element*>(params.final_o_ptr);
    kernel_params.final_lse = params.final_lse_ptr;
    kernel_params.final_lse_head_stride = params.final_lse_head_stride;
    kernel_params.attention = reinterpret_cast<AttentionWorkDesc const*>(metadata + header.attention_offset);
    kernel_params.q_tasks = reinterpret_cast<QTaskDesc const*>(metadata + header.q_tasks_offset);
    kernel_params.publish = reinterpret_cast<PublishWorkDesc const*>(metadata + header.publish_offset);
    kernel_params.history_combine = reinterpret_cast<HistoryCombineWorkDesc const*>(
        metadata + header.history_combine_offset);
    kernel_params.final = reinterpret_cast<FinalWorkDesc const*>(metadata + header.final_offset);
    kernel_params.q_dependencies = metadata + header.q_dependencies_offset;
    kernel_params.publish_dependencies = metadata + header.publish_dependencies_offset;
    kernel_params.final_dependencies = metadata + header.final_dependencies_offset;
    kernel_params.chunk_sequence_splits = metadata + header.chunk_splits_offset;
    kernel_params.history_sequence_splits = metadata + header.history_splits_offset;
    kernel_params.q_ready = params.q_ready;
    kernel_params.attention_done = params.attention_done;
    kernel_params.publish_ready = params.publish_ready;
    kernel_params.receive_ready = params.receive_ready;
    kernel_params.queue_state = params.queue_state;
    kernel_params.phase_timestamps = params.phase_timestamps;
    kernel_params.graph_post_phase = params.graph_post_phase;
    kernel_params.cu_seqlens_q = params.chunk.cu_seqlens_q;
    kernel_params.total_q = header.total_q;
    kernel_params.total_vectors = header.total_vectors;
    kernel_params.batch_size = header.batch_size;
    kernel_params.hq_local = params.hq_local;
    kernel_params.q_row_stride = params.chunk.q_row_stride;
    kernel_params.q_head_stride = params.chunk.q_head_stride;
    kernel_params.q_group_row_stride = params.history.q_row_stride;
    kernel_params.history_send_rank_stride = params.ipc_vector_capacity * 128;
    kernel_params.signal_rank_stride = params.ipc_vector_capacity;
    kernel_params.num_q_tasks = header.q_task_count;
    kernel_params.chunk_attention_count = header.chunk_attention_count;
    kernel_params.history_attention_count = header.history_attention_count;
    kernel_params.publish_count = header.publish_count;
    kernel_params.history_combine_count = header.history_combine_count;
    kernel_params.final_count = header.final_count;
    kernel_params.effective_num_splits = header.effective_num_splits;
    kernel_params.dcp_rank = params.dcp_rank;
    kernel_params.tile_ready_phase = params.tile_ready_phase;
    kernel_params.num_sms = params.num_sms;
    kernel_params.num_comm_sm = params.num_comm_sm;
    kernel_params.return_lse = params.return_lse;
    kernel_params.history_send_o = static_cast<Element*>(
        params.ipc_history_send_o_ptrs[params.dcp_rank]);
    kernel_params.history_send_lse
        = params.ipc_history_send_lse_ptrs[params.dcp_rank];
    kernel_params.token_block_capacity = params.ipc_token_block_capacity;

    auto kernel = dcp_mega_varlen_kernel<Config>;
    int const smem_size = Config::SharedStorageSize;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<params.num_sms, Config::MaxThreadsPerBlock, smem_size, stream>>>(
        kernel_params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace min_fa3_varlen_demo::dcp_mega::detail

namespace min_fa3_varlen_demo::dcp_mega {

void run_pack_split_bn128(DCPMega_fwd_params&, cudaStream_t);
void run_pack_split_bn176(DCPMega_fwd_params&, cudaStream_t);
void run_pack_nosplit_bn128(DCPMega_fwd_params&, cudaStream_t);
void run_pack_nosplit_bn176(DCPMega_fwd_params&, cudaStream_t);

}  // namespace min_fa3_varlen_demo::dcp_mega
