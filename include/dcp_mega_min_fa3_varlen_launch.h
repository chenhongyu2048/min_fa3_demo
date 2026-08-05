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

template <bool Split, int BlockN, int DCPSize, int CommHeads>
struct DCPMegaKernelConfig {
    static_assert(BlockN == 128 || BlockN == 176);
    static_assert(DCPSize == 2 || DCPSize == 4 || DCPSize == 8);
    static_assert(CommHeads == 4 || CommHeads == 8);
    static constexpr int kDCPSize = DCPSize;
    static constexpr int kCommHeads = CommHeads;

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
        kVColMajor>;

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
        int reserved;
        int receive_task_ids[kNumCommChunks];
        int receive_scan_cursors[kNumCommChunks];
        int receive_pending[kNumCommChunks];
        float source_lse[kNumCommChunks][16 * CommHeads];
        alignas(128) CommTile comm_tiles[kNumCommChunks];
        alignas(16) kittens::semaphore arrived[kNumCommChunks];
        alignas(16) kittens::semaphore finished[kNumCommChunks];
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

CUTLASS_DEVICE void update_online_state(
    float state_lse,
    float value,
    float& max_lse,
    float& denominator,
    float& accumulator) {
    if (!isfinite(state_lse)) {
        return;
    }
    float const next_max = max_lse > state_lse ? max_lse : state_lse;
    float const previous_scale = isfinite(max_lse) ? expf(max_lse - next_max) : 0.0f;
    float const state_scale = expf(state_lse - next_max);
    accumulator = accumulator * previous_scale + value * state_scale;
    denominator = denominator * previous_scale + state_scale;
    max_lse = next_max;
}

template <typename Config>
CUTLASS_DEVICE int publish_id_from_ticket(
    typename Config::KernelParams const& params,
    int ticket) {
    int const final_id = ticket / Config::kDCPSize;
    int const dst_rank = ticket - final_id * Config::kDCPSize;
    return dst_rank * params.final_count + final_id;
}

template <typename Config>
CUTLASS_DEVICE bool history_combine_dependencies_ready(
    typename Config::KernelParams const& params,
    PublishWorkDesc const& work) {
    for (int dep = 0; dep < work.dependency_count; ++dep) {
        int const completion_id = params.publish_dependencies[
            work.dependency_begin + dep];
        if (min_fa3_varlen_demo::mega_ring::load_acquire(
                params.attention_done + completion_id) < 1) {
            return false;
        }
    }
    return true;
}

template <typename Config>
CUTLASS_DEVICE void run_history_combine_task(
    typename Config::KernelParams const& params,
    typename Config::HelperSharedStorage& shared,
    int publish_id) {
    PublishWorkDesc const work = params.publish[publish_id];
    constexpr int kVectorsPerWave = 32;
    constexpr int kWaves = Config::kCommHeads / 2;
    if (threadIdx.x < 256) {
        int const vector_in_wave = int(threadIdx.x) / 8;
        int const lane = int(threadIdx.x) % 8;
        #pragma unroll
        for (int wave = 0; wave < kWaves; ++wave) {
            int const vector_in_task = wave * kVectorsPerWave + vector_in_wave;
            if (vector_in_task < work.valid_vectors) {
                int const vector = work.vector_begin + vector_in_task;
                int const token = vector / params.hq_local;
                int const local_head = vector - token * params.hq_local;
                int const history_head
                    = work.dst_rank * params.hq_local + local_head;
                int const batch = batch_for_token(
                    token, params.cu_seqlens_q, params.batch_size);
                int const splits = Config::HistoryKernel::Split
                    ? params.history_sequence_splits[batch] : 1;
                float max_lse = -INFINITY;
                float denominator = 0.0f;
                float accum[16]{};
                for (int split = 0; split < splits; ++split) {
                    float state_lse;
                    if constexpr (Config::HistoryKernel::Split) {
                        if (splits > 1) {
                            state_lse = params.history.epilogue.ptr_LSE_partial[
                                split * get<3>(params.history.epilogue.stride_LSE_partial)
                                + history_head * get<1>(params.history.epilogue.stride_LSE_partial)
                                + token];
                        } else {
                            state_lse = params.history.epilogue.ptr_LSE[
                                history_head * get<1>(params.history.epilogue.stride_LSE)
                                + token];
                        }
                    } else {
                        state_lse = params.history.epilogue.ptr_LSE[
                            history_head * get<1>(params.history.epilogue.stride_LSE)
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
                    for (int item = 0; item < 16; ++item) {
                        int const dim = lane * 16 + item;
                        float value;
                        if constexpr (Config::HistoryKernel::Split) {
                            if (splits > 1) {
                                value = params.history.epilogue.ptr_O_partial[
                                    split * get<4>(params.history.epilogue.stride_O_partial)
                                    + history_head * get<2>(params.history.epilogue.stride_O_partial)
                                    + token * get<0>(params.history.epilogue.stride_O_partial)
                                    + dim];
                            } else {
                                value = static_cast<float>(
                                    params.history.epilogue.ptr_O[
                                        token * params.q_group_row_stride
                                        + history_head * 128 + dim]);
                            }
                        } else {
                            value = static_cast<float>(
                                params.history.epilogue.ptr_O[
                                    token * params.q_group_row_stride
                                    + history_head * 128 + dim]);
                        }
                        accum[item]
                            = accum[item] * previous_scale + value * state_scale;
                    }
                    denominator = denominator * previous_scale + state_scale;
                    if (isfinite(state_lse)) { max_lse = next_max; }
                }
                #pragma unroll
                for (int item = 0; item < 16; ++item) {
                    int const dim = lane * 16 + item;
                    float const value = denominator > 0.0f
                        ? accum[item] / denominator : 0.0f;
                    int const token_in_task
                        = vector_in_task / Config::kCommHeads;
                    int const head_in_task
                        = vector_in_task - token_in_task * Config::kCommHeads;
                    shared.comm_tiles[0][make_int2(
                        token_in_task, head_in_task * 128 + dim)]
                        = __float2bfloat16(value);
                }
                if (lane == 0) {
                    float const combined_lse = denominator > 0.0f
                        ? max_lse + logf(denominator) : -INFINITY;
                    params.history_send_lse[
                        work.dst_rank * params.signal_rank_stride + vector]
                        = combined_lse;
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        kittens::tma::store_async(
            params.history_send_local, shared.comm_tiles[0],
            {0, work.dst_rank,
             work.vector_begin / (16 * Config::kCommHeads), 0});
        kittens::tma::store_async_wait();
        if (work.dst_rank == params.dcp_rank) {
            min_fa3_varlen_demo::mega_ring::store_release(
                params.publish_ready + publish_id, 1);
        } else {
            int const token_block
                = work.vector_begin / (16 * Config::kCommHeads);
            store_release_system_s32(
                params.tile_ready_remote[work.dst_rank]
                    + params.dcp_rank * params.token_block_capacity
                    + token_block,
                params.graph_post_phase != nullptr
                    ? *params.graph_post_phase - 1
                    : params.tile_ready_phase);
        }
    }
    __syncthreads();
}

template <typename Config>
CUTLASS_DEVICE void run_history_combine(
    typename Config::KernelParams const& params,
    typename Config::HelperSharedStorage& shared) {
    while (true) {
        if (threadIdx.x == 0) {
            int const ticket = atomicAdd(
                params.queue_state + kHistoryCombineCounter, 1);
            shared.work_id = ticket < params.publish_count
                ? publish_id_from_ticket<Config>(params, ticket) : -1;
        }
        __syncthreads();
        int const publish_id = shared.work_id;
        if (publish_id < 0) {
            break;
        }
        PublishWorkDesc const work = params.publish[publish_id];
        if (threadIdx.x == 0) {
            for (int dep = 0; dep < work.dependency_count; ++dep) {
                int const completion_id = params.publish_dependencies[
                    work.dependency_begin + dep];
                min_fa3_varlen_demo::mega_ring::wait_until_at_least_acquire(
                    params.attention_done + completion_id, 1);
            }
        }
        __syncthreads();
        run_history_combine_task<Config>(params, shared, publish_id);
    }
}

// Returns a publish id, -1 when the head task is not ready, or -2 once all
// combine tickets have been claimed.
template <typename Config>
CUTLASS_DEVICE int try_run_ready_history_combine(
    typename Config::KernelParams const& params,
    typename Config::HelperSharedStorage& shared) {
    if (threadIdx.x == 0) {
        int32_t* counter = params.queue_state + kHistoryCombineCounter;
        int const ticket = atomicAdd(counter, 0);
        if (ticket >= params.publish_count) {
            shared.work_id = -2;
        } else {
            int const publish_id = publish_id_from_ticket<Config>(params, ticket);
            PublishWorkDesc const work = params.publish[publish_id];
            bool const ready
                = history_combine_dependencies_ready<Config>(params, work);
            shared.work_id = ready
                    && min_fa3_varlen_demo::mega_ring::compare_exchange_acquire(
                        counter, ticket, ticket + 1) == ticket
                ? publish_id : -1;
        }
    }
    __syncthreads();
    int const publish_id = shared.work_id;
    if (publish_id >= 0) {
        run_history_combine_task<Config>(params, shared, publish_id);
    }
    return publish_id;
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
            shared.receive_task_ids[chunk] = -1;
            shared.receive_scan_cursors[chunk] = 0;
            shared.receive_pending[chunk] = 0;
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
    uint32_t phasebits = 0xFFFF0000;
    bool receive_recorded = false;
    bool combine_recorded = false;

    while (!receive_recorded || !combine_recorded) {
        if (!receive_recorded
            && warp_id < Config::kNumCommChunks && lane == 0) {
            int const chunk = warp_id;
            int const base_task
                = Config::kNumCommChunks * int(blockIdx.x) + chunk;
            int const task_count = base_task < total_receive_tasks
                ? 1 + (total_receive_tasks - 1 - base_task) / receive_stride
                : 0;
            int selected_task = -1;
            bool pending = false;
            int const cursor = task_count > 0
                ? shared.receive_scan_cursors[chunk] % task_count : 0;
            for (int offset = 0; offset < task_count; ++offset) {
                int const ordinal = (cursor + offset) % task_count;
                int const task_id = base_task + ordinal * receive_stride;
                if (min_fa3_varlen_demo::mega_ring::load_acquire(
                        params.receive_ready + task_id) >= 1) {
                    continue;
                }
                pending = true;
                int const final_id = task_id / kReceiveSources;
                int const source_ordinal
                    = task_id - final_id * kReceiveSources;
                int const source
                    = source_ordinal + (source_ordinal >= params.dcp_rank);
                FinalWorkDesc const work = params.final[final_id];
                int const token_block
                    = work.vector_begin / (16 * Config::kCommHeads);
                if (load_acquire_system_s32(
                        params.tile_ready_remote[params.dcp_rank]
                            + source * params.token_block_capacity + token_block)
                    >= expected_phase) {
                    selected_task = task_id;
                    shared.receive_scan_cursors[chunk]
                        = (ordinal + 1) % task_count;
                    break;
                }
            }
            shared.receive_task_ids[chunk] = selected_task;
            shared.receive_pending[chunk] = pending ? 1 : 0;
        } else if (receive_recorded
                   && warp_id < Config::kNumCommChunks && lane == 0) {
            shared.receive_task_ids[warp_id] = -1;
            shared.receive_pending[warp_id] = 0;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            bool has_ready_receive = false;
            bool has_pending_receive = false;
            #pragma unroll
            for (int chunk = 0; chunk < Config::kNumCommChunks; ++chunk) {
                has_ready_receive |= shared.receive_task_ids[chunk] >= 0;
                has_pending_receive |= shared.receive_pending[chunk] != 0;
            }
            shared.reserved = (has_ready_receive ? 1 : 0)
                | (has_pending_receive ? 2 : 0);
        }
        __syncthreads();

        bool const has_ready_receive = (shared.reserved & 1) != 0;
        bool const has_pending_receive = (shared.reserved & 2) != 0;
        if (has_ready_receive) {
            if (warp_id < Config::kNumCommChunks) {
                int const chunk = warp_id;
                int const task_id = shared.receive_task_ids[chunk];
                if (task_id >= 0) {
                    int const final_id = task_id / kReceiveSources;
                    int const source_ordinal
                        = task_id - final_id * kReceiveSources;
                    int const source
                        = source_ordinal + (source_ordinal >= params.dcp_rank);
                    FinalWorkDesc const work = params.final[final_id];
                    if (lane == 0) {
                        kittens::wait(
                            shared.finished[chunk],
                            kittens::get_phasebit<1>(phasebits, 0));
                        kittens::update_phasebit<1>(phasebits, 0);
                    }
                    __syncwarp();
                    for (int vector_in_task = lane;
                         vector_in_task < work.valid_vectors;
                         vector_in_task += cutlass::NumThreadsPerWarp) {
                        int const vector = work.vector_begin + vector_in_task;
                        shared.source_lse[chunk][vector_in_task]
                            = params.history_send_lse_remote[source][
                                params.dcp_rank * params.signal_rank_stride
                                + vector];
                    }
                    __syncwarp();
                    if (lane == 0) {
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        kittens::tma::expect_bytes(
                            shared.arrived[chunk],
                            sizeof(typename Config::CommTile));
                        kittens::tma::load_async(
                            shared.comm_tiles[chunk],
                            params.history_send_remote[source],
                            {0, params.dcp_rank,
                             work.vector_begin / (16 * Config::kCommHeads), 0},
                            shared.arrived[chunk]);
                    }
                }
            } else if (warp_id < 2 * Config::kNumCommChunks) {
                int const chunk = warp_id - Config::kNumCommChunks;
                int const task_id = shared.receive_task_ids[chunk];
                if (task_id >= 0) {
                    int const final_id = task_id / kReceiveSources;
                    int const source_ordinal
                        = task_id - final_id * kReceiveSources;
                    int const source
                        = source_ordinal + (source_ordinal >= params.dcp_rank);
                    FinalWorkDesc const work = params.final[final_id];
                    if (lane == 0) {
                        kittens::wait(
                            shared.arrived[chunk],
                            kittens::get_phasebit<0>(phasebits, 0));
                        kittens::update_phasebit<0>(phasebits, 0);
                    }
                    __syncwarp();
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
                        kittens::arrive(shared.finished[chunk]);
                        kittens::tma::store_async_wait();
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        min_fa3_varlen_demo::mega_ring::store_release(
                            params.receive_ready + task_id, 1);
                    }
                }
            }
            __syncthreads();
            continue;
        }

        if (!receive_recorded && !has_pending_receive) {
            record_phase_completion(
                params.queue_state, params.phase_timestamps,
                kReceivePhaseCounter, kReceiveDoneTimestamp,
                params.num_comm_sm);
            receive_recorded = true;
        }

        if (!combine_recorded) {
            int const combine_result
                = try_run_ready_history_combine<Config>(params, shared);
            if (combine_result >= 0) {
                continue;
            }
            if (combine_result == -2) {
                record_fused_history_publish_completion(
                    params.queue_state, params.phase_timestamps,
                    params.num_sms);
                combine_recorded = true;
                continue;
            }
        }

        if (!receive_recorded || !combine_recorded) {
            __nanosleep(64);
        }
    }
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
            min_fa3_varlen_demo::mega_ring::wait_until_at_least_acquire(
                params.publish_ready + publish_id, 1);
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
    record_fused_history_publish_completion(
        params.queue_state, params.phase_timestamps,
        params.num_sms);
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

template <bool Split, int BlockN, int DCPSize, int CommHeads>
void launch_dcp_mega_instance(
    DCPMega_fwd_params& params,
    cudaStream_t stream) {
    using Config = DCPMegaKernelConfig<Split, BlockN, DCPSize, CommHeads>;
    using ChunkMainloop = typename Config::template Mainloop<true>;
    using ChunkEpilogue = typename Config::template Epilogue<true>;
    using HistoryMainloop = typename Config::template Mainloop<false>;
    using HistoryEpilogue = typename Config::template Epilogue<false>;

    int32_t const* metadata = params.metadata;
    MetadataHeader const& header = params.metadata_header;
    TORCH_CHECK(params.hq_local == CommHeads,
                "DCP mega communication specialization does not match Hq_local");
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
