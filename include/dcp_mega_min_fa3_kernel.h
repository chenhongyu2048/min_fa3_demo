// Copied and trimmed from include/min_fa3_kernel.h for the DCP mega path.
// Chunk and history keep separate compile-time mainloops while sharing one
// Hopper pipeline, scheduler handshake, and monotonically increasing work_idx.

#pragma once

#include <cstddef>
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "dcp_mega_min_fa3_varlen_params.h"
#include "min_fa3_named_barrier.h"
#include "softmax.h"

namespace flash {

using namespace cute;

template <class ChunkMainloop_, class HistoryMainloop_,
          class CollectiveEpilogue_, class TileScheduler_>
class DCPMegaFlashAttnFwdSm90 {
public:
    using ChunkMainloop = ChunkMainloop_;
    using HistoryMainloop = HistoryMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;
    using TileScheduler = TileScheduler_;

    using ArchTag = typename ChunkMainloop::ArchTag;
    using ClusterShape = typename ChunkMainloop::ClusterShape;
    using TileShape_MNK_PV = typename ChunkMainloop::TileShape_MNK_PV;
    using TiledMmaPV = typename ChunkMainloop::TiledMmaPV;
    using SeqlenInfo_t = typename ChunkMainloop::SeqlenInfo_t;
    using BarrierQ = std::conditional_t<
        ChunkMainloop::Use_TMA_Q || HistoryMainloop::Use_TMA_Q,
        cutlass::arch::ClusterTransactionBarrier,
        cutlass::arch::ClusterBarrier>;

    using MainloopPipelineK = typename ChunkMainloop::MainloopPipelineK;
    using MainloopPipelineV = typename ChunkMainloop::MainloopPipelineV;
    using MainloopPipelineVt = typename ChunkMainloop::MainloopPipelineVt;
    using MainloopPipelineKVNew = typename ChunkMainloop::MainloopPipelineKVNew;
    using PipelineState = typename ChunkMainloop::PipelineState;
    using PipelineParamsK = typename MainloopPipelineK::Params;
    using PipelineParamsV = typename MainloopPipelineV::Params;
    using PipelineParamsVt = typename MainloopPipelineVt::Params;

    static constexpr int NumProducerThreads = ChunkMainloop::NumProducerThreads;
    static constexpr uint32_t NumLoadWarpGroups = 1;
    static constexpr uint32_t NumMmaWarpGroups
        = CUTE_STATIC_V(size(TiledMmaPV{})) / cutlass::NumThreadsPerWarpGroup;
    static constexpr uint32_t NumMmaThreads
        = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr uint32_t MaxThreadsPerBlock
        = NumMmaThreads + NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
    static constexpr uint32_t LoadRegisterRequirement
        = NumMmaWarpGroups == 1 ? 56
        : (NumMmaWarpGroups == 2
            ? (ChunkMainloop::Use_TMA_KV ? 24 : 40) : 32);
    static constexpr uint32_t MmaRegisterRequirement
        = NumMmaWarpGroups == 1 ? 256
        : (NumMmaWarpGroups == 2
            ? (ChunkMainloop::Use_TMA_KV ? 240 : 232) : 160);

private:
    using ChunkTensorStorage = typename ChunkMainloop::TensorStorage;
    using HistoryTensorStorage = typename HistoryMainloop::TensorStorage;

    static_assert(ChunkMainloop::Is_causal);
    static_assert(!HistoryMainloop::Is_causal);
    static_assert(ChunkMainloop::Varlen && HistoryMainloop::Varlen);
    static_assert(!ChunkMainloop::AppendKV && !HistoryMainloop::AppendKV);
    static_assert(!ChunkMainloop::HasQv && !HistoryMainloop::HasQv);
    static_assert(ChunkMainloop::Use_TMA_KV && HistoryMainloop::Use_TMA_KV);
    static_assert(!ChunkMainloop::Use_TMA_Q);
    static_assert(HistoryMainloop::Use_TMA_Q);
    static_assert(!ChunkMainloop::UseTmaPackGQAQ);
    static_assert(HistoryMainloop::UseTmaPackGQAQ);
    static_assert(ChunkMainloop::Transpose_V == HistoryMainloop::Transpose_V);
    static_assert(ChunkMainloop::SameHeadDim == HistoryMainloop::SameHeadDim);
    static_assert(ChunkMainloop::LargeHeadDimV == HistoryMainloop::LargeHeadDimV);
    static_assert(ChunkMainloop::PackGQA == HistoryMainloop::PackGQA);
    static_assert(ChunkMainloop::Split == HistoryMainloop::Split);
    static_assert(ChunkMainloop::kStages == HistoryMainloop::kStages);
    static_assert(ChunkMainloop::kBlockM == HistoryMainloop::kBlockM);
    static_assert(ChunkMainloop::kBlockN == HistoryMainloop::kBlockN);
    static_assert(ChunkMainloop::kHeadDim == HistoryMainloop::kHeadDim);
    static_assert(ChunkMainloop::NumProducerThreads
                  == HistoryMainloop::NumProducerThreads);
    static_assert(ChunkMainloop::NumProducerThreads
                  == cutlass::NumThreadsPerWarpGroup);
    static_assert(ChunkMainloop::QueryBarrierArrivalCount
                  == HistoryMainloop::QueryBarrierArrivalCount);
    static_assert(ChunkMainloop::QBarrierArrivalCount
                  == HistoryMainloop::QBarrierArrivalCount);
    static_assert(ChunkMainloop::QBarrierArrivalCount
                  == cutlass::NumThreadsPerWarpGroup);
    static_assert(ChunkMainloop::NumMmaThreads
                  == HistoryMainloop::NumMmaThreads);
    static_assert(CollectiveEpilogue::NumEpilogueThreads
                  == ChunkMainloop::NumMmaThreads);
    static_assert(CollectiveEpilogue::Varlen == ChunkMainloop::Varlen);
    static_assert(CollectiveEpilogue::PackGQA == ChunkMainloop::PackGQA);
    static_assert(CollectiveEpilogue::Split == ChunkMainloop::Split);

    static_assert(std::is_same_v<typename ChunkMainloop::ClusterShape,
                                 typename HistoryMainloop::ClusterShape>);
    static_assert(std::is_same_v<typename ChunkMainloop::TileShape_MNK,
                                 typename HistoryMainloop::TileShape_MNK>);
    static_assert(std::is_same_v<typename ChunkMainloop::TileShape_MNK_PV,
                                 typename HistoryMainloop::TileShape_MNK_PV>);
    static_assert(std::is_same_v<typename ChunkMainloop::SmemLayoutQ,
                                 typename HistoryMainloop::SmemLayoutQ>);
    static_assert(std::is_same_v<typename ChunkMainloop::SmemLayoutK,
                                 typename HistoryMainloop::SmemLayoutK>);
    static_assert(std::is_same_v<typename ChunkMainloop::SmemLayoutVt,
                                 typename HistoryMainloop::SmemLayoutVt>);
    static_assert(std::is_same_v<typename ChunkMainloop::SmemLayoutVtMma,
                                 typename HistoryMainloop::SmemLayoutVtMma>);
    static_assert(std::is_same_v<typename ChunkMainloop::SmemLayoutP,
                                 typename HistoryMainloop::SmemLayoutP>);
    static_assert(std::is_same_v<typename ChunkMainloop::TiledMmaQK,
                                 typename HistoryMainloop::TiledMmaQK>);
    static_assert(std::is_same_v<typename ChunkMainloop::TiledMmaPV,
                                 typename HistoryMainloop::TiledMmaPV>);
    static_assert(std::is_same_v<typename ChunkMainloop::MainloopPipelineK,
                                 typename HistoryMainloop::MainloopPipelineK>);
    static_assert(std::is_same_v<typename ChunkMainloop::MainloopPipelineV,
                                 typename HistoryMainloop::MainloopPipelineV>);
    static_assert(std::is_same_v<typename ChunkMainloop::MainloopPipelineVt,
                                 typename HistoryMainloop::MainloopPipelineVt>);
    static_assert(std::is_same_v<typename ChunkMainloop::PipelineState,
                                 typename HistoryMainloop::PipelineState>);
    static_assert(std::is_same_v<typename MainloopPipelineK::SharedStorage,
                                 typename HistoryMainloop::MainloopPipelineK::SharedStorage>);
    static_assert(std::is_same_v<typename MainloopPipelineV::SharedStorage,
                                 typename HistoryMainloop::MainloopPipelineV::SharedStorage>);
    static_assert(std::is_same_v<typename MainloopPipelineVt::SharedStorage,
                                 typename HistoryMainloop::MainloopPipelineVt::SharedStorage>);
    static_assert(ChunkMainloop::TmaTransactionBytesQ
                  == HistoryMainloop::TmaTransactionBytesQ);
    static_assert(ChunkMainloop::TmaTransactionBytesK
                  == HistoryMainloop::TmaTransactionBytesK);
    static_assert(ChunkMainloop::TmaTransactionBytesV
                  == HistoryMainloop::TmaTransactionBytesV);
    static_assert(sizeof(ChunkTensorStorage) == sizeof(HistoryTensorStorage));
    static_assert(alignof(ChunkTensorStorage) == alignof(HistoryTensorStorage));
    static_assert(offsetof(ChunkTensorStorage, smem_v)
                  == offsetof(HistoryTensorStorage, smem_v));
    static_assert(offsetof(ChunkTensorStorage, smem_q)
                  == offsetof(HistoryTensorStorage, smem_q));
    static_assert(offsetof(ChunkTensorStorage, smem_k)
                  == offsetof(HistoryTensorStorage, smem_k));
    static_assert(offsetof(ChunkTensorStorage, smem_qv)
                  == offsetof(HistoryTensorStorage, smem_qv));

public:
    static constexpr int mainloop_smem_padding_ =
        int(sizeof(typename CollectiveEpilogue::TensorStorage))
        - int(sizeof(decltype((ChunkTensorStorage{}).smem_v)));
    static constexpr int mainloop_smem_padding
        = mainloop_smem_padding_ < 0 ? 0 : mainloop_smem_padding_;

    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128, _1> {
            union {
                struct {
                    cute::array<uint32_t,
                                mainloop_smem_padding / sizeof(uint32_t)> padding_;
                    ChunkTensorStorage mainloop;
                };
                typename CollectiveEpilogue::TensorStorage epilogue;
            };
        } tensors;
        struct PipelineStorage : cute::aligned_struct<16, _1> {
            alignas(16) BarrierQ barrier_Q;
            alignas(16) BarrierQ barrier_Qv;
            alignas(16) cutlass::arch::ClusterBarrier barrier_O;
            alignas(16) typename MainloopPipelineK::SharedStorage pipeline_k;
            alignas(16) typename MainloopPipelineV::SharedStorage pipeline_v;
            alignas(16) typename MainloopPipelineVt::SharedStorage pipeline_vt;
            alignas(16) typename MainloopPipelineKVNew::SharedStorage pipeline_k_new;
            alignas(16) typename MainloopPipelineKVNew::SharedStorage pipeline_v_new;
            alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
        } pipelines;
    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);

private:
    template <typename AttentionParams>
    CUTLASS_DEVICE static SeqlenInfo_t make_seqlen_info(
        AttentionParams const& attention,
        int bidb) {
        auto const& mainloop = attention.mainloop;
        return {
            bidb,
            get<0>(mainloop.shape_Q),
            !mainloop.ptr_pagetable
                ? size<0>(mainloop.shape_K)
                : size<0>(mainloop.shape_K) * size<1>(mainloop.shape_pagetable),
            get<0>(mainloop.shape_K_new),
            mainloop.cu_seqlens_q,
            mainloop.cu_seqlens_k,
            mainloop.cu_seqlens_k_new,
            mainloop.seqused_q,
            mainloop.seqused_k,
            mainloop.leftpad_k,
            mainloop.seqlens_rotary};
    }

    template <typename Mainloop, typename AttentionParams>
    CUTLASS_DEVICE static void prefetch_descriptors(
        AttentionParams const& attention) {
        Mainloop::prefetch_tma_descriptors(attention.mainloop);
        CollectiveEpilogue::prefetch_tma_descriptors(attention.epilogue);
    }

public:
    template <typename Params>
    CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
        static constexpr int MmaThreadOffset
            = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
        static constexpr bool Split = ChunkMainloop::Split;
        static constexpr bool Varlen = ChunkMainloop::Varlen;
        static constexpr bool IsFP8 = ChunkMainloop::Is_FP8;
        static constexpr bool LargeHeadDimV = ChunkMainloop::LargeHeadDimV;

        SharedStorage& shared_storage
            = *reinterpret_cast<SharedStorage*>(smem_buf);
        auto const& scheduler_params = params.chunk.scheduler;

        int const lane_predicate = cute::elect_one_sync();
        int const warp_idx = cutlass::canonical_warp_idx_sync();
        int const warp_group_thread_idx
            = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
        int const warp_group_idx = cutlass::canonical_warp_group_idx();

        if (warp_idx == 0 && lane_predicate) {
            int const initial_idx
                = TileScheduler::initial_descriptor_idx(scheduler_params);
            typename TileScheduler::WorkTileInfo const initial{initial_idx};
            if (initial.is_valid(scheduler_params)) {
                if (initial.kind(scheduler_params)
                    == min_fa3_varlen_demo::dcp_mega::kHistory) {
                    prefetch_descriptors<HistoryMainloop>(params.history);
                } else {
                    prefetch_descriptors<ChunkMainloop>(params.chunk);
                }
            }
            shared_storage.pipelines.barrier_Q.init(
                ChunkMainloop::QBarrierArrivalCount);
            shared_storage.pipelines.barrier_O.init(
                cute::size(ClusterShape{})
                * (CollectiveEpilogue::Use_TMA_O ? 1 : NumMmaThreads));
        }

        PipelineParamsK pipeline_params_k;
        pipeline_params_k.role = warp_group_idx == 0
            ? MainloopPipelineK::ThreadCategory::Producer
            : MainloopPipelineK::ThreadCategory::Consumer;
        if constexpr (ChunkMainloop::Use_TMA_KV) {
            pipeline_params_k.transaction_bytes
                = ChunkMainloop::TmaTransactionBytesK;
            pipeline_params_k.is_leader = warp_group_thread_idx == 0;
            pipeline_params_k.num_consumers = !LargeHeadDimV
                ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;
        } else {
            pipeline_params_k.consumer_arv_count = !LargeHeadDimV
                ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;
            pipeline_params_k.producer_arv_count = NumProducerThreads;
        }

        static_assert(std::is_same_v<PipelineParamsK, PipelineParamsVt>);
        PipelineParamsVt pipeline_params_vt = pipeline_params_k;
        if constexpr (ChunkMainloop::Use_TMA_KV
                      && !ChunkMainloop::SameHeadDim) {
            pipeline_params_vt.transaction_bytes
                = ChunkMainloop::TmaTransactionBytesV;
            if constexpr (LargeHeadDimV) {
                pipeline_params_vt.num_consumers = NumMmaThreads;
            }
        } else if constexpr (LargeHeadDimV) {
            pipeline_params_vt.consumer_arv_count = NumMmaThreads;
        }

        MainloopPipelineK pipeline_k = [&] {
            if constexpr (ChunkMainloop::Use_TMA_KV) {
                return MainloopPipelineK(
                    shared_storage.pipelines.pipeline_k,
                    pipeline_params_k,
                    ClusterShape{});
            } else {
                return MainloopPipelineK(
                    shared_storage.pipelines.pipeline_k, pipeline_params_k);
            }
        }();
        MainloopPipelineV pipeline_v = [&] {
            if constexpr (!ChunkMainloop::Transpose_V) {
                static_assert(std::is_same_v<PipelineParamsK, PipelineParamsV>);
                if constexpr (ChunkMainloop::Use_TMA_KV) {
                    return MainloopPipelineV(
                        shared_storage.pipelines.pipeline_v,
                        pipeline_params_vt,
                        ClusterShape{});
                } else {
                    return MainloopPipelineV(
                        shared_storage.pipelines.pipeline_v,
                        pipeline_params_vt);
                }
            } else {
                PipelineParamsV pipeline_params_v;
                pipeline_params_v.role = warp_group_idx == 0
                    ? MainloopPipelineV::ThreadCategory::Producer
                    : MainloopPipelineV::ThreadCategory::Consumer;
                pipeline_params_v.producer_arv_count = NumProducerThreads;
                pipeline_params_v.consumer_arv_count = NumMmaThreads;
                return MainloopPipelineV(
                    shared_storage.pipelines.pipeline_v, pipeline_params_v);
            }
        }();
        MainloopPipelineVt pipeline_vt = [&] {
            if constexpr (ChunkMainloop::Use_TMA_KV) {
                pipeline_params_vt.num_consumers = NumProducerThreads;
                return MainloopPipelineVt(
                    shared_storage.pipelines.pipeline_vt,
                    pipeline_params_vt,
                    ClusterShape{});
            } else {
                pipeline_params_vt.consumer_arv_count = NumProducerThreads;
                return MainloopPipelineVt(
                    shared_storage.pipelines.pipeline_vt,
                    pipeline_params_vt);
            }
        }();

        ChunkMainloop chunk_mainloop;
        HistoryMainloop history_mainloop;
        CollectiveEpilogue epilogue;

        if constexpr (size(ClusterShape{}) > 1) {
            cute::cluster_arrive_relaxed();
            cute::cluster_wait();
        } else {
            __syncthreads();
        }

        TileScheduler scheduler(
            reinterpret_cast<typename TileScheduler::SharedStorage*>(
                &shared_storage.pipelines.smem_scheduler));

        if (warp_group_idx == 0) {
            cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
            PipelineState smem_pipe_write
                = cutlass::make_producer_start_state<MainloopPipelineK>();
            int work_idx = 0;
            int const warp_idx_in_warpgroup = __shfl_sync(
                0xffffffff, (threadIdx.x / 32) % 4, 0);
            static constexpr bool SingleProducerWarp
                = NumProducerThreads == cutlass::NumThreadsPerWarp;
            bool const active_producer
                = !SingleProducerWarp || warp_idx_in_warpgroup == 0;

            if (active_producer) {
                if (!SingleProducerWarp && warp_idx_in_warpgroup != 0) {
                    scheduler.init_consumer();
                }
                cutlass::arch::wait_on_dependent_grids();

                int const initial_idx
                    = TileScheduler::initial_descriptor_idx(scheduler_params);
                bool history_prefetched = initial_idx >= 0
                    && initial_idx < scheduler_params.attention_count
                    && scheduler_params.descriptors[initial_idx].kind
                        == min_fa3_varlen_demo::dcp_mega::kHistory;

                auto work_tile_info = SingleProducerWarp
                        || warp_idx_in_warpgroup == 0
                    ? scheduler.template get_initial_work<true>(scheduler_params)
                    : scheduler.template get_initial_work<false>(scheduler_params);
                while (work_tile_info.is_valid(scheduler_params)) {
                    auto block_coord
                        = work_tile_info.get_block_coord(scheduler_params);
                    auto scheduler_prefetch = [&] {
                        scheduler.prefetch_next_work(
                            scheduler_params,
                            work_tile_info,
                            [&](typename TileScheduler::WorkTileInfo const& next) {
                                if (!history_prefetched
                                    && next.kind(scheduler_params)
                                        == min_fa3_varlen_demo::dcp_mega::kHistory) {
                                    prefetch_descriptors<HistoryMainloop>(
                                        params.history);
                                    history_prefetched = true;
                                }
                            });
                    };
                    if (work_tile_info.kind(scheduler_params)
                        == min_fa3_varlen_demo::dcp_mega::kHistory) {
                        SeqlenInfo_t const seqlen_info
                            = make_seqlen_info(params.history, get<2>(block_coord));
                        history_mainloop.template load<false>(
                            params.history.mainloop,
                            pipeline_k,
                            pipeline_v,
                            pipeline_vt,
                            smem_pipe_write,
                            shared_storage,
                            scheduler_prefetch,
                            seqlen_info,
                            block_coord,
                            work_idx,
                            0);
                    } else {
                        SeqlenInfo_t const seqlen_info
                            = make_seqlen_info(params.chunk, get<2>(block_coord));
                        chunk_mainloop.template load<false>(
                            params.chunk.mainloop,
                            pipeline_k,
                            pipeline_v,
                            pipeline_vt,
                            smem_pipe_write,
                            shared_storage,
                            scheduler_prefetch,
                            seqlen_info,
                            block_coord,
                            work_idx,
                            0);
                    }
                    work_tile_info = SingleProducerWarp
                            || warp_idx_in_warpgroup == 0
                        ? scheduler.template get_next_work<true>(
                            scheduler_params, work_tile_info)
                        : scheduler.template get_next_work<false>(
                            scheduler_params, work_tile_info);
                }
                chunk_mainloop.load_tail(
                    pipeline_k,
                    pipeline_v,
                    pipeline_vt,
                    smem_pipe_write,
                    shared_storage,
                    work_idx);
            }
        } else {
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
            TiledMmaPV tiled_mma_pv;
            PipelineState smem_pipe_read;

            scheduler.init_consumer();
            chunk_mainloop.mma_init();

            int work_idx = 0;
            auto work_tile_info
                = scheduler.template get_initial_work<false>(scheduler_params);
            CUTLASS_PRAGMA_NO_UNROLL
            while (work_tile_info.is_valid(scheduler_params)) {
                auto const completed_work_tile_info = work_tile_info;
                auto block_coord
                    = work_tile_info.get_block_coord(scheduler_params);
                bool const history = work_tile_info.kind(scheduler_params)
                    == min_fa3_varlen_demo::dcp_mega::kHistory;
                float const softmax_scale_log2 = history
                    ? params.history.mainloop.softmax_scale_log2
                    : params.chunk.mainloop.softmax_scale_log2;
                flash::Softmax<
                    !LargeHeadDimV
                        ? 2 * (2 * kBlockM / NumMmaThreads) : 2,
                    !IsFP8 ? 0 : 8> softmax(softmax_scale_log2);
                Tensor tOrO = partition_fragment_C(
                    tiled_mma_pv, select<0, 1>(TileShape_MNK_PV{}));

                bool tile_valid;
                if (history) {
                    SeqlenInfo_t const seqlen_info
                        = make_seqlen_info(params.history, get<2>(block_coord));
                    tile_valid = history_mainloop.template mma<false>(
                        params.history.mainloop,
                        pipeline_k,
                        pipeline_v,
                        smem_pipe_read,
                        tOrO,
                        softmax,
                        threadIdx.x - MmaThreadOffset,
                        work_idx,
                        seqlen_info,
                        block_coord,
                        shared_storage,
                        0);
                } else {
                    SeqlenInfo_t const seqlen_info
                        = make_seqlen_info(params.chunk, get<2>(block_coord));
                    tile_valid = chunk_mainloop.template mma<false>(
                        params.chunk.mainloop,
                        pipeline_k,
                        pipeline_v,
                        smem_pipe_read,
                        tOrO,
                        softmax,
                        threadIdx.x - MmaThreadOffset,
                        work_idx,
                        seqlen_info,
                        block_coord,
                        shared_storage,
                        0);
                }

                work_tile_info = scheduler.template get_next_work<false>(
                    scheduler_params, work_tile_info);
                if constexpr (Split && Varlen) {
                    if (!work_tile_info.is_valid(scheduler_params)) {
                        cutlass::arch::launch_dependent_grids();
                    }
                }

                auto const& epilogue_params = history
                    ? params.history.epilogue : params.chunk.epilogue;
                if (tile_valid) {
                    epilogue.store(
                        epilogue_params,
                        tOrO,
                        softmax.row_sum,
                        shared_storage,
                        tiled_mma_pv,
                        threadIdx.x - MmaThreadOffset,
                        block_coord);
                } else {
                    epilogue.store_zero(
                        epilogue_params,
                        threadIdx.x - MmaThreadOffset,
                        block_coord);
                }

                flash::named_barrier_sync(
                    NumMmaThreads,
                    cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
                if (threadIdx.x == MmaThreadOffset) {
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    scheduler.publish_completion(
                        scheduler_params, completed_work_tile_info);
                }
            }
            epilogue.store_tail();
        }
    }
};

}  // namespace flash
