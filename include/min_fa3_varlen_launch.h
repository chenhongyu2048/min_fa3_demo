// Copied and trimmed from Hopper forward sources:
// - hopper/flash_fwd_launch_template.h
// - hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu
// This varlen launch path is fixed to the original Hopper SM90 bf16 head_dim=128
// forward kernel family with flattened varlen Q/K/V inputs.

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/kernel_launch.h"

#include "cuda_check.h"
#include "min_fa3_launch_override.h"
#include "min_fa3_varlen_params.h"
#include "min_fa3_varlen_traits.h"
#include "min_fa3_varlen_scheduler.h"
#include "min_fa3_kernel.h"
#include "min_fa3_mainloop.h"
#include "min_fa3_epilogue.h"

namespace min_fa3_varlen_demo {

using namespace cute;

template <bool IsCausal, bool PackGQA = false, bool Split = false>
void run_min_fa3_varlen_sm90(
    Flash_fwd_params& params,
    cudaStream_t stream,
    std::optional<int> manual_block_count) {
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
        PackGQA,
        Split,
        kVColMajor>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<
        TileShape_MNK_PV,
        ClusterShape,
        ElementOut,
        ArchTag,
        CollectiveMainloop::NumMmaThreads,
        kVarlen,
        PackGQA,
        Split,
        false,
        false>;
    using Scheduler = flash::VarlenDynamicPersistentTileScheduler<
        Config::kBlockM,
        Config::kBlockN,
        CollectiveMainloop::NumMmaThreads,
        CollectiveMainloop::NumProducerThreads,
        Split,
        PackGQA,
        true,
        IsCausal,
        true,
        true>;
    using AttnKernel = flash::enable_sm90<flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    bool const is_varlen_q = params.cu_seqlens_q != nullptr;
    bool const is_varlen_k = params.cu_seqlens_k != nullptr;
    int const seqlen_q = !is_varlen_q ? params.seqlen_q : params.total_q;
    int const batch_q = !is_varlen_q ? params.b : 1;
    int const batch_k = !is_varlen_k ? params.b : 1;
    using Index = typename Flash_fwd_params::index_t;

    typename CollectiveMainloop::StrideV v_strides = make_stride(
        params.v_row_stride,
        _1{},
        params.v_head_stride,
        !is_varlen_k ? params.v_batch_stride : Index{0});
    typename CollectiveMainloop::Arguments mainloop_args{
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},
        {params.q_row_stride,
         _1{},
         params.q_head_stride,
         !is_varlen_q ? params.q_batch_stride : Index{0}},
        static_cast<Element*>(params.k_ptr),
        {!is_varlen_k ? params.seqlen_k : params.total_k, params.d, params.h_k, batch_k},
        {params.k_row_stride,
         _1{},
         params.k_head_stride,
         !is_varlen_k ? params.k_batch_stride : Index{0}},
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
        nullptr};

    typename CollectiveEpilogue::Arguments epilogue_args{
        static_cast<ElementOut*>(params.o_ptr),
        {seqlen_q, params.dv, params.h, batch_q, params.num_splits},
        {params.o_row_stride,
         _1{},
         params.o_head_stride,
         !is_varlen_q ? params.o_batch_stride : Index{0},
         Index{0}},
        static_cast<float*>(params.oaccum_ptr),
        {params.oaccum_row_stride,
         _1{},
         params.oaccum_head_stride,
         !is_varlen_q ? params.oaccum_batch_stride : Index{0},
         params.oaccum_split_stride},
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen_q, !is_varlen_q ? params.h * seqlen_q : Index{0}, Index{0}},
        static_cast<float*>(params.softmax_lseaccum_ptr),
        {_1{},
         params.lseaccum_head_stride,
         !is_varlen_q ? params.lseaccum_batch_stride : Index{0},
         params.lseaccum_split_stride},
        params.h_k,
        params.cu_seqlens_q,
        params.seqused_q};

    int const qhead_per_khead = !PackGQA ? 1 : cutlass::ceil_div(params.h, params.h_k);
    int num_blocks_m = cutlass::ceil_div(params.seqlen_q * qhead_per_khead, Config::kBlockM);
    typename flash::TileSchedulerArguments scheduler_args{
        num_blocks_m,
        !PackGQA ? params.h : params.h_k,
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
        params.num_nheads_in_l2_ptr};

    if (!params.skip_scheduler_metadata_computation) {
        prepare_varlen_num_blocks(
            params,
            stream,
            PackGQA,
            Config::kBlockM,
            Config::kBlockN,
            params.prepare_varlen_pdl);
        CHECK_CUDA_KERNEL_LAUNCH();
    }

    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args,
        epilogue_args,
        {device, params.num_sm},
        scheduler_args});

    dim3 grid_dims = min_fa3_detail::resolve_launch_grid_shape(
        AttnKernel::get_grid_shape(kernel_params),
        manual_block_count);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    auto kernel = cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    CHECK_CUTLASS(cutlass::kernel_launch<AttnKernel>(
        grid_dims,
        block_dims,
        smem_size,
        stream,
        kernel_params,
        params.prepare_varlen_pdl && !params.skip_scheduler_metadata_computation));
}

}  // namespace min_fa3_varlen_demo

extern template void min_fa3_varlen_demo::run_min_fa3_varlen_sm90<false, false, false>(
    min_fa3_varlen_demo::Flash_fwd_params&, cudaStream_t, std::optional<int>);
extern template void min_fa3_varlen_demo::run_min_fa3_varlen_sm90<true, false, false>(
    min_fa3_varlen_demo::Flash_fwd_params&, cudaStream_t, std::optional<int>);
extern template void min_fa3_varlen_demo::run_min_fa3_varlen_sm90<false, true, false>(
    min_fa3_varlen_demo::Flash_fwd_params&, cudaStream_t, std::optional<int>);
extern template void min_fa3_varlen_demo::run_min_fa3_varlen_sm90<true, true, false>(
    min_fa3_varlen_demo::Flash_fwd_params&, cudaStream_t, std::optional<int>);
extern template void min_fa3_varlen_demo::run_min_fa3_varlen_sm90<false, true, true>(
    min_fa3_varlen_demo::Flash_fwd_params&, cudaStream_t, std::optional<int>);
extern template void min_fa3_varlen_demo::run_min_fa3_varlen_sm90<true, true, true>(
    min_fa3_varlen_demo::Flash_fwd_params&, cudaStream_t, std::optional<int>);
