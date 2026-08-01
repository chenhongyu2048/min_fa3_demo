// Copied and trimmed from Hopper forward sources:
// - hopper/flash_fwd_combine_launch_template.h
// Fixed to SM90, BF16 output, FP32 partials, and Dv=128. Retains the
// upstream dense/packed-Q Varlen dispatch used by Split-KV.

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/kernel_launch.h"

#include "cuda_check.h"
#include "min_fa3_fwd_combine_kernel.h"
#include "min_fa3_varlen_params.h"

namespace min_fa3_varlen_demo {

using namespace cute;

template <int kLogMaxSplits, bool Varlen>
void run_min_fa3_kvcache_combine_sm90(
    Flash_fwd_params& params,
    cudaStream_t stream,
    bool enable_pdl) {
    using TileShape_MK = Shape<Int<8>, Int<128>>;
    using CombineKernel = flash::FlashAttnFwdCombine<
        TileShape_MK,
        kLogMaxSplits,
        256,
        1,
        false,
        Varlen,
        cutlass::bfloat16_t,
        float,
        cutlass::arch::Sm90>;

    typename CombineKernel::Arguments args{
        params.b,
        static_cast<float const*>(params.oaccum_ptr),
        {!Varlen ? params.seqlen_q : params.total_q,
         params.dv,
         params.num_splits,
         params.h,
         !Varlen ? params.b : 1},
        {params.oaccum_row_stride,
         _1{},
         params.oaccum_split_stride,
         params.oaccum_head_stride,
         !Varlen ? params.oaccum_batch_stride : Flash_fwd_params::index_t{0}},
        static_cast<float*>(params.softmax_lseaccum_ptr),
        {!Varlen ? params.seqlen_q : params.total_q,
         params.num_splits,
         params.h,
         !Varlen ? params.b : 1},
        {_1{},
         params.lseaccum_split_stride,
         params.lseaccum_head_stride,
         !Varlen ? params.lseaccum_batch_stride : Flash_fwd_params::index_t{0}},
        static_cast<cutlass::bfloat16_t*>(params.o_ptr),
        {params.o_row_stride,
         _1{},
         params.o_head_stride,
         !Varlen ? params.o_batch_stride : Flash_fwd_params::index_t{0}},
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{},
         !Varlen ? params.seqlen_q : params.total_q,
         !Varlen ? params.h * params.seqlen_q : 0},
        params.cu_seqlens_q,
        params.seqused_q,
        params.num_splits_dynamic_ptr,
        params.varlen_batch_idx_ptr,
        params.tile_count_semaphore};

    typename CombineKernel::SchedulerArguments scheduler_args{
        params.b,
        params.seqlen_q,
        params.total_q,
        params.h,
        params.h_k,
        params.dv,
        params.pack_gqa,
        params.cu_seqlens_q,
        params.seqused_q,
        nullptr,
        params.varlen_batch_idx_ptr};

    typename CombineKernel::Params kernel_params{
        CombineKernel::to_underlying_arguments(args),
        CombineKernel::TileScheduler::to_underlying_arguments(scheduler_args)};

    dim3 grid = CombineKernel::TileScheduler::get_grid_shape(scheduler_args);
    auto kernel = cutlass::device_kernel<CombineKernel>;
    int smem_size = CombineKernel::SharedStorageSize;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    CHECK_CUTLASS(cutlass::kernel_launch<CombineKernel>(
        grid,
        CombineKernel::MaxThreadsPerBlock,
        smem_size,
        stream,
        kernel_params,
        enable_pdl));
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace min_fa3_varlen_demo
