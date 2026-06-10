// Copied and trimmed from Hopper forward sources:
// - hopper/flash_fwd_launch_template.h
// Ring-attention-specific wrapper around the minimal SM90 varlen forward path.

#pragma once

#include <torch/extension.h>

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

// We include ThunderKittens torchutils only for parallel_tensor_to_pgl(...).
// That header also defines CHECK_CUDA / CHECK_CONTIGUOUS / CHECK_INPUT as
// tensor-property assertions, which collides with the FA3-side CHECK_CUDA(call)
// runtime error-checking macros pulled in via min_fa3_varlen_launch.h.
// Undefine the TK versions here so the rest of this launch path keeps using the
// local Hopper-compatible cuda_check.h macro semantics.
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

namespace min_fa3_varlen_demo {

using namespace cute;

namespace ring_detail {

using namespace kittens;

template <bool IsCausal, int NumDevices>
struct RingKernelConfig {
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
        false>;
    using Scheduler = flash::VarlenDynamicPersistentTileScheduler<
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

    static constexpr int kVecLength = kHeadDim;
    using shared_vec = sv_bf<kVecLength>;
    using staging_gl = gl<bf16, 1, 1, -1, kVecLength, shared_vec>;
    using remote_pgl = pgl<staging_gl, NumDevices, false>;

    struct KernelParams {
        typename AttnKernel::Params compute{};
        remote_pgl remote_k;
        remote_pgl remote_v;
        staging_gl local_k;
        staging_gl local_v;
        int num_comp_sm;
        int num_comm_sm;
        int src_dev;
        int rows;
        int ring_rank;
        int ring_world_size;
        int ring_step;
    };
};

template <typename RingConfig>
__device__ inline void run_ring_remote_load(
    const typename RingConfig::KernelParams& params,
    int comm_bid,
    char* smem_buf) {
    if (threadIdx.x != 0 || comm_bid >= params.num_comm_sm) {
        return;
    }

    tma_swizzle_allocator allocator(reinterpret_cast<int*>(smem_buf));
    typename RingConfig::shared_vec (&vec)[2] = allocator.allocate<typename RingConfig::shared_vec, 2>();
    __shared__ semaphore arrived[2];

    const int total_tasks = params.rows * 2;
    int stage = 0;
    int iter = 0;
    for (int task_id = comm_bid; task_id < total_tasks; task_id += params.num_comm_sm, stage ^= 1, ++iter) {
        if (iter >= 2) {
            tma::store_async_read_wait<1>();
        }

        const bool is_v = task_id >= params.rows;
        const int row_idx = is_v ? task_id - params.rows : task_id;

        init_semaphore(arrived[stage], 0, 1);
        tma::expect_bytes(arrived[stage], sizeof(vec[stage]));
        if (!is_v) {
            tma::load_async(vec[stage], params.remote_k[params.src_dev], {row_idx, 0}, arrived[stage]);
        } else {
            tma::load_async(vec[stage], params.remote_v[params.src_dev], {row_idx, 0}, arrived[stage]);
        }
        wait(arrived[stage], 0);

        if (!is_v) {
            tma::store_async(params.local_k, vec[stage], {row_idx, 0});
        } else {
            tma::store_async(params.local_v, vec[stage], {row_idx, 0});
        }
    }
    if (iter > 0) {
        tma::store_async_read_wait<0>();
    }
}

template <typename RingConfig>
CUTLASS_GLOBAL
#ifdef __CUDACC__
__launch_bounds__(
    RingConfig::AttnKernel::MaxThreadsPerBlock,
    RingConfig::AttnKernel::MinBlocksPerMultiprocessor)
#endif
void ring_flash_attn_varlen_kernel(CUTLASS_GRID_CONSTANT typename RingConfig::KernelParams const params) {
    extern __shared__ char smem_buf[];

    if (int(blockIdx.x) >= params.num_comp_sm) {
        run_ring_remote_load<RingConfig>(params, int(blockIdx.x) - params.num_comp_sm, smem_buf);
    } else {
        typename RingConfig::AttnKernel attn_kernel;
        attn_kernel(params.compute, smem_buf);
    }
}

template <bool IsCausal, int NumDevices>
void run_min_fa3_varlen_ring_sm90(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    cudaStream_t stream) {
    using RingConfig = RingKernelConfig<IsCausal, NumDevices>;
    using AttnKernel = typename RingConfig::AttnKernel;

    int const seqlen_q = params.total_q;
    int const batch_q = 1;
    int const batch_k = 1;
    int const remote_rows = params.total_k * params.h_k;
    using Index = typename Flash_fwd_params::index_t;

    typename RingConfig::CollectiveMainloop::StrideV v_strides = make_stride(
        params.v_row_stride, _1{}, params.v_head_stride, Index{0});
    typename RingConfig::CollectiveMainloop::Arguments mainloop_args{
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},
        {params.q_row_stride, _1{}, params.q_head_stride, Index{0}},
        static_cast<Element*>(params.k_ptr),
        {params.total_k, params.d, params.h_k, batch_k},
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
        nullptr};

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
        true};

    if (!params.skip_scheduler_metadata_computation) {
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

    auto kernel = ring_flash_attn_varlen_kernel<RingConfig>;
    dim3 grid_dims(uint32_t(params.num_comp_sm + params.num_comm_sm), 1, 1);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;

    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    uint64_t local_k_dst = reinterpret_cast<uint64_t>(
        params.local_k_staging_ptr != nullptr ? params.local_k_staging_ptr : params.k_ptr);
    uint64_t local_v_dst = reinterpret_cast<uint64_t>(
        params.local_v_staging_ptr != nullptr ? params.local_v_staging_ptr : params.v_ptr);

    typename RingConfig::KernelParams kernel_params{
        compute_params,
        kittens::py::parallel_tensor_to_pgl<typename RingConfig::remote_pgl>(remote_k, 1, 1, remote_rows, RingConfig::kVecLength),
        kittens::py::parallel_tensor_to_pgl<typename RingConfig::remote_pgl>(remote_v, 1, 1, remote_rows, RingConfig::kVecLength),
        kittens::make_gl<typename RingConfig::staging_gl>(local_k_dst, 1, 1, remote_rows, RingConfig::kVecLength),
        kittens::make_gl<typename RingConfig::staging_gl>(local_v_dst, 1, 1, remote_rows, RingConfig::kVecLength),
        params.num_comp_sm,
        params.num_comm_sm,
        params.src_dev,
        remote_rows,
        params.ring_rank,
        params.ring_world_size,
        params.ring_step};
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
        TORCH_CHECK(false, "Ring varlen PDL launch requires CUDA >= 11.8");
#endif
    }
}

template <bool IsCausal>
void dispatch_ring_world_size(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    cudaStream_t stream) {
    switch (remote_k.local_world_size_) {
        case 1:
            run_min_fa3_varlen_ring_sm90<IsCausal, 1>(params, remote_k, remote_v, stream);
            break;
        case 2:
            run_min_fa3_varlen_ring_sm90<IsCausal, 2>(params, remote_k, remote_v, stream);
            break;
        case 3:
            run_min_fa3_varlen_ring_sm90<IsCausal, 3>(params, remote_k, remote_v, stream);
            break;
        case 4:
            run_min_fa3_varlen_ring_sm90<IsCausal, 4>(params, remote_k, remote_v, stream);
            break;
        case 5:
            run_min_fa3_varlen_ring_sm90<IsCausal, 5>(params, remote_k, remote_v, stream);
            break;
        case 6:
            run_min_fa3_varlen_ring_sm90<IsCausal, 6>(params, remote_k, remote_v, stream);
            break;
        case 7:
            run_min_fa3_varlen_ring_sm90<IsCausal, 7>(params, remote_k, remote_v, stream);
            break;
        case 8:
            run_min_fa3_varlen_ring_sm90<IsCausal, 8>(params, remote_k, remote_v, stream);
            break;
        default:
            TORCH_CHECK(false, "Unsupported local_world_size for ring varlen path: ", remote_k.local_world_size_);
    }
}

}  // namespace ring_detail

void run_min_fa3_varlen_ring_fwd(
    Ring_fwd_params& params,
    kittens::py::TKParallelTensor& remote_k,
    kittens::py::TKParallelTensor& remote_v,
    cudaStream_t stream);

}  // namespace min_fa3_varlen_demo
