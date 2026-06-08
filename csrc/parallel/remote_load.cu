#include "parallel/remote_load.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

namespace min_fa3_parallel {
namespace {

using namespace kittens;

constexpr int kMaxSupportedWorldSize = 8;
constexpr int kTileSide = 128;
constexpr int kVecCols = 4096;

template <int NumDevices>
struct RemoteLoadTileKernel {
    struct config {
        static constexpr int CLUSTER_SIZE = 1;
        static constexpr int MIN_BLOCKS_PER_SM = 6;
        static constexpr int NUM_THREADS = 1;
    };

    struct globals {
        using shared_tile = st_bf<kTileSide, kTileSide>;
        using output_layout = gl<bf16, 1, 1, -1, -1, shared_tile>;
        using input_layout = pgl<gl<bf16, 1, 1, -1, -1, shared_tile>, NumDevices, false>;

        output_layout output;
        input_layout input;
        int src_rank;
        int num_blocks;
        int total_tiles;
        int col_tiles;

        __host__ inline dim3 grid() const {
            return dim3(static_cast<uint32_t>(num_blocks), 1, 1);
        }

        __host__ inline int dynamic_shared_memory() const {
            return 2 * sizeof(shared_tile) + 1024;
        }
    };

    __device__ static inline void kernel(const globals& g) {
        extern __shared__ int __shm[];
        tma_swizzle_allocator allocator(reinterpret_cast<int*>(&__shm[0]));
        typename globals::shared_tile (&tile)[2] = allocator.allocate<typename globals::shared_tile, 2>();

        __shared__ semaphore arrived[2];
        int stage = 0;
        int iter = 0;
        for (int task_id = static_cast<int>(blockIdx.x); task_id < g.total_tiles; task_id += g.num_blocks, stage ^= 1, ++iter) {
            if (iter >= 2) {
                tma::store_async_read_wait<1>();
            }
            const int row_block_idx = task_id / g.col_tiles;
            const int col_block_idx = task_id % g.col_tiles;

            init_semaphore(arrived[stage], 0, 1);
            tma::expect_bytes(arrived[stage], sizeof(tile[stage]));
            tma::load_async(tile[stage], g.input[g.src_rank], {row_block_idx, col_block_idx}, arrived[stage]);
            wait(arrived[stage], 0);
            tma::store_async(g.output, tile[stage], {row_block_idx, col_block_idx});
        }
        if (iter > 0) {
            tma::store_async_read_wait<0>();
        }
    }
};

template <int NumDevices, int VecLength>
struct RemoteLoadVecKernel {
    struct config {
        static constexpr int CLUSTER_SIZE = 1;
        static constexpr int MIN_BLOCKS_PER_SM = 6;
        static constexpr int NUM_THREADS = 1;
    };

    struct globals {
        using shared_vec = sv_bf<VecLength>;
        using output_layout = gl<bf16, 1, 1, -1, VecLength, shared_vec>;
        using input_layout = pgl<gl<bf16, 1, 1, -1, VecLength, shared_vec>, NumDevices, false>;

        output_layout output;
        input_layout input;
        int src_rank;
        int num_blocks;
        int rows;

        __host__ inline dim3 grid() const {
            return dim3(static_cast<uint32_t>(num_blocks), 1, 1);
        }

        __host__ inline int dynamic_shared_memory() const {
            return 2 * sizeof(shared_vec) + 1024;
        }
    };

    __device__ static inline void kernel(const globals& g) {
        extern __shared__ int __shm[];
        tma_swizzle_allocator allocator(reinterpret_cast<int*>(&__shm[0]));
        typename globals::shared_vec (&vec)[2] = allocator.allocate<typename globals::shared_vec, 2>();

        __shared__ semaphore arrived[2];
        int stage = 0;
        int iter = 0;
        for (int row_idx = static_cast<int>(blockIdx.x); row_idx < g.rows; row_idx += g.num_blocks, stage ^= 1, ++iter) {
            if (iter >= 2) {
                tma::store_async_read_wait<1>();
            }

            init_semaphore(arrived[stage], 0, 1);
            tma::expect_bytes(arrived[stage], sizeof(vec[stage]));
            tma::load_async(vec[stage], g.input[g.src_rank], {row_idx, 0}, arrived[stage]);
            wait(arrived[stage], 0);
            tma::store_async(g.output, vec[stage], {row_idx, 0});
        }
        if (iter > 0) {
            tma::store_async_read_wait<0>();
        }
    }
};

template <int NumDevices>
void launch_remote_load_impl(
    torch::Tensor& output,
    kittens::py::TKParallelTensor& input,
    int src_rank,
    int num_blocks) {
    using remote_kernel = RemoteLoadTileKernel<NumDevices>;
    using input_layout = typename remote_kernel::globals::input_layout;
    using output_layout = typename remote_kernel::globals::output_layout;

    const int rows = static_cast<int>(input.data_.size(0));
    const int cols = static_cast<int>(input.data_.size(1));
    const int row_tiles = rows / kTileSide;
    const int col_tiles = cols / kTileSide;
    const int total_tiles = row_tiles * col_tiles;

    typename remote_kernel::globals remote_g{
        .output = kittens::py::tensor_to_gl<output_layout>(
            output, 1, 1, rows, cols),
        .input = kittens::py::parallel_tensor_to_pgl<input_layout>(
            input, 1, 1, rows, cols),
        .src_rank = src_rank,
        .num_blocks = num_blocks,
        .total_tiles = total_tiles,
        .col_tiles = col_tiles,
    };

    kittens::py::launch_kernel<
        typename remote_kernel::config,
        typename remote_kernel::globals,
        remote_kernel::kernel>(remote_g);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int NumDevices, int VecLength>
void launch_remote_load_vec_impl(
    torch::Tensor& output,
    kittens::py::TKParallelTensor& input,
    int src_rank,
    int num_blocks) {
    using remote_kernel = RemoteLoadVecKernel<NumDevices, VecLength>;
    using input_layout = typename remote_kernel::globals::input_layout;
    using output_layout = typename remote_kernel::globals::output_layout;

    const int rows = static_cast<int>(input.data_.size(0));

    typename remote_kernel::globals remote_g{
        .output = kittens::py::tensor_to_gl<output_layout>(
            output, 1, 1, rows, VecLength),
        .input = kittens::py::parallel_tensor_to_pgl<input_layout>(
            input, 1, 1, rows, VecLength),
        .src_rank = src_rank,
        .num_blocks = num_blocks,
        .rows = rows,
    };

    kittens::py::launch_kernel<
        typename remote_kernel::config,
        typename remote_kernel::globals,
        remote_kernel::kernel>(remote_g);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_tile_world_size(
    torch::Tensor& output,
    kittens::py::TKParallelTensor& input,
    int src_rank,
    int num_blocks) {
    switch (input.local_world_size_) {
        case 1:
            launch_remote_load_impl<1>(output, input, src_rank, num_blocks);
            break;
        case 2:
            launch_remote_load_impl<2>(output, input, src_rank, num_blocks);
            break;
        case 3:
            launch_remote_load_impl<3>(output, input, src_rank, num_blocks);
            break;
        case 4:
            launch_remote_load_impl<4>(output, input, src_rank, num_blocks);
            break;
        case 5:
            launch_remote_load_impl<5>(output, input, src_rank, num_blocks);
            break;
        case 6:
            launch_remote_load_impl<6>(output, input, src_rank, num_blocks);
            break;
        case 7:
            launch_remote_load_impl<7>(output, input, src_rank, num_blocks);
            break;
        case 8:
            launch_remote_load_impl<8>(output, input, src_rank, num_blocks);
            break;
        default:
            TORCH_CHECK(false, "Unsupported local_world_size: ", input.local_world_size_);
    }
}

template <int VecLength>
void dispatch_vec_world_size(
    torch::Tensor& output,
    kittens::py::TKParallelTensor& input,
    int src_rank,
    int num_blocks) {
    switch (input.local_world_size_) {
        case 1:
            launch_remote_load_vec_impl<1, VecLength>(output, input, src_rank, num_blocks);
            break;
        case 2:
            launch_remote_load_vec_impl<2, VecLength>(output, input, src_rank, num_blocks);
            break;
        case 3:
            launch_remote_load_vec_impl<3, VecLength>(output, input, src_rank, num_blocks);
            break;
        case 4:
            launch_remote_load_vec_impl<4, VecLength>(output, input, src_rank, num_blocks);
            break;
        case 5:
            launch_remote_load_vec_impl<5, VecLength>(output, input, src_rank, num_blocks);
            break;
        case 6:
            launch_remote_load_vec_impl<6, VecLength>(output, input, src_rank, num_blocks);
            break;
        case 7:
            launch_remote_load_vec_impl<7, VecLength>(output, input, src_rank, num_blocks);
            break;
        case 8:
            launch_remote_load_vec_impl<8, VecLength>(output, input, src_rank, num_blocks);
            break;
        default:
            TORCH_CHECK(false, "Unsupported local_world_size: ", input.local_world_size_);
    }
}

void dispatch_vec_cols(
    torch::Tensor& output,
    kittens::py::TKParallelTensor& input,
    int src_rank,
    int num_blocks) {
    switch (input.data_.size(1)) {
        case kVecCols:
            dispatch_vec_world_size<kVecCols>(output, input, src_rank, num_blocks);
            break;
        default:
            TORCH_CHECK(false, "parallel_remote_load_vec currently supports cols=", kVecCols, " only. Got cols=", input.data_.size(1));
    }
}

void check_base_inputs(
    const kittens::py::TKParallelTensor& input,
    int64_t src_rank,
    int64_t num_blocks,
    const char* op_name) {
    TORCH_CHECK(input.data_.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.data_.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.data_.numel() > 0, "input must be non-empty");
    TORCH_CHECK(input.data_.dim() == 2, op_name, " currently supports 2D tensors only");
    TORCH_CHECK(input.data_.scalar_type() == torch::kBFloat16, op_name, " currently supports torch.bfloat16 only");
    TORCH_CHECK(input.local_world_size_ >= 1 && input.local_world_size_ <= kMaxSupportedWorldSize, op_name, " currently supports local_world_size in [1, ", kMaxSupportedWorldSize, "]. Got ", input.local_world_size_);
    TORCH_CHECK(src_rank >= 0 && src_rank < input.local_world_size_, "src_rank must be in [0, local_world_size). Got src_rank=", src_rank, ", local_world_size=", input.local_world_size_);
    TORCH_CHECK(input.data_.numel() <= std::numeric_limits<int>::max(), op_name, " only supports tensors with numel <= INT_MAX. Got ", input.data_.numel());
    TORCH_CHECK(num_blocks > 0, op_name, " requires num_blocks > 0. Got ", num_blocks);
    TORCH_CHECK(num_blocks <= std::numeric_limits<int>::max(), op_name, " requires num_blocks <= INT_MAX. Got ", num_blocks);
}

void check_tile_inputs(const kittens::py::TKParallelTensor& input, const char* op_name) {
    TORCH_CHECK(input.data_.size(0) % kTileSide == 0, op_name, " requires rows to be a multiple of ", kTileSide, ". Got rows=", input.data_.size(0));
    TORCH_CHECK(input.data_.size(1) % kTileSide == 0, op_name, " requires cols to be a multiple of ", kTileSide, ". Got cols=", input.data_.size(1));
}

void check_vec_inputs(const kittens::py::TKParallelTensor& input, const char* op_name) {
    TORCH_CHECK(input.data_.size(1) == kVecCols, op_name, " currently supports cols=", kVecCols, " only. Got cols=", input.data_.size(1));
}

void check_output_tensor(
    const torch::Tensor& output,
    const kittens::py::TKParallelTensor& input) {
    TORCH_CHECK(output.device() == input.data_.device(), "output must be on the same device as the local input shard");
    TORCH_CHECK(output.sizes().vec() == input.data_.sizes().vec(), "output must have the same shape as input");
    TORCH_CHECK(output.scalar_type() == input.data_.scalar_type(), "output must have the same dtype as input");
}

}  // namespace

void parallel_remote_load_out(
    torch::Tensor& output,
    kittens::py::TKParallelTensor& input,
    int64_t src_rank,
    int64_t num_blocks) {
    c10::cuda::CUDAGuard device_guard(input.data_.device());
    check_base_inputs(input, src_rank, num_blocks, "parallel_remote_load");
    check_tile_inputs(input, "parallel_remote_load");
    check_output_tensor(output, input);

    dispatch_tile_world_size(output, input, static_cast<int>(src_rank), static_cast<int>(num_blocks));
}

torch::Tensor parallel_remote_load(
    kittens::py::TKParallelTensor& input,
    int64_t src_rank,
    int64_t num_blocks) {
    auto output = torch::empty_like(input.data_);
    parallel_remote_load_out(output, input, src_rank, num_blocks);
    return output;
}

void parallel_remote_load_vec_out(
    torch::Tensor& output,
    kittens::py::TKParallelTensor& input,
    int64_t src_rank,
    int64_t num_blocks) {
    c10::cuda::CUDAGuard device_guard(input.data_.device());
    check_base_inputs(input, src_rank, num_blocks, "parallel_remote_load_vec");
    check_vec_inputs(input, "parallel_remote_load_vec");
    check_output_tensor(output, input);

    dispatch_vec_cols(output, input, static_cast<int>(src_rank), static_cast<int>(num_blocks));
}

torch::Tensor parallel_remote_load_vec(
    kittens::py::TKParallelTensor& input,
    int64_t src_rank,
    int64_t num_blocks) {
    auto output = torch::empty_like(input.data_);
    parallel_remote_load_vec_out(output, input, src_rank, num_blocks);
    return output;
}

}  // namespace min_fa3_parallel
