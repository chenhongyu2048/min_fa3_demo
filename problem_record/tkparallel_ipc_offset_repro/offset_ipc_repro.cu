#include <array>
#include <cstdint>
#include <limits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/csrc/utils/pybind.h>

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace {

constexpr int kMaxDevices = 8;

template <int NumDevices>
struct RemoteCopyKernel {
    struct config {
        static constexpr int CLUSTER_SIZE = 1;
        static constexpr int MIN_BLOCKS_PER_SM = 1;
        static constexpr int NUM_THREADS = 256;
        static constexpr int DYNAMIC_SHARED_MEMORY = 0;
    };

    struct globals {
        using output_layout = gl<bf16, 1, 1, -1, -1>;
        using input_layout = pgl<gl<bf16, 1, 1, -1, -1>, NumDevices, false>;

        output_layout output;
        input_layout input;
        int src_rank;
        int total_elems;

        __host__ inline dim3 grid() const {
            return dim3(static_cast<uint32_t>((total_elems + config::NUM_THREADS - 1) / config::NUM_THREADS));
        }
    };

    __device__ static inline void kernel(const globals& g) {
        int const idx = int(blockIdx.x) * config::NUM_THREADS + int(threadIdx.x);
        if (idx < g.total_elems) {
            g.output.raw_ptr[idx] = g.input[g.src_rank].raw_ptr[idx];
        }
    }
};

void check_common(at::Tensor& output,
                  kittens::py::TKParallelTensor& input,
                  int64_t src_rank,
                  int64_t row_offset) {
    TORCH_CHECK(input.data_.is_cuda(), "input must wrap a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(input.data_.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(input.data_.scalar_type() == at::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");
    TORCH_CHECK(input.data_.dim() == 2, "input must be a 2D tensor or flattened 2D combined tensor");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");
    TORCH_CHECK(output.size(1) == input.data_.size(1), "output cols must match input cols");
    TORCH_CHECK(row_offset >= 0, "row_offset must be non-negative");
    TORCH_CHECK(row_offset + output.size(0) <= input.data_.size(0),
                "row_offset + output rows must fit in input rows");
    TORCH_CHECK(input.local_world_size_ >= 1 && input.local_world_size_ <= kMaxDevices,
                "local_world_size must be in [1, ", kMaxDevices, "]");
    TORCH_CHECK(src_rank >= 0 && src_rank < input.local_world_size_,
                "src_rank must be in [0, local_world_size)");
    TORCH_CHECK(input.local_rank_ == input.data_.device().index(),
                "input local_rank must match input device index");
    TORCH_CHECK(output.device() == input.data_.device(),
                "output must be on the same local CUDA device as input");
    TORCH_CHECK(output.numel() <= std::numeric_limits<int>::max(),
                "output numel must fit in int32");
}

template <int NumDevices>
typename RemoteCopyKernel<NumDevices>::globals::input_layout make_input_pgl(
    kittens::py::TKParallelTensor& input,
    int rows,
    int cols,
    int row_offset) {
    using Kernel = RemoteCopyKernel<NumDevices>;
    using input_layout = typename Kernel::globals::input_layout;

    std::array<uint64_t, NumDevices> ptrs{};
    uint64_t const byte_offset = uint64_t(row_offset) * uint64_t(cols) * sizeof(bf16);
    TORCH_CHECK(input.raw_ptrs_.size() == NumDevices,
                "raw_ptrs_ size must match local_world_size");
    TORCH_CHECK(input.raw_ptrs_[input.local_rank_] == reinterpret_cast<void*>(input.data_.data_ptr()),
                "local raw pointer must match input.data_ptr()");
    for (int i = 0; i < NumDevices; ++i) {
        ptrs[i] = reinterpret_cast<uint64_t>(input.raw_ptrs_[i]) + byte_offset;
    }
    return make_pgl<input_layout>(ptrs.data(), 1, 1, rows, cols);
}

template <int NumDevices>
void remote_copy_impl(at::Tensor& output,
                      kittens::py::TKParallelTensor& input,
                      int64_t src_rank,
                      int64_t row_offset) {
    using Kernel = RemoteCopyKernel<NumDevices>;
    using output_layout = typename Kernel::globals::output_layout;

    int const rows = static_cast<int>(output.size(0));
    int const cols = static_cast<int>(output.size(1));

    typename Kernel::globals g{
        .output = kittens::py::tensor_to_gl<output_layout>(output, 1, 1, rows, cols),
        .input = make_input_pgl<NumDevices>(input, rows, cols, static_cast<int>(row_offset)),
        .src_rank = static_cast<int>(src_rank),
        .total_elems = static_cast<int>(output.numel()),
    };

    kittens::py::launch_kernel<typename Kernel::config, typename Kernel::globals, Kernel::kernel>(g);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void remote_copy(at::Tensor output,
                 kittens::py::TKParallelTensor& input,
                 int64_t src_rank,
                 int64_t row_offset) {
    c10::cuda::CUDAGuard device_guard(input.data_.device());
    check_common(output, input, src_rank, row_offset);

    switch (input.local_world_size_) {
        case 1:
            remote_copy_impl<1>(output, input, src_rank, row_offset);
            break;
        case 2:
            remote_copy_impl<2>(output, input, src_rank, row_offset);
            break;
        case 3:
            remote_copy_impl<3>(output, input, src_rank, row_offset);
            break;
        case 4:
            remote_copy_impl<4>(output, input, src_rank, row_offset);
            break;
        case 5:
            remote_copy_impl<5>(output, input, src_rank, row_offset);
            break;
        case 6:
            remote_copy_impl<6>(output, input, src_rank, row_offset);
            break;
        case 7:
            remote_copy_impl<7>(output, input, src_rank, row_offset);
            break;
        case 8:
            remote_copy_impl<8>(output, input, src_rank, row_offset);
            break;
        default:
            TORCH_CHECK(false, "unsupported local_world_size: ", input.local_world_size_);
    }
}

}  // namespace

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def(
        "remote_copy",
        &remote_copy,
        pybind11::arg("output"),
        pybind11::arg("input"),
        pybind11::arg("src_rank"),
        pybind11::arg("row_offset") = 0,
        "Copy rows from a TKParallelTensor on src_rank into output. row_offset is applied to every raw_ptr_.");
}
