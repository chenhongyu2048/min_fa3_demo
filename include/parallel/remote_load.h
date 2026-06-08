#pragma once

#include <torch/extension.h>

#include "kittens.cuh"
#include "pyutils/parallel_tensor.cuh"

namespace min_fa3_parallel {

void parallel_remote_load_out(
    torch::Tensor& output,
    kittens::py::TKParallelTensor& input,
    int64_t src_rank,
    int64_t num_blocks);

torch::Tensor parallel_remote_load(
    kittens::py::TKParallelTensor& input,
    int64_t src_rank,
    int64_t num_blocks);

void parallel_remote_load_vec_out(
    torch::Tensor& output,
    kittens::py::TKParallelTensor& input,
    int64_t src_rank,
    int64_t num_blocks);

torch::Tensor parallel_remote_load_vec(
    kittens::py::TKParallelTensor& input,
    int64_t src_rank,
    int64_t num_blocks);

}  // namespace min_fa3_parallel
