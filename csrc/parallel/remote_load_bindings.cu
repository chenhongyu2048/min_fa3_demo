#include <torch/csrc/utils/pybind.h>

#include "parallel/remote_load.h"

namespace py = pybind11;

void bind_parallel_remote_load(py::module_& m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def(
        "parallel_remote_load_out",
        py::overload_cast<torch::Tensor&, kittens::py::TKParallelTensor&, int64_t, int64_t>(
            &min_fa3_parallel::parallel_remote_load_out),
        py::arg("output"),
        py::arg("input"),
        py::arg("src_rank"),
        py::arg("num_blocks"),
        "Internal out-variant for preallocated remote-load testing.");
    m.def(
        "parallel_remote_load",
        py::overload_cast<kittens::py::TKParallelTensor&, int64_t, int64_t>(
            &min_fa3_parallel::parallel_remote_load),
        py::arg("input"),
        py::arg("src_rank"),
        py::arg("num_blocks"),
        "Remote-load a contiguous bfloat16 tensor from src_rank into local device memory using ThunderKittens IPC + TMA.");
    m.def(
        "parallel_remote_load_vec_out",
        py::overload_cast<torch::Tensor&, kittens::py::TKParallelTensor&, int64_t, int64_t>(
            &min_fa3_parallel::parallel_remote_load_vec_out),
        py::arg("output"),
        py::arg("input"),
        py::arg("src_rank"),
        py::arg("num_blocks"),
        "Internal out-variant for preallocated row-vector remote-load testing.");
    m.def(
        "parallel_remote_load_vec",
        py::overload_cast<kittens::py::TKParallelTensor&, int64_t, int64_t>(
            &min_fa3_parallel::parallel_remote_load_vec),
        py::arg("input"),
        py::arg("src_rank"),
        py::arg("num_blocks"),
        "Remote-load a contiguous bfloat16 tensor from src_rank row-by-row using a ThunderKittens vector TMA path.");
}
