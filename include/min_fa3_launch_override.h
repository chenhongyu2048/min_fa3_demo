// Shared host-side launch override helpers for the minimal FA3 demo.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace min_fa3_detail {

inline int
validate_manual_block_count(int64_t manual_block_count, char const* arg_name = "manual_block_count") {
    if (manual_block_count <= 0) {
        throw std::invalid_argument(
            std::string(arg_name) +
            " must be greater than 0. The value is a thread-block count (grid.x), not a thread count.");
    }
    if (manual_block_count > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            std::string(arg_name) +
            " must fit in a positive 32-bit grid.x thread-block count. Got " +
            std::to_string(manual_block_count) + ".");
    }
    return static_cast<int>(manual_block_count);
}

inline std::string
format_grid_shape(dim3 grid_dims) {
    return "(" + std::to_string(grid_dims.x) + ", " +
        std::to_string(grid_dims.y) + ", " +
        std::to_string(grid_dims.z) + ")";
}

// By default the launch grid comes from AttnKernel::get_grid_shape(...).
// When manual_block_count is provided it overrides the current 1D persistent
// grid.x thread-block count while leaving the rest of the launch config alone.
inline dim3
resolve_launch_grid_shape(dim3 auto_grid_dims, std::optional<int> manual_block_count) {
    if (!manual_block_count.has_value()) {
        return auto_grid_dims;
    }
    if (auto_grid_dims.y != 1 || auto_grid_dims.z != 1) {
        throw std::invalid_argument(
            "manual_block_count override only supports the current 1D persistent grid. "
            "Automatic grid shape was " + format_grid_shape(auto_grid_dims) + ".");
    }
    return dim3(static_cast<uint32_t>(*manual_block_count), auto_grid_dims.y, auto_grid_dims.z);
}

}  // namespace min_fa3_detail
