// Copied and trimmed from Hopper forward sources:
// - hopper/tile_size.h
// This file fixes the demo to SM90 + bf16 + head_dim=128 and preserves
// the original Hopper tile choices for causal and non-causal forward.

#pragma once

#include "cute/tensor.hpp"
#include <cutlass/cutlass.h>

namespace min_fa3_demo {

using Element = cutlass::bfloat16_t;
using ElementOut = cutlass::bfloat16_t;

static constexpr int kHeadDim = 128;
static constexpr int kHeadDimV = 128;
static constexpr int kStages = 2;
static constexpr bool kHasSoftcap = false;
static constexpr bool kIsLocal = false;
static constexpr bool kVarlen = false;
static constexpr bool kPagedKVNonTMA = false;
static constexpr bool kAppendKV = false;
static constexpr bool kHasQv = false;
static constexpr bool kPackGQA = false;
static constexpr bool kSplit = false;
static constexpr bool kVColMajor = false;
static constexpr int kClusterM = 1;

template <bool IsCausal>
struct FwdConfig {
    static constexpr int kBlockM = 128;
    static constexpr int kBlockN = IsCausal ? 128 : 176;
    static constexpr bool MmaPV_is_RS = true;
    static constexpr bool IntraWGOverlap = true;
};

}  // namespace min_fa3_demo
