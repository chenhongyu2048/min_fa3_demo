// Copied and trimmed from Hopper backward launch configuration:
// - hopper/flash_bwd_launch_template.h

#pragma once

#include <cutlass/cutlass.h>

namespace min_fa3_backward {

using Element = cutlass::bfloat16_t;
using ElementAccum = float;

static constexpr int kHeadDim = 128;
static constexpr int kHeadDimV = 128;
static constexpr int kBlockN = 128;
static constexpr int kStages = 2;
static constexpr int kStagesdO = 2;
static constexpr int kStagesdS = 2;
static constexpr bool kSdPSwapAB = true;
static constexpr bool kdKVSwapAB = false;
static constexpr int kNumMmaWarpGroups = 2;
static constexpr int kAtomLayoutMSdP = 1;
static constexpr int kAtomLayoutNdKV = 2;
static constexpr int kAtomLayoutMdQ = 1;
static constexpr bool kVInRegs = false;

template <bool IsCausal>
struct BwdConfig {
    static constexpr int kBlockM = IsCausal ? 64 : 80;
    static constexpr bool kdQSwapAB = !IsCausal;
};

}  // namespace min_fa3_backward
