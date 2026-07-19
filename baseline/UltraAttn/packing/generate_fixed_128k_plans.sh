#!/usr/bin/env bash

# Generate UltraAttn ILP plans for five fixed 128K-token packed workloads:
# 1x128K, 2x64K, 4x32K, 8x16K, and 16x8K.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

WORLD_SIZE=${WORLD_SIZE:-8}
QHEAD=${QHEAD:-32}
KVHEAD=${KVHEAD:-8}
HEADDIM=${HEADDIM:-128}
BLOCK_TOKENS=${BLOCK_TOKENS:-8192}
TIME_LIMIT=${TIME_LIMIT:-1800}
SOLVER_SEED=${SOLVER_SEED:-0}
PLAN_DIR=${PLAN_DIR:-$REPO_ROOT/baseline/UltraAttn/packing_plans}
PLANNER_PY=${PLANNER_PY:-python}

die() {
    echo "error: $*" >&2
    exit 1
}

[[ "$WORLD_SIZE" == 8 ]] || \
    die "the fixed 128K hierarchy is defined for WORLD_SIZE=8, got '$WORLD_SIZE'"
for name in QHEAD KVHEAD HEADDIM; do
    value=${!name}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || \
        die "$name must be a positive integer, got '$value'"
done
[[ "$HEADDIM" == 128 ]] || die "HEADDIM must be 128, got '$HEADDIM'"
[[ "$QHEAD" == 32 && "$KVHEAD" == 8 ]] || \
    die "the UltraAttn graph fixed suite requires QHEAD=32 and KVHEAD=8"
[[ "$BLOCK_TOKENS" == 8192 ]] || \
    die "the UltraAttn graph fixed suite requires BLOCK_TOKENS=8192"

if [[ "$PLANNER_PY" == */* ]]; then
    [[ -x "$PLANNER_PY" ]] || die "PLANNER_PY is not executable: $PLANNER_PY"
else
    command -v "$PLANNER_PY" >/dev/null || die "PLANNER_PY was not found: $PLANNER_PY"
fi

mkdir -p "$PLAN_DIR"

labels=(
    "1x128K"
    "2x64K"
    "4x32K"
    "8x16K"
    "16x8K"
)
global_seqlens=(
    "131072"
    "65536,65536"
    "32768,32768,32768,32768"
    "16384,16384,16384,16384,16384,16384,16384,16384"
    "8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192"
)

echo "UltraAttn fixed 128K plan generation"
echo "Repository: $REPO_ROOT"
echo "World size: $WORLD_SIZE"
echo "Heads: QH=$QHEAD, KVH=$KVHEAD, D=$HEADDIM"
echo "Scheduling block: $BLOCK_TOKENS tokens"
echo "Plan directory: $PLAN_DIR"
echo "Planner Python: $PLANNER_PY"
echo "Gurobi: time_limit=${TIME_LIMIT}s per case, solver_seed=$SOLVER_SEED, threads=${GUROBI_NUM_THREADS:-64}"
echo "Cases: ${labels[*]}"

for index in "${!labels[@]}"; do
    echo
    echo "Generating case $((index + 1))/${#labels[@]}: ${labels[index]}"
    "$PLANNER_PY" baseline/UltraAttn/packing/export_packed_causal_plan.py \
        --global-seqlens "${global_seqlens[index]}" \
        --world-size "$WORLD_SIZE" \
        --qhead "$QHEAD" \
        --kvhead "$KVHEAD" \
        --headdim "$HEADDIM" \
        --block-tokens "$BLOCK_TOKENS" \
        --time-limit "$TIME_LIMIT" \
        --solver-seed "$SOLVER_SEED" \
        --output-dir "$PLAN_DIR"
done

echo
echo "All fixed 128K plans are ready in: $PLAN_DIR"
