#!/usr/bin/env bash

# Supplementary Ulysses/USP benchmark suite.
#
# This mirrors the CP workloads previously used under
# benchmark_logs/bench_cp and benchmark_logs/experiment_suite/cp-suite while
# selecting only the two all-to-all baselines.  Existing benchmark wrappers
# remain responsible for workload generation, timing, summaries, and logs.
#
# This suite deliberately excludes forward ablation, tile/load analysis, and
# the separate runtime load-balance suite: those tests measure scheduler or
# fused-kernel behavior that is not an Ulysses/USP comparison target.
#
# The suite is intentionally configurable because the complete historical
# matrix is large.  Set RUN_*_TEST=0 to skip a section, or DRY_RUN=1 to print
# all underlying torchrun commands without launching CUDA work.

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

WORLD_SIZE=${WORLD_SIZE:-8}
GPU_LIST=${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}}
TORCHRUN=${TORCHRUN:-"$ROOT_DIR/.venv/bin/torchrun"}
METHODS=${METHODS:-"ulysses,usp"}
QHEAD=${QHEAD:-32}
HEADDIM=${HEADDIM:-128}
KVHEAD=${KVHEAD:-8}
ALLGATHER_OVERLAPPING_HEADS_K_STRIDE=${ALLGATHER_OVERLAPPING_HEADS_K_STRIDE:-4}
SM_CONFIGS=${SM_CONFIGS:-"128:4,124:8,120:12,116:16"}
DATASETS=${DATASETS:-"arxiv freelaw github pile prolong"}
SEED=${SEED:-0}
NUM_CASES=${NUM_CASES:-20}
WARMUP_ITERS=${WARMUP_ITERS:-10}
NUM_ITERS=${NUM_ITERS:-40}
CHECK=${CHECK:-0}
MODE=${MODE:-causal}

# Workloads from the historical bench_cp/cp-suite runs.
UNIFORM_CONTEXT_LENGTHS=${UNIFORM_CONTEXT_LENGTHS:-"65536 131072 262144"}
UNIFORM_BATCH_SIZES=${UNIFORM_BATCH_SIZES:-"1 2 4 8 16"}
DATASET_TARGET_TOKENS=${DATASET_TARGET_TOKENS:-"65536 131072 262144"}
KVH_MATRIX_TARGET_TOKENS=${KVH_MATRIX_TARGET_TOKENS:-131072}
KVH_MATRIX_KVHEADS=${KVH_MATRIX_KVHEADS:-"1,2,4,8"}

# The old KVH matrix used a separate per-direction timing policy.  These can
# be overridden independently without changing the workload matrix.
KVH_FORWARD_WARMUP_ITERS=${KVH_FORWARD_WARMUP_ITERS:-20}
KVH_FORWARD_NUM_ITERS=${KVH_FORWARD_NUM_ITERS:-20}
KVH_BACKWARD_WARMUP_ITERS=${KVH_BACKWARD_WARMUP_ITERS:-10}
KVH_BACKWARD_NUM_ITERS=${KVH_BACKWARD_NUM_ITERS:-40}

RUN_UNIFORM_TEST=${RUN_UNIFORM_TEST:-1}
RUN_DATASET_TEST=${RUN_DATASET_TEST:-1}
RUN_KVH_MATRIX_TEST=${RUN_KVH_MATRIX_TEST:-1}
RUN_TRANSFORMER_LAYER_TEST=${RUN_TRANSFORMER_LAYER_TEST:-0}
DRY_RUN=${DRY_RUN:-0}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
SUITE_ROOT=${SUITE_ROOT:-"$ROOT_DIR/benchmark_logs/experiment_suite/cp-suite/03_ulysses_usp/$RUN_ID"}

die() {
    echo "error: $*" >&2
    exit 2
}

[[ "$WORLD_SIZE" == 8 ]] || die "WORLD_SIZE must be 8 for the CP suite"
[[ "$QHEAD" =~ ^[1-9][0-9]*$ ]] || die "QHEAD must be a positive integer"
[[ "$HEADDIM" == 128 ]] || die "HEADDIM must be 128"
[[ "$KVHEAD" =~ ^[1-9][0-9]*$ ]] || die "KVHEAD must be a positive integer"
((QHEAD % KVHEAD == 0)) || die "QHEAD must be divisible by KVHEAD"
[[ "$NUM_CASES" =~ ^[1-9][0-9]*$ ]] || die "NUM_CASES must be positive"
[[ "$WARMUP_ITERS" =~ ^[0-9]+$ ]] || die "WARMUP_ITERS must be non-negative"
[[ "$NUM_ITERS" =~ ^[1-9][0-9]*$ ]] || die "NUM_ITERS must be positive"
[[ "$CHECK" =~ ^[01]$ ]] || die "CHECK must be 0 or 1"
[[ "$DRY_RUN" =~ ^[01]$ ]] || die "DRY_RUN must be 0 or 1"

case "$MODE" in
    causal|noncausal|both) ;;
    *) die "MODE must be causal, noncausal, or both" ;;
esac
if [[ "$MODE" != causal ]]; then
    die "the supplementary suite includes backward and therefore requires MODE=causal"
fi

for toggle in \
    RUN_UNIFORM_TEST \
    RUN_DATASET_TEST \
    RUN_KVH_MATRIX_TEST \
    RUN_TRANSFORMER_LAYER_TEST; do
    value=${!toggle}
    [[ "$value" =~ ^[01]$ ]] || die "$toggle must be 0 or 1"
done

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
(( ${#GPU_ARRAY[@]} >= WORLD_SIZE )) || \
    die "GPU_LIST must expose at least ${WORLD_SIZE} comma-separated GPUs"

command -v "$TORCHRUN" >/dev/null 2>&1 || die "torchrun executable not found: $TORCHRUN"

case "$CHECK" in
    0) CHECK_ARGS=(--no-check) ;;
    1) CHECK_ARGS=(--check) ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export TORCHRUN
export DRY_RUN
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NCCL_CGA_CLUSTER_SIZE=${NCCL_CGA_CLUSTER_SIZE:-1}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}

if ((DRY_RUN == 0)); then
    mkdir -p "$SUITE_ROOT"
fi

run_section() {
    local label=$1
    shift
    echo
    echo "================ ${label} ================"
    "$@"
}

run_uniform() {
    local output_dir="$SUITE_ROOT/01_cp_uniform"
    run_section "UNIFORM CP MATRIX (forward/backward, KVH=${KVHEAD})" \
        env \
        CONTEXT_LENGTHS="$UNIFORM_CONTEXT_LENGTHS" \
        BATCH_SIZES="$UNIFORM_BATCH_SIZES" \
        DIRECTION=both \
        MODE=causal \
        METHODS="$METHODS" \
        QHEAD="$QHEAD" KVHEAD="$KVHEAD" HEADDIM="$HEADDIM" \
        ALLGATHER_OVERLAPPING_HEADS_K_STRIDE="$ALLGATHER_OVERLAPPING_HEADS_K_STRIDE" \
        SM_CONFIGS="$SM_CONFIGS" \
        WARMUP_ITERS="$WARMUP_ITERS" NUM_ITERS="$NUM_ITERS" \
        SEED="$SEED" CHECK="$CHECK" \
        LOG_DIR="$output_dir" \
        LOG_FILE="$output_dir/benchmark_uniform_both.log" \
        scripts/benchmark_cp_uniform.sh
}

run_dataset() {
    local target direction output_dir
    local dataset_targets_spec=${DATASET_TARGET_TOKENS//,/ }
    for target in $dataset_targets_spec; do
        for direction in forward backward; do
            output_dir="$SUITE_ROOT/02_dataset_${target}_${direction}"
            run_section "DATASET MATRIX (${direction}, target=${target}, KVH=${KVHEAD})" \
                env \
                GPU_COUNTS="$WORLD_SIZE" \
                DATASETS="$DATASETS" \
                DIRECTION="$direction" \
                TARGET_TOKENS="$target" \
                NUM_CASES="$NUM_CASES" \
                MODE=causal \
                METHODS="$METHODS" \
                QHEAD="$QHEAD" KVHEAD="$KVHEAD" HEADDIM="$HEADDIM" \
                ALLGATHER_OVERLAPPING_HEADS_K_STRIDE="$ALLGATHER_OVERLAPPING_HEADS_K_STRIDE" \
                SM_CONFIGS="$SM_CONFIGS" \
                WARMUP_ITERS="$WARMUP_ITERS" NUM_ITERS="$NUM_ITERS" \
                SEED="$SEED" CHECK="$CHECK" \
                LOG_DIR="$output_dir" \
                LOG_FILE="$output_dir/benchmark_dataset_${direction}.log" \
                ./benchmark_dataset.sh
        done
    done
}

run_kvh_matrix() {
    local direction output_dir warmup iterations
    for direction in forward backward; do
        if [[ "$direction" == forward ]]; then
            warmup=$KVH_FORWARD_WARMUP_ITERS
            iterations=$KVH_FORWARD_NUM_ITERS
        else
            warmup=$KVH_BACKWARD_WARMUP_ITERS
            iterations=$KVH_BACKWARD_NUM_ITERS
        fi
        output_dir="$SUITE_ROOT/03_kvh_matrix/$direction"
        run_section "KVH MATRIX (${direction}, target=${KVH_MATRIX_TARGET_TOKENS})" \
            env \
            NPROC_PER_NODE="$WORLD_SIZE" \
            DIRECTIONS="$direction" \
            DATASETS="$DATASETS" \
            KVHEADS="$KVH_MATRIX_KVHEADS" \
            TARGET_TOKENS="$KVH_MATRIX_TARGET_TOKENS" \
            NUM_CASES="$NUM_CASES" \
            METHODS="$METHODS" \
            QHEAD="$QHEAD" HEADDIM="$HEADDIM" \
            ALLGATHER_OVERLAPPING_HEADS_K_STRIDE="$ALLGATHER_OVERLAPPING_HEADS_K_STRIDE" \
            SM_CONFIGS="$SM_CONFIGS" \
            WARMUP_ITERS="$warmup" NUM_ITERS="$iterations" \
            SEED="$SEED" CHECK="$CHECK" \
            LOG_DIR="$output_dir" \
            scripts/benchmark_dataset_kvh_matrix.sh
    done
}

run_transformer_layer() {
    local output_dir="$SUITE_ROOT/04_transformer_layer"
    run_section "TRANSFORMER LAYER (${METHODS})" \
        env \
        DATASETS="${DATASETS// /,}" \
        TARGET_TOKENS=131072 \
        NUM_CASES="$NUM_CASES" \
        WORLD_SIZE="$WORLD_SIZE" \
        METHODS="$METHODS" \
        QHEAD="$QHEAD" HEADDIM="$HEADDIM" \
        SM_CONFIGS="$SM_CONFIGS" \
        WARMUP_ITERS="$WARMUP_ITERS" NUM_ITERS="$NUM_ITERS" \
        SEED="$SEED" \
        OUTPUT_DIR="$output_dir" \
        ./benchmark_transformer_layer.sh
}

if ((RUN_UNIFORM_TEST)); then
    run_uniform
fi
if ((RUN_DATASET_TEST)); then
    run_dataset
fi
if ((RUN_KVH_MATRIX_TEST)); then
    run_kvh_matrix
fi
if ((RUN_TRANSFORMER_LAYER_TEST)); then
    run_transformer_layer
fi

echo
echo "Ulysses/USP supplementary benchmark suite completed."
echo "Logs: $SUITE_ROOT"
