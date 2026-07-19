#!/usr/bin/env bash

# Dataset-shaped hierarchical hybrid forward/backward benchmark.
# Run inside a single-node allocation exposing 2, 4, or 8 SM90 GPUs.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

GPU_COUNTS=${GPU_COUNTS:-"8"}
DATASETS=${DATASETS:-"arxiv freelaw github pile"}
DIRECTION=${DIRECTION:-forward}
TARGET_TOKENS=${TARGET_TOKENS:-131072}
COMPUTE_BALANCE_TOLERANCE=${COMPUTE_BALANCE_TOLERANCE:-0.05}
TOKEN_BALANCE_TOLERANCE=${TOKEN_BALANCE_TOLERANCE:-0.10}
BEAM_WIDTH=${BEAM_WIDTH:-64}
FINALIST_COUNT=${FINALIST_COUNT:-8}
STRUCTURE_THRESHOLD=${STRUCTURE_THRESHOLD:-0.5}
MAX_REPAIR_ITERATIONS=${MAX_REPAIR_ITERATIONS:-32}
SEED=${SEED:-0}
NUM_CASES=${NUM_CASES:-1}
METHODS=${METHODS:-all}
ZEPPLIN_THRESHOLD=${ZEPPLIN_THRESHOLD:-4096}
MODE=${MODE:-causal}
QHEAD=${QHEAD:-32}
KVHEAD=${KVHEAD:-8}
HEADDIM=${HEADDIM:-128}
SM_CONFIGS=${SM_CONFIGS:-"128:4,124:8,120:12,116:16"}
WARMUP_ITERS=${WARMUP_ITERS:-10}
NUM_ITERS=${NUM_ITERS:-40}
CHECK=${CHECK:-0}
DRY_RUN=${DRY_RUN:-0}
TORCHRUN=${TORCHRUN:-torchrun}
LOG_DIR=${LOG_DIR:-"benchmark_logs/$(date +%Y%m%d-%H%M%S)"}
if [[ -z ${LOG_FILE:-} ]]; then
    if [[ "$DIRECTION" == forward ]]; then
        LOG_FILE="$LOG_DIR/benchmark_hybrid_dataset.log"
    else
        LOG_FILE="$LOG_DIR/benchmark_hybrid_dataset_backward.log"
    fi
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

die() {
    echo "error: $*" >&2
    exit 1
}

case "$CHECK" in
    0) CHECK_ARGS=(--no-check) ;;
    1) CHECK_ARGS=(--check) ;;
    *) die "CHECK must be 0 or 1, got '$CHECK'" ;;
esac

case "$DRY_RUN" in
    0|1) ;;
    *) die "DRY_RUN must be 0 or 1, got '$DRY_RUN'" ;;
esac

case "$DIRECTION" in
    forward|backward) ;;
    *) die "DIRECTION must be forward or backward, got '$DIRECTION'" ;;
esac

case "$MODE" in
    noncausal|causal|both) ;;
    *) die "MODE must be noncausal, causal, or both, got '$MODE'" ;;
esac
if [[ "$DIRECTION" == backward && "$MODE" != causal ]]; then
    die "hybrid backward supports only MODE=causal"
fi

[[ "$TARGET_TOKENS" =~ ^[1-9][0-9]*$ ]] || \
    die "TARGET_TOKENS must be a positive integer, got '$TARGET_TOKENS'"
[[ "$ZEPPLIN_THRESHOLD" =~ ^[1-9][0-9]*$ ]] || \
    die "ZEPPLIN_THRESHOLD must be a positive integer, got '$ZEPPLIN_THRESHOLD'"
[[ "$NUM_CASES" =~ ^[1-9][0-9]*$ ]] || \
    die "NUM_CASES must be a positive integer, got '$NUM_CASES'"
[[ "$BEAM_WIDTH" =~ ^[1-9][0-9]*$ ]] || \
    die "BEAM_WIDTH must be a positive integer, got '$BEAM_WIDTH'"
[[ "$FINALIST_COUNT" =~ ^[1-9][0-9]*$ ]] || \
    die "FINALIST_COUNT must be a positive integer, got '$FINALIST_COUNT'"
[[ "$MAX_REPAIR_ITERATIONS" =~ ^[0-9]+$ ]] || \
    die "MAX_REPAIR_ITERATIONS must be a non-negative integer, got '$MAX_REPAIR_ITERATIONS'"

gpu_counts_spec=${GPU_COUNTS//,/ }
read -r -a GPU_COUNT_LIST <<< "$gpu_counts_spec"
((${#GPU_COUNT_LIST[@]} > 0)) || die "GPU_COUNTS must not be empty"

max_gpu_count=0
for world_size in "${GPU_COUNT_LIST[@]}"; do
    case "$world_size" in
        2|4|8) ;;
        *) die "GPU_COUNTS must contain only 2, 4, or 8, got '$world_size'" ;;
    esac
    ((world_size > max_gpu_count)) && max_gpu_count=$world_size
done

datasets_spec=${DATASETS//,/ }
read -r -a DATASET_LIST <<< "$datasets_spec"
((${#DATASET_LIST[@]} > 0)) || die "DATASETS must not be empty"
for dataset in "${DATASET_LIST[@]}"; do
    case "$dataset" in
        arxiv|github|pile|freelaw) ;;
        *) die "DATASETS must contain only arxiv, freelaw, github, or pile, got '$dataset'" ;;
    esac
done

if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    IFS=',' read -r -a VISIBLE_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
    ((${#VISIBLE_DEVICES[@]} >= max_gpu_count)) || die \
        "CUDA_VISIBLE_DEVICES exposes ${#VISIBLE_DEVICES[@]} GPUs, but $max_gpu_count are required"
else
    VISIBLE_DEVICES=()
    for ((gpu = 0; gpu < max_gpu_count; ++gpu)); do
        VISIBLE_DEVICES+=("$gpu")
    done
fi

select_devices() {
    local world_size=$1
    local selected=("${VISIBLE_DEVICES[@]:0:world_size}")
    local IFS=,
    SELECTED_DEVICES=${selected[*]}
}

print_command() {
    printf 'CUDA_VISIBLE_DEVICES=%q' "$1"
    shift
    printf ' %q' "$@"
    printf '\n'
}

run_benchmark() {
    local dataset=$1
    local world_size=$2
    local visible_devices=$3
    local entrypoint
    if [[ "$DIRECTION" == forward ]]; then
        entrypoint=ring_test/benchmark_hybrid_dataset_forward.py
    else
        entrypoint=ring_test/benchmark_hybrid_dataset_backward.py
    fi
    local -a command=(
        "$TORCHRUN" --standalone --nproc_per_node="$world_size"
        "$entrypoint"
        --dataset "$dataset"
        --target-tokens "$TARGET_TOKENS"
        --compute-balance-tolerance "$COMPUTE_BALANCE_TOLERANCE"
        --token-balance-tolerance "$TOKEN_BALANCE_TOLERANCE"
        --beam-width "$BEAM_WIDTH"
        --finalist-count "$FINALIST_COUNT"
        --structure-threshold "$STRUCTURE_THRESHOLD"
        --max-repair-iterations "$MAX_REPAIR_ITERATIONS"
        --seed "$SEED"
        --num-cases "$NUM_CASES"
        --qhead "$QHEAD" --kvhead "$KVHEAD" --headdim "$HEADDIM"
        --zepplin-threshold "$ZEPPLIN_THRESHOLD"
        --sm-configs "$SM_CONFIGS"
        --warmup-iters "$WARMUP_ITERS" --num-iters "$NUM_ITERS"
        "${CHECK_ARGS[@]}"
    )
    if [[ "$DIRECTION" == forward ]]; then
        command+=(--mode "$MODE" --methods "$METHODS")
    else
        command+=(--methods "$METHODS")
    fi

    if ((DRY_RUN)); then
        printf '\n================================================================================\n'
        printf '[hybrid_dataset_%s] dataset=%s GPUs=%s visible=%s\n' \
            "$DIRECTION" "$dataset" "$world_size" "$visible_devices"
        printf '================================================================================\n'
        print_command "$visible_devices" "${command[@]}"
        return
    fi

    {
        printf '\n================================================================================\n'
        printf '[hybrid_dataset_%s] dataset=%s GPUs=%s visible=%s\n' \
            "$DIRECTION" "$dataset" "$world_size" "$visible_devices"
        printf '================================================================================\n'
        print_command "$visible_devices" "${command[@]}"
    } | tee -a "$LOG_FILE"
    CUDA_VISIBLE_DEVICES="$visible_devices" "${command[@]}" 2>&1 | tee -a "$LOG_FILE"
}

if ((DRY_RUN == 0)); then
    mkdir -p "$(dirname -- "$LOG_FILE")"
    : > "$LOG_FILE"
fi

echo "Log: $LOG_FILE"
echo "Datasets: ${DATASET_LIST[*]}"
echo "Config: direction=$DIRECTION, target_tokens=$TARGET_TOKENS, compute_tolerance=$COMPUTE_BALANCE_TOLERANCE, token_tolerance=$TOKEN_BALANCE_TOLERANCE, beam_width=$BEAM_WIDTH, finalist_count=$FINALIST_COUNT, structure_threshold=$STRUCTURE_THRESHOLD, max_repair_iterations=$MAX_REPAIR_ITERATIONS, seed=$SEED, num_cases=$NUM_CASES, mode=$MODE, zepplin_threshold=$ZEPPLIN_THRESHOLD"
echo "Methods: $METHODS"

for world_size in "${GPU_COUNT_LIST[@]}"; do
    select_devices "$world_size"
    for dataset in "${DATASET_LIST[@]}"; do
        run_benchmark "$dataset" "$world_size" "$SELECTED_DEVICES"
    done
done

if ((DRY_RUN == 0)); then
    echo "Results written to $LOG_FILE"
fi
