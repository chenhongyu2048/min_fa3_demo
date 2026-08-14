#!/usr/bin/env bash

# Run one causal 128K torchrun per direction, KV-head count, and dataset.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NCCL_CGA_CLUSTER_SIZE=${NCCL_CGA_CLUSTER_SIZE:-1}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE=${MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
DIRECTIONS=${DIRECTIONS:-forward}
DATASETS=${DATASETS:-arxiv,github,pile,freelaw,prolong}
KVHEADS=${KVHEADS:-1}
TARGET_TOKENS=${TARGET_TOKENS:-131072}
METHODS=${METHODS:-all}
SM_CONFIGS=${SM_CONFIGS:-128:4,124:8,120:12,116:16}
QHEAD=${QHEAD:-32}
HEADDIM=${HEADDIM:-128}
ALLGATHER_OVERLAPPING_HEADS_K_STRIDE=${ALLGATHER_OVERLAPPING_HEADS_K_STRIDE:-4}
COMPUTE_BALANCE_TOLERANCE=${COMPUTE_BALANCE_TOLERANCE:-0.05}
TOKEN_BALANCE_TOLERANCE=${TOKEN_BALANCE_TOLERANCE:-0.05}
BEAM_WIDTH=${BEAM_WIDTH:-64}
FINALIST_COUNT=${FINALIST_COUNT:-8}
STRUCTURE_THRESHOLD=${STRUCTURE_THRESHOLD:-0.5}
MAX_REPAIR_ITERATIONS=${MAX_REPAIR_ITERATIONS:-32}
SEED=${SEED:-0}
NUM_CASES=${NUM_CASES:-20}
ZEPPELIN_THRESHOLD=${ZEPPELIN_THRESHOLD:-4096}
MEGATRON_MAX_SEQLEN_PER_RANK=${MEGATRON_MAX_SEQLEN_PER_RANK:-8192}
MAGI_OVERLAP_DEGREE=${MAGI_OVERLAP_DEGREE:-2}
WARMUP_ITERS=${WARMUP_ITERS:-20}
NUM_ITERS=${NUM_ITERS:-20}
CHECK=${CHECK:-0}
COLLECT_MEGA_RING_STATS=${COLLECT_MEGA_RING_STATS:-0}
DRY_RUN=${DRY_RUN:-0}
TORCHRUN=${TORCHRUN:-torchrun}
LOG_DIR=${LOG_DIR:-benchmark_logs/dataset_kvh_matrix/$(date +%Y%m%d-%H%M%S)}

die() {
    echo "error: $*" >&2
    exit 1
}

case "$NPROC_PER_NODE" in
    2|4|8) ;;
    *) die "NPROC_PER_NODE must be 2, 4, or 8, got '$NPROC_PER_NODE'" ;;
esac
case "$CHECK" in
    0) CHECK_ARGS=(--no-check) ;;
    1) CHECK_ARGS=(--check) ;;
    *) die "CHECK must be 0 or 1, got '$CHECK'" ;;
esac
case "$COLLECT_MEGA_RING_STATS" in
    0|1) ;;
    *) die "COLLECT_MEGA_RING_STATS must be 0 or 1, got '$COLLECT_MEGA_RING_STATS'" ;;
esac
case "$DRY_RUN" in
    0|1) ;;
    *) die "DRY_RUN must be 0 or 1, got '$DRY_RUN'" ;;
esac
case "$MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE" in
    0|1) ;;
    *) die "MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE must be 0 or 1, got '$MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE'" ;;
esac

[[ "$TARGET_TOKENS" =~ ^[1-9][0-9]*$ ]] || \
    die "TARGET_TOKENS must be a positive integer, got '$TARGET_TOKENS'"
[[ "$NUM_CASES" =~ ^[1-9][0-9]*$ ]] || \
    die "NUM_CASES must be a positive integer, got '$NUM_CASES'"
[[ "$WARMUP_ITERS" =~ ^[0-9]+$ ]] || \
    die "WARMUP_ITERS must be a non-negative integer, got '$WARMUP_ITERS'"
[[ "$NUM_ITERS" =~ ^[1-9][0-9]*$ ]] || \
    die "NUM_ITERS must be a positive integer, got '$NUM_ITERS'"

if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    IFS=',' read -r -a VISIBLE_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
    ((${#VISIBLE_DEVICES[@]} >= NPROC_PER_NODE)) || die \
        "CUDA_VISIBLE_DEVICES exposes ${#VISIBLE_DEVICES[@]} GPUs, but $NPROC_PER_NODE are required"
fi

directions_spec=${DIRECTIONS//,/ }
read -r -a DIRECTION_LIST <<< "$directions_spec"
((${#DIRECTION_LIST[@]} > 0)) || die "DIRECTIONS must not be empty"
for direction in "${DIRECTION_LIST[@]}"; do
    case "$direction" in
        forward|backward) ;;
        *) die "DIRECTIONS must contain only forward or backward, got '$direction'" ;;
    esac
done

datasets_spec=${DATASETS//,/ }
read -r -a DATASET_LIST <<< "$datasets_spec"
((${#DATASET_LIST[@]} > 0)) || die "DATASETS must not be empty"
for dataset in "${DATASET_LIST[@]}"; do
    case "$dataset" in
        arxiv|github|pile|freelaw|prolong) ;;
        *) die "unsupported dataset '$dataset'" ;;
    esac
done

kvheads_spec=${KVHEADS//,/ }
read -r -a KVHEAD_LIST <<< "$kvheads_spec"
((${#KVHEAD_LIST[@]} > 0)) || die "KVHEADS must not be empty"
for kvhead in "${KVHEAD_LIST[@]}"; do
    case "$kvhead" in
        1|2|4|8) ;;
        *) die "KVHEADS must contain only 1, 2, 4, or 8, got '$kvhead'" ;;
    esac
done

COMMON_ARGS=(
    --target-tokens "$TARGET_TOKENS"
    --compute-balance-tolerance "$COMPUTE_BALANCE_TOLERANCE"
    --token-balance-tolerance "$TOKEN_BALANCE_TOLERANCE"
    --beam-width "$BEAM_WIDTH"
    --finalist-count "$FINALIST_COUNT"
    --structure-threshold "$STRUCTURE_THRESHOLD"
    --max-repair-iterations "$MAX_REPAIR_ITERATIONS"
    --seed "$SEED"
    --num-cases "$NUM_CASES"
    --qhead "$QHEAD"
    --headdim "$HEADDIM"
    --allgather-overlapping-heads-k-stride "$ALLGATHER_OVERLAPPING_HEADS_K_STRIDE"
    --methods "$METHODS"
    --zeppelin-threshold "$ZEPPELIN_THRESHOLD"
    --megatron-max-seqlen-per-rank "$MEGATRON_MAX_SEQLEN_PER_RANK"
    --magi-overlap-degree "$MAGI_OVERLAP_DEGREE"
    --sm-configs "$SM_CONFIGS"
    --warmup-iters "$WARMUP_ITERS"
    --num-iters "$NUM_ITERS"
    "${CHECK_ARGS[@]}"
)

print_command() {
    printf '%q' "$1"
    shift
    printf ' %q' "$@"
    printf '\n'
}

run_case() {
    local direction=$1
    local kvhead=$2
    local dataset=$3
    local entrypoint="ring_test/benchmark_dataset_${direction}.py"
    local case_log_dir="$LOG_DIR/$direction/kvh$kvhead"
    local log_file="$case_log_dir/$dataset.log"
    local -a command=(
        "$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE"
        "$entrypoint"
        --dataset "$dataset"
        --kvhead "$kvhead"
        "${COMMON_ARGS[@]}"
    )
    if [[ "$direction" == forward ]]; then
        command+=(--mode causal)
        if ((COLLECT_MEGA_RING_STATS)); then
            command+=(--collect-mega-ring-stats)
        fi
    fi

    printf '\n[%s] direction=%s kvhead=%s dataset=%s\n' \
        "$direction/kvh$kvhead/$dataset" "$direction" "$kvhead" "$dataset"
    print_command "${command[@]}"
    if ((DRY_RUN)); then
        return
    fi
    mkdir -p "$case_log_dir"
    "${command[@]}" 2>&1 | tee "$log_file"
}

echo "Dataset KVH matrix: datasets=$DATASETS, kvheads=$KVHEADS, target_tokens=$TARGET_TOKENS"
echo "Methods=$METHODS; SM configs=$SM_CONFIGS; GPUs=$NPROC_PER_NODE"
echo "Magi backward high-precision reduce: $MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE (1=FP32 reduction)"
launch_count=$((${#DIRECTION_LIST[@]} * ${#KVHEAD_LIST[@]} * ${#DATASET_LIST[@]}))
echo "Execution: one torchrun per direction/KVH/dataset; launches=$launch_count"

if ((DRY_RUN == 0)); then
    mkdir -p "$LOG_DIR"
    echo "Logs: $LOG_DIR"
fi

for direction in "${DIRECTION_LIST[@]}"; do
    for kvhead in "${KVHEAD_LIST[@]}"; do
        for dataset in "${DATASET_LIST[@]}"; do
            run_case "$direction" "$kvhead" "$dataset"
        done
    done
done
