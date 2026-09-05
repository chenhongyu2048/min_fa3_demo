#!/usr/bin/env bash

# CP: Ten-method uniform-workload forward/backward benchmark on one eight-GPU SM90 node.
# Each case keeps total context tokens fixed and sets S=context_length/batch_size.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

# Use the checkout-local environment when present.  The historical benchmark
# used an absolute path from the original author's workspace, which is not
# available in every clone and is unnecessary when TORCHRUN is supplied.
VENV_ACTIVATE=${VENV_ACTIVATE:-"$ROOT_DIR/.venv/bin/activate"}
if [[ -f "$VENV_ACTIVATE" ]]; then
    source "$VENV_ACTIVATE"
fi

# Match the distributed-attention settings used by benchmark_dataset.sh.
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NCCL_CGA_CLUSTER_SIZE=${NCCL_CGA_CLUSTER_SIZE:-1}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE=${MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

WORLD_SIZE=8
CONTEXT_LENGTHS=${CONTEXT_LENGTHS:-"65536 131072 262144 "}
BATCH_SIZES=${BATCH_SIZES:-"1 2 4 8 16"}
DIRECTION=${DIRECTION:-both}
MODE=${MODE:-causal}
METHODS=${METHODS:-"allgather_attention,llama3_allgather_attention,ulysses,usp,fa3_ring,megatron_hybrid_cp,magi_attention,zeppelin,mega_ring_all_cp,mega_ring_hybrid"}
QHEAD=${QHEAD:-32}
KVHEAD=${KVHEAD:-8}
HEADDIM=${HEADDIM:-128}
ALLGATHER_OVERLAPPING_HEADS_K_STRIDE=${ALLGATHER_OVERLAPPING_HEADS_K_STRIDE:-4}
ZEPPELIN_THRESHOLD=${ZEPPELIN_THRESHOLD:-4096}
MEGATRON_MAX_SEQLEN_PER_RANK=${MEGATRON_MAX_SEQLEN_PER_RANK:-8192}
MAGI_OVERLAP_DEGREE=${MAGI_OVERLAP_DEGREE:-2}
SM_CONFIGS=${SM_CONFIGS:-"128:4,124:8,120:12,116:16"}
WARMUP_ITERS=${WARMUP_ITERS:-10}
NUM_ITERS=${NUM_ITERS:-40}
SEED=${SEED:-0}
CHECK=${CHECK:-0}
DRY_RUN=${DRY_RUN:-0}
TORCHRUN=${TORCHRUN:-torchrun}
GPU_IDLE_CHECK_INTERVAL_SECONDS=${GPU_IDLE_CHECK_INTERVAL_SECONDS:-5}
LOG_DIR=${LOG_DIR:-"benchmark_logs/$(date +%Y%m%d-%H%M%S)-uniform-$DIRECTION"}
LOG_FILE=${LOG_FILE:-"$LOG_DIR/benchmark_uniform_${DIRECTION}.log"}

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

case "$MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE" in
    0|1) ;;
    *) die "MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE must be 0 or 1, got '$MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE'" ;;
esac

case "$DIRECTION" in
    forward) DIRECTION_LIST=(forward) ;;
    backward) DIRECTION_LIST=(backward) ;;
    both) DIRECTION_LIST=(forward backward) ;;
    *) die "DIRECTION must be forward, backward, or both, got '$DIRECTION'" ;;
esac

case "$MODE" in
    noncausal|causal|both) ;;
    *) die "MODE must be noncausal, causal, or both, got '$MODE'" ;;
esac
if [[ "$DIRECTION" != forward && "$MODE" != causal ]]; then
    die "DIRECTION=$DIRECTION includes topology backward, which supports only MODE=causal"
fi

[[ "$QHEAD" =~ ^[1-9][0-9]*$ ]] || die "QHEAD must be a positive integer"
[[ "$KVHEAD" =~ ^[1-9][0-9]*$ ]] || die "KVHEAD must be a positive integer"
[[ "$HEADDIM" == 128 ]] || die "HEADDIM must be 128"
((QHEAD % KVHEAD == 0)) || die "QHEAD must be divisible by KVHEAD"
((KVHEAD * HEADDIM == 1024)) || die "the fused forward/backward methods require KVHEAD * HEADDIM == 1024"
[[ "$ALLGATHER_OVERLAPPING_HEADS_K_STRIDE" =~ ^[1-9][0-9]*$ ]] || \
    die "ALLGATHER_OVERLAPPING_HEADS_K_STRIDE must be a positive integer"
((KVHEAD % ALLGATHER_OVERLAPPING_HEADS_K_STRIDE == 0)) || \
    die "ALLGATHER_OVERLAPPING_HEADS_K_STRIDE must divide KVHEAD"
[[ "$ZEPPELIN_THRESHOLD" =~ ^[1-9][0-9]*$ ]] || \
    die "ZEPPELIN_THRESHOLD must be a positive integer"
[[ "$MEGATRON_MAX_SEQLEN_PER_RANK" =~ ^[1-9][0-9]*$ ]] || \
    die "MEGATRON_MAX_SEQLEN_PER_RANK must be a positive integer"
[[ "$MAGI_OVERLAP_DEGREE" =~ ^[1-8]$ ]] || \
    die "MAGI_OVERLAP_DEGREE must be an integer in [1, 8]"
[[ "$WARMUP_ITERS" =~ ^[0-9]+$ ]] || die "WARMUP_ITERS must be a non-negative integer"
[[ "$NUM_ITERS" =~ ^[1-9][0-9]*$ ]] || die "NUM_ITERS must be a positive integer"
[[ "$GPU_IDLE_CHECK_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
    die "GPU_IDLE_CHECK_INTERVAL_SECONDS must be a positive integer"
[[ "$SM_CONFIGS" =~ ^[1-9][0-9]*:[1-9][0-9]*(,[1-9][0-9]*:[1-9][0-9]*)*$ ]] || \
    die "SM_CONFIGS must contain comma-separated COMP:COMM pairs"

context_lengths_spec=${CONTEXT_LENGTHS//,/ }
read -r -a CONTEXT_LENGTH_LIST <<< "$context_lengths_spec"
((${#CONTEXT_LENGTH_LIST[@]} > 0)) || die "CONTEXT_LENGTHS must not be empty"
for context_length in "${CONTEXT_LENGTH_LIST[@]}"; do
    case "$context_length" in
        65536|131072|262144) ;;
        *) die "CONTEXT_LENGTHS must contain only 65536, 131072, or 262144, got '$context_length'" ;;
    esac
done

batch_sizes_spec=${BATCH_SIZES//,/ }
read -r -a BATCH_SIZE_LIST <<< "$batch_sizes_spec"
((${#BATCH_SIZE_LIST[@]} > 0)) || die "BATCH_SIZES must not be empty"
for batch_size in "${BATCH_SIZE_LIST[@]}"; do
    case "$batch_size" in
        1|2|4|8|16) ;;
        *) die "BATCH_SIZES must contain only 1, 2, 4, 8, or 16, got '$batch_size'" ;;
    esac
done

if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    IFS=',' read -r -a VISIBLE_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
    ((${#VISIBLE_DEVICES[@]} >= WORLD_SIZE)) || die \
        "CUDA_VISIBLE_DEVICES exposes ${#VISIBLE_DEVICES[@]} GPUs, but $WORLD_SIZE are required"
else
    VISIBLE_DEVICES=(0 1 2 3 4 5 6 7)
fi
selected_devices=("${VISIBLE_DEVICES[@]:0:WORLD_SIZE}")
SELECTED_DEVICES=$(IFS=,; echo "${selected_devices[*]}")
CONTEXT_LENGTHS_CSV=$(IFS=,; echo "${CONTEXT_LENGTH_LIST[*]}")
BATCH_SIZES_CSV=$(IFS=,; echo "${BATCH_SIZE_LIST[*]}")

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

all_gpus_idle() {
    local gpu output compact
    local busy=0

    for gpu in "${selected_devices[@]}"; do
        if ! output=$(nvidia-smi --id="$gpu" --query-compute-apps=pid \
            --format=csv,noheader,nounits 2>&1); then
            printf '[%s] GPU %s query failed; treating it as busy: %s\n' \
                "$(timestamp)" "$gpu" "$output"
            busy=1
            continue
        fi
        compact=${output//$'\n'/,}
        compact=${compact//[[:space:]]/}
        if [[ -n "$compact" ]]; then
            printf '[%s] GPU %s is busy (compute PID(s): %s)\n' \
                "$(timestamp)" "$gpu" "${compact%,}"
            busy=1
        fi
    done
    return "$busy"
}

wait_for_all_gpus() {
    command -v nvidia-smi >/dev/null 2>&1 || die "required command not found: nvidia-smi"
    while ! all_gpus_idle; do
        printf '[%s] At least one selected GPU is busy; checking again in %ss\n' \
            "$(timestamp)" "$GPU_IDLE_CHECK_INTERVAL_SECONDS"
        sleep "$GPU_IDLE_CHECK_INTERVAL_SECONDS"
    done
    printf '[%s] All selected GPUs are idle; starting benchmark: %s\n' \
        "$(timestamp)" "$SELECTED_DEVICES"
}

print_command() {
    printf 'CUDA_VISIBLE_DEVICES=%q' "$1"
    shift
    printf ' %q' "$@"
    printf '\n'
}

if ((DRY_RUN == 0)); then
    wait_for_all_gpus
fi

run_direction() {
    local direction=$1
    local direction_index=$2
    local direction_count=$3
    local entrypoint
    local -a command

    if [[ "$direction" == forward ]]; then
        entrypoint=ring_test/benchmark_topology_forward.py
    else
        entrypoint=ring_test/benchmark_topology_backward.py
    fi
    command=(
        "$TORCHRUN" --standalone --nproc_per_node="$WORLD_SIZE"
        "$entrypoint"
        --context-lengths "$CONTEXT_LENGTHS_CSV"
        --batch-sizes "$BATCH_SIZES_CSV"
        --qhead "$QHEAD" --kvhead "$KVHEAD" --headdim "$HEADDIM"
        --allgather-overlapping-heads-k-stride "$ALLGATHER_OVERLAPPING_HEADS_K_STRIDE"
        --mode "$MODE" --methods "$METHODS"
        --zeppelin-threshold "$ZEPPELIN_THRESHOLD"
        --megatron-max-seqlen-per-rank "$MEGATRON_MAX_SEQLEN_PER_RANK"
        --magi-overlap-degree "$MAGI_OVERLAP_DEGREE"
        --sm-configs "$SM_CONFIGS"
        --warmup-iters "$WARMUP_ITERS" --num-iters "$NUM_ITERS"
        --seed "$SEED"
        "${CHECK_ARGS[@]}"
    )

    if ((DRY_RUN)); then
        printf '\n================================================================================\n'
        printf '[uniform_%s %d/%d] %d workloads in one torchrun\n' \
            "$direction" "$direction_index" "$direction_count" "$workload_cases"
        printf '================================================================================\n'
        print_command "$SELECTED_DEVICES" "${command[@]}"
        return
    fi

    {
        printf '\n================================================================================\n'
        printf '[uniform_%s %d/%d] %d workloads in one torchrun\n' \
            "$direction" "$direction_index" "$direction_count" "$workload_cases"
        printf '================================================================================\n'
        print_command "$SELECTED_DEVICES" "${command[@]}"
    } | tee -a "$LOG_FILE"
    CUDA_VISIBLE_DEVICES="$SELECTED_DEVICES" "${command[@]}" 2>&1 | tee -a "$LOG_FILE"
}

if ((DRY_RUN == 0)); then
    mkdir -p "$(dirname -- "$LOG_FILE")"
    : > "$LOG_FILE"
fi

workload_cases=$((${#CONTEXT_LENGTH_LIST[@]} * ${#BATCH_SIZE_LIST[@]}))
direction_count=${#DIRECTION_LIST[@]}
total_cases=$((workload_cases * direction_count))
echo "Log: $LOG_FILE"
echo "GPUs: $SELECTED_DEVICES (world_size=$WORLD_SIZE)"
echo "Contexts: ${CONTEXT_LENGTH_LIST[*]}"
echo "Batch sizes: ${BATCH_SIZE_LIST[*]}"
echo "Directions: ${DIRECTION_LIST[*]}"
echo "Workloads per direction: $workload_cases; total benchmark points: $total_cases; torchrun launches: $direction_count"
echo "Methods: $METHODS"
echo "Config: direction=$DIRECTION, mode=$MODE, QH=$QHEAD, KVH=$KVHEAD, D=$HEADDIM, sm_configs=$SM_CONFIGS, warmup=$WARMUP_ITERS, iters=$NUM_ITERS, check=$CHECK"
echo "Magi backward high-precision reduce: $MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE (1=FP32 reduction)"

direction_index=0
for direction in "${DIRECTION_LIST[@]}"; do
    ((direction_index += 1))
    run_direction "$direction" "$direction_index" "$direction_count"
done

if ((DRY_RUN == 0)); then
    echo "Results written to $LOG_FILE"
fi
