#!/usr/bin/env bash

# Fixed five-method dataset-shaped load-balance runtime suite.
# Run inside a single-node allocation exposing 2, 4, or 8 SM90 GPUs.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEMO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$DEMO_DIR"

export CUDA_DEVICE_MAX_CONNECTIONS=8
export NCCL_CGA_CLUSTER_SIZE=1
export TORCH_NCCL_HIGH_PRIORITY=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

GPU_COUNTS=${GPU_COUNTS:-"8"}
DATASETS=${DATASETS:-"arxiv freelaw github pile prolong"}
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
ZEPPLIN_THRESHOLD=${ZEPPLIN_THRESHOLD:-8192}
MEGATRON_MAX_SEQLEN_PER_RANK=${MEGATRON_MAX_SEQLEN_PER_RANK:-8192}
MODE=${MODE:-causal}
QHEAD=${QHEAD:-32}
KVHEAD=${KVHEAD:-8}
HEADDIM=${HEADDIM:-128}
ALLGATHER_OVERLAPPING_HEADS_K_STRIDE=${ALLGATHER_OVERLAPPING_HEADS_K_STRIDE:-4}
SM_CONFIGS=${SM_CONFIGS:-"128:4,124:8,120:12,116:16"}
WARMUP_ITERS=${WARMUP_ITERS:-10}
NUM_ITERS=${NUM_ITERS:-40}
CHECK=${CHECK:-0}
COLLECT_MEGA_RING_STATS=${COLLECT_MEGA_RING_STATS:-0}
DRY_RUN=${DRY_RUN:-0}
TORCHRUN=${TORCHRUN:-torchrun}
LOG_DIR=${LOG_DIR:-"benchmark_logs/$(date +%Y%m%d-%H%M%S)"}
LOG_FILE=${LOG_FILE:-"$LOG_DIR/load_balance_${DIRECTION}.log"}

die() {
    echo "error: $*" >&2
    exit 1
}

case "$DIRECTION" in
    forward|backward) ;;
    *) die "DIRECTION must be forward or backward, got '$DIRECTION'" ;;
esac
case "$MODE" in
    noncausal|causal|both) ;;
    *) die "MODE must be noncausal, causal, or both, got '$MODE'" ;;
esac
if [[ "$DIRECTION" == backward && "$MODE" != causal ]]; then
    die "five-method backward suite supports only MODE=causal"
fi
case "$CHECK" in
    0) CHECK_ARGS=(--no-check) ;;
    1) CHECK_ARGS=(--check) ;;
    *) die "CHECK must be 0 or 1, got '$CHECK'" ;;
esac
case "$COLLECT_MEGA_RING_STATS" in
    0|1) ;;
    *) die "COLLECT_MEGA_RING_STATS must be 0 or 1, got '$COLLECT_MEGA_RING_STATS'" ;;
esac
if [[ "$DIRECTION" == backward && "$COLLECT_MEGA_RING_STATS" == 1 ]]; then
    die "COLLECT_MEGA_RING_STATS is supported only for DIRECTION=forward"
fi
case "$DRY_RUN" in
    0|1) ;;
    *) die "DRY_RUN must be 0 or 1, got '$DRY_RUN'" ;;
esac

for variable in TARGET_TOKENS BEAM_WIDTH FINALIST_COUNT MAX_REPAIR_ITERATIONS NUM_CASES \
    ZEPPLIN_THRESHOLD MEGATRON_MAX_SEQLEN_PER_RANK QHEAD KVHEAD HEADDIM \
    ALLGATHER_OVERLAPPING_HEADS_K_STRIDE WARMUP_ITERS NUM_ITERS; do
    value=${!variable}
    [[ "$value" =~ ^[0-9]+$ ]] || die "$variable must be an integer, got '$value'"
done
((TARGET_TOKENS > 0 && BEAM_WIDTH > 0 && FINALIST_COUNT > 0 && NUM_CASES > 0)) || \
    die "TARGET_TOKENS, BEAM_WIDTH, FINALIST_COUNT, and NUM_CASES must be positive"
((ZEPPLIN_THRESHOLD > 0 && MEGATRON_MAX_SEQLEN_PER_RANK > 0)) || \
    die "planner thresholds must be positive"
((QHEAD > 0 && KVHEAD > 0 && HEADDIM > 0 && NUM_ITERS > 0)) || \
    die "attention dimensions and NUM_ITERS must be positive"
((HEADDIM == 128)) || die "this Hopper suite requires HEADDIM=128"
((QHEAD % KVHEAD == 0)) || die "QHEAD must be divisible by KVHEAD"
((KVHEAD * HEADDIM == 1024)) || die "fused Mega Ring requires KVHEAD * HEADDIM == 1024"
((KVHEAD % ALLGATHER_OVERLAPPING_HEADS_K_STRIDE == 0)) || \
    die "ALLGATHER_OVERLAPPING_HEADS_K_STRIDE must divide KVHEAD"

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
        arxiv|github|pile|freelaw|prolong) ;;
        *) die "invalid dataset '$dataset'" ;;
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
        entrypoint=ring_test/load_balance_bench/benchmark_forward.py
    else
        entrypoint=ring_test/load_balance_bench/benchmark_backward.py
    fi
    local -a command=(
        "$TORCHRUN" --standalone --nproc_per_node="$world_size" "$entrypoint"
        --dataset "$dataset" --target-tokens "$TARGET_TOKENS"
        --compute-balance-tolerance "$COMPUTE_BALANCE_TOLERANCE"
        --token-balance-tolerance "$TOKEN_BALANCE_TOLERANCE"
        --beam-width "$BEAM_WIDTH" --finalist-count "$FINALIST_COUNT"
        --structure-threshold "$STRUCTURE_THRESHOLD"
        --max-repair-iterations "$MAX_REPAIR_ITERATIONS"
        --seed "$SEED" --num-cases "$NUM_CASES"
        --qhead "$QHEAD" --kvhead "$KVHEAD" --headdim "$HEADDIM"
        --allgather-overlapping-heads-k-stride "$ALLGATHER_OVERLAPPING_HEADS_K_STRIDE"
        --zepplin-threshold "$ZEPPLIN_THRESHOLD"
        --megatron-max-seqlen-per-rank "$MEGATRON_MAX_SEQLEN_PER_RANK"
        --sm-configs "$SM_CONFIGS" --warmup-iters "$WARMUP_ITERS" --num-iters "$NUM_ITERS"
        --mode "$MODE" "${CHECK_ARGS[@]}"
    )
    if [[ "$DIRECTION" == forward && "$COLLECT_MEGA_RING_STATS" == 1 ]]; then
        command+=(--collect-mega-ring-stats)
    fi

    if ((DRY_RUN)); then
        printf '\n================================================================================\n'
        printf '[load_balance_%s] dataset=%s GPUs=%s visible=%s\n' \
            "$DIRECTION" "$dataset" "$world_size" "$visible_devices"
        printf '================================================================================\n'
        print_command "$visible_devices" "${command[@]}"
        return
    fi
    {
        printf '\n================================================================================\n'
        printf '[load_balance_%s] dataset=%s GPUs=%s visible=%s\n' \
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
echo "Fixed results: native_megatron_hybrid_cp, native_zepplin, mega_ring_hybrid_br_pbs, mega_ring_hybrid_megatron_cp, mega_ring_hybrid_zepplin"
for world_size in "${GPU_COUNT_LIST[@]}"; do
    select_devices "$world_size"
    for dataset in "${DATASET_LIST[@]}"; do
        run_benchmark "$dataset" "$world_size" "$SELECTED_DEVICES"
    done
done

if ((DRY_RUN == 0)); then
    echo "Results written to $LOG_FILE"
fi
