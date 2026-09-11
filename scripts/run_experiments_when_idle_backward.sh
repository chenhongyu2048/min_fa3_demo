#!/usr/bin/env bash

# Run only the backward portions of the eight-GPU experiment queue after all
# selected GPUs are idle. Runtime and theoretical dataset suites request every
# registered backward method; the placement suite retains its fixed five-method
# comparison.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
cd "$ROOT_DIR"

GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}}
CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-300}
START_DELAY_SECONDS=${START_DELAY_SECONDS:-28800}
DATASETS=${DATASETS:-"arxiv freelaw github pile prolong"}
NUM_CASES=${NUM_CASES:-20}
MODE=${MODE:-causal}
CHECK=${CHECK:-0}
SEED=${SEED:-0}
TOKEN_BALANCE_TOLERANCE=${TOKEN_BALANCE_TOLERANCE:-0.05}
QHEAD=${QHEAD:-32}
KVHEAD=${KVHEAD:-8}
HEADDIM=${HEADDIM:-128}
SM_CONFIGS=${SM_CONFIGS:-"128:4,124:8,120:12,116:16"}
WARMUP_ITERS=${WARMUP_ITERS:-10}
NUM_ITERS=${NUM_ITERS:-40}
MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE=${MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE:-1}
DRY_RUN=${DRY_RUN:-0}
TORCHRUN=${TORCHRUN:-torchrun}
LOG_ROOT=${LOG_ROOT:-"$ROOT_DIR/benchmark_logs/experiment_queue"}
RUN_ID=${RUN_ID:-backward-$(date +%Y%m%d-%H%M%S)}
RUN_DIR="$LOG_ROOT/$RUN_ID"
MASTER_LOG="$RUN_DIR/experiment_queue_backward.log"

die() {
    echo "error: $*" >&2
    exit 1
}

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$MASTER_LOG"
}

print_command() {
    printf '[%s] command:' "$(timestamp)" | tee -a "$MASTER_LOG"
    printf ' %q' "$@" | tee -a "$MASTER_LOG"
    printf '\n' | tee -a "$MASTER_LOG"
}

case "$DRY_RUN" in
    0|1) ;;
    *) die "DRY_RUN must be 0 or 1, got '$DRY_RUN'" ;;
esac
case "$CHECK" in
    0|1) ;;
    *) die "CHECK must be 0 or 1, got '$CHECK'" ;;
esac
case "$MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE" in
    0|1) ;;
    *) die "MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE must be 0 or 1, got '$MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE'" ;;
esac
[[ "$CHECK_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
    die "CHECK_INTERVAL_SECONDS must be a positive integer"
[[ "$START_DELAY_SECONDS" =~ ^[0-9]+$ ]] || \
    die "START_DELAY_SECONDS must be a non-negative integer"
[[ "$NUM_CASES" =~ ^[1-9][0-9]*$ ]] || die "NUM_CASES must be a positive integer"
[[ "$SM_CONFIGS" =~ ^[1-9][0-9]*:[1-9][0-9]*(,[1-9][0-9]*:[1-9][0-9]*)*$ ]] || \
    die "SM_CONFIGS must contain comma-separated COMP:COMM pairs"
[[ "$MODE" == causal ]] || die "backward experiments require MODE=causal"

IFS=',' read -r -a GPU_ID_LIST <<< "$GPU_IDS"
((${#GPU_ID_LIST[@]} == 8)) || \
    die "GPU_IDS/CUDA_VISIBLE_DEVICES must list exactly 8 GPUs, got '$GPU_IDS'"
declare -A SEEN_GPU_IDS=()
for index in "${!GPU_ID_LIST[@]}"; do
    gpu=${GPU_ID_LIST[$index]//[[:space:]]/}
    [[ -n "$gpu" ]] || die "GPU_IDS contains an empty GPU identifier"
    [[ -z ${SEEN_GPU_IDS[$gpu]+x} ]] || die "GPU_IDS contains duplicate '$gpu'"
    SEEN_GPU_IDS[$gpu]=1
    GPU_ID_LIST[$index]=$gpu
done
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_ID_LIST[*]}")

for command in nvidia-smi flock tee "$TORCHRUN"; do
    command -v "$command" >/dev/null 2>&1 || die "required command not found: $command"
done
for gpu in "${GPU_ID_LIST[@]}"; do
    nvidia-smi --id="$gpu" --query-gpu=index --format=csv,noheader,nounits \
        >/dev/null 2>&1 || die "GPU '$gpu' is not visible to nvidia-smi"
done
[[ -f "$ROOT_DIR/benchmark_dataset.sh" ]] || die "benchmark_dataset.sh is missing"
[[ -f "$SCRIPT_DIR/benchmark_load_balance.sh" ]] || \
    die "benchmark_load_balance.sh is missing"
[[ -f "$ROOT_DIR/ring_test/load_balance_bench/run.sh" ]] || \
    die "ring_test/load_balance_bench/run.sh is missing"

mkdir -p "$RUN_DIR"
touch "$MASTER_LOG"
exec 9>"$LOG_ROOT/.experiment_queue.lock"
flock -n 9 || die "another experiment queue is already running (lock: $LOG_ROOT/.experiment_queue.lock)"

handle_signal() {
    local signal=$1
    log "Backward experiment queue interrupted by $signal"
    exit 130
}

trap 'handle_signal SIGINT' INT
trap 'handle_signal SIGTERM' TERM

all_gpus_idle() {
    local gpu output compact
    local busy=0
    for gpu in "${GPU_ID_LIST[@]}"; do
        if ! output=$(nvidia-smi --id="$gpu" --query-compute-apps=pid \
            --format=csv,noheader,nounits 2>&1); then
            log "GPU $gpu query failed; treating it as busy: $output"
            busy=1
            continue
        fi
        compact=${output//$'\n'/,}
        compact=${compact//[[:space:]]/}
        if [[ -n "$compact" ]]; then
            log "GPU $gpu is busy (compute PID(s): ${compact%,})"
            busy=1
        fi
    done
    return "$busy"
}

wait_for_all_gpus() {
    while ! all_gpus_idle; do
        log "At least one selected GPU is busy; checking again in ${CHECK_INTERVAL_SECONDS}s"
        sleep "$CHECK_INTERVAL_SECONDS"
    done
    log "All selected GPUs are idle: $CUDA_VISIBLE_DEVICES"
}

declare -a FAILED_EXPERIMENTS=()

run_experiment() {
    local label=$1
    shift
    local console_log="$RUN_DIR/${label}.console.log"
    local status

    log "Queued experiment: $label"
    print_command "$@"
    if ((DRY_RUN)); then
        return 0
    fi

    wait_for_all_gpus
    log "Starting experiment: $label"
    set +e
    "$@" 2>&1 | tee -a "$console_log" "$MASTER_LOG"
    status=${PIPESTATUS[0]}
    set -e
    if ((status == 0)); then
        log "Completed experiment: $label"
    else
        log "FAILED experiment: $label (exit status $status)"
        FAILED_EXPERIMENTS+=("$label:$status")
    fi
}

run_dataset_backward() {
    local tokens=$1
    local label="dataset_${tokens}_backward"
    run_experiment "$label" env \
        MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE="$MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE" \
        GPU_COUNTS=8 DATASETS="$DATASETS" DIRECTION=backward \
        TARGET_TOKENS="$tokens" NUM_CASES="$NUM_CASES" MODE="$MODE" \
        SEED="$SEED" TOKEN_BALANCE_TOLERANCE="$TOKEN_BALANCE_TOLERANCE" \
        QHEAD="$QHEAD" KVHEAD="$KVHEAD" HEADDIM="$HEADDIM" \
        SM_CONFIGS="$SM_CONFIGS" WARMUP_ITERS="$WARMUP_ITERS" \
        NUM_ITERS="$NUM_ITERS" METHODS=all COLLECT_MEGA_RING_STATS=0 \
        CHECK="$CHECK" TORCHRUN="$TORCHRUN" \
        LOG_DIR="$RUN_DIR/$label.results" LOG_FILE="$RUN_DIR/$label.results.log" \
        bash "$ROOT_DIR/benchmark_dataset.sh"
}

log "Backward experiment queue created"
log "Run directory: $RUN_DIR"
log "Datasets: $DATASETS; num_cases=$NUM_CASES; mode=$MODE; correctness_check=$CHECK"
log "Registered backward methods: all"
log "Magi backward high-precision reduce: $MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE (1=FP32 reduction)"
log "Scheduled start delay: ${START_DELAY_SECONDS}s; GPU polling interval: ${CHECK_INTERVAL_SECONDS}s; dry_run=$DRY_RUN"
if ((DRY_RUN)); then
    log "DRY_RUN=1: skipping the scheduled start delay"
else
    log "Delaying backward experiment queue start for ${START_DELAY_SECONDS}s"
    sleep "$START_DELAY_SECONDS"
    log "Scheduled start delay complete; beginning backward experiment queue"
fi

# 1. All registered backward methods and all datasets at 128K, 64K, and 256K.
for tokens in 131072 65536 262144; do
    run_dataset_backward "$tokens"
done

# 2. Metadata-only backward load analysis for every registered method at 128K.
run_experiment theoretical_load_131072_backward env \
    MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE="$MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE" \
    GPU_COUNTS=8 DATASETS="$DATASETS" DIRECTION=backward \
    TARGET_TOKENS=131072 NUM_CASES="$NUM_CASES" MODE="$MODE" \
    METHODS=all TORCHRUN="$TORCHRUN" \
    LOG_DIR="$RUN_DIR/theoretical_load_131072_backward.results" \
    LOG_FILE="$RUN_DIR/theoretical_load_131072_backward.results.log" \
    bash "$SCRIPT_DIR/benchmark_load_balance.sh"

# 3. Fixed backward placement comparison: native Megatron/Zeppelin and the
# three MegaRing hybrid placement algorithms.
run_experiment load_balance_algorithms_131072_backward env \
    MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE="$MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE" \
    GPU_COUNTS=8 DATASETS="$DATASETS" DIRECTION=backward \
    TARGET_TOKENS=131072 NUM_CASES="$NUM_CASES" MODE="$MODE" \
    SEED="$SEED" TOKEN_BALANCE_TOLERANCE="$TOKEN_BALANCE_TOLERANCE" \
    QHEAD="$QHEAD" KVHEAD="$KVHEAD" HEADDIM="$HEADDIM" \
    SM_CONFIGS="$SM_CONFIGS" WARMUP_ITERS="$WARMUP_ITERS" \
    NUM_ITERS="$NUM_ITERS" COLLECT_MEGA_RING_STATS=0 CHECK="$CHECK" \
    TORCHRUN="$TORCHRUN" \
    LOG_DIR="$RUN_DIR/load_balance_algorithms_131072_backward.results" \
    LOG_FILE="$RUN_DIR/load_balance_algorithms_131072_backward.results.log" \
    bash "$ROOT_DIR/ring_test/load_balance_bench/run.sh"

if ((${#FAILED_EXPERIMENTS[@]})); then
    log "Backward experiment queue finished with ${#FAILED_EXPERIMENTS[@]} failure(s): ${FAILED_EXPERIMENTS[*]}"
    exit 1
fi

log "Backward experiment queue finished successfully"
