#!/usr/bin/env bash

# Run the requested eight-GPU experiment matrix after all selected GPUs are idle.
# This is only an orchestrator; every measurement is delegated to an existing
# benchmark entry point in this repository.

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
export MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE=${MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE:-1}
DRY_RUN=${DRY_RUN:-0}
TORCHRUN=${TORCHRUN:-torchrun}
LOG_ROOT=${LOG_ROOT:-"$ROOT_DIR/benchmark_logs/experiment_queue"}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
RUN_DIR="$LOG_ROOT/$RUN_ID"
MASTER_LOG="$RUN_DIR/experiment_queue.log"

# The strict six-level ablation uses its own reproducible ArXiv suite instead
# of the matrix-wide NUM_CASES setting.
ABLATION_DATASET=${ABLATION_DATASET:-arxiv}
ABLATION_TARGET_TOKENS=${ABLATION_TARGET_TOKENS:-"131072 65536 262144"}
ABLATION_NUM_CASES=${ABLATION_NUM_CASES:-20}
ABLATION_SM_CONFIGS=${ABLATION_SM_CONFIGS:-"$SM_CONFIGS"}

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
[[ "$CHECK_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
    die "CHECK_INTERVAL_SECONDS must be a positive integer"
[[ "$START_DELAY_SECONDS" =~ ^[0-9]+$ ]] || \
    die "START_DELAY_SECONDS must be a non-negative integer"
[[ "$NUM_CASES" =~ ^[1-9][0-9]*$ ]] || die "NUM_CASES must be a positive integer"
[[ "$ABLATION_NUM_CASES" =~ ^[1-9][0-9]*$ ]] || \
    die "ABLATION_NUM_CASES must be a positive integer"
[[ "$ABLATION_SM_CONFIGS" =~ ^[1-9][0-9]*:[1-9][0-9]*(,[1-9][0-9]*:[1-9][0-9]*)*$ ]] || \
    die "ABLATION_SM_CONFIGS must contain comma-separated COMP:COMM pairs"
[[ "$ABLATION_DATASET" == arxiv ]] || \
    die "the strict forward ablation only supports ABLATION_DATASET=arxiv"
case "$MODE" in
    causal) ;;
    *) die "this forward/backward matrix requires MODE=causal" ;;
esac
case "$CHECK" in
    0) ABLATION_CHECK_ARG=--no-check ;;
    1) ABLATION_CHECK_ARG=--check ;;
    *) die "CHECK must be 0 or 1, got '$CHECK'" ;;
esac
case "$MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE" in
    0|1) ;;
    *) die "MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE must be 0 or 1, got '$MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE'" ;;
esac

ablation_targets_spec=${ABLATION_TARGET_TOKENS//,/ }
read -r -a ABLATION_TARGET_LIST <<< "$ablation_targets_spec"
((${#ABLATION_TARGET_LIST[@]} > 0)) || \
    die "ABLATION_TARGET_TOKENS must provide at least one target"
for target_tokens in "${ABLATION_TARGET_LIST[@]}"; do
    [[ "$target_tokens" =~ ^[1-9][0-9]*$ ]] || \
        die "ABLATION_TARGET_TOKENS must contain positive integers"
done

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
    log "Experiment queue interrupted by $signal"
    exit 130
}

trap 'handle_signal SIGINT' INT
trap 'handle_signal SIGTERM' TERM

all_gpus_idle() {
    local gpu output compact
    local idle=0
    for gpu in "${GPU_ID_LIST[@]}"; do
        if ! output=$(nvidia-smi --id="$gpu" --query-compute-apps=pid \
            --format=csv,noheader,nounits 2>&1); then
            log "GPU $gpu query failed; treating it as busy: $output"
            idle=1
            continue
        fi
        compact=${output//$'\n'/,}
        compact=${compact//[[:space:]]/}
        if [[ -n "$compact" ]]; then
            log "GPU $gpu is busy (compute PID(s): ${compact%,})"
            idle=1
        fi
    done
    return "$idle"
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

run_dataset_benchmark() {
    local tokens=$1
    local direction=$2
    local label="dataset_${tokens}_${direction}"
    local -a magi_backward_env=()
    if [[ "$direction" == backward ]]; then
        magi_backward_env+=(
            "MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE=$MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE"
        )
    fi
    run_experiment "$label" env "${magi_backward_env[@]}" \
        GPU_COUNTS=8 DATASETS="$DATASETS" DIRECTION="$direction" \
        TARGET_TOKENS="$tokens" NUM_CASES="$NUM_CASES" MODE="$MODE" \
        SEED="$SEED" TOKEN_BALANCE_TOLERANCE="$TOKEN_BALANCE_TOLERANCE" \
        QHEAD="$QHEAD" KVHEAD="$KVHEAD" HEADDIM="$HEADDIM" \
        SM_CONFIGS="$SM_CONFIGS" WARMUP_ITERS="$WARMUP_ITERS" \
        NUM_ITERS="$NUM_ITERS" \
        METHODS=all COLLECT_MEGA_RING_STATS=0 CHECK="$CHECK" TORCHRUN="$TORCHRUN" \
        LOG_DIR="$RUN_DIR/$label.results" LOG_FILE="$RUN_DIR/$label.results.log" \
        bash "$ROOT_DIR/benchmark_dataset.sh"
}

log "Experiment queue created"
log "Run directory: $RUN_DIR"
log "Datasets: $DATASETS; num_cases=$NUM_CASES; mode=$MODE; correctness_check=$CHECK"
log "Magi backward high-precision reduce: $MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE (1=FP32 reduction)"
log "Forward ablation: dataset=$ABLATION_DATASET; target_tokens=${ABLATION_TARGET_LIST[*]}; cases=$ABLATION_NUM_CASES; seed=$SEED; token_tolerance=$TOKEN_BALANCE_TOLERANCE; sm_configs=$ABLATION_SM_CONFIGS"
log "Scheduled start delay: ${START_DELAY_SECONDS}s; GPU polling interval: ${CHECK_INTERVAL_SECONDS}s; dry_run=$DRY_RUN"
if ((DRY_RUN)); then
    log "DRY_RUN=1: skipping the scheduled start delay"
else
    log "Delaying experiment queue start for ${START_DELAY_SECONDS}s"
    sleep "$START_DELAY_SECONDS"
    log "Scheduled start delay complete; beginning experiment queue"
fi

# 1-2. All registered baselines and all datasets at 64K, 128K, and 256K.
for tokens in 131072 65536 262144; do
    run_dataset_benchmark "$tokens" forward
    run_dataset_benchmark "$tokens" backward
done

# 3a. Runtime Mega Ring tile counters, forward only, at 128K.
run_experiment tile_analysis_131072_forward env \
    GPU_COUNTS=8 DATASETS="$DATASETS" DIRECTION=forward TARGET_TOKENS=131072 \
    NUM_CASES="$NUM_CASES" MODE="$MODE" \
    SEED="$SEED" TOKEN_BALANCE_TOLERANCE="$TOKEN_BALANCE_TOLERANCE" \
    QHEAD="$QHEAD" KVHEAD="$KVHEAD" HEADDIM="$HEADDIM" \
    SM_CONFIGS="$SM_CONFIGS" WARMUP_ITERS="$WARMUP_ITERS" \
    NUM_ITERS="$NUM_ITERS" \
    METHODS=mega_ring_all_cp,mega_ring_hybrid COLLECT_MEGA_RING_STATS=1 \
    CHECK="$CHECK" TORCHRUN="$TORCHRUN" \
    LOG_DIR="$RUN_DIR/tile_analysis_131072_forward.results" \
    LOG_FILE="$RUN_DIR/tile_analysis_131072_forward.results.log" \
    bash "$ROOT_DIR/benchmark_dataset.sh"

# 3b. Metadata-only theoretical load analysis for every registered baseline.
for direction in forward backward; do
    label="theoretical_load_131072_${direction}"
    magi_backward_env=()
    if [[ "$direction" == backward ]]; then
        magi_backward_env+=(
            "MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE=$MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE"
        )
    fi
    run_experiment "$label" env "${magi_backward_env[@]}" \
        GPU_COUNTS=8 DATASETS="$DATASETS" DIRECTION="$direction" \
        TARGET_TOKENS=131072 NUM_CASES="$NUM_CASES" MODE="$MODE" \
        METHODS=all TORCHRUN="$TORCHRUN" LOG_DIR="$RUN_DIR/$label.results" \
        LOG_FILE="$RUN_DIR/$label.results.log" \
        bash "$SCRIPT_DIR/benchmark_load_balance.sh"
done

# 4. Native Megatron/Zeppelin and Mega Ring with three placement algorithms.
for direction in forward backward; do
    label="load_balance_algorithms_131072_${direction}"
    magi_backward_env=()
    if [[ "$direction" == backward ]]; then
        magi_backward_env+=(
            "MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE=$MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE"
        )
    fi
    run_experiment "$label" env "${magi_backward_env[@]}" \
        GPU_COUNTS=8 DATASETS="$DATASETS" DIRECTION="$direction" \
        TARGET_TOKENS=131072 NUM_CASES="$NUM_CASES" MODE="$MODE" \
        COLLECT_MEGA_RING_STATS=0 CHECK="$CHECK" TORCHRUN="$TORCHRUN" \
        LOG_DIR="$RUN_DIR/$label.results" LOG_FILE="$RUN_DIR/$label.results.log" \
        bash "$ROOT_DIR/ring_test/load_balance_bench/run.sh"
done

# 5. Strict W8 six-level causal forward ablation on the ArXiv case suite.
for target_tokens in "${ABLATION_TARGET_LIST[@]}"; do
    run_experiment "forward_ablation_arxiv_${target_tokens}_${ABLATION_NUM_CASES}cases" env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        "$TORCHRUN" --standalone --nproc_per_node=8 \
        "$ROOT_DIR/ring_test/benchmark_forward_ablation.py" \
        --dataset "$ABLATION_DATASET" \
        --target-tokens "$target_tokens" \
        --num-cases "$ABLATION_NUM_CASES" \
        --seed "$SEED" \
        --token-balance-tolerance "$TOKEN_BALANCE_TOLERANCE" \
        --sm-configs "$ABLATION_SM_CONFIGS" \
        --warmup-iters "$WARMUP_ITERS" --num-iters "$NUM_ITERS" \
        --qhead "$QHEAD" --kvhead "$KVHEAD" --headdim "$HEADDIM" \
        --mode causal "$ABLATION_CHECK_ARG"
done

if ((${#FAILED_EXPERIMENTS[@]})); then
    log "Experiment queue finished with ${#FAILED_EXPERIMENTS[@]} failure(s): ${FAILED_EXPERIMENTS[*]}"
    exit 1
fi

log "Experiment queue finished successfully"
