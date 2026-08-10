#!/usr/bin/env bash

# Wait for all eight selected GPUs to become idle, then run the DCP Mega
# multi-rank correctness matrix once. The script never terminates other jobs.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-3}
IDLE_MEMORY_LIMIT_MIB=${IDLE_MEMORY_LIMIT_MIB:-64}
TORCHRUN=${TORCHRUN:-torchrun}
LOG_ROOT=${LOG_ROOT:-"$REPO_ROOT/benchmark_logs/dcp_mega_matrix_when_idle"}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
RUN_DIR="$LOG_ROOT/$RUN_ID"
MASTER_LOG="$RUN_DIR/matrix_queue.log"
CONSOLE_LOG="$RUN_DIR/matrix.console.log"

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

[[ "$CHECK_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
    die "CHECK_INTERVAL_SECONDS must be a positive integer"
[[ "$IDLE_MEMORY_LIMIT_MIB" =~ ^[0-9]+$ ]] || \
    die "IDLE_MEMORY_LIMIT_MIB must be a non-negative integer"

IFS=',' read -r -a GPU_ID_LIST <<< "$GPU_IDS"
((${#GPU_ID_LIST[@]} == 8)) || \
    die "GPU_IDS must list exactly 8 GPUs, got '$GPU_IDS'"
declare -A SEEN_GPU_IDS=()
for index in "${!GPU_ID_LIST[@]}"; do
    gpu=${GPU_ID_LIST[$index]//[[:space:]]/}
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "GPU ID must be a non-negative integer, got '$gpu'"
    [[ -z ${SEEN_GPU_IDS[$gpu]+x} ]] || die "GPU_IDS contains duplicate '$gpu'"
    SEEN_GPU_IDS[$gpu]=1
    GPU_ID_LIST[$index]=$gpu
done
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_ID_LIST[*]}")
export CUDA_VISIBLE_DEVICES

for command in nvidia-smi flock sudo tee "$TORCHRUN"; do
    command -v "$command" >/dev/null 2>&1 || die "required command not found: $command"
done
for gpu in "${GPU_ID_LIST[@]}"; do
    nvidia-smi --id="$gpu" --query-gpu=index --format=csv,noheader,nounits \
        >/dev/null 2>&1 || die "GPU '$gpu' is not visible to nvidia-smi"
done
[[ -f "$REPO_ROOT/scripts/test_min_fa3/test_dcp_mega_varlen_multi_rank.py" ]] || \
    die "DCP Mega matrix test entry point was not found"

mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/.matrix_queue.lock"
flock -n 9 || die "another DCP Mega matrix queue is already running"
mkdir -p "$RUN_DIR"
touch "$MASTER_LOG" "$CONSOLE_LOG"

handle_signal() {
    local signal=$1
    log "Matrix queue interrupted by $signal"
    exit 130
}

trap 'handle_signal SIGINT' INT
trap 'handle_signal SIGTERM' TERM

# Return success only when every selected GPU has no compute process, zero
# reported utilization, and no more than the configured driver-memory residue.
all_gpus_idle() {
    local gpu apps metrics index uuid util memory app_uuid pid pids
    local busy=0
    local found=0
    local -a states=()
    local -A uuid_by_gpu=()
    local -A util_by_gpu=()
    local -A memory_by_gpu=()
    local -A pids_by_uuid=()

    if ! metrics=$(nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used \
        --format=csv,noheader,nounits 2>&1); then
        log "GPU metric query failed; treating all GPUs as busy: $metrics"
        return 1
    fi
    if ! apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid \
        --format=csv,noheader,nounits 2>&1); then
        log "GPU process query failed; treating all GPUs as busy: $apps"
        return 1
    fi

    while IFS=',' read -r index uuid util memory; do
        index=${index//[[:space:]]/}
        [[ -n ${SEEN_GPU_IDS[$index]+x} ]] || continue
        uuid=${uuid//[[:space:]]/}
        util=${util//[[:space:]]/}
        memory=${memory//[[:space:]]/}
        if [[ -z "$uuid" || ! "$util" =~ ^[0-9]+$ || ! "$memory" =~ ^[0-9]+$ ]]; then
            log "Could not parse nvidia-smi metrics for GPU $index"
            return 1
        fi
        uuid_by_gpu[$index]=$uuid
        util_by_gpu[$index]=$util
        memory_by_gpu[$index]=$memory
        ((found += 1))
    done <<< "$metrics"
    ((found == ${#GPU_ID_LIST[@]})) || {
        log "Expected metrics for ${#GPU_ID_LIST[@]} GPUs, found $found"
        return 1
    }

    while IFS=',' read -r app_uuid pid; do
        app_uuid=${app_uuid//[[:space:]]/}
        pid=${pid//[[:space:]]/}
        [[ -n "$app_uuid" && -n "$pid" ]] || continue
        [[ "$pid" =~ ^[0-9]+$ ]] || {
            log "Could not parse nvidia-smi compute PID '$pid'"
            return 1
        }
        if [[ -n ${pids_by_uuid[$app_uuid]:-} ]]; then
            pids_by_uuid[$app_uuid]+=",$pid"
        else
            pids_by_uuid[$app_uuid]=$pid
        fi
    done <<< "$apps"

    for gpu in "${GPU_ID_LIST[@]}"; do
        uuid=${uuid_by_gpu[$gpu]}
        util=${util_by_gpu[$gpu]}
        memory=${memory_by_gpu[$gpu]}
        pids=${pids_by_uuid[$uuid]:-none}
        states+=("gpu${gpu}:util=${util}%,mem=${memory}MiB,pids=${pids}")
        if [[ "$pids" != none ]] || ((util != 0)) || ((memory > IDLE_MEMORY_LIMIT_MIB)); then
            busy=1
        fi
    done

    log "GPU state: ${states[*]}"
    return "$busy"
}

ensure_exclusive_process() {
    local gpu mode
    for gpu in "${GPU_ID_LIST[@]}"; do
        mode=$(nvidia-smi --id="$gpu" --query-gpu=compute_mode \
            --format=csv,noheader,nounits 2>&1) || \
            die "failed to query compute mode for GPU $gpu: $mode"
        mode=${mode//[[:space:]]/}
        if [[ "$mode" != "Exclusive_Process" ]]; then
            log "GPU $gpu is in '$mode'; switching it to EXCLUSIVE_PROCESS"
            sudo -n nvidia-smi --id="$gpu" -c EXCLUSIVE_PROCESS 2>&1 | tee -a "$MASTER_LOG"
        fi
    done
}

wait_for_all_gpus() {
    while true; do
        if all_gpus_idle; then
            log "All selected GPUs appear idle; confirming Exclusive Process mode"
            ensure_exclusive_process
            if all_gpus_idle; then
                log "All selected GPUs are still idle after the mode check"
                return 0
            fi
            log "A GPU became busy during the final check"
        fi
        log "Waiting ${CHECK_INTERVAL_SECONDS}s before the next check"
        sleep "$CHECK_INTERVAL_SECONDS"
    done
}

MATRIX_COMMAND=(
    "$TORCHRUN"
    --standalone
    --nproc_per_node=8
    scripts/test_min_fa3/test_dcp_mega_varlen_multi_rank.py
    --matrix
)

log "DCP Mega matrix queue started"
log "Run directory: $RUN_DIR"
log "Selected GPUs: $CUDA_VISIBLE_DEVICES; polling interval: ${CHECK_INTERVAL_SECONDS}s"
wait_for_all_gpus
print_command "${MATRIX_COMMAND[@]}"
log "Starting DCP Mega correctness matrix"

set +e
"${MATRIX_COMMAND[@]}" 2>&1 | tee -a "$CONSOLE_LOG" "$MASTER_LOG"
status=${PIPESTATUS[0]}
set -e

if ((status == 0)); then
    log "DCP Mega correctness matrix completed successfully"
else
    log "DCP Mega correctness matrix failed with exit status $status"
fi
exit "$status"
