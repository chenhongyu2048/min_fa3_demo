#!/usr/bin/env bash

# Profile Mega phase timestamps and graph-only baseline phase breakdowns for
# the arrival-rate=4 trace at DCP sizes 2, 4, and 8.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
cd "$ROOT_DIR"

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NCCL_CGA_CLUSTER_SIZE=${NCCL_CGA_CLUSTER_SIZE:-1}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

PYTHON=${PYTHON:-"$ROOT_DIR/.venv/bin/python"}
TORCHRUN=${TORCHRUN:-"$ROOT_DIR/.venv/bin/torchrun"}
TRACE_CONFIG=${TRACE_CONFIG:-dcp_test/trace/example_config.json}
BATCH_CONFIG=${BATCH_CONFIG:-dcp_test/configs/dcp_mega_six_loads.json}
LOG_DIR=${LOG_DIR:-"$ROOT_DIR/benchmark_logs/bench_dcp/$(date +%Y%m%d-%H%M%S)-arrival4-phases-graph"}
RESULT_DIR=${RESULT_DIR:-"$LOG_DIR/results"}
MASTER_LOG=${MASTER_LOG:-"$LOG_DIR/benchmark.log"}
MEGA_NUM_COMM_SMS=${MEGA_NUM_COMM_SMS:-"4,8,12,16,20"}
NUM_CASES=${NUM_CASES:-100}
WARMUP=${WARMUP:-40}
ITERS=${ITERS:-60}

readonly ARRIVAL_RATE=4
TP_SIZE=${TP_SIZE:-8}
QHEAD=${QHEAD:-}
KVHEAD=${KVHEAD:-}
readonly BASELINE_IMPLEMENTATIONS="ours,vllm,sglang,full"
DCP_SIZES_SPEC=${DCP_SIZES:-"2,4,8"}
DCP_SIZES_SPEC=${DCP_SIZES_SPEC//,/ }
read -r -a DCP_SIZES <<< "$DCP_SIZES_SPEC"

case "$TP_SIZE" in
    2|4|8) ;;
    *) printf 'error: TP_SIZE must be 2, 4, or 8, got %q\n' "$TP_SIZE" >&2; exit 2 ;;
esac
if [[ -n "$QHEAD" || -n "$KVHEAD" ]]; then
    [[ "$QHEAD" =~ ^[1-9][0-9]*$ && "$KVHEAD" =~ ^[1-9][0-9]*$ ]] || {
        printf 'error: QHEAD and KVHEAD must be positive integers and provided together\n' >&2
        exit 2
    }
fi
for dcp_size in "${DCP_SIZES[@]}"; do
    case "$dcp_size" in
        2|4|8) ;;
        *) printf 'error: DCP_SIZES must contain only 2, 4, or 8\n' >&2; exit 2 ;;
    esac
    ((dcp_size <= TP_SIZE && TP_SIZE % dcp_size == 0)) || {
        printf 'error: every DCP size must divide TP_SIZE=%s and cannot exceed it\n' "$TP_SIZE" >&2
        exit 2
    }
done
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    case "$TP_SIZE" in
        2) CUDA_VISIBLE_DEVICES=0,1 ;;
        4) CUDA_VISIBLE_DEVICES=0,1,2,3 ;;
        8) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ;;
    esac
fi
export CUDA_VISIBLE_DEVICES

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$MASTER_LOG"
}

run_logged() {
    local label=$1
    local log_path=$2
    shift 2
    local -a command=("$@")

    log "Starting $label"
    printf 'CUDA_VISIBLE_DEVICES=%q' "$CUDA_VISIBLE_DEVICES" | tee -a "$MASTER_LOG" "$log_path"
    printf ' %q' "${command[@]}" | tee -a "$MASTER_LOG" "$log_path"
    printf '\n' | tee -a "$MASTER_LOG" "$log_path"
    if "${command[@]}" 2>&1 | tee -a "$MASTER_LOG" "$log_path"; then
        log "Completed $label"
    else
        local status=${PIPESTATUS[0]}
        log "Failed $label with status $status"
        return "$status"
    fi
}

finish() {
    local status=$?
    trap - EXIT
    if ((status == 0)); then
        log "All arrival=$ARRIVAL_RATE phase-timing launches completed"
    else
        log "Phase-timing run exited with status $status"
    fi
    exit "$status"
}
trap finish EXIT

mkdir -p "$RESULT_DIR" "$LOG_DIR/logs"
{
    printf 'Mega DCP phase-timing benchmark\n'
    printf 'started=%s\n' "$(date --iso-8601=seconds)"
    printf 'arrival_rate=%s dcp_sizes=2,4,8 mega_num_comm_sms=%s\n' \
        "$ARRIVAL_RATE" "$MEGA_NUM_COMM_SMS"
    printf 'num_cases=%s warmup=%s iters=%s\n' "$NUM_CASES" "$WARMUP" "$ITERS"
    printf 'mega_phase_timestamps=1 baseline_mode=graph baseline_phase_timing=1\n'
    printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
} | tee "$MASTER_LOG"

for dcp_size in "${DCP_SIZES[@]}"; do
    combo_dir="$RESULT_DIR/arrival_${ARRIVAL_RATE}/dcp_${dcp_size}"
    cases_path="$combo_dir/trace/cases.jsonl"
    mkdir -p "$combo_dir/trace" "$combo_dir/mega" "$combo_dir/baseline_graph"

    run_logged "generate trace arrival=$ARRIVAL_RATE dcp=$dcp_size" \
        "$LOG_DIR/logs/arrival_${ARRIVAL_RATE}_dcp_${dcp_size}_trace.log" \
        "$PYTHON" -m dcp_test.trace.generate \
        --config "$TRACE_CONFIG" \
        --output "$cases_path" \
        --num-cases "$NUM_CASES" \
        --arrival-time-scale "$ARRIVAL_RATE" \
        --dcp-size "$dcp_size"

    common_args=(
        --config "$BATCH_CONFIG"
        --trace-config "$TRACE_CONFIG"
        --trace-cases "$cases_path"
        --trace-arrival-time-scale "$ARRIVAL_RATE"
        --trace-dcp-size "$dcp_size"
        --num-cases "$NUM_CASES"
        --dcp-sizes "$dcp_size"
        --num-splits 0
        --mega-block-n auto
        --warmup "$WARMUP"
        --iters "$ITERS"
        --no-check
    )
    if [[ -n "$QHEAD" ]]; then
        common_args+=(
            --tp-size "$TP_SIZE" --dcp-size "$dcp_size"
            --qhead "$QHEAD" --kvhead "$KVHEAD"
        )
    fi

    run_logged "arrival=$ARRIVAL_RATE dcp=$dcp_size suite=mega_timestamps" \
        "$LOG_DIR/logs/arrival_${ARRIVAL_RATE}_dcp_${dcp_size}_mega.log" \
        "$TORCHRUN" --standalone --nproc_per_node="$TP_SIZE" \
        --module dcp_test.benchmark_dcp_mega_batch \
        "${common_args[@]}" \
        --implementations mega \
        --no-cuda-graph \
        --mega-phase-timestamps \
        --no-baseline-phase-timing \
        --mega-num-comm-sms "$MEGA_NUM_COMM_SMS" \
        --output-dir "$combo_dir/mega" \
        --manifest "$combo_dir/mega/manifest.json"

    run_logged "arrival=$ARRIVAL_RATE dcp=$dcp_size suite=baseline_graph_phases" \
        "$LOG_DIR/logs/arrival_${ARRIVAL_RATE}_dcp_${dcp_size}_baseline_graph.log" \
        "$TORCHRUN" --standalone --nproc_per_node="$TP_SIZE" \
        --module dcp_test.benchmark_dcp_mega_batch \
        "${common_args[@]}" \
        --implementations "$BASELINE_IMPLEMENTATIONS" \
        --cuda-graph \
        --no-mega-phase-timestamps \
        --baseline-phase-timing \
        --output-dir "$combo_dir/baseline_graph" \
        --manifest "$combo_dir/baseline_graph/manifest.json"
done
