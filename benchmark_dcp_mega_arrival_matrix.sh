#!/usr/bin/env bash

# Benchmark independent arrival-time-scale/DCP trace workloads. Each combination
# uses one Mega comm-SM sweep torchrun and eager/graph baseline torchruns.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NCCL_CGA_CLUSTER_SIZE=${NCCL_CGA_CLUSTER_SIZE:-1}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

PYTHON=${PYTHON:-python}
TORCHRUN=${TORCHRUN:-torchrun}
TRACE_CONFIG=${TRACE_CONFIG:-dcp_test/trace/example_config.json}
BATCH_CONFIG=${BATCH_CONFIG:-dcp_test/configs/dcp_mega_six_loads.json}
ARRIVAL_TIME_SCALES=${ARRIVAL_TIME_SCALES:-"1,2,4"}
DCP_SIZES=${DCP_SIZES:-"2,4,8"}
MEGA_NUM_COMM_SMS=${MEGA_NUM_COMM_SMS:-"4,8,12,16,20"}
BASELINE_IMPLEMENTATIONS=${BASELINE_IMPLEMENTATIONS:-"ours,vllm,sglang,full"}
NUM_CASES=${NUM_CASES:-100}
WARMUP=${WARMUP:-40}
ITERS=${ITERS:-60}
CHECK=${CHECK:-0}
NUM_SPLITS=${NUM_SPLITS:-0}
MEGA_BLOCK_N=${MEGA_BLOCK_N:-auto}
MEGA_PHASE_TIMESTAMPS=${MEGA_PHASE_TIMESTAMPS:-0}
BASELINE_PHASE_TIMING=${BASELINE_PHASE_TIMING:-0}
GENERATE_TRACE=${GENERATE_TRACE:-1}
FORCE_TRACE=${FORCE_TRACE:-0}
DRY_RUN=${DRY_RUN:-0}
LOG_DIR=${LOG_DIR:-"benchmark_logs/bench_dcp/$(date +%Y%m%d-%H%M%S)"}
RESULT_DIR=${RESULT_DIR:-"$LOG_DIR/results"}
MASTER_LOG=${MASTER_LOG:-"$LOG_DIR/benchmark.log"}
SUMMARY_JSON=${SUMMARY_JSON:-"$LOG_DIR/matrix_manifest.json"}
SUMMARY_CSV=${SUMMARY_CSV:-"$LOG_DIR/matrix_summary.csv"}
TP_SIZE=${TP_SIZE:-8}
QHEAD=${QHEAD:-}
KVHEAD=${KVHEAD:-}
SUMMARY_COMPLETE=0

die() {
    echo "error: $*" >&2
    exit 1
}

require_nonnegative_integer() {
    local name=$1
    local value=$2
    [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer, got '$value'"
}

require_positive_integer() {
    local name=$1
    local value=$2
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer, got '$value'"
}

parse_csv() {
    local name=$1
    local spec=$2
    local -n output=$3
    local normalized=${spec//,/ }
    read -r -a output <<< "$normalized"
    ((${#output[@]} > 0)) || die "$name must not be empty"
}

check_unique() {
    local name=$1
    shift
    local value
    local -A seen=()
    for value in "$@"; do
        [[ -z ${seen[$value]+x} ]] || die "$name contains duplicate '$value'"
        seen[$value]=1
    done
}

print_command() {
    printf 'CUDA_VISIBLE_DEVICES=%q' "$CUDA_VISIBLE_DEVICES"
    printf ' %q' "$@"
    printf '\n'
}

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$MASTER_LOG"
}

case "$CHECK" in
    0) CHECK_ARG=--no-check ;;
    1) CHECK_ARG=--check ;;
    *) die "CHECK must be 0 or 1, got '$CHECK'" ;;
esac
case "$MEGA_PHASE_TIMESTAMPS" in
    0) MEGA_PHASE_ARG=--no-mega-phase-timestamps ;;
    1) MEGA_PHASE_ARG=--mega-phase-timestamps ;;
    *) die "MEGA_PHASE_TIMESTAMPS must be 0 or 1" ;;
esac
case "$BASELINE_PHASE_TIMING" in
    0) BASELINE_PHASE_ARG=--no-baseline-phase-timing ;;
    1) BASELINE_PHASE_ARG=--baseline-phase-timing ;;
    *) die "BASELINE_PHASE_TIMING must be 0 or 1" ;;
esac
for flag_name in GENERATE_TRACE FORCE_TRACE DRY_RUN; do
    flag_value=${!flag_name}
    [[ "$flag_value" == 0 || "$flag_value" == 1 ]] || \
        die "$flag_name must be 0 or 1, got '$flag_value'"
done

require_positive_integer NUM_CASES "$NUM_CASES"
require_positive_integer TP_SIZE "$TP_SIZE"
require_nonnegative_integer WARMUP "$WARMUP"
require_positive_integer ITERS "$ITERS"
require_nonnegative_integer NUM_SPLITS "$NUM_SPLITS"
((NUM_SPLITS <= 128)) || die "NUM_SPLITS must be at most 128"
case "$TP_SIZE" in
    2|4|8) ;;
    *) die "TP_SIZE must be 2, 4, or 8, got '$TP_SIZE'" ;;
esac
if [[ -n "$QHEAD" || -n "$KVHEAD" ]]; then
    [[ -n "$QHEAD" && -n "$KVHEAD" ]] || die \
        "QHEAD and KVHEAD must be provided together"
    require_positive_integer QHEAD "$QHEAD"
    require_positive_integer KVHEAD "$KVHEAD"
fi
[[ "$BASELINE_IMPLEMENTATIONS" == "ours,vllm,sglang,full" ]] || die \
    "BASELINE_IMPLEMENTATIONS must be ours,vllm,sglang,full for matrix summaries"
case "$MEGA_BLOCK_N" in
    auto|128|176) ;;
    *) die "MEGA_BLOCK_N must be auto, 128, or 176" ;;
esac

parse_csv ARRIVAL_TIME_SCALES "$ARRIVAL_TIME_SCALES" ARRIVAL_LIST
parse_csv DCP_SIZES "$DCP_SIZES" DCP_LIST
parse_csv MEGA_NUM_COMM_SMS "$MEGA_NUM_COMM_SMS" COMM_SM_LIST
check_unique ARRIVAL_TIME_SCALES "${ARRIVAL_LIST[@]}"
check_unique DCP_SIZES "${DCP_LIST[@]}"
check_unique MEGA_NUM_COMM_SMS "${COMM_SM_LIST[@]}"
EXPECTED_LAUNCH_COUNT=$((${#ARRIVAL_LIST[@]} * ${#DCP_LIST[@]} * 3))

arrival_validation_code='import math,sys; value=float(sys.argv[1]); '
arrival_validation_code+='raise SystemExit(not math.isfinite(value) or value <= 0)'
for arrival in "${ARRIVAL_LIST[@]}"; do
    [[ "$arrival" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
        die "arrival_time_scale must be a positive number, got '$arrival'"
    "$PYTHON" -c "$arrival_validation_code" "$arrival" || \
        die "arrival_time_scale must be positive and finite, got '$arrival'"
done
for dcp_size in "${DCP_LIST[@]}"; do
    case "$dcp_size" in
        2|4|8) ;;
        *) die "DCP_SIZES must contain only 2, 4, or 8, got '$dcp_size'" ;;
    esac
    ((dcp_size <= TP_SIZE && TP_SIZE % dcp_size == 0)) || die \
        "every DCP size must divide TP_SIZE=$TP_SIZE and cannot exceed it"
done
for comm_sm in "${COMM_SM_LIST[@]}"; do
    require_positive_integer MEGA_NUM_COMM_SMS "$comm_sm"
    ((comm_sm < 132)) || die "comm-SM values must be less than 132"
done

[[ -f "$TRACE_CONFIG" ]] || die "TRACE_CONFIG was not found: '$TRACE_CONFIG'"
[[ -f "$BATCH_CONFIG" ]] || die "BATCH_CONFIG was not found: '$BATCH_CONFIG'"
command -v "$PYTHON" >/dev/null 2>&1 || die "PYTHON command was not found: '$PYTHON'"
command -v "$TORCHRUN" >/dev/null 2>&1 || die "TORCHRUN command was not found: '$TORCHRUN'"

if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    case "$TP_SIZE" in
        2) CUDA_VISIBLE_DEVICES=0,1 ;;
        4) CUDA_VISIBLE_DEVICES=0,1,2,3 ;;
        8) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ;;
    esac
else
    IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
    ((${#visible_devices[@]} >= TP_SIZE)) || die \
        "CUDA_VISIBLE_DEVICES exposes ${#visible_devices[@]} GPUs, but $TP_SIZE are required"
fi
export CUDA_VISIBLE_DEVICES

summary_command() {
    SUMMARY_COMMAND=(
        "$PYTHON" -m dcp_test.summarize_dcp_mega_matrix
        --result-dir "$RESULT_DIR"
        --arrival-time-scales "$ARRIVAL_TIME_SCALES"
        --dcp-sizes "$DCP_SIZES"
        --mega-num-comm-sms "$MEGA_NUM_COMM_SMS"
        --num-cases "$NUM_CASES"
        --output-json "$SUMMARY_JSON"
        --output-csv "$SUMMARY_CSV"
    )
}

finalize_partial() {
    local status=$1
    trap - EXIT
    if ((DRY_RUN == 0 && SUMMARY_COMPLETE == 0)); then
        summary_command
        "${SUMMARY_COMMAND[@]}" --allow-incomplete 2>&1 | tee -a "$MASTER_LOG" || true
    fi
    exit "$status"
}
trap 'finalize_partial $?' EXIT

run_logged() {
    local label=$1
    local log_path=$2
    shift 2
    local -a command=("$@")
    if ((DRY_RUN)); then
        printf '\n[%s]\n' "$label"
        print_command "${command[@]}"
        return
    fi
    log "Starting $label"
    print_command "${command[@]}" | tee -a "$MASTER_LOG" "$log_path"
    if "${command[@]}" 2>&1 | tee -a "$MASTER_LOG" "$log_path"; then
        log "Completed $label"
    else
        local status=${PIPESTATUS[0]}
        log "Failed $label with status $status"
        return "$status"
    fi
}

generate_cases() {
    local arrival=$1
    local dcp_size=$2
    local cases_path=$3
    local combo_log=$4
    if ((GENERATE_TRACE == 0)); then
        [[ -f "$cases_path" ]] || die \
            "trace cases do not exist with GENERATE_TRACE=0: '$cases_path'"
        if ((DRY_RUN)); then
            printf '\n[reuse trace arrival=%s dcp=%s]\n%s\n' \
                "$arrival" "$dcp_size" "$cases_path"
        else
            log "Reusing trace cases for arrival=$arrival DCP=$dcp_size: $cases_path"
        fi
        return
    fi
    local -a command=(
        "$PYTHON" -m dcp_test.trace.generate
        --config "$TRACE_CONFIG"
        --output "$cases_path"
        --num-cases "$NUM_CASES"
        --arrival-time-scale "$arrival"
        --dcp-size "$dcp_size"
    )
    if ((FORCE_TRACE)); then
        command+=(--force)
    fi
    run_logged "generate trace arrival=$arrival dcp=$dcp_size" "$combo_log" \
        "${command[@]}"
}

run_combo_mode() {
    local arrival=$1
    local dcp_size=$2
    local cases_path=$3
    local suite=$4
    local output_dir=$5
    local manifest=$6
    local mode_log=$7
    local implementations graph_arg phase_arg
    local -a extra_args=()
    local -a topology_args=()
    if [[ -n "$QHEAD" ]]; then
        topology_args=(
            --tp-size "$TP_SIZE" --dcp-size "$dcp_size"
            --qhead "$QHEAD" --kvhead "$KVHEAD"
        )
    fi
    case "$suite" in
        mega)
            implementations=mega
            graph_arg=--no-cuda-graph
            phase_arg=$MEGA_PHASE_ARG
            extra_args=(--mega-num-comm-sms "$MEGA_NUM_COMM_SMS")
            ;;
        baseline_eager)
            implementations=$BASELINE_IMPLEMENTATIONS
            graph_arg=--no-cuda-graph
            phase_arg=--no-mega-phase-timestamps
            ;;
        baseline_graph)
            implementations=$BASELINE_IMPLEMENTATIONS
            graph_arg=--cuda-graph
            phase_arg=--no-mega-phase-timestamps
            ;;
        *) die "unknown suite '$suite'" ;;
    esac

    local -a command=(
        "$TORCHRUN" --standalone --nproc_per_node="$TP_SIZE"
        --module dcp_test.benchmark_dcp_mega_batch
        --config "$BATCH_CONFIG"
        --trace-config "$TRACE_CONFIG"
        --trace-cases "$cases_path"
        --trace-arrival-time-scale "$arrival"
        --trace-dcp-size "$dcp_size"
        --num-cases "$NUM_CASES"
        --dcp-sizes "$dcp_size"
        "${topology_args[@]}"
        --implementations "$implementations"
        --num-splits "$NUM_SPLITS"
        --mega-block-n "$MEGA_BLOCK_N"
        --warmup "$WARMUP"
        --iters "$ITERS"
        "$CHECK_ARG"
        "$graph_arg"
        "$phase_arg"
        "$BASELINE_PHASE_ARG"
        --output-dir "$output_dir"
        --manifest "$manifest"
        "${extra_args[@]}"
    )
    run_logged "arrival=$arrival dcp=$dcp_size suite=$suite" "$mode_log" \
        "${command[@]}"
}

if ((DRY_RUN == 0)); then
    mkdir -p "$RESULT_DIR" "$LOG_DIR/logs"
    {
        printf 'Mega DCP arrival/DCP matrix benchmark\n'
        printf 'started=%s\n' "$(date --iso-8601=seconds)"
        printf 'arrival_time_scales=%s dcp_sizes=%s mega_num_comm_sms=%s\n' \
            "$ARRIVAL_TIME_SCALES" "$DCP_SIZES" "$MEGA_NUM_COMM_SMS"
        printf 'num_cases=%s warmup=%s iters=%s check=%s\n' \
            "$NUM_CASES" "$WARMUP" "$ITERS" "$CHECK"
        printf 'baseline_implementations=%s baseline_phase_timing=%s\n' \
            "$BASELINE_IMPLEMENTATIONS" "$BASELINE_PHASE_TIMING"
        printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
    } | tee "$MASTER_LOG"
fi

for arrival in "${ARRIVAL_LIST[@]}"; do
    for dcp_size in "${DCP_LIST[@]}"; do
        combo_dir="$RESULT_DIR/arrival_${arrival}/dcp_${dcp_size}"
        cases_path="$combo_dir/trace/cases.jsonl"
        combo_log="$LOG_DIR/logs/arrival_${arrival}_dcp_${dcp_size}_trace.log"
        if ((DRY_RUN == 0)); then
            mkdir -p "$combo_dir/trace" "$combo_dir/mega" \
                "$combo_dir/baseline_eager" "$combo_dir/baseline_graph"
        fi
        generate_cases "$arrival" "$dcp_size" "$cases_path" "$combo_log"
        run_combo_mode "$arrival" "$dcp_size" "$cases_path" mega \
            "$combo_dir/mega" "$combo_dir/mega/manifest.json" \
            "$LOG_DIR/logs/arrival_${arrival}_dcp_${dcp_size}_mega.log"
        run_combo_mode "$arrival" "$dcp_size" "$cases_path" baseline_eager \
            "$combo_dir/baseline_eager" "$combo_dir/baseline_eager/manifest.json" \
            "$LOG_DIR/logs/arrival_${arrival}_dcp_${dcp_size}_baseline_eager.log"
        run_combo_mode "$arrival" "$dcp_size" "$cases_path" baseline_graph \
            "$combo_dir/baseline_graph" "$combo_dir/baseline_graph/manifest.json" \
            "$LOG_DIR/logs/arrival_${arrival}_dcp_${dcp_size}_baseline_graph.log"
    done
done

summary_command
if ((DRY_RUN)); then
    printf '\n[matrix summary]\n'
    printf '%q' "${SUMMARY_COMMAND[0]}"
    printf ' %q' "${SUMMARY_COMMAND[@]:1}"
    printf '\n'
else
    "${SUMMARY_COMMAND[@]}" 2>&1 | tee -a "$MASTER_LOG"
    SUMMARY_COMPLETE=1
    log "All $EXPECTED_LAUNCH_COUNT matrix launches completed; JSON=$SUMMARY_JSON CSV=$SUMMARY_CSV"
fi
