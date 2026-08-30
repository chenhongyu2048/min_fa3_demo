#!/usr/bin/env bash

# Reproduce the 01_dcp_wave-b scheduler ablation with Mega DCP phase
# timestamps enabled. Results are written to a new timestamped directory.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NCCL_CGA_CLUSTER_SIZE=${NCCL_CGA_CLUSTER_SIZE:-1}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
TORCHRUN=${TORCHRUN:-"$SCRIPT_DIR/.venv/bin/torchrun"}
TRACE_CONFIG=${TRACE_CONFIG:-dcp_test/trace/example_config.json}
BATCH_CONFIG=${BATCH_CONFIG:-dcp_test/configs/dcp_mega_six_loads.json}
ARRIVAL_TIME_SCALES=${ARRIVAL_TIME_SCALES:-"1,2,4"}
DCP_SIZES=${DCP_SIZES:-"2,4,8"}
NUM_CASES=${NUM_CASES:-100}
WARMUP=${WARMUP:-40}
ITERS=${ITERS:-60}
CHECK=${CHECK:-0}
GENERATE_TRACE=${GENERATE_TRACE:-1}
FORCE_TRACE=${FORCE_TRACE:-0}
DRY_RUN=${DRY_RUN:-0}
DCP2_COMM_SM=${DCP2_COMM_SM:-12}
DCP4_COMM_SM=${DCP4_COMM_SM:-16}
DCP8_COMM_SM=${DCP8_COMM_SM:-16}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
RUN_DIR=${RUN_DIR:-"benchmark_logs/experiment_suite/dcp-suite/01_dcp_wave-b-timestamps-$RUN_ID"}
RESULT_DIR=${RESULT_DIR:-"$RUN_DIR/results"}
LOG_DIR=${LOG_DIR:-"$RUN_DIR/logs"}
MASTER_LOG=${MASTER_LOG:-"$RUN_DIR/benchmark.log"}
TRACE_CASES_ROOT=${TRACE_CASES_ROOT:-"$RESULT_DIR"}
TP_SIZE=${TP_SIZE:-8}
QHEAD=${QHEAD:-}
KVHEAD=${KVHEAD:-}
readonly -a STRATEGIES=(fa3_native_fifo critical_wave_auto_lpt)

die() {
    echo "error: $*" >&2
    exit 1
}

require_positive_integer() {
    local name=$1
    local value=$2
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || \
        die "$name must be a positive integer, got '$value'"
}

require_nonnegative_integer() {
    local name=$1
    local value=$2
    [[ "$value" =~ ^[0-9]+$ ]] || \
        die "$name must be a non-negative integer, got '$value'"
}

parse_csv() {
    local name=$1
    local value=$2
    local -n destination=$3
    local normalized=${value//,/ }
    read -r -a destination <<< "$normalized"
    ((${#destination[@]} > 0)) || die "$name must not be empty"
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

comm_sm_for_dcp() {
    case "$1" in
        2) printf '%s\n' "$DCP2_COMM_SM" ;;
        4) printf '%s\n' "$DCP4_COMM_SM" ;;
        8) printf '%s\n' "$DCP8_COMM_SM" ;;
        *) die "unsupported DCP size '$1'" ;;
    esac
}

print_command() {
    printf 'CUDA_VISIBLE_DEVICES=%q' "$CUDA_VISIBLE_DEVICES"
    printf ' %q' "$@"
    printf '\n'
}

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$MASTER_LOG"
}

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
    local trace_log=$4

    if ((GENERATE_TRACE == 0)); then
        [[ -f "$cases_path" ]] || die \
            "trace cases do not exist with GENERATE_TRACE=0: '$cases_path'"
        if ((DRY_RUN)); then
            printf '\n[reuse trace arrival=%s dcp=%s]\n%s\n' \
                "$arrival" "$dcp_size" "$cases_path"
        else
            log "Reusing trace cases for arrival=$arrival dcp=$dcp_size: $cases_path"
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
    run_logged "generate trace arrival=$arrival dcp=$dcp_size" "$trace_log" \
        "${command[@]}"
}

run_strategy() {
    local arrival=$1
    local dcp_size=$2
    local comm_sm=$3
    local cases_path=$4
    local strategy=$5
    local strategy_dir="$RESULT_DIR/arrival_${arrival}/dcp_${dcp_size}/$strategy"
    local strategy_log="$LOG_DIR/arrival_${arrival}_dcp_${dcp_size}_${strategy}.log"
    local -a scheduler_args
    local -a topology_args=()
    if [[ -n "$QHEAD" ]]; then
        topology_args=(
            --tp-size "$TP_SIZE" --dcp-size "$dcp_size"
            --qhead "$QHEAD" --kvhead "$KVHEAD"
        )
    fi

    case "$strategy" in
        fa3_native_fifo)
            scheduler_args=(--no-mega-scheduler-heuristic --mega-history-order fifo)
            ;;
        critical_wave_auto_lpt)
            scheduler_args=(--mega-scheduler-heuristic --mega-history-order auto)
            ;;
        *) die "unknown strategy '$strategy'" ;;
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
        --implementations mega
        --num-splits 0
        --mega-block-n auto
        --mega-num-comm-sms "$comm_sm"
        --warmup "$WARMUP"
        --iters "$ITERS"
        "$CHECK_ARG"
        --no-cuda-graph
        --mega-phase-timestamps
        --no-baseline-phase-timing
        "${scheduler_args[@]}"
        --output-dir "$strategy_dir"
        --manifest "$strategy_dir/manifest.json"
    )
    run_logged \
        "arrival=$arrival dcp=$dcp_size strategy=$strategy comm_sm=$comm_sm timestamps=on" \
        "$strategy_log" "${command[@]}"
}

case "$CHECK" in
    0) CHECK_ARG=--no-check ;;
    1) CHECK_ARG=--check ;;
    *) die "CHECK must be 0 or 1, got '$CHECK'" ;;
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
require_positive_integer DCP2_COMM_SM "$DCP2_COMM_SM"
require_positive_integer DCP4_COMM_SM "$DCP4_COMM_SM"
require_positive_integer DCP8_COMM_SM "$DCP8_COMM_SM"
for comm_sm in "$DCP2_COMM_SM" "$DCP4_COMM_SM" "$DCP8_COMM_SM"; do
    ((comm_sm < 132)) || die "comm-SM values must be less than 132, got '$comm_sm'"
done
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

parse_csv ARRIVAL_TIME_SCALES "$ARRIVAL_TIME_SCALES" ARRIVAL_LIST
parse_csv DCP_SIZES "$DCP_SIZES" DCP_LIST
check_unique ARRIVAL_TIME_SCALES "${ARRIVAL_LIST[@]}"
check_unique DCP_SIZES "${DCP_LIST[@]}"

arrival_validation_code='import math,sys; value=float(sys.argv[1]); '
arrival_validation_code+='raise SystemExit(not math.isfinite(value) or value <= 0)'
for arrival in "${ARRIVAL_LIST[@]}"; do
    [[ "$arrival" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
        die "arrival time scale must be a positive number, got '$arrival'"
    "$PYTHON" -c "$arrival_validation_code" "$arrival" || \
        die "arrival time scale must be positive and finite, got '$arrival'"
done
for dcp_size in "${DCP_LIST[@]}"; do
    case "$dcp_size" in
        2|4|8) ;;
        *) die "DCP_SIZES must contain only 2, 4, or 8, got '$dcp_size'" ;;
    esac
    ((dcp_size <= TP_SIZE && TP_SIZE % dcp_size == 0)) || die \
        "every DCP size must divide TP_SIZE=$TP_SIZE and cannot exceed it"
done

[[ -f "$TRACE_CONFIG" ]] || die "TRACE_CONFIG was not found: '$TRACE_CONFIG'"
[[ -f "$BATCH_CONFIG" ]] || die "BATCH_CONFIG was not found: '$BATCH_CONFIG'"
[[ -x "$PYTHON" ]] || command -v "$PYTHON" >/dev/null 2>&1 || \
    die "PYTHON command was not found: '$PYTHON'"
[[ -x "$TORCHRUN" ]] || command -v "$TORCHRUN" >/dev/null 2>&1 || \
    die "TORCHRUN command was not found: '$TORCHRUN'"

if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    case "$TP_SIZE" in
        2) CUDA_VISIBLE_DEVICES=0,1 ;;
        4) CUDA_VISIBLE_DEVICES=0,1,2,3 ;;
        8) CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ;;
    esac
fi
export CUDA_VISIBLE_DEVICES
IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
((${#visible_devices[@]} == TP_SIZE)) || die \
    "CUDA_VISIBLE_DEVICES must expose exactly $TP_SIZE GPUs, got '$CUDA_VISIBLE_DEVICES'"

EXPECTED_TRACES=$((${#ARRIVAL_LIST[@]} * ${#DCP_LIST[@]}))
EXPECTED_LAUNCHES=$((EXPECTED_TRACES * ${#STRATEGIES[@]}))

if ((DRY_RUN == 0)); then
    mkdir -p "$RESULT_DIR" "$LOG_DIR"
    {
        printf 'Mega DCP trace wave-scheduler ablation with phase timestamps\n'
        printf 'started=%s\n' "$(date --iso-8601=seconds)"
        printf 'arrival_time_scales=%s dcp_sizes=%s num_cases=%s\n' \
            "$ARRIVAL_TIME_SCALES" "$DCP_SIZES" "$NUM_CASES"
        printf 'strategies=%s\n' "$(IFS=,; echo "${STRATEGIES[*]}")"
        printf 'comm_sm_by_dcp=dcp2:%s,dcp4:%s,dcp8:%s execution=eager block_n=auto\n' \
            "$DCP2_COMM_SM" "$DCP4_COMM_SM" "$DCP8_COMM_SM"
        printf 'expected_traces=%s expected_strategy_launches=%s warmup=%s iters=%s check=%s\n' \
            "$EXPECTED_TRACES" "$EXPECTED_LAUNCHES" "$WARMUP" "$ITERS" "$CHECK"
        printf 'mega_phase_timestamps=1 baseline_phase_timing=0\n'
        printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
    } | tee "$MASTER_LOG"
fi

for arrival in "${ARRIVAL_LIST[@]}"; do
    for dcp_size in "${DCP_LIST[@]}"; do
        combo_dir="$RESULT_DIR/arrival_${arrival}/dcp_${dcp_size}"
        cases_path="$TRACE_CASES_ROOT/arrival_${arrival}/dcp_${dcp_size}/trace/cases.jsonl"
        trace_log="$LOG_DIR/arrival_${arrival}_dcp_${dcp_size}_trace.log"
        comm_sm=$(comm_sm_for_dcp "$dcp_size")
        if ((DRY_RUN == 0)); then
            mkdir -p "$(dirname -- "$cases_path")" "$combo_dir"
        fi
        generate_cases "$arrival" "$dcp_size" "$cases_path" "$trace_log"
        for strategy in "${STRATEGIES[@]}"; do
            run_strategy "$arrival" "$dcp_size" "$comm_sm" "$cases_path" "$strategy"
        done
    done
done

if ((DRY_RUN)); then
    printf '\nDry run complete: %s traces and %s timestamp-enabled strategy launches.\n' \
        "$EXPECTED_TRACES" "$EXPECTED_LAUNCHES"
else
    log "Completed $EXPECTED_LAUNCHES timestamp-enabled wave ablation launches; results=$RESULT_DIR"
fi
