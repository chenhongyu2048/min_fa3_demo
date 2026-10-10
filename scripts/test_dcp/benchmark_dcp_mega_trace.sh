#!/usr/bin/env bash

# Generate trace cases, then run eager Mega/baselines and graph baselines.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NCCL_CGA_CLUSTER_SIZE=${NCCL_CGA_CLUSTER_SIZE:-1}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

PYTHON=${PYTHON:-python}
TORCHRUN=${TORCHRUN:-torchrun}
TRACE_CONFIG=${TRACE_CONFIG:-dcp_test/trace/example_config.json}
BATCH_CONFIG=${BATCH_CONFIG:-dcp_test/configs/dcp_mega_six_loads.json}
NUM_CASES=${NUM_CASES:-20}
MODES=${MODES:-"eager,graph"}
EAGER_IMPLEMENTATIONS=${EAGER_IMPLEMENTATIONS:-"mega,ours,vllm,sglang"}
GRAPH_IMPLEMENTATIONS=${GRAPH_IMPLEMENTATIONS:-"ours,vllm,sglang"}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-40}
CHECK=${CHECK:-0}
NUM_SPLITS=${NUM_SPLITS:-0}
MEGA_BLOCK_N=${MEGA_BLOCK_N:-auto}
MEGA_NUM_COMM_SM=${MEGA_NUM_COMM_SM:-8}
MEGA_PHASE_TIMESTAMPS=${MEGA_PHASE_TIMESTAMPS:-0}
BASELINE_PHASE_TIMING=${BASELINE_PHASE_TIMING:-0}
GENERATE_TRACE=${GENERATE_TRACE:-1}
FORCE_TRACE=${FORCE_TRACE:-0}
DRY_RUN=${DRY_RUN:-0}
LOG_DIR=${LOG_DIR:-"benchmark_logs/dcp_mega_trace_$(date +%Y%m%d-%H%M%S)"}
RESULT_DIR=${RESULT_DIR:-"$LOG_DIR/results"}
TRACE_CASES=${TRACE_CASES:-"$RESULT_DIR/trace_cases.jsonl"}
MASTER_LOG=${MASTER_LOG:-"$LOG_DIR/benchmark_dcp_mega_trace.log"}
TP_SIZE=${TP_SIZE:-8}
DCP_SIZE=${DCP_SIZE:-}
QHEAD=${QHEAD:-}
KVHEAD=${KVHEAD:-}

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

case "$CHECK" in
    0) CHECK_ARG=--no-check ;;
    1) CHECK_ARG=--check ;;
    *) die "CHECK must be 0 or 1, got '$CHECK'" ;;
esac

case "$MEGA_PHASE_TIMESTAMPS" in
    0) EAGER_PHASE_ARG=--no-mega-phase-timestamps ;;
    1) EAGER_PHASE_ARG=--mega-phase-timestamps ;;
    *) die "MEGA_PHASE_TIMESTAMPS must be 0 or 1, got '$MEGA_PHASE_TIMESTAMPS'" ;;
esac

case "$BASELINE_PHASE_TIMING" in
    0) BASELINE_PHASE_ARG=--no-baseline-phase-timing ;;
    1) BASELINE_PHASE_ARG=--baseline-phase-timing ;;
    *) die "BASELINE_PHASE_TIMING must be 0 or 1, got '$BASELINE_PHASE_TIMING'" ;;
esac

case "$FORCE_TRACE" in
    0|1) ;;
    *) die "FORCE_TRACE must be 0 or 1, got '$FORCE_TRACE'" ;;
esac

case "$GENERATE_TRACE" in
    0|1) ;;
    *) die "GENERATE_TRACE must be 0 or 1, got '$GENERATE_TRACE'" ;;
esac

case "$DRY_RUN" in
    0|1) ;;
    *) die "DRY_RUN must be 0 or 1, got '$DRY_RUN'" ;;
esac

require_positive_integer NUM_CASES "$NUM_CASES"
require_positive_integer TP_SIZE "$TP_SIZE"
require_nonnegative_integer WARMUP "$WARMUP"
require_positive_integer ITERS "$ITERS"
require_nonnegative_integer NUM_SPLITS "$NUM_SPLITS"
require_positive_integer MEGA_NUM_COMM_SM "$MEGA_NUM_COMM_SM"
((NUM_SPLITS <= 128)) || die "NUM_SPLITS must be at most 128, got '$NUM_SPLITS'"
((MEGA_NUM_COMM_SM < 132)) || die \
    "MEGA_NUM_COMM_SM must leave at least one of 132 SMs for compute"
case "$TP_SIZE" in
    2|4|8) ;;
    *) die "TP_SIZE must be 2, 4, or 8, got '$TP_SIZE'" ;;
esac

TOPOLOGY_ARGS=()
if [[ -n "$DCP_SIZE" || -n "$QHEAD" || -n "$KVHEAD" ]]; then
    [[ -n "$DCP_SIZE" && -n "$QHEAD" && -n "$KVHEAD" ]] || die \
        "DCP_SIZE, QHEAD, and KVHEAD must be provided together"
    require_positive_integer DCP_SIZE "$DCP_SIZE"
    require_positive_integer QHEAD "$QHEAD"
    require_positive_integer KVHEAD "$KVHEAD"
    case "$DCP_SIZE" in
        2|4|8) ;;
        *) die "DCP_SIZE must be 2, 4, or 8, got '$DCP_SIZE'" ;;
    esac
    ((DCP_SIZE <= TP_SIZE && TP_SIZE % DCP_SIZE == 0)) || die \
        "DCP_SIZE must divide TP_SIZE and cannot exceed it"
    TOPOLOGY_ARGS=(
        --tp-size "$TP_SIZE" --dcp-size "$DCP_SIZE"
        --qhead "$QHEAD" --kvhead "$KVHEAD"
    )
fi
case "$MEGA_BLOCK_N" in
    auto|128|176) ;;
    *) die "MEGA_BLOCK_N must be auto, 128, or 176, got '$MEGA_BLOCK_N'" ;;
esac

mode_spec=${MODES//,/ }
read -r -a MODE_LIST <<< "$mode_spec"
((${#MODE_LIST[@]} > 0)) || die "MODES must not be empty"
for mode in "${MODE_LIST[@]}"; do
    case "$mode" in
        eager|graph) ;;
        *) die "MODES must contain only eager or graph, got '$mode'" ;;
    esac
done

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

print_command() {
    printf '%q' "$1"
    shift
    printf ' %q' "$@"
    printf '\n'
}

print_cuda_command() {
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
    print_command "$@"
}

build_generate_command() {
    GENERATE_COMMAND=(
        "$PYTHON" -m dcp_test.trace.generate
        --config "$TRACE_CONFIG"
        --output "$TRACE_CASES"
        --num-cases "$NUM_CASES"
    )
    if [[ -n "$DCP_SIZE" ]]; then
        GENERATE_COMMAND+=(--dcp-size "$DCP_SIZE")
    fi
    if ((FORCE_TRACE)); then
        GENERATE_COMMAND+=(--force)
    fi
}

build_batch_args() {
    local mode=$1
    local implementations
    local graph_arg
    local phase_arg
    if [[ "$mode" == eager ]]; then
        implementations=$EAGER_IMPLEMENTATIONS
        graph_arg=--no-cuda-graph
        phase_arg=$EAGER_PHASE_ARG
    else
        implementations=$GRAPH_IMPLEMENTATIONS
        graph_arg=--cuda-graph
        phase_arg=--no-mega-phase-timestamps
    fi
    BATCH_ARGS=(
        --config "$BATCH_CONFIG"
        --trace-config "$TRACE_CONFIG"
        --trace-cases "$TRACE_CASES"
        --num-cases "$NUM_CASES"
        "${TOPOLOGY_ARGS[@]}"
        --implementations "$implementations"
        --num-splits "$NUM_SPLITS"
        --mega-block-n "$MEGA_BLOCK_N"
        --mega-num-comm-sm "$MEGA_NUM_COMM_SM"
        --warmup "$WARMUP"
        --iters "$ITERS"
        "$CHECK_ARG"
        "$graph_arg"
        "$phase_arg"
        "$BASELINE_PHASE_ARG"
        --output-dir "$RESULT_DIR/$mode"
        --manifest "$RESULT_DIR/${mode}_manifest.json"
    )
}

generate_trace_cases() {
    if ((GENERATE_TRACE == 0)); then
        [[ -f "$TRACE_CASES" ]] || die \
            "TRACE_CASES does not exist with GENERATE_TRACE=0: '$TRACE_CASES'"
        printf '\n[trace input]\nUsing existing cases: %s\n' "$TRACE_CASES"
        return
    fi

    build_generate_command
    if ((DRY_RUN)); then
        printf '\n[trace generation]\n'
        print_command "${GENERATE_COMMAND[@]}"
        return
    fi

    {
        printf '\n================================================================================\n'
        printf '[trace generation] started=%s\n' "$(date --iso-8601=seconds)"
        printf '================================================================================\n'
        print_command "${GENERATE_COMMAND[@]}"
    } | tee -a "$MASTER_LOG"
    if "${GENERATE_COMMAND[@]}" 2>&1 | tee -a "$MASTER_LOG"; then
        printf '[trace generation] completed=%s\n' "$(date --iso-8601=seconds)" \
            | tee -a "$MASTER_LOG"
    else
        local status=$?
        printf '[trace generation] failed status=%s time=%s\n' \
            "$status" "$(date --iso-8601=seconds)" | tee -a "$MASTER_LOG"
        return "$status"
    fi
}

run_mode() {
    local mode=$1
    local mode_log="$LOG_DIR/${mode}.log"
    local -a command
    build_batch_args "$mode"
    command=(
        "$TORCHRUN" --standalone --nproc_per_node="$TP_SIZE"
        --module dcp_test.benchmark_dcp_mega_batch
        "${BATCH_ARGS[@]}"
    )

    if ((DRY_RUN)); then
        printf '\n[%s trace batch]\n' "$mode"
        print_cuda_command "${command[@]}"
        return
    fi

    {
        printf '\n================================================================================\n'
        printf '[%s trace batch] started=%s\n' "$mode" "$(date --iso-8601=seconds)"
        printf '================================================================================\n'
        print_cuda_command "${command[@]}"
    } | tee -a "$MASTER_LOG" "$mode_log"

    if "${command[@]}" 2>&1 | tee -a "$MASTER_LOG" "$mode_log"; then
        printf '[%s trace batch] completed=%s\n' "$mode" "$(date --iso-8601=seconds)" \
            | tee -a "$MASTER_LOG" "$mode_log"
    else
        local status=$?
        printf '[%s trace batch] failed status=%s time=%s\n' \
            "$mode" "$status" "$(date --iso-8601=seconds)" \
            | tee -a "$MASTER_LOG" "$mode_log"
        return "$status"
    fi
}

if ((DRY_RUN == 0)); then
    mkdir -p "$RESULT_DIR"
    {
        printf 'DCP Mega trace benchmark\n'
        printf 'started=%s\n' "$(date --iso-8601=seconds)"
        printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
        printf 'trace_config=%s trace_cases=%s num_cases=%s modes=%s\n' \
            "$TRACE_CONFIG" "$TRACE_CASES" "$NUM_CASES" "$MODES"
        printf 'generate_trace=%s force_trace=%s\n' "$GENERATE_TRACE" "$FORCE_TRACE"
        printf 'warmup=%s iters=%s check=%s\n' "$WARMUP" "$ITERS" "$CHECK"
        printf 'eager_implementations=%s\n' "$EAGER_IMPLEMENTATIONS"
        printf 'graph_implementations=%s\n' "$GRAPH_IMPLEMENTATIONS"
        printf 'num_splits=%s block_n=%s comm_sm=%s mega_phase_timestamps=%s baseline_phase_timing=%s\n' \
            "$NUM_SPLITS" "$MEGA_BLOCK_N" "$MEGA_NUM_COMM_SM" \
            "$MEGA_PHASE_TIMESTAMPS" "$BASELINE_PHASE_TIMING"
    } | tee "$MASTER_LOG"
fi

generate_trace_cases
for mode in "${MODE_LIST[@]}"; do
    run_mode "$mode"
done

if ((DRY_RUN == 0)); then
    printf '\nAll trace batches completed at %s\nResults: %s\n' \
        "$(date --iso-8601=seconds)" "$LOG_DIR" | tee -a "$MASTER_LOG"
fi
