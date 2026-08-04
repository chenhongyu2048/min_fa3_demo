#!/usr/bin/env bash

# Reproduce the six packed-varlen chunk workloads across DCP=2/4/8.
# Run Mega and the orchestration baselines in both eager and CUDA Graph modes.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NCCL_CGA_CLUSTER_SIZE=${NCCL_CGA_CLUSTER_SIZE:-1}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

TORCHRUN=${TORCHRUN:-torchrun}
LOADS=${LOADS:-"small1,small2,medium1,medium2,large1,large2"}
DCP_SIZES=${DCP_SIZES:-"2,4,8"}
MODES=${MODES:-"eager,graph"}
EAGER_IMPLEMENTATIONS=${EAGER_IMPLEMENTATIONS:-"mega,ours,vllm,sglang"}
GRAPH_IMPLEMENTATIONS=${GRAPH_IMPLEMENTATIONS:-"mega,ours,vllm,sglang"}
WARMUP=${WARMUP:-500}
ITERS=${ITERS:-100}
CHECK=${CHECK:-1}
NUM_SPLITS=${NUM_SPLITS:-0}
MEGA_BLOCK_N=${MEGA_BLOCK_N:-128}
MEGA_NUM_COMM_SM=${MEGA_NUM_COMM_SM:-8}
MEGA_PHASE_TIMESTAMPS=${MEGA_PHASE_TIMESTAMPS:-0}
DRY_RUN=${DRY_RUN:-0}
LOG_DIR=${LOG_DIR:-"benchmark_logs/dcp_mega_six_loads_$(date +%Y%m%d-%H%M%S)"}
RESULT_DIR=${RESULT_DIR:-"$LOG_DIR/results"}
MASTER_LOG=${MASTER_LOG:-"$LOG_DIR/benchmark_dcp_mega_six_loads.log"}

TP_SIZE=8
QHEAD=32
HEADDIM=128

WORKLOADS=(
    "small1 1 8 4096"
    "small2 4 8 4096"
    "medium1 4 32 16384"
    "medium2 16 32 16384"
    "large1 4 128 65536"
    "large2 16 128 65536"
)

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
    0) CHECK_ARGS=(--no-check) ;;
    1) CHECK_ARGS=(--check) ;;
    *) die "CHECK must be 0 or 1, got '$CHECK'" ;;
esac

case "$MEGA_PHASE_TIMESTAMPS" in
    0) PHASE_ARGS=(--no-mega-phase-timestamps) ;;
    1) PHASE_ARGS=(--mega-phase-timestamps) ;;
    *) die "MEGA_PHASE_TIMESTAMPS must be 0 or 1, got '$MEGA_PHASE_TIMESTAMPS'" ;;
esac

case "$DRY_RUN" in
    0|1) ;;
    *) die "DRY_RUN must be 0 or 1, got '$DRY_RUN'" ;;
esac

require_nonnegative_integer WARMUP "$WARMUP"
require_positive_integer ITERS "$ITERS"
require_nonnegative_integer NUM_SPLITS "$NUM_SPLITS"
require_positive_integer MEGA_NUM_COMM_SM "$MEGA_NUM_COMM_SM"
((NUM_SPLITS <= 128)) || die "NUM_SPLITS must be at most 128, got '$NUM_SPLITS'"
((MEGA_NUM_COMM_SM < 132)) || die \
    "MEGA_NUM_COMM_SM must leave at least one of 132 SMs for compute"
case "$MEGA_BLOCK_N" in
    128|176) ;;
    *) die "MEGA_BLOCK_N must be 128 or 176, got '$MEGA_BLOCK_N'" ;;
esac

load_spec=${LOADS//,/ }
read -r -a LOAD_LIST <<< "$load_spec"
((${#LOAD_LIST[@]} > 0)) || die "LOADS must not be empty"
for load in "${LOAD_LIST[@]}"; do
    case "$load" in
        small1|small2|medium1|medium2|large1|large2) ;;
        *) die "LOADS contains unknown workload '$load'" ;;
    esac
done

dcp_spec=${DCP_SIZES//,/ }
read -r -a DCP_SIZE_LIST <<< "$dcp_spec"
((${#DCP_SIZE_LIST[@]} > 0)) || die "DCP_SIZES must not be empty"
for dcp_size in "${DCP_SIZE_LIST[@]}"; do
    case "$dcp_size" in
        2|4|8) ;;
        *) die "DCP_SIZES must contain only 2, 4, or 8, got '$dcp_size'" ;;
    esac
done

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
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
    ((${#visible_devices[@]} >= TP_SIZE)) || die \
        "CUDA_VISIBLE_DEVICES exposes ${#visible_devices[@]} GPUs, but $TP_SIZE are required"
fi
export CUDA_VISIBLE_DEVICES

kv_heads_for_dcp() {
    case "$1" in
        2) echo 4 ;;
        4) echo 2 ;;
        8) echo 1 ;;
        *) return 1 ;;
    esac
}

load_is_enabled() {
    local candidate=$1
    local selected
    for selected in "${LOAD_LIST[@]}"; do
        [[ "$candidate" == "$selected" ]] && return 0
    done
    return 1
}

print_command() {
    printf 'CUDA_VISIBLE_DEVICES=%q' "$CUDA_VISIBLE_DEVICES"
    printf ' %q' "$@"
    printf '\n'
}

run_job() {
    local workload_name=$1
    local batch=$2
    local sq=$3
    local history=$4
    local dcp_size=$5
    local kvhead=$6
    local mode=$7
    local case_name="${workload_name}_b${batch}_q${sq}_h${history}_dcp${dcp_size}_hkv${kvhead}_${mode}"
    local case_log="$LOG_DIR/${case_name}.log"
    local output_json="$RESULT_DIR/${case_name}.json"
    local -a command=(
        "$TORCHRUN" --standalone --nproc_per_node="$TP_SIZE"
        --module dcp_test.benchmark_dcp_varlen
        --b "$batch" --sq "$sq" --seqlen "$history"
        --qhead "$QHEAD" --kvhead "$kvhead" --headdim "$HEADDIM"
        --tp-size "$TP_SIZE" --dcp-size "$dcp_size"
        --workload chunk
        --num-splits "$NUM_SPLITS"
        --mega-block-n "$MEGA_BLOCK_N"
        --mega-num-comm-sm "$MEGA_NUM_COMM_SM"
        --warmup "$WARMUP" --iters "$ITERS"
        "${CHECK_ARGS[@]}"
        --output-json "$output_json"
    )

    if [[ "$mode" == eager ]]; then
        command+=(
            --implementations "$EAGER_IMPLEMENTATIONS"
            --no-cuda-graph
            "${PHASE_ARGS[@]}"
        )
    else
        command+=(
            --implementations "$GRAPH_IMPLEMENTATIONS"
            --cuda-graph
        )
    fi

    if ((DRY_RUN)); then
        printf '\n[%s]\n' "$case_name"
        print_command "${command[@]}"
        return
    fi

    {
        printf '\n================================================================================\n'
        printf '[%s] started=%s\n' "$case_name" "$(date --iso-8601=seconds)"
        printf '================================================================================\n'
        print_command "${command[@]}"
    } | tee -a "$MASTER_LOG" "$case_log"

    if "${command[@]}" 2>&1 | tee -a "$MASTER_LOG" "$case_log"; then
        printf '[%s] completed=%s\n' "$case_name" "$(date --iso-8601=seconds)" \
            | tee -a "$MASTER_LOG" "$case_log"
    else
        local status=$?
        printf '[%s] failed status=%s time=%s\n' \
            "$case_name" "$status" "$(date --iso-8601=seconds)" \
            | tee -a "$MASTER_LOG" "$case_log"
        return "$status"
    fi
}

if ((DRY_RUN == 0)); then
    mkdir -p "$RESULT_DIR"
    {
        printf 'DCP mega six-load benchmark\n'
        printf 'started=%s\n' "$(date --iso-8601=seconds)"
        printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
        printf 'loads=%s dcp_sizes=%s modes=%s warmup=%s iters=%s check=%s\n' \
            "$LOADS" "$DCP_SIZES" "$MODES" "$WARMUP" "$ITERS" "$CHECK"
        printf 'eager_implementations=%s\n' "$EAGER_IMPLEMENTATIONS"
        printf 'graph_implementations=%s\n' "$GRAPH_IMPLEMENTATIONS"
        printf 'num_splits=%s block_n=%s comm_sm=%s phase_timestamps=%s\n' \
            "$NUM_SPLITS" "$MEGA_BLOCK_N" "$MEGA_NUM_COMM_SM" "$MEGA_PHASE_TIMESTAMPS"
    } | tee "$MASTER_LOG"
fi

for workload in "${WORKLOADS[@]}"; do
    read -r workload_name batch sq history <<< "$workload"
    load_is_enabled "$workload_name" || continue
    for dcp_size in "${DCP_SIZE_LIST[@]}"; do
        kvhead=$(kv_heads_for_dcp "$dcp_size")
        for mode in "${MODE_LIST[@]}"; do
            run_job "$workload_name" "$batch" "$sq" "$history" \
                "$dcp_size" "$kvhead" "$mode"
        done
    done
done

if ((DRY_RUN == 0)); then
    printf '\nAll jobs completed at %s\nResults: %s\n' \
        "$(date --iso-8601=seconds)" "$LOG_DIR" | tee -a "$MASTER_LOG"
fi
