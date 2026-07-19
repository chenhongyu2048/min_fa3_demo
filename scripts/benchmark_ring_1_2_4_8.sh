#!/usr/bin/env bash

# Single-node 1/2/4/8-GPU varlen causal benchmark sweep.
# Run this inside an allocation that exposes at least eight SM90 GPUs.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

GPU_COUNTS=${GPU_COUNTS:-"1 2 4 8"}
WARMUP_ITERS=${WARMUP_ITERS:-10}
NUM_ITERS=${NUM_ITERS:-40}
QHEAD=${QHEAD:-32}
KVHEAD=${KVHEAD:-8}
HEADDIM=${HEADDIM:-128}
ALLGATHER_OVERLAPPING_HEADS_K_STRIDE=${ALLGATHER_OVERLAPPING_HEADS_K_STRIDE:-1}
CHECK=${CHECK:-0}
DRY_RUN=${DRY_RUN:-0}
TORCHRUN=${TORCHRUN:-torchrun}
LOG_DIR=${LOG_DIR:-"benchmark_logs/$(date +%Y%m%d-%H%M%S)"}
LOG_FILE=${LOG_FILE:-"$LOG_DIR/benchmark_ring_1_2_4_8.log"}

# Each setting keeps 16K tokens per rank: B=16,8,4,2,1 is paired with local
# S=1K,2K,4K,8K,16K. Global sequence lengths therefore scale with world size.
ALL_CP_BATCH=${ALL_CP_BATCH:-"16,8,4,2,1"}
ALL_CP_GLOBAL_SEQLENS_1=${ALL_CP_GLOBAL_SEQLENS_1:-"1024,2048,4096,8192,16384"}
ALL_CP_GLOBAL_SEQLENS_2=${ALL_CP_GLOBAL_SEQLENS_2:-"2048,4096,8192,16384,32768"}
ALL_CP_GLOBAL_SEQLENS_4=${ALL_CP_GLOBAL_SEQLENS_4:-"4096,8192,16384,32768,65536"}
ALL_CP_GLOBAL_SEQLENS_8=${ALL_CP_GLOBAL_SEQLENS_8:-"8192,16384,32768,65536,131072"}
ALL_CP_METHODS=${ALL_CP_METHODS:-all}

BACKWARD_BATCH=${BACKWARD_BATCH:-"16,8,4,2,1"}
BACKWARD_GLOBAL_SEQLENS_1=${BACKWARD_GLOBAL_SEQLENS_1:-"1024,2048,4096,8192,16384"}
BACKWARD_GLOBAL_SEQLENS_2=${BACKWARD_GLOBAL_SEQLENS_2:-"2048,4096,8192,16384,32768"}
BACKWARD_GLOBAL_SEQLENS_4=${BACKWARD_GLOBAL_SEQLENS_4:-"4096,8192,16384,32768,65536"}
BACKWARD_GLOBAL_SEQLENS_8=${BACKWARD_GLOBAL_SEQLENS_8:-"8192,16384,32768,65536,131072"}
BACKWARD_METHODS=${BACKWARD_METHODS:-all}

# Hierarchical hybrid metadata for each supported physical world size. Each
# case fuses every legal group size for that world in one persistent launch.
HYBRID_GLOBAL_SEQLENS_2=${HYBRID_GLOBAL_SEQLENS_2:-"8192,1024,1024"}
HYBRID_RING_SIZES_2=${HYBRID_RING_SIZES_2:-"2,1,1"}
HYBRID_RING_STARTS_2=${HYBRID_RING_STARTS_2:-"0,0,1"}

HYBRID_GLOBAL_SEQLENS_4=${HYBRID_GLOBAL_SEQLENS_4:-"49152,4096,2048,2048,1024,1024,1024,1024,1024,1024,1024,1024"}
HYBRID_RING_SIZES_4=${HYBRID_RING_SIZES_4:-"4,4,2,2,1,1,1,1,1,1,1,1"}
HYBRID_RING_STARTS_4=${HYBRID_RING_STARTS_4:-"0,0,0,2,0,0,1,1,2,2,3,3"}

HYBRID_GLOBAL_SEQLENS_8=${HYBRID_GLOBAL_SEQLENS_8:-"98304,8192,4096,4096,2048,2048,2048,2048,2048,2048,2048,2048"}
HYBRID_RING_SIZES_8=${HYBRID_RING_SIZES_8:-"8,8,4,4,1,1,1,1,1,1,1,1"}
HYBRID_RING_STARTS_8=${HYBRID_RING_STARTS_8:-"0,0,0,4,0,1,2,3,4,5,6,7"}

# Defaults follow the H200 configurations used by the existing Slurm scripts.
# A one-GPU forward run does not need communication CTAs.
FORWARD_SM_CONFIGS_SINGLE=${FORWARD_SM_CONFIGS_SINGLE:-"132:0"}
FORWARD_SM_CONFIGS_MULTI=${FORWARD_SM_CONFIGS_MULTI:-"128:4,124:8,120:12,116:16"}
BACKWARD_SM_CONFIGS=${BACKWARD_SM_CONFIGS:-"128:4,124:8,120:12,116:16"}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

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

gpu_counts_spec=${GPU_COUNTS//,/ }
read -r -a GPU_COUNT_LIST <<< "$gpu_counts_spec"
((${#GPU_COUNT_LIST[@]} > 0)) || die "GPU_COUNTS must not be empty"

max_gpu_count=0
for world_size in "${GPU_COUNT_LIST[@]}"; do
    [[ "$world_size" =~ ^[1-8]$ ]] || die "GPU counts must be integers in [1, 8], got '$world_size'"
    ((world_size > max_gpu_count)) && max_gpu_count=$world_size
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

divide_global_seqlens() {
    local spec=$1
    local world_size=$2
    local alignment=$3
    local label=$4
    local token value local_value
    local -a global_values local_values

    IFS=',' read -r -a global_values <<< "$spec"
    ((${#global_values[@]} > 0)) || die "$label must not be empty"
    local_values=()
    for token in "${global_values[@]}"; do
        token=${token//[[:space:]]/}
        [[ "$token" =~ ^[0-9]+$ ]] || die "$label contains a non-integer length '$token'"
        value=$((10#$token))
        ((value > 0)) || die "$label lengths must be positive"
        ((value % world_size == 0)) || die \
            "$label length $value is not divisible by world size $world_size"
        local_value=$((value / world_size))
        ((local_value % alignment == 0)) || die \
            "$label produces local length $local_value, which is not aligned to $alignment"
        local_values+=("$local_value")
    done

    local IFS=,
    LOCAL_SEQLENS=${local_values[*]}
}

print_command() {
    printf 'CUDA_VISIBLE_DEVICES=%q' "$1"
    shift
    printf ' %q' "$@"
    printf '\n'
}

run_benchmark() {
    local label=$1
    local world_size=$2
    local visible_devices=$3
    local entrypoint=$4
    shift 4

    local -a command=(
        "$TORCHRUN" --standalone --nproc_per_node="$world_size" "$entrypoint" "$@"
    )

    if ((DRY_RUN)); then
        printf '\n================================================================================\n'
        printf '[%s] GPUs=%s, visible=%s\n' "$label" "$world_size" "$visible_devices"
        printf '================================================================================\n'
        print_command "$visible_devices" "${command[@]}"
        return
    fi

    {
        printf '\n================================================================================\n'
        printf '[%s] GPUs=%s, visible=%s\n' "$label" "$world_size" "$visible_devices"
        printf '================================================================================\n'
        print_command "$visible_devices" "${command[@]}"
    } | tee -a "$LOG_FILE"
    CUDA_VISIBLE_DEVICES="$visible_devices" "${command[@]}" 2>&1 | tee -a "$LOG_FILE"
}

run_hierarchical_hybrid() {
    local world_size=$1
    local visible_devices=$2
    local sm_configs=$3
    local global_seqlens ring_sizes ring_starts

    case "$world_size" in
        2)
            global_seqlens=$HYBRID_GLOBAL_SEQLENS_2
            ring_sizes=$HYBRID_RING_SIZES_2
            ring_starts=$HYBRID_RING_STARTS_2
            ;;
        4)
            global_seqlens=$HYBRID_GLOBAL_SEQLENS_4
            ring_sizes=$HYBRID_RING_SIZES_4
            ring_starts=$HYBRID_RING_STARTS_4
            ;;
        8)
            global_seqlens=$HYBRID_GLOBAL_SEQLENS_8
            ring_sizes=$HYBRID_RING_SIZES_8
            ring_starts=$HYBRID_RING_STARTS_8
            ;;
        1)
            echo "Skipping hierarchical hybrid forward for world_size=1"
            return
            ;;
        *) die "hierarchical hybrid forward does not support world_size=$world_size" ;;
    esac

    run_benchmark forward_hybrid "$world_size" "$visible_devices" \
        ring_test/benchmark_topology_forward.py \
        --global-seqlens "$global_seqlens" \
        --ring-sizes "$ring_sizes" \
        --ring-starts "$ring_starts" \
        --qhead "$QHEAD" --kvhead "$KVHEAD" --headdim "$HEADDIM" \
        --allgather-overlapping-heads-k-stride "$ALLGATHER_OVERLAPPING_HEADS_K_STRIDE" \
        --mode causal --methods all --sm-configs "$sm_configs" \
        --warmup-iters "$WARMUP_ITERS" --num-iters "$NUM_ITERS" \
        "${CHECK_ARGS[@]}"
}

mkdir -p "$(dirname -- "$LOG_FILE")"
if ((DRY_RUN == 0)); then
    : > "$LOG_FILE"
fi

echo "Log: $LOG_FILE"
echo "Common config: causal varlen, QH=$QHEAD, KVH=$KVHEAD, D=$HEADDIM, allgather_overlapping_heads_k_stride=$ALLGATHER_OVERLAPPING_HEADS_K_STRIDE, warmup=$WARMUP_ITERS, iters=$NUM_ITERS, check=$CHECK"

for world_size in "${GPU_COUNT_LIST[@]}"; do
    select_devices "$world_size"
    visible_devices=$SELECTED_DEVICES

    case "$world_size" in
        1)
            forward_sm_configs=$FORWARD_SM_CONFIGS_SINGLE
            all_cp_global_seqlens=$ALL_CP_GLOBAL_SEQLENS_1
            backward_global_seqlens=$BACKWARD_GLOBAL_SEQLENS_1
            ;;
        2)
            forward_sm_configs=$FORWARD_SM_CONFIGS_MULTI
            all_cp_global_seqlens=$ALL_CP_GLOBAL_SEQLENS_2
            backward_global_seqlens=$BACKWARD_GLOBAL_SEQLENS_2
            ;;
        4)
            forward_sm_configs=$FORWARD_SM_CONFIGS_MULTI
            all_cp_global_seqlens=$ALL_CP_GLOBAL_SEQLENS_4
            backward_global_seqlens=$BACKWARD_GLOBAL_SEQLENS_4
            ;;
        8)
            forward_sm_configs=$FORWARD_SM_CONFIGS_MULTI
            all_cp_global_seqlens=$ALL_CP_GLOBAL_SEQLENS_8
            backward_global_seqlens=$BACKWARD_GLOBAL_SEQLENS_8
            ;;
        *)
            die "no all-CP workload is configured for world size $world_size"
            ;;
    esac

    # Ordinary all-CP forward takes a rank-local --seqlen list. Each value is
    # the selected global sequence length divided by physical world size.
    divide_global_seqlens "$all_cp_global_seqlens" "$world_size" 256 "ALL_CP_GLOBAL_SEQLENS_$world_size"
    all_cp_local_seqlens=$LOCAL_SEQLENS
    run_benchmark forward_all_cp "$world_size" "$visible_devices" \
        ring_test/benchmark_ring_forward.py \
        --b "$ALL_CP_BATCH" \
        --seqlen "$all_cp_local_seqlens" \
        --qhead "$QHEAD" --kvhead "$KVHEAD" --headdim "$HEADDIM" \
        --allgather-overlapping-heads-k-stride "$ALLGATHER_OVERLAPPING_HEADS_K_STRIDE" \
        --mode causal --methods "$ALL_CP_METHODS" \
        --sm-configs "$forward_sm_configs" \
        --warmup-iters "$WARMUP_ITERS" --num-iters "$NUM_ITERS" \
        "${CHECK_ARGS[@]}"

    # Hierarchical hybrid forward takes a global --global-seqlens list. Each
    # entry's rank-local length is global_seqlen / ring_size on ranks belonging
    # to [ring_start, ring_start + ring_size), and zero on all other ranks.
    run_hierarchical_hybrid "$world_size" "$visible_devices" "$forward_sm_configs"

    # Ordinary all-CP backward takes a rank-local --seqlen list, with the same
    # world-size selection and division as ordinary forward.
    divide_global_seqlens "$backward_global_seqlens" "$world_size" 256 "BACKWARD_GLOBAL_SEQLENS_$world_size"
    backward_local_seqlens=$LOCAL_SEQLENS
    run_benchmark backward "$world_size" "$visible_devices" \
        ring_test/benchmark_ring_backward.py \
        --b "$BACKWARD_BATCH" \
        --seqlen "$backward_local_seqlens" \
        --qhead "$QHEAD" --kvhead "$KVHEAD" --headdim "$HEADDIM" \
        --allgather-overlapping-heads-k-stride "$ALLGATHER_OVERLAPPING_HEADS_K_STRIDE" \
        --methods "$BACKWARD_METHODS" \
        --sm-configs "$BACKWARD_SM_CONFIGS" \
        --warmup-iters "$WARMUP_ITERS" --num-iters "$NUM_ITERS" \
        "${CHECK_ARGS[@]}"
done

echo
echo "Benchmark sweep complete. Log: $LOG_FILE"
