#!/usr/bin/env bash

# Minimal but comprehensive Ulysses/USP smoke test.
#
# One eight-GPU torchrun is launched for each KV-head count.  Forward runs both
# noncausal and causal cases; backward runs its causal varlen path.  With
# KVH=1,2,4,8 this covers Ulysses KV replication and USP's U=min(CP, KVH),
# R=CP/U topology, including the causal USP ring/zigzag path when R>1.

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

WORLD_SIZE=${WORLD_SIZE:-8}
GPU_LIST=${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}}
QHEAD=${QHEAD:-32}
HEADDIM=${HEADDIM:-128}
BATCH_SIZE=${BATCH_SIZE:-1}
LOCAL_SEQLEN=${LOCAL_SEQLEN:-256}
WARMUP_ITERS=${WARMUP_ITERS:-0}
NUM_ITERS=${NUM_ITERS:-1}
SEED=${SEED:-1234}
ATOL_FORWARD=${ATOL_FORWARD:-0.2}
RTOL_FORWARD=${RTOL_FORWARD:-0.2}
ATOL_BACKWARD=${ATOL_BACKWARD:-0.3}
RTOL_BACKWARD=${RTOL_BACKWARD:-0.3}
ARXIV_TARGET_TOKENS=${ARXIV_TARGET_TOKENS:-16385}
ARXIV_MODE=${ARXIV_MODE:-causal}
ARXIV_KVHEADS=${ARXIV_KVHEADS:-1,2,4,8}
TORCHRUN=${TORCHRUN:-"$ROOT_DIR/.venv/bin/torchrun"}
PYTHONPATH_EXTRA=${PYTHONPATH_EXTRA:-"$ROOT_DIR"}
CHECK=${CHECK:-1}

die() {
    echo "error: $*" >&2
    exit 2
}

[[ "$WORLD_SIZE" == 8 ]] || die "WORLD_SIZE must be 8 for the KVH=1,2,4,8 smoke matrix"
[[ "$QHEAD" =~ ^[1-9][0-9]*$ ]] || die "QHEAD must be a positive integer"
[[ "$HEADDIM" == 128 ]] || die "HEADDIM must be 128"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "BATCH_SIZE must be a positive integer"
[[ "$LOCAL_SEQLEN" =~ ^[1-9][0-9]*$ ]] || die "LOCAL_SEQLEN must be a positive integer"
((LOCAL_SEQLEN % 256 == 0)) || die "LOCAL_SEQLEN must be divisible by 256"
[[ "$WARMUP_ITERS" =~ ^[0-9]+$ ]] || die "WARMUP_ITERS must be non-negative"
[[ "$NUM_ITERS" =~ ^[1-9][0-9]*$ ]] || die "NUM_ITERS must be positive"
[[ "$ARXIV_TARGET_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "ARXIV_TARGET_TOKENS must be positive"
arxiv_kvheads_spec=${ARXIV_KVHEADS//,/ }
read -r -a ARXIV_KVHEAD_LIST <<< "$arxiv_kvheads_spec"
(( ${#ARXIV_KVHEAD_LIST[@]} > 0 )) || die "ARXIV_KVHEADS must not be empty"
for arxiv_kvhead in "${ARXIV_KVHEAD_LIST[@]}"; do
    case "$arxiv_kvhead" in
        1|2|4|8) ;;
        *) die "ARXIV_KVHEADS must contain only 1, 2, 4, or 8" ;;
    esac
done
case "$ARXIV_MODE" in
    noncausal|causal|both) ;;
    *) die "ARXIV_MODE must be noncausal, causal, or both" ;;
esac
[[ "$CHECK" =~ ^[01]$ ]] || die "CHECK must be 0 or 1"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
(( ${#GPU_ARRAY[@]} >= WORLD_SIZE )) || \
    die "GPU_LIST must expose at least ${WORLD_SIZE} comma-separated GPUs"

command -v "$TORCHRUN" >/dev/null 2>&1 || die "torchrun executable not found: $TORCHRUN"

if [[ "$CHECK" == 1 ]]; then
    CHECK_ARGS=(--check)
else
    CHECK_ARGS=(--no-check)
fi

export PYTHONPATH="$PYTHONPATH_EXTRA${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NCCL_CGA_CLUSTER_SIZE=${NCCL_CGA_CLUSTER_SIZE:-1}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}

run_forward() {
    local kvhead=$1
    echo
    echo "================ ULYSSES/USP FORWARD: KVH=${kvhead} ================"
    CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" \
        --standalone \
        --nproc_per_node="$WORLD_SIZE" \
        ring_test/homogeneous_all_cp_microbench/benchmark_ring_forward.py \
        --b "$BATCH_SIZE" \
        --seqlen "$LOCAL_SEQLEN" \
        --qhead "$QHEAD" \
        --kvhead "$kvhead" \
        --headdim "$HEADDIM" \
        --mode both \
        --methods ulysses,usp \
        --warmup-iters "$WARMUP_ITERS" \
        --num-iters "$NUM_ITERS" \
        --seed "$SEED" \
        --atol "$ATOL_FORWARD" \
        --rtol "$RTOL_FORWARD" \
        "${CHECK_ARGS[@]}"
}

run_backward() {
    local kvhead=$1
    echo
    echo "================ ULYSSES/USP BACKWARD: KVH=${kvhead} ================"
    CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" \
        --standalone \
        --nproc_per_node="$WORLD_SIZE" \
        ring_test/homogeneous_all_cp_microbench/benchmark_ring_backward.py \
        --b "$BATCH_SIZE" \
        --seqlen "$LOCAL_SEQLEN" \
        --qhead "$QHEAD" \
        --kvhead "$kvhead" \
        --headdim "$HEADDIM" \
        --methods ulysses,usp \
        --sm-configs 64:8 \
        --warmup-iters "$WARMUP_ITERS" \
        --num-iters "$NUM_ITERS" \
        --seed "$SEED" \
        --atol "$ATOL_BACKWARD" \
        --rtol "$RTOL_BACKWARD" \
        "${CHECK_ARGS[@]}"
}

run_arxiv_case() {
    echo
    echo "================ ARXIV DATASET: ONE CASE, KVH=${ARXIV_KVHEADS} ================"
    CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" \
        --standalone \
        --nproc_per_node="$WORLD_SIZE" \
        ring_test/benchmark_dataset_forward.py \
        --dataset arxiv \
        --target-tokens "$ARXIV_TARGET_TOKENS" \
        --seed "$SEED" \
        --num-cases 1 \
        --world-size "$WORLD_SIZE" \
        --qhead "$QHEAD" \
        --kvheads "$ARXIV_KVHEADS" \
        --headdim "$HEADDIM" \
        --mode "$ARXIV_MODE" \
        --methods ulysses,usp \
        --sm-configs 64:8 \
        --warmup-iters "$WARMUP_ITERS" \
        --num-iters "$NUM_ITERS" \
        --atol "$ATOL_FORWARD" \
        --rtol "$RTOL_FORWARD" \
        "${CHECK_ARGS[@]}"
    CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" \
        --standalone \
        --nproc_per_node="$WORLD_SIZE" \
        ring_test/benchmark_dataset_backward.py \
        --dataset arxiv \
        --target-tokens "$ARXIV_TARGET_TOKENS" \
        --seed "$SEED" \
        --num-cases 1 \
        --world-size "$WORLD_SIZE" \
        --qhead "$QHEAD" \
        --kvheads "$ARXIV_KVHEADS" \
        --headdim "$HEADDIM" \
        --mode "$ARXIV_MODE" \
        --methods ulysses,usp \
        --sm-configs 64:8 \
        --warmup-iters "$WARMUP_ITERS" \
        --num-iters "$NUM_ITERS" \
        --atol "$ATOL_FORWARD" \
        --rtol "$RTOL_FORWARD" \
        "${CHECK_ARGS[@]}"
}

for kvhead in 1 2 4 8; do
    run_forward "$kvhead"
    run_backward "$kvhead"
done

# Exercise the dataset sampler and explicit-topology frontend with exactly one
# sampled Arxiv workload.  --kvheads keeps this to one torchrun while running
# the sampled case for all four supported KV-head counts.
run_arxiv_case

echo
echo "Ulysses/USP KVH=1,2,4,8 plus one Arxiv dataset case completed successfully."
