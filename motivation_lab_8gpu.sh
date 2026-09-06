#!/usr/bin/env bash

# Complete eight-GPU MegaCP motivation batch. This script is self-contained.

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
TORCHRUN_BIN=${TORCHRUN_BIN:-.venv/bin/torchrun}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
RUN_ID=${RUN_ID:-motivation_8gpu_$(date -u +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-benchmark_logs/motivation/${RUN_ID}}

RUN_T1=${RUN_T1:-1}
RUN_T2=${RUN_T2:-1}
RUN_T3=${RUN_T3:-1}
RUN_D1=${RUN_D1:-1}

T1_CASE_MANIFEST=${T1_CASE_MANIFEST:-motivation/t1_uniform_cases_cp8.json}
T1_CASE_IDS=${T1_CASE_IDS:-context65k_b1,context65k_b2,context65k_b4,context65k_b8,context65k_b16}
T1_WARMUP=${T1_WARMUP:-40}
T1_ITERS=${T1_ITERS:-60}
T1_NUM_COMM_SM=${T1_NUM_COMM_SM:-8}

T2_CASE_MANIFEST=${T2_CASE_MANIFEST:-$T1_CASE_MANIFEST}
T2_CASE_IDS=${T2_CASE_IDS:-$T1_CASE_IDS}
T2_WARMUP=${T2_WARMUP:-40}
T2_ITERS=${T2_ITERS:-60}

T3_DATASETS=${T3_DATASETS:-arxiv,github,pile,freelaw,prolong}
T3_TARGET_TOKENS=${T3_TARGET_TOKENS:-131072}
T3_NUM_CASES=${T3_NUM_CASES:-30}
T3_SEED=${T3_SEED:-0}

D1_CASE_MANIFEST=${D1_CASE_MANIFEST:-motivation/d1_cases_dcp4.json}
D1_TOPOLOGIES=${D1_TOPOLOGIES:-8:1,8:2,8:4}
D1_WARMUP=${D1_WARMUP:-20}
D1_ITERS=${D1_ITERS:-30}
D1_NUM_COMM_SM=${D1_NUM_COMM_SM:-4}

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NCCL_CGA_CLUSTER_SIZE=${NCCL_CGA_CLUSTER_SIZE:-1}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

usage() {
    cat <<'EOF'
Usage: ./motivation_lab_8gpu.sh

Runs complete CP8 T1/T2/T3/D1 experiments on GPU_IDS=0,...,7 by default.
D1 topologies are CP8/KVH1 (DCP8), CP8/KVH2 (two DCP4 groups), and
CP8/KVH4 (four DCP2 groups). Use RUN_T1/RUN_T2/RUN_T3/RUN_D1=0 to select.
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then usage; exit 0; fi
if (($#)); then echo "error: use environment variables, not positional arguments" >&2; exit 2; fi

for flag in RUN_T1 RUN_T2 RUN_T3 RUN_D1; do
    [[ ${!flag} == 0 || ${!flag} == 1 ]] || { echo "error: $flag must be 0 or 1" >&2; exit 1; }
done
for setting in T1_WARMUP T1_ITERS T1_NUM_COMM_SM T2_WARMUP T2_ITERS \
    T3_TARGET_TOKENS T3_NUM_CASES D1_WARMUP D1_ITERS D1_NUM_COMM_SM; do
    [[ ${!setting} =~ ^[0-9]+$ ]] || { echo "error: $setting must be non-negative" >&2; exit 1; }
done
((T1_ITERS && T2_ITERS && T3_TARGET_TOKENS && T3_NUM_CASES && D1_ITERS)) || {
    echo "error: iteration counts and T3 sizes must be positive" >&2
    exit 1
}
[[ $T3_SEED =~ ^-?[0-9]+$ ]] || { echo "error: T3_SEED must be an integer" >&2; exit 1; }

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
((${#GPU_LIST[@]} == 8)) || { echo "error: GPU_IDS must contain exactly eight devices" >&2; exit 1; }
for gpu_id in "${GPU_LIST[@]}"; do
    [[ $gpu_id =~ ^[0-9]+$ ]] || { echo "error: invalid GPU_IDS '$GPU_IDS'" >&2; exit 1; }
done
[[ -x $PYTHON_BIN && -x $TORCHRUN_BIN ]] || { echo "error: Python/torchrun executable is missing" >&2; exit 1; }
if [[ $RUN_T1 == 1 || $RUN_T2 == 1 || $RUN_D1 == 1 ]]; then
    [[ -f _min_fa3_op.so ]] || { echo "error: _min_fa3_op.so is missing" >&2; exit 1; }
fi
[[ $RUN_T1 == 0 || -f $T1_CASE_MANIFEST ]] || { echo "error: missing T1 manifest" >&2; exit 1; }
[[ $RUN_T2 == 0 || -f $T2_CASE_MANIFEST ]] || { echo "error: missing T2 manifest" >&2; exit 1; }
[[ $RUN_D1 == 0 || -f $D1_CASE_MANIFEST ]] || { echo "error: missing D1 manifest" >&2; exit 1; }

IFS=',' read -r -a D1_TOPOLOGY_LIST <<< "$D1_TOPOLOGIES"
for topology in "${D1_TOPOLOGY_LIST[@]}"; do
    [[ $topology =~ ^8:(1|2|4)$ ]] || { echo "error: CP8 D1 topology must be 8:1, 8:2, or 8:4" >&2; exit 1; }
done

mkdir -p "$OUT_DIR/t1_training_overlap" "$OUT_DIR/t2_training_step" \
    "$OUT_DIR/t3_load_balance" "$OUT_DIR/d1_decode_sm_trace"
[[ $RUN_T1 == 0 ]] || cp "$T1_CASE_MANIFEST" "$OUT_DIR/t1_training_overlap/case_manifest.json"
[[ $RUN_T2 == 0 ]] || cp "$T2_CASE_MANIFEST" "$OUT_DIR/t2_training_step/case_manifest.json"
[[ $RUN_D1 == 0 ]] || cp "$D1_CASE_MANIFEST" "$OUT_DIR/d1_decode_sm_trace/case_manifest.json"

run_distributed() {
    local label=$1 log_file=$2
    shift 2
    local -a command=("$TORCHRUN_BIN" --standalone --nproc_per_node=8 "$@")
    {
        printf '\n[%s] CUDA_VISIBLE_DEVICES=%q' "$label" "$GPU_IDS"
        printf ' %q' "${command[@]}"
        printf '\n'
        CUDA_VISIBLE_DEVICES="$GPU_IDS" "${command[@]}"
    } 2>&1 | tee "$log_file"
}

run_local() {
    local label=$1 log_file=$2
    shift 2
    { printf '\n[%s]' "$label"; printf ' %q' "$@"; printf '\n'; "$@"; } 2>&1 | tee "$log_file"
}

echo "Formal eight-GPU motivation batch: $OUT_DIR"
echo "CP8 allocation: CUDA_VISIBLE_DEVICES=$GPU_IDS"
echo "T1/T2 warmup/iters: $T1_WARMUP/$T1_ITERS and $T2_WARMUP/$T2_ITERS"
echo "T3: datasets=$T3_DATASETS target_tokens=$T3_TARGET_TOKENS cases=$T3_NUM_CASES"

if [[ $RUN_T1 == 1 ]]; then
    IFS=',' read -r -a CASES <<< "$T1_CASE_IDS"
    for case_id in "${CASES[@]}"; do
        [[ $case_id =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "error: invalid T1 case" >&2; exit 1; }
        run_distributed "T1_${case_id}_CP8" "$OUT_DIR/t1_training_overlap/t1_${case_id}.log" \
            motivation/training_overlap.py --qhead 32 --kvhead 8 --headdim 128 \
            --mode causal --methods ring,allgather,mega \
            --case-manifest "$T1_CASE_MANIFEST" --case-id "$case_id" \
            --num-comp-sm 0 --num-comm-sm "$T1_NUM_COMM_SM" \
            --warmup "$T1_WARMUP" --iters "$T1_ITERS" \
            --output-json "$OUT_DIR/t1_training_overlap/t1_${case_id}.json"
    done
else echo "T1 skipped"; fi

if [[ $RUN_T2 == 1 ]]; then
    IFS=',' read -r -a CASES <<< "$T2_CASE_IDS"
    for case_id in "${CASES[@]}"; do
        [[ $case_id =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "error: invalid T2 case" >&2; exit 1; }
        run_distributed "T2_${case_id}_CP8" "$OUT_DIR/t2_training_step/t2_${case_id}.log" \
            motivation/training_step.py --qhead 32 --kvhead 8 --headdim 128 \
            --case-manifest "$T2_CASE_MANIFEST" --case-id "$case_id" \
            --num-comp-sm 0 --num-comm-sm 0 --warmup "$T2_WARMUP" \
            --iters "$T2_ITERS" --output-json "$OUT_DIR/t2_training_step/t2_${case_id}.json"
    done
else echo "T2 skipped"; fi

if [[ $RUN_T3 == 1 ]]; then
    run_local "T3_CP8" "$OUT_DIR/t3_load_balance/t3_cp8.log" \
        "$PYTHON_BIN" motivation/load_balance.py --datasets "$T3_DATASETS" \
        --target-tokens "$T3_TARGET_TOKENS" --num-cases "$T3_NUM_CASES" \
        --world-size 8 --qhead 32 --kvhead 8 --headdim 128 --seed "$T3_SEED" \
        --output-json "$OUT_DIR/t3_load_balance/t3_cp8.json"
else echo "T3 skipped"; fi

if [[ $RUN_D1 == 1 ]]; then
    for topology in "${D1_TOPOLOGY_LIST[@]}"; do
        kvh=${topology#*:}
        run_distributed "D1_CP8_KVH${kvh}" "$OUT_DIR/d1_decode_sm_trace/d1_cp8_kvh${kvh}.log" \
            motivation/decode_sm_trace.py --case-manifest "$D1_CASE_MANIFEST" \
            --qhead 32 --kvhead "$kvh" --tp-size 8 --num-comm-sm "$D1_NUM_COMM_SM" \
            --warmup "$D1_WARMUP" --iters "$D1_ITERS" \
            --output-json "$OUT_DIR/d1_decode_sm_trace/d1_cp8_kvh${kvh}.json" \
            --output-jsonl "$OUT_DIR/d1_decode_sm_trace/d1_cp8_kvh${kvh}.jsonl"
    done
else echo "D1 skipped"; fi

cat <<EOF

Completed formal eight-GPU motivation batch: $OUT_DIR
D2 status: pending; this runner does not emit HBM/DRAM conclusions.
EOF
