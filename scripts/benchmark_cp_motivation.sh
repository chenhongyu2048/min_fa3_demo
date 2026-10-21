#!/usr/bin/env bash

# Megatron hybrid CP: normal, resident-KV compute-only, and one-step compute.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

VENV_ACTIVATE=${VENV_ACTIVATE:-"$ROOT_DIR/.venv/bin/activate"}
if [[ -f "$VENV_ACTIVATE" ]]; then
    source "$VENV_ACTIVATE"
fi

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NCCL_CGA_CLUSTER_SIZE=${NCCL_CGA_CLUSTER_SIZE:-1}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

GPUS=${GPUS:-8}
DATASETS=${DATASETS:-"arxiv freelaw github pile prolong"}
LOG_DIR=${LOG_DIR:-"benchmark_logs/$(date +%Y%m%d-%H%M%S)-cp-motivation"}
read -r -a DATASET_LIST <<< "${DATASETS//,/ }"
ARGS=(
    --datasets "${DATASET_LIST[@]}" --world-size "$GPUS"
    --target-tokens "${TARGET_TOKENS:-131072}" --num-cases "${NUM_CASES:-4}"
    --qhead "${QHEAD:-32}" --kvhead "${KVHEAD:-8}" --headdim 128
    --megatron-max-seqlen-per-rank "${MEGATRON_MAX_SEQLEN_PER_RANK:-8192}"
    --seed "${SEED:-0}" --warmup-iters "${WARMUP_ITERS:-10}" --num-iters "${NUM_ITERS:-40}"
)
case "${CHECK:-1}" in
    1) ARGS+=(--check) ;;
    0) ARGS+=(--no-check) ;;
    *) echo 'CHECK must be 0 or 1' >&2; exit 1 ;;
esac

if [[ ${DRY_RUN:-0} == 1 ]]; then
    "${PYTHON:-python}" -m ring_test.benchmark_cp_motivation "${ARGS[@]}" --print-workload
else
    mkdir -p "$LOG_DIR"
    "${TORCHRUN:-torchrun}" --standalone --nproc_per_node="$GPUS" \
        -m ring_test.benchmark_cp_motivation "${ARGS[@]}" --output "$LOG_DIR/results.jsonl" \
        2>&1 | tee "$LOG_DIR/console.log"
fi
