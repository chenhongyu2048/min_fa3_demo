#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export NCCL_CGA_CLUSTER_SIZE=${NCCL_CGA_CLUSTER_SIZE:-1}
export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
export MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE=${MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

PYTHON=${PYTHON:-$ROOT_DIR/.venv/bin/python}
TORCHRUN=${TORCHRUN:-$ROOT_DIR/.venv/bin/torchrun}
MEGATRON_PATH=${MEGATRON_PATH:-$ROOT_DIR/third_party/Megatron-LM}
DATASETS=${DATASETS:-arxiv,github,pile,freelaw,prolong}
NUM_CASES=${NUM_CASES:-20}
SEED=${SEED:-0}
TARGET_TOKENS=${TARGET_TOKENS:-131072}
WARMUP_ITERS=${WARMUP_ITERS:-10}
NUM_ITERS=${NUM_ITERS:-40}
# Python defines the random-weight Qwen3-30B-A3B layer, uniform MoE routing,
# and TP=1, CP=EP=WORLD_SIZE. This experiment excludes Ulysses/USP by default.
METHODS=${METHODS:-"allgather_attention,llama3_allgather_attention,fa3_ring,megatron_hybrid_cp,magi_attention,zeppelin,mega_ring_all_cp,mega_ring_hybrid"}
WORLD_SIZE=${WORLD_SIZE:-8}
SM_CONFIGS=${SM_CONFIGS:-"128:4,124:8,120:12,116:16"}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/results/transformer_layer_cp}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
DRY_RUN=${DRY_RUN:-0}

case "$DRY_RUN" in
    0|1) ;;
    *)
        echo "error: DRY_RUN must be 0 or 1, got '$DRY_RUN'" >&2
        exit 2
        ;;
esac

SITE_PACKAGES=$(
    "$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)
NVIDIA_LIBRARY_PATH=$(
    "$PYTHON" - <<'PY'
import sysconfig
from pathlib import Path

root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()))
PY
)
export LD_LIBRARY_PATH="$NVIDIA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if ((DRY_RUN == 0)); then
    mkdir -p "$OUTPUT_DIR"
fi

IFS=',' read -r -a dataset_list <<< "$DATASETS"
for dataset in "${dataset_list[@]}"; do
    dataset=${dataset//[[:space:]]/}
    if [[ -z "$dataset" ]]; then
        continue
    fi
    output_jsonl="$OUTPUT_DIR/${RUN_ID}-${dataset}.jsonl"
    args=(
        --standalone
        --nnodes=1
        --nproc-per-node="$WORLD_SIZE"
        ring_test/benchmark_transformer_layer.py
        --dataset "$dataset"
        --target-tokens "$TARGET_TOKENS"
        --seed "$SEED"
        --num-cases "$NUM_CASES"
        --world-size "$WORLD_SIZE"
        --methods "$METHODS"
        --warmup-iters "$WARMUP_ITERS"
        --num-iters "$NUM_ITERS"
        --output-jsonl "$output_jsonl"
        --megatron-path "$MEGATRON_PATH"
        --compute-balance-tolerance 0.05
        --token-balance-tolerance 0.05
        --beam-width 64
        --finalist-count 8
        --structure-threshold 0.5
        --max-repair-iterations 32
        --megatron-max-seqlen-per-rank 8192
        --zeppelin-threshold 4096
        --magi-overlap-degree 2
        --allgather-heads-k-stride 4
    )
    args+=(--sm-configs "$SM_CONFIGS")
    echo "Launching dataset=$dataset -> $output_jsonl"
    if ((DRY_RUN)); then
        printf 'CUDA_VISIBLE_DEVICES=%q' "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
        printf ' %q' "$TORCHRUN" "${args[@]}"
        printf '\n'
    else
        "$TORCHRUN" "${args[@]}"
    fi
done
