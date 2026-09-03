#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR/infer${PYTHONPATH:+:$PYTHONPATH}"
TRACE=${TRACE:-$ROOT_DIR/dcp_test/trace/conversation_trace.jsonl}
WORKLOAD=${WORKLOAD:-$ROOT_DIR/.cache/vllm_dcp/smoke-workload.json}
RESULT_DIR=${RESULT_DIR:-$ROOT_DIR/benchmark_logs/vllm_dcp/smoke}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.05}
NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-1}
KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-1073741824}
PORT=${PORT:-18000}

mkdir -p "$(dirname -- "$WORKLOAD")" "$RESULT_DIR"
"$PYTHON" -m vllm_bench.workload \
    --trace "$TRACE" \
    --output "$WORKLOAD" \
    --warmup-requests 1 \
    --num-requests 2

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
    "$PYTHON" -m vllm_bench.matrix \
        --workload "$WORKLOAD" \
        --result-dir "$RESULT_DIR" \
        --port "$PORT" \
        --kv-heads 1 \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
        --num-hidden-layers "$NUM_HIDDEN_LAYERS" \
        --arrival-time-scales 4 \
        --backends vllm-ag-rs vllm-a2a mega
