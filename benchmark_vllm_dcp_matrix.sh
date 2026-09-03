#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR/infer${PYTHONPATH:+:$PYTHONPATH}"
TRACE=${TRACE:-$ROOT_DIR/dcp_test/trace/conversation_trace.jsonl}
WORKLOAD=${WORKLOAD:-$ROOT_DIR/.cache/vllm_dcp/workload-100-1000.json}
RESULT_DIR=${RESULT_DIR:-$ROOT_DIR/benchmark_logs/vllm_dcp/kvh1}
KV_HEADS=${KV_HEADS:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-32}
PORT=${PORT:-18000}

mkdir -p "$(dirname -- "$WORKLOAD")" "$RESULT_DIR"
if [[ ! -f "$WORKLOAD" ]]; then
    "$PYTHON" -m vllm_bench.workload \
        --trace "$TRACE" \
        --output "$WORKLOAD" \
        --warmup-requests 100 \
        --num-requests 1000
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
    "$PYTHON" -m vllm_bench.matrix \
        --workload "$WORKLOAD" \
        --result-dir "$RESULT_DIR" \
        --port "$PORT" \
        --kv-heads "$KV_HEADS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --num-hidden-layers "$NUM_HIDDEN_LAYERS" \
        --arrival-time-scales 1 2 4 \
        --backends vllm-ag-rs vllm-a2a mega

"$PYTHON" -m vllm_bench.summarize --result-dir "$RESULT_DIR"
