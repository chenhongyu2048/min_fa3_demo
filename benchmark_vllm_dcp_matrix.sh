#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR/infer${PYTHONPATH:+:$PYTHONPATH}"
TRACE=${TRACE:-$ROOT_DIR/dcp_test/trace/conversation_trace.jsonl}
WORKLOAD=${WORKLOAD:-$ROOT_DIR/.cache/vllm_dcp/workload-100-1000.json}
RESULT_DIR=${RESULT_DIR:-$ROOT_DIR/benchmark_logs/vllm_dcp}
KV_HEADS=${KV_HEADS:-1,2,4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-32}
PORT=${PORT:-18000}

kv_heads_spec=${KV_HEADS//,/ }
read -r -a KV_HEAD_LIST <<< "$kv_heads_spec"
if ((${#KV_HEAD_LIST[@]} == 0)); then
    echo "KV_HEADS must not be empty" >&2
    exit 2
fi
for kv_heads in "${KV_HEAD_LIST[@]}"; do
    case "$kv_heads" in
        1|2|4) ;;
        *)
            echo "KV_HEADS must contain only 1, 2, or 4, got '$kv_heads'" >&2
            exit 2
            ;;
    esac
done

mkdir -p "$(dirname -- "$WORKLOAD")" "$RESULT_DIR"
if [[ ! -f "$WORKLOAD" ]]; then
    "$PYTHON" -m vllm_bench.workload \
        --trace "$TRACE" \
        --output "$WORKLOAD" \
        --warmup-requests 100 \
        --num-requests 1000
fi

for kv_heads in "${KV_HEAD_LIST[@]}"; do
    kv_result_dir="$RESULT_DIR/kvh$kv_heads"
    mkdir -p "$kv_result_dir"

    echo "Running vLLM DCP matrix with KV_HEADS=$kv_heads; results=$kv_result_dir"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
        "$PYTHON" -m vllm_bench.matrix \
            --workload "$WORKLOAD" \
            --result-dir "$kv_result_dir" \
            --port "$PORT" \
            --kv-heads "$kv_heads" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --num-hidden-layers "$NUM_HIDDEN_LAYERS" \
            --arrival-time-scales 1 2 4 \
            --backends vllm-ag-rs vllm-a2a mega

    "$PYTHON" -m vllm_bench.summarize --result-dir "$kv_result_dir"
done
