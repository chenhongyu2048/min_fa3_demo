#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${PYTHON:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR/infer${PYTHONPATH:+:$PYTHONPATH}"
# FlashInfer's sampler is unrelated to this CUSTOM attention benchmark and is
# disabled by default in ``serve.py``.  Only require a complete toolkit when a
# caller explicitly opts that sampler back in.
VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
case "$VLLM_USE_FLASHINFER_SAMPLER" in
    0|1) export VLLM_USE_FLASHINFER_SAMPLER ;;
    *)
        echo "error: VLLM_USE_FLASHINFER_SAMPLER must be 0 or 1, got '$VLLM_USE_FLASHINFER_SAMPLER'" >&2
        exit 2
        ;;
esac
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
if [[ "$VLLM_USE_FLASHINFER_SAMPLER" == 1 && \
    ! -f "$CUDA_HOME/include/cuda_runtime.h" ]]; then
    echo "error: CUDA_HOME='$CUDA_HOME' is not a complete CUDA toolkit; expected" \
        "'$CUDA_HOME/bin/nvcc' and '$CUDA_HOME/include/cuda_runtime.h'." >&2
    echo "Set CUDA_HOME to the CUDA toolkit used to build min-FA3, or leave" \
        "VLLM_USE_FLASHINFER_SAMPLER=0 for this CUSTOM-attention benchmark." >&2
    exit 2
fi
if "$CUDA_HOME/bin/nvcc" --version >/dev/null 2>&1; then
    export CUDA_HOME CUDA_PATH="$CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [[ "$VLLM_USE_FLASHINFER_SAMPLER" == 1 ]]; then
    echo "error: CUDA compiler cannot be started: $CUDA_HOME/bin/nvcc" >&2
    exit 2
fi
TRACE=${TRACE:-$ROOT_DIR/dcp_test/trace/conversation_trace.jsonl}
WORKLOAD=${WORKLOAD:-$ROOT_DIR/.cache/vllm_dcp/smoke-workload.json}
RESULT_DIR=${RESULT_DIR:-$ROOT_DIR/benchmark_logs/vllm_dcp/smoke}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.05}
NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-1}
KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-1073741824}
MEGA_NUM_COMM_SMS=${MEGA_NUM_COMM_SMS:-"4,8,12,16,20"}
MEGA_MAX_NUM_SPLITS=${MEGA_MAX_NUM_SPLITS:-128}
PORT=${PORT:-18000}
TP_SIZE=${TP_SIZE:-8}

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
        --tp-size "$TP_SIZE" \
        --kv-heads 1 \
        --mega-num-comm-sms "$MEGA_NUM_COMM_SMS" \
        --mega-max-num-splits "$MEGA_MAX_NUM_SPLITS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
        --num-hidden-layers "$NUM_HIDDEN_LAYERS" \
        --arrival-time-scales 4 \
        --backends vllm-ag-rs vllm-a2a mega-fa3-native mega
