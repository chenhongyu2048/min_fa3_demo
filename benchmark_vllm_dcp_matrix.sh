#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-$ROOT_DIR/.venv/bin/python}
export PYTHONPATH="$ROOT_DIR/infer${PYTHONPATH:+:$PYTHONPATH}"
TRACE=${TRACE:-$ROOT_DIR/dcp_test/trace/conversation_trace.jsonl}
WORKLOAD=${WORKLOAD:-$ROOT_DIR/.cache/vllm_dcp/workload-100-1000.json}
RESULT_DIR=${RESULT_DIR:-$ROOT_DIR/benchmark_logs/vllm_dcp}
KV_HEADS=${KV_HEADS:-1,2,4}
MEGA_NUM_COMM_SMS=${MEGA_NUM_COMM_SMS:-"4,8,12,16,20"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-32}
VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
PORT=${PORT:-18000}

case "$VLLM_USE_FLASHINFER_SAMPLER" in
    0|1) export VLLM_USE_FLASHINFER_SAMPLER ;;
    *)
        echo "VLLM_USE_FLASHINFER_SAMPLER must be 0 or 1, got '$VLLM_USE_FLASHINFER_SAMPLER'" >&2
        exit 2
        ;;
esac

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

mega_num_comm_sms_spec=${MEGA_NUM_COMM_SMS//,/ }
read -r -a MEGA_NUM_COMM_SM_LIST <<< "$mega_num_comm_sms_spec"
if ((${#MEGA_NUM_COMM_SM_LIST[@]} == 0)); then
    echo "MEGA_NUM_COMM_SMS must not be empty" >&2
    exit 2
fi
declare -A seen_comm_sms=()
for comm_sm in "${MEGA_NUM_COMM_SM_LIST[@]}"; do
    if [[ ! "$comm_sm" =~ ^[1-9][0-9]*$ ]] || ((comm_sm >= 132)); then
        echo "MEGA_NUM_COMM_SMS values must be positive integers below 132, got '$comm_sm'" >&2
        exit 2
    fi
    if [[ -n "${seen_comm_sms[$comm_sm]+x}" ]]; then
        echo "MEGA_NUM_COMM_SMS contains duplicate '$comm_sm'" >&2
        exit 2
    fi
    seen_comm_sms[$comm_sm]=1
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

    echo "Running vLLM DCP matrix with KV_HEADS=$kv_heads MEGA_NUM_COMM_SMS=$MEGA_NUM_COMM_SMS VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER; results=$kv_result_dir"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
        "$PYTHON" -m vllm_bench.matrix \
            --workload "$WORKLOAD" \
            --result-dir "$kv_result_dir" \
            --port "$PORT" \
            --kv-heads "$kv_heads" \
            --mega-num-comm-sms "$MEGA_NUM_COMM_SMS" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --num-hidden-layers "$NUM_HIDDEN_LAYERS" \
            --arrival-time-scales 1 2 4 \
            --backends vllm-ag-rs vllm-a2a mega

    "$PYTHON" -m vllm_bench.summarize --result-dir "$kv_result_dir"
done
