#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
NUM_CASES=20
TRACE_CASES="$REPO_ROOT/dcp_test/mega_dcp_trace_cases.jsonl"

python -m dcp_test.trace.generate \
    --config dcp_test/trace/example_config.json \
    --output "$TRACE_CASES" \
    --num-cases "$NUM_CASES" --force

GENERATE_TRACE=0 \
  TRACE_CONFIG=dcp_test/trace/example_config.json \
  TRACE_CASES="$TRACE_CASES" \
  NUM_CASES="$NUM_CASES" \
  MODES=eager,graph \
  WARMUP=40 \
  ITERS=60 \
  CHECK=0 \
  MEGA_PHASE_TIMESTAMPS=1 \
  BASELINE_PHASE_TIMING=1 \
  MEGA_NUM_COMM_SM=16 \
  ./scripts/test_dcp/benchmark_dcp_mega_trace.sh

########################################
# GENERATE_TRACE=1
# TRACE_CONFIG=dcp_test/trace/example_config.json
# NUM_CASES=20

# LOG_DIR=benchmark_logs/dcp_mega_trace_<timestamp>
# RESULT_DIR="$LOG_DIR/results"
# TRACE_CASES="$RESULT_DIR/trace_cases.jsonl"
# NUM_CASES=20 GENERATE_TRACE=1 ./scripts/test_dcp/benchmark_dcp_mega_trace.sh

########################################
# script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# repo_root=$(cd -- "$script_dir/../.." && pwd)
# cd "$repo_root"

# run_command() {
#     if [[ ${DRY_RUN:-0} == 1 ]]; then
#         printf '%q ' "$@"
#         printf '\n'
#         return
#     fi
#     "$@"
# }

# torchrun_bin=${TORCHRUN:-torchrun}
# stamp=$(date +%Y%m%d_%H%M%S)
# sq_lengths=16,16,16,16,16,16,16,16,16,16,32,32,32,32,64,128
# dcp_size=2

# run_command "$torchrun_bin" --standalone --nproc_per_node=8 \
#     --module dcp_test.benchmark_dcp_varlen \
#     --b 16 \
#     --sq "$sq_lengths" \
#     --seqlen 65536 \
#     --qhead 32 \
#     --kvhead 4 \
#     --headdim 128 \
#     --tp-size 8 \
#     --dcp-size "$dcp_size" \
#     --workload chunk \
#     --implementations mega \
#     --no-cuda-graph \
#     --mega-phase-timestamps \
#     --mega-block-n 128 \
#     --mega-num-comm-sm 8 \
#     --num-splits 0 \
#     --warmup 500 \
#     --iters 100 \
#     --check \
#     --output-json "benchmarks/results/dcp_mega_b16_h64k_dcp${dcp_size}_${stamp}.json"

# run_command "$torchrun_bin" --standalone --nproc_per_node=8 \
#     --module dcp_test.benchmark_dcp_varlen \
#     --b 16 \
#     --sq "$sq_lengths" \
#     --seqlen 65536 \
#     --qhead 32 \
#     --kvhead 4 \
#     --headdim 128 \
#     --tp-size 8 \
#     --dcp-size "$dcp_size" \
#     --workload chunk \
#     --implementations ours,vllm,sglang,full \
#     --cuda-graph \
#     --num-splits 0 \
#     --warmup 500 \
#     --iters 100 \
#     --check \
#     --output-json "benchmarks/results/dcp_baselines_graph_b16_h64k_dcp${dcp_size}_${stamp}.json"
