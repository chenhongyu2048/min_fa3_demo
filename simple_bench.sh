#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_dir"

run_command() {
    if [[ ${DRY_RUN:-0} == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
        return
    fi
    "$@"
}

torchrun_bin=${TORCHRUN:-torchrun}
stamp=$(date +%Y%m%d_%H%M%S)
sq_lengths=16,16,16,16,16,16,16,16,16,16,32,32,32,32,64,128
dcp_size=2

run_command "$torchrun_bin" --standalone --nproc_per_node=8 \
    --module dcp_test.benchmark_dcp_varlen \
    --b 16 \
    --sq "$sq_lengths" \
    --seqlen 65536 \
    --qhead 32 \
    --kvhead 4 \
    --headdim 128 \
    --tp-size 8 \
    --dcp-size "$dcp_size" \
    --workload chunk \
    --implementations mega \
    --no-cuda-graph \
    --mega-phase-timestamps \
    --mega-block-n 128 \
    --mega-num-comm-sm 8 \
    --num-splits 0 \
    --warmup 500 \
    --iters 100 \
    --check \
    --output-json "benchmarks/results/dcp_mega_b16_h64k_dcp${dcp_size}_${stamp}.json"

run_command "$torchrun_bin" --standalone --nproc_per_node=8 \
    --module dcp_test.benchmark_dcp_varlen \
    --b 16 \
    --sq "$sq_lengths" \
    --seqlen 65536 \
    --qhead 32 \
    --kvhead 4 \
    --headdim 128 \
    --tp-size 8 \
    --dcp-size "$dcp_size" \
    --workload chunk \
    --implementations ours,vllm,sglang,full \
    --cuda-graph \
    --num-splits 0 \
    --warmup 500 \
    --iters 100 \
    --check \
    --output-json "benchmarks/results/dcp_baselines_graph_b16_h64k_dcp${dcp_size}_${stamp}.json"
