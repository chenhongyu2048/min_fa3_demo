#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_dir"

python_bin=${PYTHON:-python}
torchrun_bin=${TORCHRUN:-torchrun}
run_gpu=0

case ${1:-} in
    ""|--static-only)
        ;;
    --gpu)
        run_gpu=1
        ;;
    -h|--help)
        printf 'Usage: %s [--static-only|--gpu]\n' "$0"
        printf '  --static-only  Run CPU/static checks only (default).\n'
        printf '  --gpu          Run CPU/static checks, then the 8-GPU SM90 smoke suite.\n'
        exit 0
        ;;
    *)
        printf 'Unknown argument: %s\n' "$1" >&2
        printf 'Usage: %s [--static-only|--gpu]\n' "$0" >&2
        exit 2
        ;;
esac

printf '[1/6] CPU unit tests\n'
"$python_bin" -m unittest \
    scripts.test_min_fa3.test_dcp_topology \
    scripts.test_min_fa3.test_dcp_mega_metadata

printf '[2/6] Python compile checks\n'
"$python_bin" -m py_compile \
    min_fa3_dcp.py \
    dcp_test/__init__.py \
    dcp_test/baselines.py \
    dcp_test/utils.py \
    dcp_test/benchmark_dcp.py \
    dcp_test/benchmark_dcp_varlen.py \
    scripts/test_min_fa3/test_dcp_topology.py \
    scripts/test_min_fa3/test_min_fa3_dcp.py \
    scripts/test_min_fa3/test_min_fa3_dcp_varlen.py

printf '[3/6] CLI import checks\n'
"$python_bin" -m dcp_test.benchmark_dcp --help >/dev/null
"$python_bin" -m dcp_test.benchmark_dcp_varlen --help >/dev/null
"$python_bin" scripts/test_min_fa3/test_min_fa3_dcp.py --help >/dev/null
"$python_bin" scripts/test_min_fa3/test_min_fa3_dcp_varlen.py --help >/dev/null
"$python_bin" -m scripts.test_min_fa3.test_dcp_mega_varlen_multi_rank \
    --help >/dev/null

printf '[4/6] Shell syntax and benchmark dry-run checks\n'
bash -n simple_bench.sh simple_test.sh
DRY_RUN=1 bash simple_bench.sh >/dev/null

printf '[5/6] Patch whitespace check\n'
git diff --check

printf '[6/6] Core import and public API checks\n'
"$python_bin" -c '
import inspect
import min_fa3_dcp as dcp

baseline_names = {
    "VLLMDCPAttentionRunner",
    "VLLMA2ADCPAttentionRunner",
    "SGLangDCPAttentionRunner",
}
assert baseline_names.isdisjoint(dcp.__all__)
assert all(not hasattr(dcp, name) for name in baseline_names)
assert not hasattr(dcp.DCPAttentionRunner, "last_timing_ms")
for name in (
    "forward_decode",
    "forward_chunk_prefill",
    "forward_decode_varlen",
    "forward_chunk_prefill_varlen",
):
    assert "_record_timing" not in inspect.signature(
        getattr(dcp.DCPAttentionRunner, name)
    ).parameters
for name in (
    "capture_decode",
    "capture_chunk_prefill",
    "capture_decode_varlen",
    "capture_chunk_prefill_varlen",
):
    assert "record_timing" not in inspect.signature(
        getattr(dcp.DCPAttentionRunner, name)
    ).parameters
'

if ((run_gpu == 0)); then
    printf 'Static checks passed. GPU smoke tests were not requested.\n'
    printf 'Run %s --gpu after eight SM90 GPUs are available.\n' "$0"
    exit 0
fi

result_dir=${SIMPLE_TEST_RESULT_DIR:-benchmarks/results/simple_test_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$result_dir"

run_distributed() {
    "$torchrun_bin" --standalone --nproc_per_node=8 --module "$@"
}

printf '[GPU 1/7] Fixed BSHD correctness, eager validation and CUDA Graph\n'
run_distributed scripts.test_min_fa3.test_min_fa3_dcp \
    --qhead 32 --kvhead 4 --tp-size 8 --dcp-sizes 2 \
    --sq 2 --num-splits 1 --repeat 1 --cuda-graph \
    --output-json "$result_dir/fixed_correctness.json"

printf '[GPU 2/7] Packed-varlen correctness, eager validation and CUDA Graph\n'
run_distributed scripts.test_min_fa3.test_min_fa3_dcp_varlen \
    --b 2 --sq 1,8 --seqlen 129,258 \
    --qhead 32 --kvhead 4 --headdim 128 --tp-size 8 --dcp-sizes 2 \
    --num-splits 1 --repeat 1 --cuda-graph \
    --output-json "$result_dir/varlen_correctness.json"

printf '[GPU 3/7] DCP Mega eager, prepared replay, timestamps and CUDA Graph\n'
run_distributed scripts.test_min_fa3.test_dcp_mega_varlen_multi_rank \
    --dcp-size 2 --q-lengths 1,8 --history-lengths 129,258 \
    --hq-local 4 --num-splits 1 --block-n 128 --num-comm-sm 8 --repeat 1

printf '[GPU 4/7] Dense benchmark eager timing smoke\n'
run_distributed dcp_test.benchmark_dcp \
    --qhead 32 --kvhead 4 --tp-size 8 --dcp-sizes 2 \
    --implementations ours,vllm,sglang --workload both \
    --decode-b 1 --chunk-b 1 --seqlen 129 --sq 8 \
    --num-splits 1 --warmup 1 --iters 2 --no-cuda-graph --no-mqa-control \
    --output-json "$result_dir/dense_eager.json"

printf '[GPU 5/7] Dense benchmark CUDA Graph timing smoke\n'
run_distributed dcp_test.benchmark_dcp \
    --qhead 32 --kvhead 4 --tp-size 8 --dcp-sizes 2 \
    --implementations ours,vllm,sglang --workload both \
    --decode-b 1 --chunk-b 1 --seqlen 129 --sq 8 \
    --num-splits 1 --warmup 1 --iters 2 --cuda-graph --no-mqa-control \
    --output-json "$result_dir/dense_graph.json"

printf '[GPU 6/7] Packed-varlen benchmark eager timing smoke\n'
run_distributed dcp_test.benchmark_dcp_varlen \
    --b 2 --sq 1,8 --seqlen 129,258 \
    --qhead 32 --kvhead 4 --headdim 128 --tp-size 8 --dcp-size 2 \
    --workload chunk --implementations ours,vllm,sglang,full \
    --num-splits 1 --warmup 1 --iters 2 --no-cuda-graph --check \
    --output-json "$result_dir/varlen_eager.json"

printf '[GPU 7/7] Packed-varlen benchmark CUDA Graph timing smoke\n'
run_distributed dcp_test.benchmark_dcp_varlen \
    --b 2 --sq 1,8 --seqlen 129,258 \
    --qhead 32 --kvhead 4 --headdim 128 --tp-size 8 --dcp-size 2 \
    --workload chunk --implementations ours,vllm,sglang,full \
    --num-splits 1 --warmup 1 --iters 2 --cuda-graph --check \
    --output-json "$result_dir/varlen_graph.json"

"$python_bin" - "$result_dir" <<'PY'
import json
import pathlib
import sys

result_dir = pathlib.Path(sys.argv[1])
expected = {
    "dense_eager.json": 4,
    "dense_graph.json": 4,
    "varlen_eager.json": 3,
    "varlen_graph.json": 3,
}
for name, schema_version in expected.items():
    payload = json.loads((result_dir / name).read_text(encoding="utf-8"))
    assert payload["schema_version"] == schema_version, name
PY

printf 'All static and GPU smoke checks passed. Results: %s\n' "$result_dir"
