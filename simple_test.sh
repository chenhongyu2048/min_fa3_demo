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
    scripts.test_min_fa3.test_dcp_mega_metadata \
    scripts.test_min_fa3.test_dcp_mega_batch \
    dcp_test.trace.tests.test_trace_workload \
    dcp_test.trace.tests.test_summarize_dcp_mega_matrix \
    dcp_test.trace.tests.test_plot_dcp_mega_latency

printf '[2/6] Python compile checks\n'
"$python_bin" -m py_compile \
    min_fa3_dcp.py \
    dcp_test/__init__.py \
    dcp_test/baselines.py \
    dcp_test/utils.py \
    dcp_test/benchmark_dcp.py \
    dcp_test/benchmark_dcp_varlen.py \
    dcp_test/benchmark_dcp_mega_batch.py \
    dcp_test/summarize_dcp_mega_matrix.py \
    dcp_test/plot_dcp_mega_latency.py \
    dcp_test/trace/models.py \
    dcp_test/trace/generate.py \
    scripts/test_min_fa3/test_dcp_topology.py \
    scripts/test_min_fa3/test_dcp_mega_batch.py \
    scripts/test_min_fa3/test_min_fa3_dcp.py \
    scripts/test_min_fa3/test_min_fa3_dcp_varlen.py

printf '[3/6] CLI import checks\n'
"$python_bin" -m dcp_test.benchmark_dcp --help >/dev/null
"$python_bin" -m dcp_test.benchmark_dcp_varlen --help >/dev/null
"$python_bin" -m dcp_test.benchmark_dcp_mega_batch --help >/dev/null
"$python_bin" -m dcp_test.benchmark_dcp_mega_batch \
    --workloads small1 --dcp-sizes 2,8 --print-cases >/dev/null
"$python_bin" -m dcp_test.trace.generate --help >/dev/null
"$python_bin" -m dcp_test.summarize_dcp_mega_matrix --help >/dev/null
"$python_bin" -m dcp_test.plot_dcp_mega_latency --help >/dev/null
"$python_bin" scripts/test_min_fa3/test_min_fa3_dcp.py --help >/dev/null
"$python_bin" scripts/test_min_fa3/test_min_fa3_dcp_varlen.py --help >/dev/null
"$python_bin" -m scripts.test_min_fa3.test_dcp_mega_varlen_multi_rank \
    --help >/dev/null

printf '[4/6] Shell syntax and benchmark dry-run checks\n'
bash -n benchmark_dcp_mega_six_loads.sh benchmark_dcp_mega_trace.sh \
    benchmark_dcp_mega_arrival_matrix.sh \
    simple_bench.sh simple_test.sh
DRY_RUN=1 bash simple_bench.sh >/dev/null
batch_dry_run=$(DRY_RUN=1 LOADS=small1 DCP_SIZES=2,8 WARMUP=0 ITERS=1 \
    bash benchmark_dcp_mega_six_loads.sh)
[[ $(rg -c -- '--module dcp_test.benchmark_dcp_mega_batch' <<< "$batch_dry_run") == 2 ]]
[[ $(rg -c -- '^\[[12]/2\]' <<< "$batch_dry_run") == 4 ]]
rg -Fq "mode=eager, cases=2, implementations=['mega', 'ours', 'vllm', 'sglang']" \
    <<< "$batch_dry_run"
rg -Fq "mode=graph, cases=2, implementations=['ours', 'vllm', 'sglang']" \
    <<< "$batch_dry_run"
eager_dry_run=$(DRY_RUN=1 BASELINE_PHASE_TIMING=0 LOADS=small1 DCP_SIZES=2,8 MODES=eager \
    WARMUP=0 ITERS=1 bash benchmark_dcp_mega_six_loads.sh)
[[ $(rg -c -- '--module dcp_test.benchmark_dcp_mega_batch' <<< "$eager_dry_run") == 1 ]]
[[ $(rg -c -- '--no-baseline-phase-timing' <<< "$eager_dry_run") == 1 ]]
trace_dry_run=$(DRY_RUN=1 NUM_CASES=2 MODES=eager WARMUP=0 ITERS=1 \
    bash benchmark_dcp_mega_trace.sh)
[[ $(rg -c -- '--module dcp_test.benchmark_dcp_mega_batch' <<< "$trace_dry_run") == 1 ]]
[[ $(rg -c -- '--num-cases 2' <<< "$trace_dry_run") == 2 ]]
[[ $(rg -c -- '--trace-cases' <<< "$trace_dry_run") == 1 ]]
trace_reuse_dry_run=$(DRY_RUN=1 GENERATE_TRACE=0 \
    TRACE_CASES=dcp_test/trace/conversation_trace.jsonl NUM_CASES=2 \
    MODES=eager WARMUP=0 ITERS=1 bash benchmark_dcp_mega_trace.sh)
[[ $(rg -c -- '^Using existing cases:' <<< "$trace_reuse_dry_run") == 1 ]]
[[ $(rg -c -- '--module dcp_test.benchmark_dcp_mega_batch' <<< "$trace_reuse_dry_run") == 1 ]]
! rg -q -- 'dcp_test.trace.generate' <<< "$trace_reuse_dry_run"
trace_graph_dry_run=$(DRY_RUN=1 NUM_CASES=2 MODES=graph WARMUP=0 ITERS=1 \
    bash benchmark_dcp_mega_trace.sh)
rg -Fq -- '--implementations ours\,vllm\,sglang' <<< "$trace_graph_dry_run"
! rg -Fq -- '--implementations mega\,ours\,vllm\,sglang' <<< "$trace_graph_dry_run"
matrix_dry_run=$(DRY_RUN=1 ARRIVAL_TIME_SCALES=4 DCP_SIZES=2 \
    MEGA_NUM_COMM_SMS=4,8 NUM_CASES=2 WARMUP=0 ITERS=1 \
    bash benchmark_dcp_mega_arrival_matrix.sh)
[[ $(rg -c -- '--module dcp_test.benchmark_dcp_mega_batch' <<< "$matrix_dry_run") == 3 ]]
[[ $(rg -c -- '--mega-num-comm-sms 4\\,8' <<< "$matrix_dry_run") == 2 ]]
[[ $(rg -c -- '--implementations ours\\,vllm\\,sglang\\,full' <<< "$matrix_dry_run") == 2 ]]
[[ $(rg -c -- '--arrival-time-scale 4 --dcp-size 2' <<< "$matrix_dry_run") == 1 ]]
[[ $(rg -c -- 'dcp_test.summarize_dcp_mega_matrix' <<< "$matrix_dry_run") == 1 ]]

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

printf '[GPU 6/7] Packed-varlen multi-case benchmark eager timing smoke\n'
run_distributed dcp_test.benchmark_dcp_mega_batch \
    --workloads small1 --dcp-sizes 2,8 \
    --implementations mega,ours,vllm,sglang \
    --num-splits 1 --warmup 1 --iters 2 --no-cuda-graph --check \
    --no-baseline-phase-timing \
    --output-dir "$result_dir/batch_eager" \
    --manifest "$result_dir/batch_eager_manifest.json"

printf '[GPU 7/7] Packed-varlen multi-case benchmark CUDA Graph timing smoke\n'
run_distributed dcp_test.benchmark_dcp_mega_batch \
    --workloads small1 --dcp-sizes 2,8 \
    --implementations ours,vllm,sglang \
    --num-splits 1 --warmup 1 --iters 2 --cuda-graph --check \
    --output-dir "$result_dir/batch_graph" \
    --manifest "$result_dir/batch_graph_manifest.json"

"$python_bin" - "$result_dir" <<'PY'
import json
import pathlib
import sys

result_dir = pathlib.Path(sys.argv[1])
expected = {
    "dense_eager.json": 4,
    "dense_graph.json": 4,
}
for name, schema_version in expected.items():
    payload = json.loads((result_dir / name).read_text(encoding="utf-8"))
    assert payload["schema_version"] == schema_version, name
for mode in ("eager", "graph"):
    manifest_path = result_dir / f"batch_{mode}_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1, mode
    assert manifest["status"] == "complete", mode
    assert manifest["case_total"] == 2, mode
    assert manifest["completed_case_count"] == 2, mode
    assert manifest["weighted_summary"] is not None, mode
    expected_phase_timing = mode == "graph"
    assert manifest["parameters"]["baseline_phase_timing"] is expected_phase_timing, mode
    assert [entry["dcp_size"] for entry in manifest["cases"]] == [2, 8], mode
    for topology in manifest["weighted_summary"]["topologies"].values():
        for method in topology["methods"].values():
            assert method["case_count"] == 1, (mode, topology)
            assert method["workload_weighted_effective_tflops"] > 0
    for entry in manifest["cases"]:
        if mode == "eager":
            assert "dcp_mega_varlen" in entry["methods"], entry["case_id"]
        else:
            assert "dcp_mega_varlen" not in entry["methods"], entry["case_id"]
        result_path = pathlib.Path(entry["output_json"])
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 3, result_path
        assert payload["parameters"]["baseline_phase_timing"] is expected_phase_timing
        for method, report in payload["methods"].items():
            if method == "dcp_mega_varlen":
                continue
            execution = report["execution"]
            assert execution["cuda_event_phase_timing_enabled"] is expected_phase_timing
            if not expected_phase_timing:
                assert all(
                    summary["p50"] == 0.0 and summary["p90"] == 0.0
                    for stage, summary in report["stages_ms"].items()
                    if stage != "attention_end_to_end_ms"
                ), method
PY

printf 'All static and GPU smoke checks passed. Results: %s\n' "$result_dir"
