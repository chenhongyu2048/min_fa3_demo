#!/usr/bin/env bash
# Run after installing the project environment and building the CUDA extension.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
python_bin=${PYTHON:-"$repo_root/.venv/bin/python"}
output_dir="$repo_root/benchmark_logs/motivation_v2/full_8gpu_$(date -u +%Y%m%dT%H%M%SZ)"
dry_run=false

usage() {
    cat <<'EOF'
Usage: bash motivation/run_full_8gpu.sh [--output-dir DIR] [--dry-run]

Runs full T1/T2/T3, three independent uninstrumented D1 candidate runs,
selection, selected-case trace diagnostics, and per-run summaries/plots.
Uses .venv/bin/python by default; set PYTHON to use another project interpreter.
CUDA_VISIBLE_DEVICES is inherited. Provide eight full SM90 GPUs.
Relative output paths are resolved from the repository root; DIR must not exist.
--dry-run prints commands without importing CUDA, creating files, or running tests.
EOF
}

while (($#)); do
    case "$1" in
        --output-dir) output_dir=${2:?--output-dir requires a directory}; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

if ! "$dry_run"; then
    mkdir -p "$(dirname "$output_dir")"
    mkdir "$output_dir"
    exec > >(tee "$output_dir/run.log") 2>&1
fi

run() {
    printf '\n+'
    printf ' %q' "$@"
    printf '\n'
    if ! "$dry_run"; then
        "$@"
    fi
}

analyze() {
    run "$python_bin" -m motivation.summarize "$1" --output "$1/summary.csv"
    run "$python_bin" -m motivation.plot "$1" --output-dir "$1/figures"
}

# Keep all cases and the experiment-specific default warmup/measurement counts.
run "$python_bin" -m motivation.run --gpus 8 --experiments T1,T2,T3 \
    --output-dir "$output_dir/training"

for run_id in 0 1 2; do
    run "$python_bin" -m motivation.run --gpus 8 --experiments D1 \
        --d1-run-id "$run_id" --output-dir "$output_dir/d1_run_$run_id"
done

# Preserve separate summaries for all candidates, even if selection finds too
# few stable winners. Do not merge independent runs or selected-case reruns.
analyze "$output_dir/training"
for run_id in 0 1 2; do
    analyze "$output_dir/d1_run_$run_id"
done

# A failed selector stops before trace capture.
run "$python_bin" -m motivation.select_d1_cases \
    "$output_dir/d1_run_0/D1/cases.jsonl" \
    "$output_dir/d1_run_1/D1/cases.jsonl" \
    "$output_dir/d1_run_2/D1/cases.jsonl" \
    --output-json "$output_dir/d1_selected.json"

# This reruns uninstrumented timing on the selected cases, then collects trace
# with separate diagnostic graphs. The original three runs determine selection.
run "$python_bin" -m motivation.run --gpus 8 --experiments D1 \
    --d1-run-id selected --d1-manifest "$output_dir/d1_selected.json" \
    --d1-trace --output-dir "$output_dir/d1_selected_trace"
analyze "$output_dir/d1_selected_trace"

if "$dry_run"; then
    printf '\nDry-run only; no experiments executed. Planned output: %s\n' "$output_dir"
else
    printf '\nCompleted full eight-GPU motivation run: %s\n' "$output_dir"
fi
