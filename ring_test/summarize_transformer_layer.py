#!/usr/bin/env python3
"""Print Transformer-layer JSONL cases and per-dataset attention throughput.

Usage (one benchmark run, all datasets):
    python3 ring_test/summarize_transformer_layer.py results/transformer_layer_cp/RUN_ID-*.jsonl

Only the standard library is required. FLOPs follow the dataset benchmarks:
4 * visible_scores * QH * D for forward, 10 * visible_scores * QH * D for
backward (including attention recomputation). Padding work is not credited.
Each case contributes its recorded mean latency once, irrespective of the
number of measurement iterations. SM configurations are summarized separately.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


def sm_label(record: dict[str, Any]) -> str:
    sm = record["sm_config"]
    return "-" if sm is None else f"{sm['num_comp_sm']}:{sm['num_comm_sm']}"


def attention_flops(record: dict[str, Any]) -> tuple[int, int]:
    lengths = record["workload"]["global_lengths"]
    scores = sum(
        n * (n + 1) // 2 if record["causal"] else n * n for n in lengths
    )
    model = record["model"]
    work = scores * model["q_heads"] * model["head_dim"]
    return 4 * work, 10 * work


def core_times(record: dict[str, Any]) -> tuple[float, float]:
    timing = record["timing"]
    return (
        float(timing["core_attn_forward_cuda_critical_rank_avg_ms"]),
        float(timing["core_attn_backward_cuda_critical_rank_avg_ms"]),
    )


def gpu_count(record: dict[str, Any]) -> int:
    parallel = record["parallelism"]
    # EP reuses the attention ranks; it is not another world-size factor.
    return math.prod(parallel[axis] for axis in ("tp", "cp", "pp", "dp"))


def weighted_tflops(records: Sequence[dict[str, Any]]) -> tuple[float, float, float]:
    fwd_flops, bwd_flops = map(sum, zip(*(attention_flops(r) for r in records)))
    fwd_ms, bwd_ms = map(sum, zip(*(core_times(r) for r in records)))
    return (
        fwd_flops / fwd_ms / 1e9,
        bwd_flops / bwd_ms / 1e9,
        (fwd_flops + bwd_flops) / (fwd_ms + bwd_ms) / 1e9,
    )


def load_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    seen = set()
    configurations = {}
    workloads = {}
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record["schema"] != "min_fa3.megatron_transformer_layer_cp.v2":
                        raise ValueError("requires v2 core-attention timing records")
                    dataset = record["dataset"]
                    config = {key: record[key] for key in (
                        "model", "parallelism", "seed", "target_tokens", "num_cases", "causal"
                    )}
                    if configurations.setdefault(dataset, config) != config:
                        raise ValueError(f"mixed configurations for {dataset}; select one run")
                    case = record["case_index"]
                    if not 0 <= case < record["num_cases"]:
                        raise ValueError(f"case_index out of range: {case}")
                    key = (dataset, case, record["method"], sm_label(record))
                    if key in seen:
                        raise ValueError(f"duplicate case/method/SM {key}; select one run")
                    lengths = record["workload"]["global_lengths"]
                    if workloads.setdefault((dataset, case), lengths) != lengths:
                        raise ValueError(f"inconsistent original lengths for {dataset} case {case + 1}")
                    if any(not math.isfinite(t) or t <= 0 for t in core_times(record)):
                        raise ValueError("core-attention times must be finite and positive")
                    if attention_flops(record)[0] <= 0 or gpu_count(record) <= 0:
                        raise ValueError("workload FLOPs and GPU count must be positive")
                    seen.add(key)
                    records.append(record)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    if not records:
        raise ValueError("no benchmark results found")
    return records


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    for row in (headers, *rows):
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def print_report(records: Sequence[dict[str, Any]]) -> None:
    print("TFLOPS scope: core attention, original useful sequence lengths (padding excluded).")
    print("FLOPs: forward=4*S*QH*D, backward=10*S*QH*D; causal S=sum(L*(L+1)/2).")
    print("Weighted TFLOPS=sum(case FLOPs)/sum(case mean core ms)/1e9; F+B uses both sums.")
    print("/GPU divides aggregate throughput by world size; EP is not multiplied again.")
    print("Core times use the recorded whole-layer critical rank; no rank reselection is possible.")
    print("Partial runs are included as recorded; compare case coverage before comparing summaries.")
    by_dataset = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)
    for dataset, dataset_records in sorted(by_dataset.items()):
        first = dataset_records[0]
        world = gpu_count(first)
        model = first["model"]
        print(f"\nDATASET {dataset} | GPUs={world} | model={model['profile']} | "
              f"routing={model.get('moe_routing', 'model_default')} | seed={first['seed']}")
        print("CASES (times in ms; F/B/FB TFLOPS below are per GPU)")
        rows = []
        grouped = defaultdict(list)
        for record in sorted(dataset_records, key=lambda r: (
            r["case_index"], r["method"], sm_label(r)
        )):
            timing = record["timing"]
            core_f, core_b = core_times(record)
            rates = weighted_tflops([record])
            rows.append([
                f"{record['case_index'] + 1}/{record['num_cases']}",
                record["method"], sm_label(record),
                str(record["tokens"]["original_global"]),
                str(record["tokens"]["execution_global"]),
                f"{core_f:.3f}", f"{core_b:.3f}",
                f"{timing['forward_cuda_critical_rank_avg_ms']:.3f}",
                f"{timing['backward_cuda_critical_rank_avg_ms']:.3f}",
                *(f"{rate / world:.3f}" for rate in rates),
            ])
            grouped[(record["method"], sm_label(record))].append(record)
        print_table([
            "Case", "Method", "SM", "Tokens", "ExecTokens", "CoreF", "CoreB",
            "LayerF", "LayerB", "F_TF/GPU", "B_TF/GPU", "FB_TF/GPU",
        ], rows)
        print(f"\nSUMMARY {dataset} (time-weighted TFLOPS across recorded cases)")
        rows = []
        for (method, sm), cases in sorted(grouped.items()):
            rates = weighted_tflops(cases)
            rows.append([
                method, sm, f"{len(cases)}/{first['num_cases']}",
                *(f"{rate:.3f}" for rate in rates),
                *(f"{rate / world:.3f}" for rate in rates),
            ])
        print_table([
            "Method", "SM", "Cases", "F_TF", "B_TF", "FB_TF",
            "F_TF/GPU", "B_TF/GPU", "FB_TF/GPU",
        ], rows)
        for (method, sm), cases in sorted(grouped.items()):
            present = {r["case_index"] for r in cases}
            missing = sorted(set(range(first["num_cases"])) - present)
            if missing:
                print(f"Missing cases: {method} SM={sm}: " + ",".join(str(i + 1) for i in missing))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path, help="JSONL files from one run (shell glob supported)")
    args = parser.parse_args(argv)
    try:
        records = load_records(args.jsonl)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print_report(records)


if __name__ == "__main__":
    main()
