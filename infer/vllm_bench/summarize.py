"""Aggregate the vLLM DCP matrix and compute Mega/baseline ratios."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _value(summary: dict[str, Any], path: str) -> float | None:
    value: Any = summary
    for component in path.split("."):
        value = value.get(component)
        if value is None:
            return None
    return float(value)


def _comm_sm(summary: dict[str, Any]) -> int | None:
    value = summary.get("mega_num_comm_sm")
    return None if value is None else int(value)


METRICS = (
    "offered_rps",
    "achieved_rps",
    "output_token_throughput",
    "ttft_s.mean",
    "ttft_s.p50",
    "ttft_s.p90",
    "ttft_s.p99",
    "pooled_itl_s.mean",
    "pooled_itl_s.p50",
    "pooled_itl_s.p90",
    "pooled_itl_s.p99",
    "per_request_mean_tbt_s.mean",
    "per_request_mean_tbt_s.p50",
    "per_request_mean_tbt_s.p90",
    "per_request_mean_tbt_s.p99",
    "e2e_s.mean",
    "e2e_s.p50",
    "e2e_s.p90",
    "e2e_s.p99",
)


def aggregate(result_dir: Path) -> dict[str, Any]:
    summaries = []
    for path in sorted(result_dir.glob("*-scale*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    by_key = {
        (
            summary["backend"],
            float(summary["arrival_time_scale"]),
            _comm_sm(summary),
        ): summary
        for summary in summaries
    }
    comparisons = []
    scales = sorted({scale for _, scale, _ in by_key})
    for scale in scales:
        mega_runs = sorted(
            [
                (comm_sm, summary)
                for (backend, run_scale, comm_sm), summary in by_key.items()
                if backend == "mega" and run_scale == scale
            ],
            key=lambda item: (item[0] is None, item[0] or 0),
        )
        for comm_sm, mega in mega_runs:
            for baseline in ("vllm-ag-rs", "vllm-a2a"):
                baseline_summary = by_key.get((baseline, scale, None))
                if baseline_summary is None:
                    continue
                ratios = {}
                for metric in METRICS:
                    numerator = _value(mega, metric)
                    denominator = _value(baseline_summary, metric)
                    ratios[metric] = (
                        numerator / denominator
                        if numerator is not None and denominator not in (None, 0)
                        else None
                    )
                comparisons.append(
                    {
                        "arrival_time_scale": scale,
                        "mega_num_comm_sm": comm_sm,
                        "comparison": f"mega/{baseline}",
                        "ratios": ratios,
                        "interpretation": {
                            "latency_and_tbt": "ratio < 1 is lower",
                            "throughput": "ratio > 1 is higher",
                        },
                    }
                )
    return {"runs": summaries, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.result_dir)
    (args.result_dir / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for summary in result["runs"]:
        row = {
            "backend": summary["backend"],
            "arrival_time_scale": summary["arrival_time_scale"],
            "mega_num_comm_sm": _comm_sm(summary),
            "successful_requests": summary["successful_requests"],
            "failed_requests": summary["failed_requests"],
        }
        row.update({metric: _value(summary, metric) for metric in METRICS})
        rows.append(row)
    if rows:
        with (args.result_dir / "comparison.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
