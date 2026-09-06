"""Select the four frozen D1 cases from the historical DCP4 Graph run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motivation.common import write_json


KINDS = ("decode", "mixed")
KIND_LABELS = {"decode": "decode_only_q16", "mixed": "mixed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--dcp-size", type=int, default=4)
    parser.add_argument("--cases-per-kind", type=int, default=2)
    return parser.parse_args()


def _case_number(case_id: str) -> int:
    prefix, number = case_id.rsplit("_", 1)
    if prefix != "case" or not number.isdigit():
        raise ValueError(f"invalid case ID: {case_id}")
    return int(number)


def _load_cases(path: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, text in enumerate(source, start=1):
            if not text.strip():
                continue
            case = json.loads(text)
            case_id = case.get("case_id")
            if not isinstance(case_id, str):
                raise ValueError(f"{path}:{line_number}: missing case_id")
            if case_id in cases:
                raise ValueError(f"{path}:{line_number}: duplicate {case_id}")
            cases[case_id] = case
    return cases


def main() -> None:
    args = parse_args()
    if args.cases_per_kind <= 0:
        raise SystemExit("--cases-per-kind must be positive")
    rows_by_kind: dict[str, list[dict[str, str]]] = {kind: [] for kind in KINDS}
    with args.summary_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {
            "record_type", "stat", "dcp_size", "case_id", "workload_kind",
            "baseline_attention_end_to_end_ms_us",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{args.summary_csv} is missing {sorted(missing)}")
        for row in reader:
            kind = row["workload_kind"]
            if (
                row["record_type"] == "baseline"
                and row["stat"] == "p50"
                and int(row["dcp_size"]) == args.dcp_size
                and kind in rows_by_kind
            ):
                rows_by_kind[kind].append(row)

    source_cases = _load_cases(args.cases_jsonl)
    selected: list[dict[str, Any]] = []
    medians: dict[str, float] = {}
    for kind in KINDS:
        candidates = rows_by_kind[kind]
        if len(candidates) < args.cases_per_kind:
            raise ValueError(f"not enough {kind} candidates in {args.summary_csv}")
        median_us = statistics.median(
            float(row["baseline_attention_end_to_end_ms_us"])
            for row in candidates
        )
        medians[KIND_LABELS[kind]] = median_us
        candidates.sort(
            key=lambda row: (
                abs(float(row["baseline_attention_end_to_end_ms_us"]) - median_us),
                _case_number(row["case_id"]),
            )
        )
        for selection_rank, row in enumerate(candidates[: args.cases_per_kind], start=1):
            case_id = row["case_id"]
            case = source_cases.get(case_id)
            if case is None:
                raise ValueError(f"selected {case_id} is absent from {args.cases_jsonl}")
            q_lengths = [int(value) for value in case["q_lens"]]
            if kind == "decode" and any(length != 16 for length in q_lengths):
                raise ValueError(f"{case_id} is not a q=16 decode-only case")
            selected.append(
                {
                    "case_id": case_id,
                    "workload_kind": KIND_LABELS[kind],
                    "selection_rank": selection_rank,
                    "selection_metric": "historical_vllm_a2a_graph_p50_us",
                    "selection_metric_value": float(
                        row["baseline_attention_end_to_end_ms_us"]
                    ),
                    "selection_median_value": median_us,
                    "selection_distance": abs(
                        float(row["baseline_attention_end_to_end_ms_us"]) - median_us
                    ),
                    "batch_size": int(case["batch_size"]),
                    "q_lengths": q_lengths,
                    "logical_q_lengths": [int(value) for value in case["logical_q_lens"]],
                    "history_lengths": [int(value) for value in case["history_lens"]],
                    "phases": list(case["phases"]),
                }
            )
    payload = {
        "schema_version": 1,
        "experiment": "D1_case_selection",
        "dcp_size": args.dcp_size,
        "cases_per_kind": args.cases_per_kind,
        "selection_rule": (
            "smallest absolute distance to the within-kind historical vLLM A2A "
            "Graph p50 median; ties use the smaller numeric case ID"
        ),
        "source_summary_csv": str(args.summary_csv),
        "source_cases_jsonl": str(args.cases_jsonl),
        "median_us": medians,
        "selected_case_ids": [case["case_id"] for case in selected],
        "cases": selected,
    }
    write_json(args.output_json, payload)


if __name__ == "__main__":
    main()
