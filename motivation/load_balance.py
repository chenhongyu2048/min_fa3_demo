"""Motivation T3: static multi-objective load accounting on one shared manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import balancer
from ring_test.forward_load_model import analyze_method, cumulative_result
from motivation.common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, choices=tuple(balancer.DATASET_WEIGHTS))
    parser.add_argument("--datasets", default=None, help="comma-separated dataset list; overrides --dataset")
    parser.add_argument("--target-tokens", type=int, default=128 * 1024)
    parser.add_argument("--num-cases", type=int, default=30)
    parser.add_argument("--world-size", type=int, default=8, choices=(2, 4, 8))
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = ("megatron_hybrid_cp", "zeppelin", "mega_ring_all_cp")
    if args.datasets:
        datasets = tuple(dict.fromkeys(item.strip() for item in args.datasets.split(",") if item.strip()))
    elif args.dataset:
        datasets = (args.dataset,)
    else:
        datasets = tuple(balancer.DATASET_WEIGHTS)
    unknown = sorted(set(datasets) - set(balancer.DATASET_WEIGHTS))
    if not datasets or unknown:
        raise SystemExit(f"unknown or empty dataset list: {unknown or datasets}")
    all_cases: dict[str, list[dict[str, object]]] = {}
    all_summaries: dict[str, dict[str, object]] = {}
    for dataset_index, dataset in enumerate(datasets):
        dataset_seed = args.seed + dataset_index
        cases = balancer.make_workloads(
            dataset=dataset, target_tokens=args.target_tokens, seed=dataset_seed,
            num_cases=args.num_cases, world_size=args.world_size, mode="causal",
            compute_balance_tolerance=0.05, token_balance_tolerance=0.10,
            beam_width=64, finalist_count=8, structure_threshold=0.5,
            max_repair_iterations=32,
        )
        case_rows: list[dict[str, object]] = []
        cumulative: dict[str, list[object]] = {method: [] for method in methods}
        for case_index, workload in enumerate(cases):
            lengths = list(workload.global_lengths)
            case_payload: dict[str, object] = {
                "case_id": case_index, "dataset": dataset, "seed": dataset_seed,
                "global_lengths": lengths, "ring_sizes": list(workload.ring_sizes),
                "ring_starts": list(workload.ring_starts), "original_tokens": sum(lengths),
            }
            for method in methods:
                result = analyze_method(
                    method, lengths, list(workload.ring_sizes), list(workload.ring_starts),
                    args.world_size, args.qhead, args.kvhead, args.headdim, True,
                    heads_k_stride=min(4, args.kvhead),
                )
                cumulative[method].append(result)
                case_payload[method] = {
                    "rank_records": [record.__dict__ for record in result.records],
                    "token_imbalance": max(record.physical_tokens for record in result.records) / (sum(record.physical_tokens for record in result.records) / args.world_size),
                    "attention_imbalance": max(record.effective_scores for record in result.records) / (sum(record.effective_scores for record in result.records) / args.world_size),
                    "communication_tx_bytes": sum(record.comm_tx_bytes for record in result.records),
                    "tile_work": sum(record.kv_tile_reads for record in result.records), "note": result.note,
                }
            case_rows.append(case_payload)
        all_cases[dataset] = case_rows
        all_summaries[dataset] = {
            method: {
                "rank_records": [record.__dict__ for record in cumulative_result(cumulative[method]).records],
                "token_imbalance": max(record.physical_tokens for record in cumulative_result(cumulative[method]).records) / (sum(record.physical_tokens for record in cumulative_result(cumulative[method]).records) / args.world_size),
                "attention_imbalance": max(record.effective_scores for record in cumulative_result(cumulative[method]).records) / (sum(record.effective_scores for record in cumulative_result(cumulative[method]).records) / args.world_size),
                "communication_tx_bytes": sum(record.comm_tx_bytes for record in cumulative_result(cumulative[method]).records),
                "tile_work": sum(record.kv_tile_reads for record in cumulative_result(cumulative[method]).records),
            } for method in methods
        }
    payload = {
        "schema_version": 2,
        "experiment": "T3_load_balance",
        "config": vars(args) | {"datasets": list(datasets)},
        "datasets": list(datasets),
        "dataset_seeds": {dataset: args.seed + index for index, dataset in enumerate(datasets)},
        "cases": all_cases,
        "summary": all_summaries,
    }
    write_json(args.output_json or Path("benchmark_logs/motivation/t3_load_balance.json"), payload)


if __name__ == "__main__":
    main()
