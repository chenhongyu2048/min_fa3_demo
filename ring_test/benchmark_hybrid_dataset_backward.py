"""Dataset-shaped frontend for the hierarchical hybrid backward benchmark."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
for path in (THIS_DIR, DEMO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import balancer
from benchmark_hybrid_dataset_forward import (
    _positive_int,
    _format_int_list,
    _requests_all,
    _world_size,
    print_workload,
)
from ring_test.utils import HybridBenchmarkCase
from zepplin import DEFAULT_ZEPPLIN_THRESHOLD


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a dataset-shaped workload and run hybrid backward"
    )
    parser.add_argument("--dataset", choices=tuple(balancer.DATASET_WEIGHTS), required=True)
    parser.add_argument("--target-tokens", type=int, default=balancer.MAX_SEQUENCE_TOKENS)
    parser.add_argument("--balance-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--token-balance-tolerance",
        type=float,
        default=0.10,
        help="Requested maximum token imbalance relative to average",
    )
    parser.add_argument(
        "--max-compute-balance-tolerance",
        type=float,
        default=0.20,
        help="Maximum compute imbalance explored by the planner",
    )
    parser.add_argument(
        "--max-token-balance-tolerance",
        type=float,
        default=0.50,
        help="Maximum token imbalance explored before emergency fallback",
    )
    parser.add_argument(
        "--communication-weight",
        type=float,
        default=0.05,
        help="Weight of estimated ring communication in placement scoring",
    )
    parser.add_argument(
        "--local-search-passes",
        type=int,
        default=4,
        help="Number of deterministic local-repair passes; zero disables repair",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-cases", type=_positive_int, default=1)
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8))
    parser.add_argument("--print-workload", action="store_true")
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--methods", default="all")
    parser.add_argument(
        "--zepplin-threshold",
        type=_positive_int,
        default=DEFAULT_ZEPPLIN_THRESHOLD,
    )
    parser.add_argument("--sm-configs", default="128:4,124:8,120:12,116:16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=40)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dq-atol", type=float, default=1.0)
    parser.add_argument("--dkv-atol", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=0.2)
    return parser.parse_args(argv)


def _benchmark_argv(
    args: argparse.Namespace, workload: balancer.HybridWorkload
) -> list[str]:
    return [
        "--global-seqlens",
        _format_int_list(workload.global_lengths),
        "--ring-sizes",
        _format_int_list(workload.ring_sizes),
        "--ring-starts",
        _format_int_list(workload.ring_starts),
        "--qhead",
        str(args.qhead),
        "--kvhead",
        str(args.kvhead),
        "--headdim",
        str(args.headdim),
        "--methods",
        args.methods,
        "--zepplin-threshold",
        str(args.zepplin_threshold),
        "--sm-configs",
        args.sm_configs,
        "--warmup-iters",
        str(args.warmup_iters),
        "--num-iters",
        str(args.num_iters),
        "--seed",
        str(args.seed),
        "--dq-atol",
        str(args.dq_atol),
        "--dkv-atol",
        str(args.dkv_atol),
        "--rtol",
        str(args.rtol),
        "--check" if args.check else "--no-check",
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    world_size = _world_size(args)
    try:
        workloads = balancer.make_workloads(
            dataset=args.dataset,
            target_tokens=args.target_tokens,
            seed=args.seed,
            num_cases=args.num_cases,
            world_size=world_size,
            mode="causal",
            balance_tolerance=args.balance_tolerance,
            token_balance_tolerance=args.token_balance_tolerance,
            max_compute_balance_tolerance=args.max_compute_balance_tolerance,
            max_token_balance_tolerance=args.max_token_balance_tolerance,
            communication_weight=args.communication_weight,
            local_search_passes=args.local_search_passes,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.print_workload or local_rank == 0:
        for case_index, workload in enumerate(workloads):
            print(f"\nDataset case {case_index + 1}/{args.num_cases}")
            print_workload(
                workload,
                args.dataset,
                args.seed,
                args.target_tokens,
                world_size,
                "causal",
            )
    if args.print_workload:
        return

    benchmark_cases = [
        HybridBenchmarkCase(
            label=f"dataset={args.dataset}, case={case_index + 1}/{args.num_cases}",
            case_index=case_index,
            num_cases=args.num_cases,
            global_lengths=tuple(workload.global_lengths),
            ring_sizes=tuple(workload.ring_sizes),
            ring_starts=tuple(workload.ring_starts),
        )
        for case_index, workload in enumerate(workloads)
    ]
    forwarded_argv = _benchmark_argv(args, workloads[0])

    import benchmark_hybrid_backward

    benchmark_hybrid_backward.main(
        forwarded_argv,
        workload_cases=benchmark_cases,
        skip_incompatible_methods=_requests_all(args.methods),
    )


if __name__ == "__main__":
    main()
