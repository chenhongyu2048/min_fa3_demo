"""Dataset-shaped frontend for the hierarchical hybrid forward benchmark."""

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


def _format_int_list(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def print_workload(
    workload: balancer.HybridWorkload,
    dataset: str,
    seed: int,
    target_tokens: int,
    world_size: int,
    mode: str = "causal",
) -> None:
    actual_tokens = sum(workload.global_lengths)
    planner_is_causal = mode in ("causal", "both")
    planner_mode = "causal-compatible" if mode == "both" else mode
    print(
        f"Planner workload: dataset={dataset}, seed={seed}, "
        f"world_size={world_size}, mode={planner_mode}"
    )
    print(
        f"Batch: B={len(workload.global_lengths)}, target_tokens={target_tokens}, "
        f"actual_tokens={actual_tokens}, "
        f"target_delta={actual_tokens - target_tokens:+d}"
    )

    print("\nSequence placement")
    print(
        f"{'Seq':>4} {'Global S':>10} {'Ring':>6} {'Ranks':>9} "
        f"{'Local S':>10} {'Est compute/member':>19}"
    )
    for index, (length, ring_size, ring_start) in enumerate(
        zip(workload.global_lengths, workload.ring_sizes, workload.ring_starts)
    ):
        ranks = (
            str(ring_start)
            if ring_size == 1
            else f"{ring_start}-{ring_start + ring_size - 1}"
        )
        compute_per_member = balancer.attention_compute(length, planner_is_causal) / ring_size
        print(
            f"{index:>4} {length:>10} {f'G{ring_size}':>6} {ranks:>9} "
            f"{length // ring_size:>10} {compute_per_member:>19.3e}"
        )

    print("\nPer-rank load")
    print(
        f"{'Rank':>4} {'Tokens':>10} {'Token/avg':>10} "
        f"{'Est compute':>14} {'Compute/avg':>12} {'Est communication':>19}"
    )
    for rank, (tokens, compute, communication) in enumerate(
        zip(
            workload.rank_tokens,
            workload.rank_compute,
            workload.rank_communication,
        )
    ):
        print(
            f"{rank:>4} {tokens:>10} "
            f"{100.0 * tokens / workload.average_tokens:>9.2f}% "
            f"{compute:>14.3e} "
            f"{100.0 * compute / workload.average_compute:>11.2f}% "
            f"{communication:>19.3e}"
        )

    cap_rows = (
        (
            "Compute",
            f"{workload.average_compute:.3e}",
            f"{workload.peak_compute:.3e}",
            f"{100.0 * workload.imbalance:.2f}%",
            f"{workload.requested_cap:.3e}",
            f"{workload.topology_lower_bound:.3e}",
            f"{workload.final_cap:.3e}",
            str(workload.topology_limited),
            str(workload.compute_cap_relaxed),
        ),
        (
            "Tokens",
            f"{workload.average_tokens:.3f}",
            str(workload.peak_tokens),
            f"{100.0 * workload.token_imbalance:.2f}%",
            f"{workload.requested_token_cap:.3f}",
            f"{workload.token_topology_lower_bound:.3f}",
            f"{workload.final_token_cap:.3f}",
            str(workload.token_topology_limited),
            str(workload.token_cap_relaxed),
        ),
    )
    print("\nBalance and caps")
    print(
        f"{'Metric':<8} {'Average':>14} {'Peak':>14} {'Imbalance':>10} "
        f"{'Requested cap':>14} {'Topology LB':>14} {'Final cap':>14} "
        f"{'Topology':>10} {'Relaxed':>9}"
    )
    for row in cap_rows:
        metric, average, peak, imbalance, requested, topology, final, limited, relaxed = row
        print(
            f"{metric:<8} {average:>14} {peak:>14} {imbalance:>10} "
            f"{requested:>14} {topology:>14} {final:>14} "
            f"{limited:>10} {relaxed:>9}"
        )

    print(
        "\nPlanner status: "
        f"placement_relaxed={workload.placement_relaxed}, "
        f"emergency_relaxed={workload.emergency_relaxed}, "
        f"local_search_moves={workload.local_search_moves}"
    )
    print(
        "Communication: "
        f"amplification={workload.communication_amplification:.6f}, "
        f"total_token_hops={sum(workload.rank_communication):.3e}"
    )

    print("\nKernel metadata")
    print(f"global_seqlens={_format_int_list(workload.global_lengths)}")
    print(f"ring_sizes={_format_int_list(workload.ring_sizes)}")
    print(
        f"ring_starts={_format_int_list(workload.ring_starts)}",
        flush=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a dataset-shaped workload and run the hybrid forward benchmark"
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
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8))
    parser.add_argument("--print-workload", action="store_true")
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--mode", choices=("noncausal", "causal", "both"), default="causal")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--sm-configs", default="128:4,124:8,120:12,116:16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=40)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("--rtol", type=float, default=2e-1)
    return parser.parse_args(argv)


def _world_size(args: argparse.Namespace) -> int:
    env_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if args.print_workload:
        if args.world_size is not None:
            return args.world_size
        if env_world_size is not None:
            return int(env_world_size)
        raise SystemExit("--print-workload requires --world-size outside torchrun")
    if env_world_size is None:
        raise SystemExit("Run this benchmark with torchrun")
    world_size = int(env_world_size)
    if args.world_size is not None and args.world_size != world_size:
        raise SystemExit(
            f"--world-size={args.world_size} does not match LOCAL_WORLD_SIZE={world_size}"
        )
    return world_size


def _requests_all(methods: str) -> bool:
    return any(token.strip() == "all" for token in methods.split(","))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    world_size = _world_size(args)
    try:
        workload = balancer.make_workload(
            dataset=args.dataset,
            target_tokens=args.target_tokens,
            seed=args.seed,
            world_size=world_size,
            mode=args.mode,
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
        print_workload(
            workload,
            args.dataset,
            args.seed,
            args.target_tokens,
            world_size,
            args.mode,
        )
    if args.print_workload:
        return

    forwarded_argv = [
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
        "--mode",
        args.mode,
        "--methods",
        args.methods,
        "--sm-configs",
        args.sm_configs,
        "--warmup-iters",
        str(args.warmup_iters),
        "--num-iters",
        str(args.num_iters),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--seed",
        str(args.seed),
        "--check" if args.check else "--no-check",
    ]

    import benchmark_hybrid_forward

    benchmark_hybrid_forward.main(
        forwarded_argv,
        skip_incompatible_methods=_requests_all(args.methods),
    )


if __name__ == "__main__":
    main()
