"""Dataset-shaped frontend for the explicit-topology forward benchmark."""

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
from ring_test.utils import HybridBenchmarkCase
from zeppelin import DEFAULT_ZEPPELIN_THRESHOLD


SUPPORTED_KVHEADS = (1, 2, 4, 8)


def _format_int_list(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _magi_overlap_degree(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 8:
        raise argparse.ArgumentTypeError("value must be an integer in [1, 8]")
    return parsed


def _csv_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return values


def _selected_datasets(args: argparse.Namespace) -> list[str]:
    if args.dataset is not None and args.datasets is not None:
        raise SystemExit("--dataset and --datasets cannot be combined")
    datasets = [args.dataset] if args.dataset is not None else args.datasets
    if datasets is None:
        raise SystemExit("one of --dataset or --datasets is required")
    unknown = [dataset for dataset in datasets if dataset not in balancer.DATASET_WEIGHTS]
    if unknown:
        raise SystemExit(
            f"unknown datasets {unknown}; expected values from {tuple(balancer.DATASET_WEIGHTS)}"
        )
    return list(dict.fromkeys(datasets))


def _selected_kvheads(args: argparse.Namespace) -> list[int]:
    kvheads = [args.kvhead] if args.kvheads is None else args.kvheads
    unsupported = [kvhead for kvhead in kvheads if kvhead not in SUPPORTED_KVHEADS]
    if unsupported:
        raise SystemExit(
            f"--kvhead/--kvheads must contain only {SUPPORTED_KVHEADS}, got {unsupported}"
        )
    if any(args.qhead % kvhead for kvhead in kvheads):
        raise SystemExit("qhead must be divisible by every requested kvhead")
    return list(dict.fromkeys(kvheads))


def _heads_k_stride(kvhead: int, requested: int) -> int:
    return max(
        stride
        for stride in range(1, min(kvhead, requested) + 1)
        if kvhead % stride == 0
    )


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

    balance_rows = (
        (
            "Compute",
            f"{min(workload.rank_compute):.3e}",
            f"{workload.average_compute:.3e}",
            f"{workload.peak_compute:.3e}",
            f"{100.0 * workload.compute_deviation:.2f}%",
            f"{100.0 * workload.compute_balance_tolerance:.2f}%",
            str(
                workload.compute_deviation
                <= workload.compute_balance_tolerance + 1e-12
            ),
        ),
        (
            "Tokens",
            str(min(workload.rank_tokens)),
            f"{workload.average_tokens:.3f}",
            str(workload.peak_tokens),
            f"{100.0 * workload.token_deviation:.2f}%",
            f"{100.0 * workload.token_balance_tolerance:.2f}%",
            str(
                workload.token_deviation
                <= workload.token_balance_tolerance + 1e-12
            ),
        ),
    )
    print("\nBalance targets")
    print(
        f"{'Metric':<8} {'Minimum':>14} {'Average':>14} {'Maximum':>14} "
        f"{'Max deviation':>14} {'Tolerance':>12} {'Met':>7}"
    )
    for row in balance_rows:
        metric, minimum, average, maximum, deviation, tolerance, met = row
        print(
            f"{metric:<8} {minimum:>14} {average:>14} {maximum:>14} "
            f"{deviation:>14} {tolerance:>12} {met:>7}"
        )

    print(
        "\nPlanner status: "
        f"feasible={workload.feasible}, "
        f"violation={workload.load_violation:.6f}, "
        f"relaxation_level={workload.relaxation_level}, "
        f"relaxation={workload.relaxation_label!r}, "
        f"repair_moves={workload.repair_moves}"
    )
    print("Split protection")
    print(f"{'Bucket':>10} {'Split count':>12} {'Split penalty':>15}")
    for label, count, penalty in zip(
        balancer.LENGTH_BUCKET_LABELS,
        workload.split_counts,
        workload.split_penalties,
    ):
        print(f"{label:>10} {count:>12} {penalty:>15}")
    print(
        "Communication: "
        f"amplification={workload.communication_amplification:.6f}, "
        f"proxy_Q={workload.communication_cost:.3e}, "
        f"active_rings={workload.active_ring_count}, "
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
        description="Generate a dataset-shaped workload and run the explicit-topology forward benchmark"
    )
    parser.add_argument("--dataset", choices=tuple(balancer.DATASET_WEIGHTS))
    parser.add_argument(
        "--datasets",
        type=_csv_values,
        help="Comma-separated datasets to benchmark in the same torchrun",
    )
    parser.add_argument("--target-tokens", type=int, default=balancer.MAX_SEQUENCE_TOKENS)
    parser.add_argument(
        "--compute-balance-tolerance",
        type=float,
        default=0.05,
        help="Maximum absolute attention-compute deviation from the rank average",
    )
    parser.add_argument(
        "--token-balance-tolerance",
        type=float,
        default=0.10,
        help="Maximum absolute token-load deviation from the rank average",
    )
    parser.add_argument(
        "--beam-width",
        type=_positive_int,
        default=64,
        help="Maximum BR-PBS states retained after each structural placement",
    )
    parser.add_argument(
        "--finalist-count",
        type=_positive_int,
        default=8,
        help="Number of complete Beam solutions passed to local repair",
    )
    parser.add_argument(
        "--structure-threshold",
        type=float,
        default=0.5,
        help="Minimum normalized sequence size included in structural search",
    )
    parser.add_argument(
        "--max-repair-iterations",
        type=_nonnegative_int,
        default=32,
        help="Maximum number of strict lexicographic local-repair moves",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-cases", type=_positive_int, default=1)
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8))
    parser.add_argument("--print-workload", action="store_true")
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument(
        "--kvheads",
        type=lambda value: [int(item) for item in _csv_values(value)],
        help="Comma-separated KV head counts to benchmark in the same torchrun",
    )
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument(
        "--allgather-overlapping-heads-k-stride",
        type=_positive_int,
        default=4,
        help="KV heads per all-gather/attention overlap pipeline chunk",
    )
    parser.add_argument("--mode", choices=("noncausal", "causal", "both"), default="causal")
    parser.add_argument("--methods", default="all")
    parser.add_argument(
        "--zeppelin-threshold",
        type=_positive_int,
        default=DEFAULT_ZEPPELIN_THRESHOLD,
    )
    parser.add_argument(
        "--megatron-max-seqlen-per-rank",
        type=_positive_int,
        default=8192,
    )
    parser.add_argument(
        "--magi-overlap-degree",
        type=_magi_overlap_degree,
        default=2,
        help="Static MagiAttention overlap degree (1-8)",
    )
    parser.add_argument("--sm-configs", default="128:4,124:8,120:12,116:16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=40)
    parser.add_argument("--collect-mega-ring-stats", action="store_true")
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
        "--allgather-overlapping-heads-k-stride",
        str(args.allgather_overlapping_heads_k_stride),
        "--mode",
        args.mode,
        "--methods",
        args.methods,
        "--zeppelin-threshold",
        str(args.zeppelin_threshold),
        "--megatron-max-seqlen-per-rank",
        str(args.megatron_max_seqlen_per_rank),
        "--magi-overlap-degree",
        str(args.magi_overlap_degree),
        "--sm-configs",
        args.sm_configs,
        "--warmup-iters",
        str(args.warmup_iters),
        "--num-iters",
        str(args.num_iters),
        *(["--collect-mega-ring-stats"] if args.collect_mega_ring_stats else []),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--seed",
        str(args.seed),
        "--check" if args.check else "--no-check",
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    world_size = _world_size(args)
    datasets = _selected_datasets(args)
    kvheads = _selected_kvheads(args)
    workloads_by_dataset: list[tuple[str, list[balancer.HybridWorkload]]] = []
    for dataset in datasets:
        try:
            workloads = balancer.make_workloads(
                dataset=dataset,
                target_tokens=args.target_tokens,
                seed=args.seed,
                num_cases=args.num_cases,
                world_size=world_size,
                mode=args.mode,
                compute_balance_tolerance=args.compute_balance_tolerance,
                token_balance_tolerance=args.token_balance_tolerance,
                beam_width=args.beam_width,
                finalist_count=args.finalist_count,
                structure_threshold=args.structure_threshold,
                max_repair_iterations=args.max_repair_iterations,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(f"dataset={dataset}: {exc}") from exc
        workloads_by_dataset.append((dataset, workloads))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.print_workload or local_rank == 0:
        for dataset, workloads in workloads_by_dataset:
            for case_index, workload in enumerate(workloads):
                print(f"\nDataset case {case_index + 1}/{args.num_cases}")
                print_workload(
                    workload,
                    dataset,
                    args.seed,
                    args.target_tokens,
                    world_size,
                    args.mode,
                )
    if args.print_workload:
        return

    total_cases = len(datasets) * args.num_cases
    benchmark_cases: list[HybridBenchmarkCase] = []
    for dataset, workloads in workloads_by_dataset:
        for dataset_case_index, workload in enumerate(workloads):
            case_index = len(benchmark_cases)
            benchmark_cases.append(
                HybridBenchmarkCase(
                    label=(
                        f"dataset={dataset}, "
                        f"case={dataset_case_index + 1}/{args.num_cases}"
                    ),
                    case_index=case_index,
                    num_cases=total_cases,
                    global_lengths=tuple(workload.global_lengths),
                    ring_sizes=tuple(workload.ring_sizes),
                    ring_starts=tuple(workload.ring_starts),
                )
            )

    import benchmark_topology_forward
    import torch
    import torch.distributed as dist

    requested_stride = args.allgather_overlapping_heads_k_stride
    try:
        for kvhead in kvheads:
            args.kvhead = kvhead
            args.allgather_overlapping_heads_k_stride = _heads_k_stride(
                kvhead, requested_stride
            )
            if local_rank == 0:
                print(
                    f"\nDataset matrix forward KVH={kvhead}: "
                    f"datasets={datasets}, cases={total_cases}",
                    flush=True,
                )
            forwarded_argv = _benchmark_argv(args, workloads_by_dataset[0][1][0])
            benchmark_topology_forward.main(
                forwarded_argv,
                workload_cases=benchmark_cases,
                skip_incompatible_methods=_requests_all(args.methods),
                manage_process_group=False,
            )
    finally:
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
