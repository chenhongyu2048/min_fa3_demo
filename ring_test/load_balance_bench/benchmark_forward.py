"""Five-method dataset-shaped forward runtime benchmark."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence


THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from ring_test.load_balance_bench.common import (
    RESULT_LABELS,
    LoadBalanceCase,
    add_shared_arguments,
    build_cases,
    comma_separated,
    mode_name,
    mode_values,
    print_cases,
    resolve_world_size,
)
from ring_test.load_balance_bench.topology import PlannerTopology, validate_with_runner


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run native Megatron/Zeppelin and three fused Mega Ring placements "
            "on the same dataset-shaped raw workload"
        )
    )
    add_shared_arguments(parser)
    parser.add_argument("--mode", choices=("noncausal", "causal", "both"), default="causal")
    parser.add_argument("--collect-mega-ring-stats", action="store_true")
    parser.add_argument("--atol", type=float, default=2e-1)
    return parser.parse_args(argv)


def _runner_argv(
    args: argparse.Namespace,
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    method: str,
    is_causal: bool,
) -> list[str]:
    return [
        "--global-seqlens",
        comma_separated(global_lengths),
        "--ring-sizes",
        comma_separated(ring_sizes),
        "--ring-starts",
        comma_separated(ring_starts),
        "--qhead",
        str(args.qhead),
        "--kvhead",
        str(args.kvhead),
        "--headdim",
        str(args.headdim),
        "--allgather-overlapping-heads-k-stride",
        str(args.allgather_overlapping_heads_k_stride),
        "--mode",
        mode_name(is_causal),
        "--methods",
        method,
        "--zeppelin-threshold",
        str(args.zeppelin_threshold),
        "--megatron-max-seqlen-per-rank",
        str(args.megatron_max_seqlen_per_rank),
        "--sm-configs",
        args.sm_configs,
        "--warmup-iters",
        str(args.warmup_iters),
        "--num-iters",
        str(args.num_iters),
        "--seed",
        str(args.seed),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--check" if args.check else "--no-check",
        *(["--collect-mega-ring-stats"] if args.collect_mega_ring_stats and method == "mega_ring_hybrid" else []),
    ]


def _native_metadata(raw_lengths: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Metadata placeholders; native planners derive placement from raw lengths."""

    return tuple(1 for _ in raw_lengths), tuple(0 for _ in raw_lengths)


def _mapped_metadata(topology: PlannerTopology) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return topology.global_lengths, topology.ring_sizes, topology.ring_starts


def _all_fused_topologies(
    mode_cases: Sequence[tuple[bool, Sequence[LoadBalanceCase]]]
) -> list[PlannerTopology]:
    return [
        topology
        for _is_causal, cases in mode_cases
        for case in cases
        for topology in case.topologies
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    world_size = resolve_world_size(args)
    mode_cases = tuple(
        (is_causal, build_cases(args, world_size, is_causal))
        for is_causal in mode_values(args.mode)
    )
    if args.print_workload:
        for is_causal, cases in mode_cases:
            print_cases(cases, is_causal)
        return

    import torch
    import torch.distributed as dist

    import ring_test.benchmark_topology_forward as runner

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("SM90 Hopper CUDA device is required")
    if args.headdim != 128:
        raise SystemExit("this benchmark requires D=128")
    if args.kvhead * args.headdim != 1024:
        raise SystemExit("the fused Mega Ring methods require KVH * D == 1024")
    if args.qhead % args.kvhead:
        raise SystemExit("qhead must be divisible by kvhead")

    rank, actual_world_size = runner.init_distributed()
    if actual_world_size != world_size:
        raise RuntimeError("LOCAL_WORLD_SIZE changed after workload construction")
    pools = None
    try:
        fused_topologies = _all_fused_topologies(mode_cases)
        for topology in fused_topologies:
            validate_with_runner(topology, "forward")
        hybrid_capacity = runner.max_hybrid_rank_capacity(
            [
                runner.HybridBenchmarkCase(
                    label=topology.planner,
                    case_index=index,
                    num_cases=len(fused_topologies),
                    global_lengths=topology.global_lengths,
                    ring_sizes=topology.ring_sizes,
                    ring_starts=topology.ring_starts,
                )
                for index, topology in enumerate(fused_topologies)
            ],
            world_size,
        )
        pools = runner.ForwardParallelPools(
            all_cp=None,
            hybrid=runner.make_mega_parallel_tensors(
                rank,
                world_size,
                hybrid_capacity,
                args.kvhead,
                args.headdim,
            ),
        )
        if rank == 0:
            for is_causal, cases in mode_cases:
                print_cases(cases, is_causal)
            print(
                "Reusable forward IPC pool: "
                f"topologies={len(fused_topologies)}, hybrid_rank_capacity={hybrid_capacity}",
                flush=True,
            )

        summary_samples = []
        for is_causal, cases in mode_cases:
            for case in cases:
                raw_sizes, raw_starts = _native_metadata(case.raw_lengths)
                requests = (
                    (
                        RESULT_LABELS[0],
                        "megatron_hybrid_cp",
                        case.raw_lengths,
                        raw_sizes,
                        raw_starts,
                        None,
                    ),
                    (
                        RESULT_LABELS[1],
                        "zeppelin",
                        case.raw_lengths,
                        raw_sizes,
                        raw_starts,
                        None,
                    ),
                    (
                        RESULT_LABELS[2],
                        "mega_ring_hybrid",
                        *_mapped_metadata(case.br_pbs),
                        case.raw_lengths,
                    ),
                    (
                        RESULT_LABELS[3],
                        "mega_ring_hybrid",
                        *_mapped_metadata(case.megatron_cp),
                        case.raw_lengths,
                    ),
                    (
                        RESULT_LABELS[4],
                        "mega_ring_hybrid",
                        *_mapped_metadata(case.zeppelin),
                        case.raw_lengths,
                    ),
                )
                for label, method, lengths, ring_sizes, ring_starts, metric_lengths in requests:
                    if rank == 0:
                        print(
                            f"\nFive-method result: {label}; "
                            f"case={case.case_index + 1}/{case.num_cases}; "
                            f"mode={mode_name(is_causal)}",
                            flush=True,
                        )
                    records = runner._main_single(
                        _runner_argv(
                            args,
                            lengths,
                            ring_sizes,
                            ring_starts,
                            method,
                            is_causal,
                        ),
                        parallel_pools=pools,
                        case_label=(
                            f"dataset={args.dataset}, case={case.case_index + 1}/"
                            f"{case.num_cases}, suite={label}"
                        ),
                        case_index=case.case_index,
                        manage_process_group=False,
                        metric_global_lengths=metric_lengths,
                    )
                    if rank == 0:
                        summary_samples.extend(
                            replace(record, method=label) for record in records
                        )
        if rank == 0:
            runner._print_forward_summary(
                summary_samples, args.num_cases, world_size
            )
    finally:
        if dist.is_initialized():
            runner.cuda_barrier()
        if pools is not None:
            pools.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
