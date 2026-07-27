"""Five-method dataset-shaped causal backward runtime benchmark."""

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
    mode_name,
    print_cases,
    resolve_world_size,
)
from ring_test.load_balance_bench.topology import validate_with_runner


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run native Megatron/Zeppelin and three fused Mega Ring placements "
            "on the same dataset-shaped causal backward workload"
        )
    )
    add_shared_arguments(parser)
    parser.add_argument(
        "--mode",
        choices=("noncausal", "causal", "both"),
        default="causal",
        help="backward accepts causal only",
    )
    parser.add_argument(
        "--dq-atol",
        type=float,
        default=3.0,
        help="Absolute dQ check tolerance for the fused backward numerical tail",
    )
    parser.add_argument("--dkv-atol", type=float, default=0.5)
    return parser.parse_args(argv)


def _native_metadata(raw_lengths: Sequence[int]) -> tuple[list[int], list[int]]:
    return [1] * len(raw_lengths), [0] * len(raw_lengths)


def _fused_workloads(cases: Sequence[LoadBalanceCase]) -> list[tuple[str, list[int], list[int], list[int]]]:
    workloads: list[tuple[str, list[int], list[int], list[int]]] = []
    for case in cases:
        for topology in case.topologies:
            workloads.append(
                (
                    f"case={case.case_index + 1}/{case.num_cases}, {topology.planner}",
                    list(topology.global_lengths),
                    list(topology.ring_sizes),
                    list(topology.ring_starts),
                )
            )
    return workloads


def _validate_common_args(args: argparse.Namespace) -> None:
    if args.mode != "causal":
        raise SystemExit("five-method backward benchmark supports only --mode causal")
    if args.headdim != 128:
        raise SystemExit("this benchmark requires D=128")
    if args.kvhead * args.headdim != 1024:
        raise SystemExit("the fused Mega Ring methods require KVH * D == 1024")
    if args.qhead % args.kvhead:
        raise SystemExit("qhead must be divisible by kvhead")
    if args.allgather_overlapping_heads_k_stride <= 0 or (
        args.kvhead % args.allgather_overlapping_heads_k_stride
    ):
        raise SystemExit(
            "--allgather-overlapping-heads-k-stride must be a positive divisor "
            "of --kvhead"
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_common_args(args)
    world_size = resolve_world_size(args)
    cases = build_cases(args, world_size, True)
    if args.print_workload:
        print_cases(cases, True)
        return

    import torch
    import torch.distributed as dist

    import ring_test.benchmark_topology_backward as runner

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("SM90 Hopper CUDA device is required")
    sm_configs = runner.parse_sm_configs(args.sm_configs)
    rank, actual_world_size = runner.init_distributed()
    if actual_world_size != world_size:
        raise RuntimeError("LOCAL_WORLD_SIZE changed after workload construction")
    pools = None
    try:
        for case in cases:
            for topology in case.topologies:
                validate_with_runner(topology, "backward")
        sm_count = torch.cuda.get_device_properties(rank).multi_processor_count
        for config in sm_configs:
            if config.num_comp_sm <= 0 or config.num_comm_sm <= 0:
                raise SystemExit("backward requires positive compute and communication SM counts")
            if config.num_comp_sm + config.num_comm_sm > sm_count:
                raise SystemExit(
                    f"SM config {config.num_comp_sm}:{config.num_comm_sm} "
                    f"exceeds device SM count {sm_count}"
                )

        allgather_backend = runner.select_fa3_backend(
            dist.group.WORLD, require_backward=True
        )
        pools = runner.make_backward_parallel_pools(
            _fused_workloads(cases),
            ["mega_ring_hybrid"],
            rank,
            world_size,
            args.kvhead,
            args.headdim,
        )
        if rank == 0:
            print_cases(cases, True)
            hybrid_pool = pools.hybrid
            print(
                "Reusable backward IPC pool: "
                f"topologies={len(cases) * 3}, "
                f"hybrid_rank_capacity="
                f"{None if hybrid_pool is None else hybrid_pool.rank_capacity}, "
                f"hybrid_accum_numel="
                f"{None if hybrid_pool is None else hybrid_pool.remote_dk.data_.numel()}",
                flush=True,
            )
            backend_name = (
                "external FA3"
                if allgather_backend == "external_fa3"
                else "in-repo min_fa3 fallback"
            )
            print(f"Block baseline backend: {backend_name}", flush=True)

        summary_samples = []
        for case in cases:
            raw_sizes, raw_starts = _native_metadata(case.raw_lengths)
            requests = (
                (
                    RESULT_LABELS[0],
                    "megatron_hybrid_cp",
                    list(case.raw_lengths),
                    raw_sizes,
                    raw_starts,
                    None,
                ),
                (
                    RESULT_LABELS[1],
                    "zeppelin",
                    list(case.raw_lengths),
                    raw_sizes,
                    raw_starts,
                    None,
                ),
                (
                    RESULT_LABELS[2],
                    "mega_ring_hybrid",
                    list(case.br_pbs.global_lengths),
                    list(case.br_pbs.ring_sizes),
                    list(case.br_pbs.ring_starts),
                    case.raw_lengths,
                ),
                (
                    RESULT_LABELS[3],
                    "mega_ring_hybrid",
                    list(case.megatron_cp.global_lengths),
                    list(case.megatron_cp.ring_sizes),
                    list(case.megatron_cp.ring_starts),
                    case.raw_lengths,
                ),
                (
                    RESULT_LABELS[4],
                    "mega_ring_hybrid",
                    list(case.zeppelin.global_lengths),
                    list(case.zeppelin.ring_sizes),
                    list(case.zeppelin.ring_starts),
                    case.raw_lengths,
                ),
            )
            for label, method, lengths, ring_sizes, ring_starts, metric_lengths in requests:
                if rank == 0:
                    print(
                        f"\nFive-method result: {label}; "
                        f"case={case.case_index + 1}/{case.num_cases}; "
                        f"mode={mode_name(True)}",
                        flush=True,
                    )
                results = runner.benchmark_topology(
                    args,
                    rank,
                    world_size,
                    lengths,
                    ring_sizes,
                    ring_starts,
                    sm_configs,
                    (
                        f"dataset={args.dataset}, case={case.case_index + 1}/"
                        f"{case.num_cases}, suite={label}"
                    ),
                    [method],
                    allgather_backend,
                    pools,
                    metric_global_lengths=metric_lengths,
                )
                if rank == 0:
                    summary_samples.extend(
                        runner.BackwardSummarySample(
                            case.case_index, replace(result, method=label)
                        )
                        for result in results
                    )
        if rank == 0:
            runner._print_backward_summary(summary_samples, args.num_cases, world_size)
    finally:
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
            # Keep the VMM-backed IPC pool alive until the shared NCCL group is
            # gone; releasing it first can race a peer entering NCCL teardown.
            dist.destroy_process_group()
        if pools is not None:
            pools.close()


if __name__ == "__main__":
    main()
