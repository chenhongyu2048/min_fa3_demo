"""Shared CPU workload construction and CLI support for the five-method suite."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Sequence

import balancer
from ring_test.load_balance_bench.topology import (
    PlannerControls,
    PlannerTopology,
    make_br_pbs_topology,
    make_megatron_cp_topology,
    make_zepplin_topology,
)
from ring_test.zepplin import DEFAULT_ZEPPLIN_THRESHOLD


RESULT_LABELS = (
    "native_megatron_hybrid_cp",
    "native_zepplin",
    "mega_ring_hybrid_br_pbs",
    "mega_ring_hybrid_megatron_cp",
    "mega_ring_hybrid_zepplin",
)


@dataclass(frozen=True)
class LoadBalanceCase:
    case_index: int
    num_cases: int
    raw_lengths: tuple[int, ...]
    br_pbs: PlannerTopology
    megatron_cp: PlannerTopology
    zepplin: PlannerTopology

    @property
    def topologies(self) -> tuple[PlannerTopology, PlannerTopology, PlannerTopology]:
        return self.br_pbs, self.megatron_cp, self.zepplin


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=tuple(balancer.DATASET_WEIGHTS), required=True)
    parser.add_argument("--target-tokens", type=positive_int, default=balancer.MAX_SEQUENCE_TOKENS)
    parser.add_argument("--compute-balance-tolerance", type=float, default=0.05)
    parser.add_argument("--token-balance-tolerance", type=float, default=0.10)
    parser.add_argument("--beam-width", type=positive_int, default=64)
    parser.add_argument("--finalist-count", type=positive_int, default=8)
    parser.add_argument("--structure-threshold", type=float, default=0.5)
    parser.add_argument("--max-repair-iterations", type=nonnegative_int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-cases", type=positive_int, default=1)
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8))
    parser.add_argument("--print-workload", action="store_true")
    parser.add_argument("--qhead", type=positive_int, default=32)
    parser.add_argument("--kvhead", type=positive_int, default=8)
    parser.add_argument("--headdim", type=positive_int, default=128)
    parser.add_argument(
        "--allgather-overlapping-heads-k-stride",
        type=positive_int,
        default=4,
        help="KV heads per all-gather/attention overlap pipeline chunk",
    )
    parser.add_argument(
        "--zepplin-threshold", type=positive_int, default=DEFAULT_ZEPPLIN_THRESHOLD
    )
    parser.add_argument(
        "--megatron-max-seqlen-per-rank", type=positive_int, default=8192
    )
    parser.add_argument("--sm-configs", default="128:4,124:8,120:12,116:16")
    parser.add_argument("--warmup-iters", type=nonnegative_int, default=10)
    parser.add_argument("--num-iters", type=positive_int, default=40)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rtol", type=float, default=0.2)


def planner_controls(args: argparse.Namespace) -> PlannerControls:
    return PlannerControls(
        compute_balance_tolerance=args.compute_balance_tolerance,
        token_balance_tolerance=args.token_balance_tolerance,
        beam_width=args.beam_width,
        finalist_count=args.finalist_count,
        structure_threshold=args.structure_threshold,
        max_repair_iterations=args.max_repair_iterations,
    )


def resolve_world_size(args: argparse.Namespace) -> int:
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


def mode_values(mode: str) -> tuple[bool, ...]:
    if mode == "noncausal":
        return (False,)
    if mode == "causal":
        return (True,)
    if mode == "both":
        return (False, True)
    raise ValueError(f"unknown forward mode {mode!r}")


def mode_name(is_causal: bool) -> str:
    return "causal" if is_causal else "noncausal"


def build_cases(
    args: argparse.Namespace, world_size: int, is_causal: bool
) -> tuple[LoadBalanceCase, ...]:
    """Sample raw lengths once, then map each planner independently."""

    try:
        raw_cases = balancer.generate_dataset_length_cases(
            args.dataset, args.target_tokens, args.seed, args.num_cases
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    controls = planner_controls(args)
    cases: list[LoadBalanceCase] = []
    mode = mode_name(is_causal)
    for case_index, raw_lengths in enumerate(raw_cases):
        lengths = tuple(raw_lengths)
        try:
            br_pbs = make_br_pbs_topology(lengths, world_size, is_causal, controls)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(
                f"BR-PBS planner failed for {mode} case {case_index + 1}: {exc}"
            ) from exc
        try:
            megatron_cp = make_megatron_cp_topology(
                lengths,
                world_size,
                is_causal,
                args.megatron_max_seqlen_per_rank,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(
                f"Megatron CP planner failed for {mode} case {case_index + 1}: {exc}"
            ) from exc
        try:
            zepplin = make_zepplin_topology(
                lengths, world_size, is_causal, args.zepplin_threshold
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(
                f"Zepllin planner failed for {mode} case {case_index + 1}: {exc}"
            ) from exc
        cases.append(
            LoadBalanceCase(
                case_index, args.num_cases, lengths, br_pbs, megatron_cp, zepplin
            )
        )
    return tuple(cases)


def print_cases(cases: Sequence[LoadBalanceCase], is_causal: bool) -> None:
    """Print raw and effective workloads without importing CUDA benchmark code."""

    print(f"\nPlanner workloads ({mode_name(is_causal)})")
    for case in cases:
        print(
            f"Case {case.case_index + 1}/{case.num_cases}: "
            f"raw_tokens={sum(case.raw_lengths)}, raw_sample_lengths="
            f"{','.join(str(length) for length in case.raw_lengths)}"
        )
        for topology in case.topologies:
            diagnostics = "; ".join(
                f"{name}={value}" for name, value in topology.diagnostics
            )
            print(
                f"  {topology.planner}: execution_tokens={topology.execution_tokens}, "
                f"padding={topology.padding_tokens}, "
                f"build_ms(rank0)={topology.planner_build_ms:.3f}; "
                f"sample_ids={','.join(str(value) for value in topology.sample_ids)}; "
                f"global_seqlens={','.join(str(value) for value in topology.global_lengths)}; "
                f"ring_sizes={','.join(str(value) for value in topology.ring_sizes)}; "
                f"ring_starts={','.join(str(value) for value in topology.ring_starts)}"
            )
            if diagnostics:
                print(f"    diagnostics: {diagnostics}")


def comma_separated(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)

