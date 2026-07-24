"""Metadata-only forward/backward load benchmark for registered baselines."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence


THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import balancer
from baseline.megatron_hybrid_cp import hybrid_cp_incompatibility
from ring_test.backward_load_model import (
    BackwardMethodLoadResult,
    analyze_backward_magi_rank,
    analyze_backward_method,
    backward_magi_result_from_records,
    cumulative_backward_result,
    global_backward_ratio,
    global_backward_score_conserved,
)
from ring_test.forward_load_model import (
    METHOD_ORDER,
    MethodLoadResult,
    analyze_magi_rank,
    analyze_method,
    cumulative_result,
    global_ratio,
    global_score_conserved,
    magi_result_from_records,
)
from ring_test.utils import parse_int_list
from ring_test.zepplin import (
    DEFAULT_ZEPPLIN_THRESHOLD,
    zepplin_incompatibility,
)


@dataclass(frozen=True)
class WorkloadCase:
    label: str
    global_lengths: tuple[int, ...]
    ring_sizes: tuple[int, ...]
    ring_starts: tuple[int, ...]


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


def _overlap_degree(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 8:
        raise argparse.ArgumentTypeError("value must be an integer in [1, 8]")
    return parsed


def parse_methods(spec: str) -> list[str]:
    methods: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "all":
            methods.extend(METHOD_ORDER)
        elif token in METHOD_ORDER:
            methods.append(token)
        else:
            raise SystemExit(
                f"unknown method {token!r}, expected one of {list(METHOD_ORDER)} or all"
            )
    deduped = list(dict.fromkeys(methods))
    if not deduped:
        raise SystemExit("--methods must provide at least one method")
    return deduped


def _requests_all(spec: str) -> bool:
    return any(token.strip() == "all" for token in spec.split(","))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report static forward or backward token, score, FLOP, communication, "
            "and tile load without running attention kernels"
        )
    )
    parser.add_argument(
        "--direction", choices=("forward", "backward"), default="forward"
    )
    workload = parser.add_mutually_exclusive_group(required=True)
    workload.add_argument("--dataset", choices=tuple(balancer.DATASET_WEIGHTS))
    workload.add_argument("--global-seqlens")
    parser.add_argument("--ring-sizes")
    parser.add_argument("--ring-starts")
    parser.add_argument("--target-tokens", type=_positive_int, default=131072)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-cases", type=_positive_int, default=1)
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8))

    parser.add_argument("--compute-balance-tolerance", type=float, default=0.05)
    parser.add_argument("--token-balance-tolerance", type=float, default=0.10)
    parser.add_argument("--beam-width", type=_positive_int, default=64)
    parser.add_argument("--finalist-count", type=_positive_int, default=8)
    parser.add_argument("--structure-threshold", type=float, default=0.5)
    parser.add_argument("--max-repair-iterations", type=_nonnegative_int, default=32)

    parser.add_argument("--qhead", type=_positive_int, default=32)
    parser.add_argument("--kvhead", type=_positive_int, default=8)
    parser.add_argument("--headdim", type=_positive_int, default=128)
    parser.add_argument(
        "--mode", choices=("noncausal", "causal", "both"), default="causal"
    )
    parser.add_argument(
        "--methods",
        default="all",
        help=f"Comma-separated methods from {list(METHOD_ORDER)}, or all",
    )
    parser.add_argument(
        "--allgather-overlapping-heads-k-stride",
        type=_positive_int,
        default=4,
    )
    parser.add_argument(
        "--zepplin-threshold",
        type=_positive_int,
        default=DEFAULT_ZEPPLIN_THRESHOLD,
    )
    parser.add_argument(
        "--megatron-max-seqlen-per-rank", type=_positive_int, default=8192
    )
    parser.add_argument(
        "--magi-overlap-degree", type=_overlap_degree, default=2
    )
    return parser.parse_args(argv)


def _resolve_world_size(args: argparse.Namespace) -> tuple[int, bool, int]:
    env_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    under_torchrun = env_world_size is not None
    if under_torchrun:
        world_size = int(env_world_size)
        if world_size not in (2, 4, 8):
            raise SystemExit(
                f"LOCAL_WORLD_SIZE must be 2, 4, or 8, got {world_size}"
            )
        if args.world_size is not None and args.world_size != world_size:
            raise SystemExit(
                f"--world-size={args.world_size} does not match LOCAL_WORLD_SIZE={world_size}"
            )
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        return world_size, True, rank
    if args.world_size is None:
        raise SystemExit("ordinary Python static analysis requires --world-size 2, 4, or 8")
    return args.world_size, False, 0


def _validate_explicit_topology(
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    world_size: int,
) -> str | None:
    if not (
        len(global_lengths) == len(ring_sizes) == len(ring_starts)
    ):
        return "global lengths, ring sizes, and ring starts must have the same length"
    previous_size = world_size
    for index, (length, ring_size, ring_start) in enumerate(
        zip(global_lengths, ring_sizes, ring_starts)
    ):
        if length <= 0:
            return f"batch {index} has non-positive global length {length}"
        if ring_size not in (1, 2, 4, 8) or ring_size > world_size:
            return f"batch {index} has unsupported ring size {ring_size}"
        if ring_size > previous_size:
            return "ring sizes must be ordered non-increasingly"
        if (
            ring_start < 0
            or ring_start % ring_size
            or ring_start + ring_size > world_size
        ):
            return f"batch {index} has invalid aligned ring start {ring_start}"
        if length % ring_size:
            return f"batch {index} length {length} is not divisible by G{ring_size}"
        previous_size = ring_size
    return None


def _mega_hybrid_incompatibility(
    case: WorkloadCase,
    world_size: int,
    is_causal: bool,
) -> str | None:
    reason = _validate_explicit_topology(
        case.global_lengths, case.ring_sizes, case.ring_starts, world_size
    )
    if reason is not None:
        return reason
    for index, (length, ring_size) in enumerate(
        zip(case.global_lengths, case.ring_sizes)
    ):
        local_length = length // ring_size
        if local_length % 128:
            return (
                f"batch {index} G{ring_size} local length {local_length} "
                "is not 128-aligned"
            )
        if is_causal and ring_size > 1 and (local_length // 2) % 128:
            return (
                f"batch {index} causal G{ring_size} local half length "
                f"{local_length // 2} is not 128-aligned"
            )
    return None


def method_incompatibility(
    method: str,
    case: WorkloadCase,
    world_size: int,
    is_causal: bool,
    args: argparse.Namespace,
    *,
    magi_available: bool,
    magi_reason: str | None,
) -> str | None:
    lengths = case.global_lengths
    if method in ("mega_ring_all_cp", "mega_ring_hybrid") and (
        args.kvhead * args.headdim != 1024
    ):
        return "fused mega-ring requires KVH * D == 1024"
    if method in ("allgather_attention", "llama3_allgather_attention") and (
        args.kvhead % args.allgather_overlapping_heads_k_stride
    ):
        return "all-gather head stride must divide KVH"
    if method == "magi_attention":
        return None if magi_available else (magi_reason or "MagiAttention is unavailable")
    if method == "megatron_hybrid_cp":
        return hybrid_cp_incompatibility(
            lengths,
            world_size,
            is_causal,
            max_seqlen_per_rank=args.megatron_max_seqlen_per_rank,
        )
    if method == "zepplin":
        return zepplin_incompatibility(
            list(lengths), world_size, is_causal, args.zepplin_threshold
        )
    if method == "mega_ring_hybrid":
        return _mega_hybrid_incompatibility(case, world_size, is_causal)
    if method == "mega_ring_all_cp":
        return None
    for index, length in enumerate(lengths):
        if length % world_size:
            return (
                f"all-CP requires batch {index} length {length} divisible by "
                f"world_size={world_size}"
            )
        local_length = length // world_size
        if is_causal and local_length % 2:
            return f"causal all-CP batch {index} local length {local_length} is odd"
    if method == "llama3_allgather_attention" and sum(lengths) % (
        2 * world_size
    ):
        return "whole-packed Llama3 layout requires total tokens divisible by 2 * world_size"
    return None


def _make_cases(args: argparse.Namespace, world_size: int) -> list[WorkloadCase]:
    if args.dataset is None:
        if args.ring_sizes is None or args.ring_starts is None:
            raise SystemExit(
                "explicit --global-seqlens requires both --ring-sizes and --ring-starts"
            )
        lengths = tuple(parse_int_list(args.global_seqlens, "--global-seqlens"))
        ring_sizes = tuple(parse_int_list(args.ring_sizes, "--ring-sizes"))
        ring_starts = tuple(parse_int_list(args.ring_starts, "--ring-starts"))
        reason = _validate_explicit_topology(
            lengths, ring_sizes, ring_starts, world_size
        )
        if reason is not None:
            raise SystemExit(reason)
        return [
            WorkloadCase("explicit topology", lengths, ring_sizes, ring_starts)
        ]
    if args.ring_sizes is not None or args.ring_starts is not None:
        raise SystemExit("dataset mode does not accept --ring-sizes or --ring-starts")
    try:
        workloads = balancer.make_workloads(
            dataset=args.dataset,
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
        raise SystemExit(str(exc)) from exc
    return [
        WorkloadCase(
            f"dataset={args.dataset}, seed={args.seed}, case={index + 1}/{len(workloads)}",
            tuple(workload.global_lengths),
            tuple(workload.ring_sizes),
            tuple(workload.ring_starts),
        )
        for index, workload in enumerate(workloads)
    ]


def _human(value: float, *, bytes_value: bool = False) -> str:
    suffixes = (
        ((1 << 40, "TiB"), (1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB"))
        if bytes_value
        else ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K"))
    )
    magnitude = abs(value)
    for scale, suffix in suffixes:
        if magnitude >= scale:
            return f"{value / scale:.3f}{suffix}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}"


def _ratio(value: float, average: float) -> str:
    return f"{value / average:.3f}x" if average else "n/a"


def _print_efficiency_rows(result: MethodLoadResult) -> None:
    for record in result.records:
        visits = f"[{record.qo_visits_worst},{record.qo_visits_best}]"
        ratio = (
            f"[{record.kv_tiles_per_qo_lower:.3f},"
            f"{record.kv_tiles_per_qo_upper:.3f}]"
        )
        print(
            f"{record.rank:>4} {record.kv_tile_reads:>14} "
            f"{visits:>25} {ratio:>24}"
        )


def _summary_row(label: str, values: Sequence[float], *, as_bytes: bool = False) -> None:
    minimum = min(values)
    average = sum(values) / len(values)
    maximum = max(values)
    print(
        f"  {label:<16} min={_human(minimum, bytes_value=as_bytes):>10} "
        f"avg={_human(average, bytes_value=as_bytes):>10} "
        f"max={_human(maximum, bytes_value=as_bytes):>10} "
        f"max/avg={_ratio(maximum, average)}"
    )


def print_summary(result: MethodLoadResult) -> None:
    print("Summary")
    _summary_row(
        "Physical tokens", [record.physical_tokens for record in result.records]
    )
    _summary_row(
        "Physical FLOPs", [record.physical_flops for record in result.records]
    )
    _summary_row(
        "Sent bytes",
        [record.comm_total_bytes for record in result.records],
        as_bytes=True,
    )
    lower, upper = global_ratio(result)
    effective_tokens = sum(record.effective_tokens for record in result.records)
    physical_tokens = sum(record.physical_tokens for record in result.records)
    effective_scores = sum(record.effective_scores for record in result.records)
    physical_scores = sum(record.physical_scores for record in result.records)
    token_overhead = (
        physical_tokens / effective_tokens - 1 if effective_tokens else 0.0
    )
    score_overhead = (
        physical_scores / effective_scores - 1 if effective_scores else 0.0
    )
    print(f"  Global KV/QO    [{lower:.3f},{upper:.3f}]")
    print(
        f"  Overhead         tokens={100 * token_overhead:.3f}%, "
        f"scores={100 * score_overhead:.3f}%"
    )
    print(f"  Note             {result.note}")


def print_complete_result(result: MethodLoadResult) -> None:
    records = result.records
    avg_flops = sum(record.physical_flops for record in records) / len(records)
    avg_comm = sum(record.comm_total_bytes for record in records) / len(records)
    print(f"\nMethod: {result.method} ({result.mode})")
    print("Load")
    print(
        f"{'Rank':>4} {'Eff tok':>10} {'Phys tok':>10} {'Eff FLOPs':>11} "
        f"{'Phys FLOPs':>11} {'Comp/avg':>9} {'Sent':>10} {'Send/avg':>9}"
    )
    for record in records:
        print(
            f"{record.rank:>4} {_human(record.effective_tokens):>10} "
            f"{_human(record.physical_tokens):>10} "
            f"{_human(record.effective_flops):>11} "
            f"{_human(record.physical_flops):>11} "
            f"{_ratio(record.physical_flops, avg_flops):>9} "
            f"{_human(record.comm_tx_bytes, bytes_value=True):>10} "
            f"{_ratio(record.comm_total_bytes, avg_comm):>9}"
        )
    print("Efficiency")
    print(
        f"{'Rank':>4} {'KV tile reads':>14} {'Q/O visits [worst,best]':>25} "
        f"{'KV/QO [lower,upper]':>24}"
    )
    _print_efficiency_rows(result)
    print_summary(result)


def print_backward_summary(result: BackwardMethodLoadResult) -> None:
    print("Summary")
    _summary_row(
        "Physical tokens", [record.physical_tokens for record in result.records]
    )
    _summary_row(
        "Physical FLOPs", [record.physical_flops for record in result.records]
    )
    _summary_row(
        "Sent bytes",
        [record.comm_total_bytes for record in result.records],
        as_bytes=True,
    )
    effective_tokens = sum(record.effective_tokens for record in result.records)
    physical_tokens = sum(record.physical_tokens for record in result.records)
    effective_scores = sum(record.effective_scores for record in result.records)
    physical_scores = sum(record.physical_scores for record in result.records)
    token_overhead = (
        physical_tokens / effective_tokens - 1 if effective_tokens else 0.0
    )
    score_overhead = (
        physical_scores / effective_scores - 1 if effective_scores else 0.0
    )
    print(f"  Global Q/K-dKV  {global_backward_ratio(result):.3f}")
    print(
        f"  Overhead         tokens={100 * token_overhead:.3f}%, "
        f"scores={100 * score_overhead:.3f}%"
    )
    print(f"  Note             {result.note}")


def print_backward_result(result: BackwardMethodLoadResult) -> None:
    records = result.records
    avg_flops = sum(record.physical_flops for record in records) / len(records)
    avg_comm = sum(record.comm_total_bytes for record in records) / len(records)
    print(f"\nMethod: {result.method} ({result.mode}, backward)")
    print("Load")
    print(
        f"{'Rank':>4} {'Eff tok':>10} {'Phys tok':>10} {'Eff bwd FLOPs':>13} "
        f"{'Phys bwd FLOPs':>14} {'Comp/avg':>9} {'Sent':>10} {'Send/avg':>9}"
    )
    for record in records:
        print(
            f"{record.rank:>4} {_human(record.effective_tokens):>10} "
            f"{_human(record.physical_tokens):>10} "
            f"{_human(record.effective_flops):>13} "
            f"{_human(record.physical_flops):>14} "
            f"{_ratio(record.physical_flops, avg_flops):>9} "
            f"{_human(record.comm_tx_bytes, bytes_value=True):>10} "
            f"{_ratio(record.comm_total_bytes, avg_comm):>9}"
        )
    print("Efficiency")
    print(
        f"{'Rank':>4} {'Q tile reads':>14} {'K/dKV visits':>14} "
        f"{'Q tiles/K-dKV':>15}"
    )
    for record in records:
        print(
            f"{record.rank:>4} {record.q_tile_reads:>14} "
            f"{record.k_dkv_visits:>14} {record.q_tiles_per_k_dkv:>15.3f}"
        )
    print_backward_summary(result)


def _init_magi(
    methods: list[str],
    methods_spec: str,
    under_torchrun: bool,
    rank: int,
) -> tuple[object | None, bool, str | None]:
    if "magi_attention" not in methods:
        return None, False, None
    if not under_torchrun:
        reason = "MagiAttention metadata requires torchrun, CUDA, and its extensions"
        if _requests_all(methods_spec):
            return None, False, reason
        raise SystemExit(reason)

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        reason = "MagiAttention metadata requires CUDA"
        if _requests_all(methods_spec):
            return None, False, reason
        raise SystemExit(reason)
    torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability(rank) != (9, 0):
        reason = "MagiAttention metadata requires an SM90 Hopper GPU"
        if _requests_all(methods_spec):
            return None, False, reason
        raise SystemExit(reason)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", device_id=torch.device("cuda", rank)
        )
    from baseline.magi_attention import probe_magi_attention_all_ranks

    available, reason = probe_magi_attention_all_ranks(dist.group.WORLD)
    if not available and not _requests_all(methods_spec):
        raise SystemExit(f"magi_attention is unavailable: {reason}")
    return dist, available, reason


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    methods = parse_methods(args.methods)
    if args.direction == "backward" and args.mode != "causal":
        raise SystemExit("--direction backward requires --mode causal")
    if args.headdim != 128:
        raise SystemExit("this benchmark supports only head dim 128")
    if args.qhead % args.kvhead:
        raise SystemExit("qhead must be divisible by kvhead")
    world_size, under_torchrun, rank = _resolve_world_size(args)
    cases = _make_cases(args, world_size)
    dist_module, magi_available, magi_reason = _init_magi(
        methods, args.methods, under_torchrun, rank
    )
    modes = {
        "noncausal": (False,),
        "causal": (True,),
        "both": (False, True),
    }[args.mode]
    skip_incompatible = _requests_all(args.methods)
    cumulative: dict[
        tuple[str, str], list[MethodLoadResult | BackwardMethodLoadResult]
    ] = {}

    try:
        if rank == 0:
            source = f"dataset={args.dataset}" if args.dataset else "explicit topology"
            if args.direction == "forward":
                print(
                    "Forward load-balance benchmark "
                    "(metadata only, BF16, logical tiles=128x128)"
                )
                print(
                    "Aligned-method token and communication counters use original "
                    "sequence lengths; FLOPs retain execution padding"
                )
                print(
                    f"Config: source={source}, world_size={world_size}, "
                    f"QH={args.qhead}, KVH={args.kvhead}, D={args.headdim}, "
                    f"mode={args.mode}, methods={methods}"
                )
            else:
                print(
                    "Backward load-balance benchmark "
                    "(metadata only, BF16, logical tiles=128x128)"
                )
                print(
                    f"Config: source={source}, world_size={world_size}, "
                    f"QH={args.qhead}, KVH={args.kvhead}, D={args.headdim}, "
                    f"direction=backward, mode={args.mode}, methods={methods}"
                )

        for case_index, case in enumerate(cases):
            if rank == 0:
                print(f"\n{'=' * 96}")
                print(f"Case {case_index + 1}/{len(cases)}: {case.label}")
                print(
                    f"Workload: B={len(case.global_lengths)}, tokens={sum(case.global_lengths)}, "
                    f"global_seqlens={list(case.global_lengths)}"
                )
                print(
                    f"Topology: ring_sizes={list(case.ring_sizes)}, "
                    f"ring_starts={list(case.ring_starts)}"
                )

            for is_causal in modes:
                mode = "causal" if is_causal else "noncausal"
                if rank == 0:
                    print(f"\nMode: {mode}")
                for method in methods:
                    reason = method_incompatibility(
                        method,
                        case,
                        world_size,
                        is_causal,
                        args,
                        magi_available=magi_available,
                        magi_reason=magi_reason,
                    )
                    if reason is not None:
                        if skip_incompatible:
                            if rank == 0:
                                print(f"SKIP {method}: {reason}")
                            continue
                        raise SystemExit(f"method {method!r} is incompatible: {reason}")

                    if method == "magi_attention":
                        if dist_module is None:
                            raise AssertionError("MagiAttention has no distributed context")
                        from baseline.magi_attention import (
                            MagiAttentionConfig,
                            build_magi_attention_metadata,
                        )

                        metadata = build_magi_attention_metadata(
                            dist_module.group.WORLD,
                            case.global_lengths,
                            args.qhead,
                            args.kvhead,
                            args.headdim,
                            is_causal,
                            config=MagiAttentionConfig(
                                overlap_degree=args.magi_overlap_degree,
                                seed=args.seed,
                            ),
                        )
                        if args.direction == "forward":
                            local_record = analyze_magi_rank(
                                metadata,
                                case.global_lengths,
                                args.qhead,
                                args.kvhead,
                                args.headdim,
                                is_causal,
                            )
                        else:
                            local_record = analyze_backward_magi_rank(
                                metadata,
                                case.global_lengths,
                                args.qhead,
                                args.kvhead,
                                args.headdim,
                            )
                        gathered: list[object | None] = [None] * world_size
                        dist_module.all_gather_object(gathered, local_record)
                        if rank == 0:
                            result = (
                                magi_result_from_records(gathered)  # type: ignore[arg-type]
                                if args.direction == "forward"
                                else backward_magi_result_from_records(gathered)  # type: ignore[arg-type]
                            )
                        else:
                            result = None
                    elif rank == 0:
                        if args.direction == "forward":
                            result = analyze_method(
                                method,
                                case.global_lengths,
                                case.ring_sizes,
                                case.ring_starts,
                                world_size,
                                args.qhead,
                                args.kvhead,
                                args.headdim,
                                is_causal,
                                heads_k_stride=args.allgather_overlapping_heads_k_stride,
                                zepplin_threshold=args.zepplin_threshold,
                                megatron_max_seqlen_per_rank=args.megatron_max_seqlen_per_rank,
                            )
                        else:
                            result = analyze_backward_method(
                                method,
                                case.global_lengths,
                                case.ring_sizes,
                                case.ring_starts,
                                world_size,
                                args.qhead,
                                args.kvhead,
                                args.headdim,
                                heads_k_stride=args.allgather_overlapping_heads_k_stride,
                                zepplin_threshold=args.zepplin_threshold,
                                megatron_max_seqlen_per_rank=args.megatron_max_seqlen_per_rank,
                            )
                    else:
                        result = None

                    if rank == 0 and result is not None:
                        score_conserved = (
                            global_score_conserved(
                                result, case.global_lengths, is_causal
                            )
                            if args.direction == "forward"
                            else global_backward_score_conserved(
                                result, case.global_lengths
                            )
                        )
                        if not score_conserved:
                            raise RuntimeError(
                                f"{method} effective score count is not globally conserved"
                            )
                        if args.direction == "forward":
                            print_complete_result(result)
                        else:
                            print_backward_result(result)
                        cumulative.setdefault((method, mode), []).append(result)

        if rank == 0 and len(cases) > 1:
            print(f"\n{'=' * 96}")
            print("Cumulative dataset summary")
            for method in methods:
                for is_causal in modes:
                    mode = "causal" if is_causal else "noncausal"
                    results = cumulative.get((method, mode), [])
                    if not results:
                        continue
                    result = (
                        cumulative_result(results)  # type: ignore[arg-type]
                        if args.direction == "forward"
                        else cumulative_backward_result(results)  # type: ignore[arg-type]
                    )
                    if len(results) != len(cases):
                        result = replace(
                            result,
                            note=(
                                f"{result.note}; represented cases="
                                f"{len(results)}/{len(cases)}"
                            ),
                        )
                    if args.direction == "forward":
                        print_complete_result(result)
                    else:
                        print_backward_result(result)
    finally:
        if dist_module is not None and dist_module.is_initialized():
            dist_module.destroy_process_group()


if __name__ == "__main__":
    main()
