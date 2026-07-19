"""Distributed hierarchical hybrid forward benchmark with all-CP baselines."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

import torch
import torch.distributed as dist

THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import min_fa3_op
from allgather_attention import (
    Llama3AllGatherAttention,
    repartition_sequence_shards_to_llama3,
    select_fa3_backend,
)
from hybrid_forward_baselines import (
    VarlenAllGatherForward,
    ZepplinForward,
    fa3_ring_forward,
)
from ring_test.utils import (
    MEGA_RING_ALL_CP_ALIGNMENT,
    HybridBenchmarkCase,
    align_mega_ring_all_cp_lengths,
    hierarchical_reference,
    init_distributed,
    local_lengths_for_rank,
    make_cu_seqlens,
    make_local_qkv,
    parse_int_list,
)
from zepplin import (
    DEFAULT_ZEPPLIN_THRESHOLD,
    ZepplinPlan,
    make_zepplin_plan,
    zepplin_incompatibility,
)


METHOD_ORDER = [
    "allgather_attention",
    "llama3_allgather_attention",
    "fa3_ring",
    "zepplin",
    "mega_ring_all_cp",
    "mega_ring_hybrid",
]
SM_SWEEP_METHODS = {"mega_ring_all_cp", "mega_ring_hybrid"}

ALL_CP_METHODS = {
    "allgather_attention",
    "llama3_allgather_attention",
    "fa3_ring",
    "mega_ring_all_cp",
}
BLOCK_ALL_CP_METHODS = ALL_CP_METHODS - {"mega_ring_all_cp"}


@dataclass(frozen=True)
class SmConfig:
    num_comp_sm: int
    num_comm_sm: int


@dataclass(frozen=True)
class TimingResult:
    local_ms: float
    max_ms: float
    rank_times_ms: list[float] | None


@dataclass(frozen=True)
class MethodRun:
    name: str
    launch: Callable[[], object]
    expected_out: torch.Tensor | None
    expected_lse: torch.Tensor | None
    note: str
    aligned_global_lengths: tuple[int, ...] | None = None


@dataclass(frozen=True)
class Result:
    time_ms: float
    aggregate_tflops: float
    avg_gpu_tflops: float
    check: str
    note: str
    rank_times_ms: list[float] | None


@dataclass(frozen=True)
class ForwardSummarySample:
    case_index: int
    method: str
    is_causal: bool
    config: SmConfig
    time_ms: float
    aggregate_tflops: float


@dataclass
class MegaKvPool:
    remote_k: min_fa3_op.TKParallelTensor
    remote_v: min_fa3_op.TKParallelTensor
    rank_capacity: int
    rank: int

    def populate(self, local_k: torch.Tensor, local_v: torch.Tensor) -> None:
        if local_k.size(0) > self.rank_capacity:
            raise RuntimeError(
                f"local K/V rows {local_k.size(0)} exceed pooled rank capacity "
                f"{self.rank_capacity}"
            )
        self.remote_k.data_.zero_()
        self.remote_v.data_.zero_()
        owner_begin = self.rank * self.rank_capacity
        owner_end = owner_begin + local_k.size(0)
        self.remote_k.data_[owner_begin:owner_end].copy_(local_k)
        self.remote_v.data_[owner_begin:owner_end].copy_(local_v)


@dataclass
class ForwardParallelPools:
    all_cp: MegaKvPool | None
    hybrid: MegaKvPool | None

    def close(self) -> None:
        self.all_cp = None
        self.hybrid = None


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
            raise SystemExit(f"unknown method '{token}', expected one of {METHOD_ORDER} or all")
    deduped: list[str] = []
    for method in methods:
        if method not in deduped:
            deduped.append(method)
    if not deduped:
        raise SystemExit("--methods must provide at least one method")
    return deduped


def parse_sm_configs(spec: str) -> list[SmConfig]:
    configs: list[SmConfig] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        fields = token.split(":")
        if len(fields) != 2:
            raise SystemExit(f"invalid SM config '{token}', expected COMP:COMM")
        configs.append(SmConfig(int(fields[0]), int(fields[1])))
    if not configs:
        raise SystemExit("--sm-configs must provide at least one COMP:COMM pair")
    return configs


def cuda_barrier() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def validate_metadata(
    global_lengths: list[int],
    ring_sizes: list[int],
    ring_starts: list[int],
    world_size: int,
    mode: str,
) -> None:
    if not (len(global_lengths) == len(ring_sizes) == len(ring_starts)):
        raise SystemExit("global lengths, ring sizes, and ring starts must have the same length")
    previous_size = 8
    for idx, (global_len, ring_size, ring_start) in enumerate(
        zip(global_lengths, ring_sizes, ring_starts)
    ):
        if ring_size not in (1, 2, 4, 8) or ring_size > previous_size:
            raise SystemExit(f"invalid ring size/order at batch {idx}")
        if ring_start < 0 or ring_start % ring_size or ring_start + ring_size > world_size:
            raise SystemExit(f"invalid ring start at batch {idx}")
        if global_len <= 0 or global_len % ring_size:
            raise SystemExit(f"invalid global length at batch {idx}")
        local_len = global_len // ring_size
        if mode in ("causal", "both") and ring_size > 1 and (
            local_len % 2 or (local_len // 2) % 128
        ):
            raise SystemExit(f"causal local half length is not 128-aligned at batch {idx}")
        previous_size = ring_size

def method_incompatibility(
    method: str,
    global_lengths: list[int],
    world_size: int,
    is_causal: bool,
    zepplin_threshold: int = DEFAULT_ZEPPLIN_THRESHOLD,
) -> str | None:
    if method == "zepplin":
        return zepplin_incompatibility(
            global_lengths, world_size, is_causal, zepplin_threshold
        )
    if method == "mega_ring_all_cp":
        # This baseline benchmarks separately padded lengths, which are valid
        # for causal all-CP mega-ring at every supported physical world size.
        return None
    if method not in ALL_CP_METHODS:
        return None
    for idx, global_len in enumerate(global_lengths):
        if global_len % world_size:
            return (
                "all-CP methods require every global length to be divisible by world_size: "
                f"batch={idx}, global_len={global_len}, world_size={world_size}"
            )
        local_len = global_len // world_size
        if is_causal and local_len % 2:
            return (
                "causal all-CP methods require even local lengths: "
                f"batch={idx}, local_len={local_len}"
            )
    if method == "llama3_allgather_attention" and sum(global_lengths) % (2 * world_size):
        return (
            "llama3_allgather_attention requires total global tokens divisible by "
            f"2 * world_size, got total={sum(global_lengths)}, world_size={world_size}"
        )
    return None


def compatible_methods_for_mode(
    methods: list[str],
    global_lengths: list[int],
    world_size: int,
    is_causal: bool,
    *,
    skip_incompatible: bool,
    zepplin_threshold: int = DEFAULT_ZEPPLIN_THRESHOLD,
) -> tuple[list[str], list[tuple[str, str]]]:
    compatible: list[str] = []
    skipped: list[tuple[str, str]] = []
    for method in methods:
        reason = method_incompatibility(
            method,
            global_lengths,
            world_size,
            is_causal,
            zepplin_threshold,
        )
        if reason is None:
            compatible.append(method)
        elif skip_incompatible:
            skipped.append((method, reason))
        else:
            raise SystemExit(f"method '{method}' is incompatible: {reason}")
    if not compatible:
        mode = "causal" if is_causal else "noncausal"
        raise SystemExit(f"no compatible methods remain for {mode} mode")
    return compatible, skipped


def make_mega_parallel_tensors(
    rank: int,
    world_size: int,
    rank_capacity: int,
    kv_heads: int,
    head_dim: int,
) -> MegaKvPool:
    arena_shape = [world_size * rank_capacity, kv_heads, head_dim]
    remote_k = min_fa3_op.TKParallelTensor(arena_shape, torch.bfloat16, rank, world_size, False)
    remote_v = min_fa3_op.TKParallelTensor(arena_shape, torch.bfloat16, rank, world_size, False)
    return MegaKvPool(remote_k, remote_v, rank_capacity, rank)


def max_hybrid_rank_capacity(
    workload_cases: Sequence[HybridBenchmarkCase], world_size: int
) -> int:
    capacity = max(
        sum(
            local_lengths_for_rank(
                list(case.global_lengths),
                list(case.ring_sizes),
                list(case.ring_starts),
                rank,
            )
        )
        for case in workload_cases
        for rank in range(world_size)
    )
    return ((capacity + 127) // 128) * 128


def max_all_cp_rank_capacity(
    workload_cases: Sequence[HybridBenchmarkCase], world_size: int
) -> int:
    return max(
        sum(align_mega_ring_all_cp_lengths(list(case.global_lengths))) // world_size
        for case in workload_cases
    )


def gather_padded_rank_tensor(tensor: torch.Tensor, rank_capacity: int) -> torch.Tensor:
    padded = torch.zeros(
        (rank_capacity, tensor.size(1), tensor.size(2)), device=tensor.device, dtype=tensor.dtype
    )
    padded[:tensor.size(0)].copy_(tensor)
    parts = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, padded)
    return torch.stack(parts)


def measure_distributed_ms(
    fn: Callable[[], object], warmup_iters: int, num_iters: int, rank: int
) -> TimingResult:
    for _ in range(warmup_iters):
        fn()
    cuda_barrier()

    local_samples: list[float] = []
    max_samples: list[float] = []
    for _ in range(num_iters):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        elapsed_ms = begin.elapsed_time(end)
        elapsed = torch.tensor([elapsed_ms], device="cuda", dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        local_samples.append(elapsed_ms)
        max_samples.append(elapsed.item())
    cuda_barrier()

    local_avg = sum(local_samples) / len(local_samples)
    max_avg = sum(max_samples) / len(max_samples)
    local_tensor = torch.tensor([local_avg], device="cuda", dtype=torch.float64)
    gathered = [torch.empty_like(local_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_tensor)
    rank_times = [value.item() for value in gathered] if rank == 0 else None
    return TimingResult(local_avg, max_avg, rank_times)


def aggregate_score_count(global_lengths: Sequence[int], is_causal: bool) -> int:
    if is_causal:
        return sum(length * (length + 1) // 2 for length in global_lengths)
    return sum(length * length for length in global_lengths)


def aggregate_tflops(
    global_lengths: Sequence[int],
    q_heads: int,
    head_dim: int,
    is_causal: bool,
    time_ms: float,
) -> float:
    flops = 4 * aggregate_score_count(global_lengths, is_causal) * q_heads * head_dim
    return float(flops) / (time_ms * 1e-3) / 1e12


def print_results(
    global_lengths: list[int],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
    methods: list[str],
    results: dict[str, Result],
) -> None:
    """Print one hybrid benchmark result table on rank 0."""
    mode = "causal" if is_causal else "noncausal"
    print(
        f"\nB={len(global_lengths)}, global_tokens={sum(global_lengths)}, "
        f"QH={q_heads}, KVH={kv_heads}, D={head_dim}, mode={mode}"
    )
    rows: list[tuple[str, str, str, str, str, str]] = []
    for method in methods:
        result = results.get(method)
        if result is None:
            continue
        if result.rank_times_ms is None:
            time_s = f"max_across_ranks={result.time_ms:.3f}"
        else:
            rank_times_s = ", ".join(
                f"t{rank}={time_ms:.3f}"
                for rank, time_ms in enumerate(result.rank_times_ms)
            )
            time_s = (
                f"{rank_times_s} | max_across_ranks={result.time_ms:.3f}"
            )
        rows.append(
            (
                method,
                time_s,
                f"{result.aggregate_tflops:.1f}",
                f"{result.avg_gpu_tflops:.1f}",
                result.check,
                result.note,
            )
        )

    method_width = max((24, *(len(row[0]) for row in rows)))
    time_width = max((64, *(len(row[1]) for row in rows)))
    print(
        f"{'Method':<{method_width}} {'Time ms':<{time_width}} "
        f"{'Agg TFLOPS':>12} {'Avg/GPU':>10} {'Check':>10}  Note"
    )
    for method, time_s, aggregate_s, per_gpu_s, check, note in rows:
        print(
            f"{method:<{method_width}} {time_s:<{time_width}} "
            f"{aggregate_s:>12} {per_gpu_s:>10} {check:>10}  {note}"
        )


def raise_if_any_rank_failed(local_error: str | None) -> None:
    failed = torch.tensor([local_error is not None], device="cuda", dtype=torch.int32)
    dist.all_reduce(failed)
    if failed.item() == 0:
        return
    if local_error is not None:
        raise AssertionError(local_error)
    raise AssertionError("another rank failed hierarchical output validation")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark hierarchical hybrid forward with all-CP baselines")
    parser.add_argument("--global-seqlens", required=True)
    parser.add_argument("--ring-sizes", required=True)
    parser.add_argument("--ring-starts", required=True)
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument(
        "--allgather-overlapping-heads-k-stride",
        type=int,
        default=4,
        help="KV heads per all-gather/attention overlap pipeline chunk",
    )
    parser.add_argument("--mode", choices=("noncausal", "causal", "both"), default="causal")
    parser.add_argument(
        "--methods",
        default="all",
        help=f"Comma-separated methods from {METHOD_ORDER}, or all",
    )
    parser.add_argument("--sm-configs", default="128:4,124:8,120:12,116:16")
    parser.add_argument(
        "--zepplin-threshold",
        type=positive_int,
        default=DEFAULT_ZEPPLIN_THRESHOLD,
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=40)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("--rtol", type=float, default=2e-1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _main_single(
    argv: Sequence[str] | None = None,
    *,
    skip_incompatible_methods: bool = False,
    parallel_pools: ForwardParallelPools | None = None,
    case_label: str | None = None,
    case_index: int = 0,
    manage_process_group: bool = True,
) -> list[ForwardSummarySample]:
    args = parse_args(argv)
    methods = parse_methods(args.methods)
    summary_samples: list[ForwardSummarySample] = []
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("SM90 Hopper CUDA device is required")
    if args.headdim != 128 or args.kvhead * args.headdim != 1024:
        raise SystemExit("hierarchical communication requires D=128 and KVH * D == 1024")
    if args.qhead % args.kvhead:
        raise SystemExit("qhead must be divisible by kvhead")
    if (
        args.allgather_overlapping_heads_k_stride <= 0
        or args.kvhead % args.allgather_overlapping_heads_k_stride
    ):
        raise SystemExit(
            "--allgather-overlapping-heads-k-stride must be a positive divisor "
            "of --kvhead, "
            f"got stride={args.allgather_overlapping_heads_k_stride}, "
            f"kvhead={args.kvhead}"
        )
    if args.warmup_iters < 0 or args.num_iters <= 0:
        raise SystemExit("warmup iterations must be non-negative and measured iterations must be positive")

    rank, world_size = init_distributed()
    try:
        global_lengths = parse_int_list(args.global_seqlens, "--global-seqlens")
        mega_ring_all_cp_global_lengths = align_mega_ring_all_cp_lengths(global_lengths)
        ring_sizes = parse_int_list(args.ring_sizes, "--ring-sizes")
        ring_starts = parse_int_list(args.ring_starts, "--ring-starts")
        if any(method != "zepplin" for method in methods):
            validate_metadata(
                global_lengths, ring_sizes, ring_starts, world_size, args.mode
            )
        elif not (len(global_lengths) == len(ring_sizes) == len(ring_starts)):
            raise SystemExit(
                "global lengths, ring sizes, and ring starts must have the same length"
            )
        sm_configs = parse_sm_configs(args.sm_configs)
        sm_count = torch.cuda.get_device_properties(rank).multi_processor_count
        for config in sm_configs:
            if config.num_comp_sm <= 0 or config.num_comm_sm <= 0:
                raise SystemExit("hierarchical benchmark requires positive compute and communication SM counts")
            if config.num_comp_sm + config.num_comm_sm > sm_count:
                raise SystemExit(
                    f"SM config {config.num_comp_sm}:{config.num_comm_sm} exceeds device SM count {sm_count}"
                )

        device = torch.device("cuda", rank)
        modes = {
            "noncausal": [False],
            "causal": [True],
            "both": [False, True],
        }[args.mode]
        methods_by_mode: dict[bool, list[str]] = {}
        skipped_by_mode: dict[bool, list[tuple[str, str]]] = {}
        for is_causal in modes:
            active_methods, skipped_methods = compatible_methods_for_mode(
                methods,
                global_lengths,
                world_size,
                is_causal,
                skip_incompatible=skip_incompatible_methods,
                zepplin_threshold=args.zepplin_threshold,
            )
            methods_by_mode[is_causal] = active_methods
            skipped_by_mode[is_causal] = skipped_methods

        zepplin_plans: dict[bool, ZepplinPlan] = {}
        for is_causal in modes:
            if "zepplin" in methods_by_mode[is_causal]:
                zepplin_plans[is_causal] = make_zepplin_plan(
                    global_lengths,
                    world_size,
                    is_causal,
                    args.zepplin_threshold,
                )

        block_backend = (
            select_fa3_backend(dist.group.WORLD, require_backward=False)
            if any(
                any(
                    method in active_methods
                    for method in (
                        "allgather_attention",
                        "llama3_allgather_attention",
                        "fa3_ring",
                        "zepplin",
                    )
                )
                for active_methods in methods_by_mode.values()
            )
            else None
        )
        if block_backend == "external_fa3":
            backend_note = "external FA3"
        elif block_backend == "min_fa3":
            backend_note = "in-repo min_fa3 fallback"
        else:
            backend_note = "not used"

        if rank == 0:
            if case_label is not None:
                print(f"\nBenchmark case: {case_label}", flush=True)
            sm_configs_s = ",".join(
                f"{config.num_comp_sm}:{config.num_comm_sm}"
                for config in sm_configs
            )
            print(
                f"Config: world_size={world_size}, methods={methods}, "
                f"QH={args.qhead}, KVH={args.kvhead}, D={args.headdim}, "
                "allgather_overlapping_heads_k_stride="
                f"{args.allgather_overlapping_heads_k_stride}, "
                f"mode={args.mode}, sm_configs={sm_configs_s}, "
                f"zepplin_threshold={args.zepplin_threshold}, "
                f"warmup={args.warmup_iters}, iters={args.num_iters}, "
                f"check={args.check}"
            )
            print(
                f"Workload: B={len(global_lengths)}, "
                f"global_tokens={sum(global_lengths)}, "
                f"global_seqlens={global_lengths}"
            )
            if any(
                "mega_ring_all_cp" in active_methods
                for active_methods in methods_by_mode.values()
            ):
                print(
                    "Mega-ring all-CP workload: "
                    f"alignment={MEGA_RING_ALL_CP_ALIGNMENT}, "
                    f"global_tokens={sum(mega_ring_all_cp_global_lengths)}, "
                    f"global_seqlens={mega_ring_all_cp_global_lengths}"
                )
            print(
                f"Hybrid rings: sizes={ring_sizes}, starts={ring_starts}"
            )
            print(
                f"FA backend: {backend_note}",
                flush=True,
            )
            for is_causal, plan in zepplin_plans.items():
                mode = "causal" if is_causal else "noncausal"
                print(
                    f"Zepplin placement ({mode}): threshold={plan.threshold}, "
                    f"G1={len(plan.short_indices)}, "
                    f"Gworld={len(plan.long_indices)}, "
                    f"G1_rank_loads={list(plan.short_loads)}"
                )
            print(
                "Agg TFLOPS uses the original workload lengths and sums visible "
                "attention work across ranks; Avg/GPU divides it by world_size."
            )
            if args.check:
                print(
                    "Checks compare each method with the matching full-rank "
                    "reference output."
                )
            for is_causal in modes:
                mode = "causal" if is_causal else "noncausal"
                skipped_methods = skipped_by_mode[is_causal]
                if skipped_methods:
                    print(f"Skipped methods ({mode}):")
                    for method, reason in skipped_methods:
                        print(f"  {method}: {reason}")

        for is_causal in modes:
            active_methods = methods_by_mode[is_causal]
            if rank == 0:
                print(
                    f"\nRunning B={len(global_lengths)}, "
                    f"global_tokens={sum(global_lengths)}, "
                    f"causal={is_causal}",
                    flush=True,
                )
            all_cp_runs: dict[str, tuple[Callable[[], object], str]] = {}
            expected_all_cp_out = None
            expected_llama3_out = None
            if any(method in BLOCK_ALL_CP_METHODS for method in active_methods):
                all_cp_lengths = [length // world_size for length in global_lengths]
                all_cp_total = sum(all_cp_lengths)
                all_cp_cu, all_cp_cu_host = make_cu_seqlens(all_cp_lengths, device)
                all_cp_q, all_cp_k, all_cp_v = make_local_qkv(
                    all_cp_total,
                    args.qhead,
                    args.kvhead,
                    args.headdim,
                    rank,
                    is_causal,
                    device,
                    base_seed=args.seed,
                )
                if "allgather_attention" in active_methods:
                    if block_backend is None:
                        raise RuntimeError("all-gather baseline requires a block backend")
                    allgather_runner = VarlenAllGatherForward(
                        dist.group.WORLD,
                        all_cp_q,
                        all_cp_k,
                        all_cp_v,
                        all_cp_lengths,
                        is_causal,
                        block_backend,
                        heads_k_stride=args.allgather_overlapping_heads_k_stride,
                    )
                    all_cp_runs["allgather_attention"] = (
                        allgather_runner.forward,
                        allgather_runner.note,
                    )
                if "llama3_allgather_attention" in active_methods:
                    if block_backend is None:
                        raise RuntimeError("Llama3 baseline requires a block backend")
                    llama3_q = repartition_sequence_shards_to_llama3(
                        dist.group.WORLD, all_cp_q, global_lengths, is_causal
                    )
                    llama3_k = repartition_sequence_shards_to_llama3(
                        dist.group.WORLD, all_cp_k, global_lengths, is_causal
                    )
                    llama3_v = repartition_sequence_shards_to_llama3(
                        dist.group.WORLD, all_cp_v, global_lengths, is_causal
                    )
                    llama3_runner = Llama3AllGatherAttention(
                        dist.group.WORLD,
                        llama3_q,
                        llama3_k,
                        llama3_v,
                        global_lengths,
                        is_causal,
                        block_backend,
                        heads_k_stride=args.allgather_overlapping_heads_k_stride,
                    )
                    all_cp_runs["llama3_allgather_attention"] = (
                        llama3_runner.forward,
                        llama3_runner.note,
                    )
                if "fa3_ring" in active_methods:
                    if block_backend is None:
                        raise RuntimeError("FA3 ring baseline requires a block backend")
                    all_cp_runs["fa3_ring"] = (
                        lambda: fa3_ring_forward(
                            dist.group.WORLD,
                            all_cp_q,
                            all_cp_k,
                            all_cp_v,
                            all_cp_cu,
                            all_cp_cu_host,
                            all_cp_lengths,
                            is_causal,
                            block_backend,
                        ),
                        f"all-CP NCCL ring; {backend_note}",
                    )
                if args.check:
                    gathered_all_cp_k = gather_padded_rank_tensor(all_cp_k, all_cp_total)
                    gathered_all_cp_v = gather_padded_rank_tensor(all_cp_v, all_cp_total)
                    expected_all_cp_out, _ = hierarchical_reference(
                        all_cp_q,
                        gathered_all_cp_k,
                        gathered_all_cp_v,
                        [all_cp_lengths for _ in range(world_size)],
                        all_cp_cu_host,
                        global_lengths,
                        [world_size] * len(global_lengths),
                        [0] * len(global_lengths),
                        rank,
                        is_causal,
                    )
                    if "llama3_allgather_attention" in active_methods:
                        expected_llama3_out = repartition_sequence_shards_to_llama3(
                            dist.group.WORLD,
                            expected_all_cp_out,
                            global_lengths,
                            is_causal,
                        )

            mega_ring_all_cp_run_data = None
            if "mega_ring_all_cp" in active_methods:
                mega_ring_all_cp_lengths = [
                    length // world_size
                    for length in mega_ring_all_cp_global_lengths
                ]
                mega_ring_all_cp_total = sum(mega_ring_all_cp_lengths)
                mega_ring_all_cp_cu, mega_ring_all_cp_cu_host = make_cu_seqlens(
                    mega_ring_all_cp_lengths, device
                )
                mega_ring_all_cp_q, mega_ring_all_cp_k, mega_ring_all_cp_v = make_local_qkv(
                    mega_ring_all_cp_total,
                    args.qhead,
                    args.kvhead,
                    args.headdim,
                    rank,
                    is_causal,
                    device,
                    base_seed=args.seed,
                )
                mega_ring_all_cp_pool = (
                    None if parallel_pools is None else parallel_pools.all_cp
                )
                if mega_ring_all_cp_pool is None:
                    mega_ring_all_cp_pool = make_mega_parallel_tensors(
                        rank,
                        world_size,
                        mega_ring_all_cp_total,
                        args.kvhead,
                        args.headdim,
                    )
                mega_ring_all_cp_pool.populate(
                    mega_ring_all_cp_k, mega_ring_all_cp_v
                )
                mega_ring_all_cp_remote_k = mega_ring_all_cp_pool.remote_k
                mega_ring_all_cp_remote_v = mega_ring_all_cp_pool.remote_v
                mega_ring_all_cp_global_host = torch.tensor(
                    mega_ring_all_cp_global_lengths, dtype=torch.int32
                )
                mega_ring_all_cp_ring_sizes_host = torch.full(
                    (len(global_lengths),), world_size, dtype=torch.int32
                )
                mega_ring_all_cp_ring_starts_host = torch.zeros(
                    len(global_lengths), dtype=torch.int32
                )
                expected_mega_ring_all_cp_out = None
                expected_mega_ring_all_cp_lse = None
                if args.check:
                    gathered_mega_ring_all_cp_k = gather_padded_rank_tensor(
                        mega_ring_all_cp_k, mega_ring_all_cp_total
                    )
                    gathered_mega_ring_all_cp_v = gather_padded_rank_tensor(
                        mega_ring_all_cp_v, mega_ring_all_cp_total
                    )
                    (
                        expected_mega_ring_all_cp_out,
                        expected_mega_ring_all_cp_lse,
                    ) = hierarchical_reference(
                        mega_ring_all_cp_q,
                        gathered_mega_ring_all_cp_k,
                        gathered_mega_ring_all_cp_v,
                        [mega_ring_all_cp_lengths for _ in range(world_size)],
                        mega_ring_all_cp_cu_host,
                        mega_ring_all_cp_global_lengths,
                        [world_size] * len(global_lengths),
                        [0] * len(global_lengths),
                        rank,
                        is_causal,
                    )
                mega_ring_all_cp_run_data = (
                    mega_ring_all_cp_q,
                    mega_ring_all_cp_cu,
                    mega_ring_all_cp_cu_host,
                    mega_ring_all_cp_remote_k,
                    mega_ring_all_cp_remote_v,
                    mega_ring_all_cp_global_host,
                    mega_ring_all_cp_ring_sizes_host,
                    mega_ring_all_cp_ring_starts_host,
                    max(mega_ring_all_cp_lengths),
                    expected_mega_ring_all_cp_out,
                    expected_mega_ring_all_cp_lse,
                )

            zepplin_run = None
            if "zepplin" in active_methods:
                if block_backend is None:
                    raise RuntimeError("zepplin baseline requires a block backend")
                zepplin_plan = zepplin_plans[is_causal]
                zepplin_local_lengths = zepplin_plan.packed_lengths_for_rank(rank)
                zepplin_local_total = sum(zepplin_local_lengths)
                zepplin_q, zepplin_k, zepplin_v = make_local_qkv(
                    zepplin_local_total,
                    args.qhead,
                    args.kvhead,
                    args.headdim,
                    rank,
                    is_causal,
                    device,
                    base_seed=args.seed + 37,
                )
                zepplin_runner = ZepplinForward(
                    dist.group.WORLD,
                    zepplin_q,
                    zepplin_k,
                    zepplin_v,
                    zepplin_plan,
                    block_backend,
                )
                expected_zepplin_out = None
                if args.check:
                    zepplin_rank_lengths = [
                        zepplin_plan.topology_lengths_for_rank(source_rank)
                        for source_rank in range(world_size)
                    ]
                    zepplin_rank_capacity = max(
                        sum(lengths) for lengths in zepplin_rank_lengths
                    )
                    gathered_zepplin_k = gather_padded_rank_tensor(
                        zepplin_k, zepplin_rank_capacity
                    )
                    gathered_zepplin_v = gather_padded_rank_tensor(
                        zepplin_v, zepplin_rank_capacity
                    )
                    _, zepplin_cu_host = make_cu_seqlens(
                        zepplin_plan.topology_lengths_for_rank(rank), device
                    )
                    expected_zepplin_out, _ = hierarchical_reference(
                        zepplin_q,
                        gathered_zepplin_k,
                        gathered_zepplin_v,
                        zepplin_rank_lengths,
                        zepplin_cu_host,
                        zepplin_plan.packed_global_lengths,
                        zepplin_plan.ring_sizes,
                        zepplin_plan.ring_starts,
                        rank,
                        is_causal,
                    )
                zepplin_run = MethodRun(
                    "zepplin",
                    zepplin_runner.forward,
                    expected_zepplin_out,
                    None,
                    zepplin_runner.note,
                )

            hybrid_run_data = None
            if "mega_ring_hybrid" in active_methods:
                hybrid_rank_lengths = [
                    local_lengths_for_rank(global_lengths, ring_sizes, ring_starts, source_rank)
                    for source_rank in range(world_size)
                ]
                hybrid_local_lengths = hybrid_rank_lengths[rank]
                hybrid_local_total = sum(hybrid_local_lengths)
                hybrid_cu, hybrid_cu_host = make_cu_seqlens(hybrid_local_lengths, device)
                hybrid_q, hybrid_local_k, hybrid_local_v = make_local_qkv(
                    hybrid_local_total,
                    args.qhead,
                    args.kvhead,
                    args.headdim,
                    rank,
                    is_causal,
                    device,
                    base_seed=args.seed + 17,
                )
                capacity = torch.tensor([hybrid_local_total], device=device, dtype=torch.int32)
                dist.all_reduce(capacity, op=dist.ReduceOp.MAX)
                hybrid_rank_capacity = int(capacity.item())
                hybrid_pool = (
                    None if parallel_pools is None else parallel_pools.hybrid
                )
                if hybrid_pool is None:
                    hybrid_pool = make_mega_parallel_tensors(
                        rank,
                        world_size,
                        hybrid_rank_capacity,
                        args.kvhead,
                        args.headdim,
                    )
                hybrid_pool.populate(hybrid_local_k, hybrid_local_v)
                hybrid_remote_k = hybrid_pool.remote_k
                hybrid_remote_v = hybrid_pool.remote_v
                hybrid_global_host = torch.tensor(global_lengths, dtype=torch.int32)
                hybrid_ring_sizes_host = torch.tensor(ring_sizes, dtype=torch.int32)
                hybrid_ring_starts_host = torch.tensor(ring_starts, dtype=torch.int32)
                hybrid_max_local_len = max(max(lengths) for lengths in hybrid_rank_lengths)
                expected_hybrid_out = None
                expected_hybrid_lse = None
                if args.check:
                    gathered_hybrid_k = gather_padded_rank_tensor(
                        hybrid_local_k, hybrid_rank_capacity
                    )
                    gathered_hybrid_v = gather_padded_rank_tensor(
                        hybrid_local_v, hybrid_rank_capacity
                    )
                    expected_hybrid_out, expected_hybrid_lse = hierarchical_reference(
                        hybrid_q,
                        gathered_hybrid_k,
                        gathered_hybrid_v,
                        hybrid_rank_lengths,
                        hybrid_cu_host,
                        global_lengths,
                        ring_sizes,
                        ring_starts,
                        rank,
                        is_causal,
                    )
                hybrid_run_data = (
                    hybrid_q,
                    hybrid_cu,
                    hybrid_cu_host,
                    hybrid_remote_k,
                    hybrid_remote_v,
                    hybrid_global_host,
                    hybrid_ring_sizes_host,
                    hybrid_ring_starts_host,
                    hybrid_max_local_len,
                    expected_hybrid_out,
                    expected_hybrid_lse,
                )

            cuda_barrier()
            for config_index, config in enumerate(sm_configs):
                config_methods = [
                    method
                    for method in active_methods
                    if config_index == 0 or method in SM_SWEEP_METHODS
                ]
                if not config_methods:
                    continue
                if rank == 0:
                    print(
                        f"\nSM config: num_comp_sm={config.num_comp_sm}, "
                        f"num_comm_sm={config.num_comm_sm}",
                        flush=True,
                    )
                runs: list[MethodRun] = []
                for method in config_methods:
                    if method == "zepplin":
                        if zepplin_run is None:
                            raise RuntimeError("zepplin run was not prepared")
                        runs.append(zepplin_run)
                    elif method in all_cp_runs:
                        launch, note = all_cp_runs[method]
                        expected_out = (
                            expected_llama3_out
                            if method == "llama3_allgather_attention"
                            else expected_all_cp_out
                        )
                        runs.append(
                            MethodRun(
                                method,
                                launch,
                                expected_out,
                                None,
                                note,
                            )
                        )
                    elif method == "mega_ring_all_cp":
                        if mega_ring_all_cp_run_data is None:
                            raise RuntimeError("mega-ring all-CP run was not prepared")
                        (
                            mega_ring_all_cp_q,
                            mega_ring_all_cp_cu,
                            mega_ring_all_cp_cu_host,
                            mega_ring_all_cp_remote_k,
                            mega_ring_all_cp_remote_v,
                            mega_ring_all_cp_global_host,
                            mega_ring_all_cp_ring_sizes_host,
                            mega_ring_all_cp_ring_starts_host,
                            mega_ring_all_cp_max_local_len,
                            expected_mega_ring_all_cp_out,
                            expected_mega_ring_all_cp_lse,
                        ) = mega_ring_all_cp_run_data

                        def launch_all_cp_mega() -> tuple[torch.Tensor, torch.Tensor]:
                            return min_fa3_op.forward_varlen_mega_ring(
                                mega_ring_all_cp_q,
                                mega_ring_all_cp_remote_k.data_,
                                mega_ring_all_cp_remote_v.data_,
                                mega_ring_all_cp_cu,
                                mega_ring_all_cp_cu,
                                mega_ring_all_cp_max_local_len,
                                mega_ring_all_cp_max_local_len,
                                is_causal,
                                cu_seqlens_q_host=mega_ring_all_cp_cu_host,
                                cu_seqlens_k_host=mega_ring_all_cp_cu_host,
                                remote_k=mega_ring_all_cp_remote_k,
                                remote_v=mega_ring_all_cp_remote_v,
                                num_comp_sm=config.num_comp_sm,
                                num_comm_sm=config.num_comm_sm,
                                global_seqlens_host=mega_ring_all_cp_global_host,
                                ring_sizes_host=mega_ring_all_cp_ring_sizes_host,
                                ring_starts_host=mega_ring_all_cp_ring_starts_host,
                                return_lse=True,
                            )

                        runs.append(
                            MethodRun(
                                method,
                                launch_all_cp_mega,
                                expected_mega_ring_all_cp_out,
                                expected_mega_ring_all_cp_lse,
                                "all-CP fused mega-ring",
                                tuple(mega_ring_all_cp_global_lengths),
                            )
                        )
                    elif method == "mega_ring_hybrid":
                        (
                            hybrid_q,
                            hybrid_cu,
                            hybrid_cu_host,
                            hybrid_remote_k,
                            hybrid_remote_v,
                            hybrid_global_host,
                            hybrid_ring_sizes_host,
                            hybrid_ring_starts_host,
                            hybrid_max_local_len,
                            expected_hybrid_out,
                            expected_hybrid_lse,
                        ) = hybrid_run_data

                        def launch_hybrid_mega() -> tuple[torch.Tensor, torch.Tensor]:
                            return min_fa3_op.forward_varlen_mega_ring(
                                hybrid_q,
                                hybrid_remote_k.data_,
                                hybrid_remote_v.data_,
                                hybrid_cu,
                                hybrid_cu,
                                hybrid_max_local_len,
                                hybrid_max_local_len,
                                is_causal,
                                cu_seqlens_q_host=hybrid_cu_host,
                                cu_seqlens_k_host=hybrid_cu_host,
                                remote_k=hybrid_remote_k,
                                remote_v=hybrid_remote_v,
                                num_comp_sm=config.num_comp_sm,
                                num_comm_sm=config.num_comm_sm,
                                global_seqlens_host=hybrid_global_host,
                                ring_sizes_host=hybrid_ring_sizes_host,
                                ring_starts_host=hybrid_ring_starts_host,
                                return_lse=True,
                            )

                        runs.append(
                            MethodRun(
                                method,
                                launch_hybrid_mega,
                                expected_hybrid_out,
                                expected_hybrid_lse,
                                "hierarchical hybrid fused mega-ring",
                            )
                        )
                    else:
                        raise RuntimeError(f"unhandled method {method}")

                results: dict[str, Result] = {}
                for run in runs:
                    timing = measure_distributed_ms(
                        run.launch, args.warmup_iters, args.num_iters, rank
                    )
                    agg_tflops = aggregate_tflops(
                        global_lengths,
                        args.qhead,
                        args.headdim,
                        is_causal,
                        timing.max_ms,
                    )
                    check_status = "skip"
                    if args.check:
                        result = run.launch()
                        torch.cuda.synchronize()
                        out = result[0] if isinstance(result, tuple) else result
                        lse = result[1] if isinstance(result, tuple) and len(result) > 1 else None
                        local_error = None
                        try:
                            torch.testing.assert_close(
                                out.float(),
                                run.expected_out.float(),
                                atol=args.atol,
                                rtol=args.rtol,
                            )
                            if run.expected_lse is not None and lse is not None:
                                torch.testing.assert_close(
                                    lse, run.expected_lse, atol=args.atol, rtol=args.rtol
                                )
                        except AssertionError as exc:
                            local_error = (
                                f"{run.name}, SM {config.num_comp_sm}:{config.num_comm_sm}: {exc}"
                            )
                        raise_if_any_rank_failed(local_error)
                        check_status = "ok"
                    cuda_barrier()

                    if rank == 0:
                        note = run.note
                        if run.aligned_global_lengths is not None:
                            aligned_agg_tflops = aggregate_tflops(
                                run.aligned_global_lengths,
                                args.qhead,
                                args.headdim,
                                is_causal,
                                timing.max_ms,
                            )
                            note = (
                                f"{note}; {MEGA_RING_ALL_CP_ALIGNMENT}-aligned "
                                f"Agg TFLOPS={aligned_agg_tflops:.1f}, "
                                f"Avg/GPU={aligned_agg_tflops / world_size:.1f}"
                            )
                        results[run.name] = Result(
                            timing.max_ms,
                            agg_tflops,
                            agg_tflops / world_size,
                            check_status,
                            note,
                            timing.rank_times_ms,
                        )
                        summary_samples.append(
                            ForwardSummarySample(
                                case_index=case_index,
                                method=run.name,
                                is_causal=is_causal,
                                config=config,
                                time_ms=timing.max_ms,
                                aggregate_tflops=agg_tflops,
                            )
                        )
                if rank == 0:
                    print_results(
                        global_lengths,
                        args.qhead,
                        args.kvhead,
                        args.headdim,
                        is_causal,
                        config_methods,
                        results,
                    )
    finally:
        if manage_process_group and dist.is_initialized():
            dist.destroy_process_group()
    return summary_samples


def _replace_option(argv: Sequence[str], name: str, value: str) -> list[str]:
    result = list(argv)
    try:
        index = result.index(name)
    except ValueError as exc:
        raise RuntimeError(f"missing required forwarded option {name}") from exc
    if index + 1 >= len(result):
        raise RuntimeError(f"missing value for forwarded option {name}")
    result[index + 1] = value
    return result


def _argv_for_case(
    argv: Sequence[str], workload_case: HybridBenchmarkCase
) -> list[str]:
    result = list(argv)
    for name, values in (
        ("--global-seqlens", workload_case.global_lengths),
        ("--ring-sizes", workload_case.ring_sizes),
        ("--ring-starts", workload_case.ring_starts),
    ):
        result = _replace_option(
            result, name, ",".join(str(value) for value in values)
        )
    return result


def _print_forward_summary(
    samples: Sequence[ForwardSummarySample], total_cases: int, world_size: int
) -> None:
    grouped: dict[tuple[str, bool, SmConfig], list[ForwardSummarySample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.method, sample.is_causal, sample.config)].append(sample)

    print("\nCross-case forward summary")
    print(
        f"{'Method':<28} {'Mode':<10} {'SM':>8} {'Cases':>8} "
        f"{'Min ms':>10} {'Mean ms':>10} {'P50 ms':>10} {'Max ms':>10} "
        f"{'Mean TFLOPS':>14} {'Weighted TFLOPS':>18} {'Weighted/GPU':>14}"
    )
    for (method, is_causal, config), records in grouped.items():
        times = [record.time_ms for record in records]
        weighted_tflops = sum(
            record.aggregate_tflops * record.time_ms for record in records
        ) / sum(times)
        mean_tflops = sum(
            record.aggregate_tflops for record in records
        ) / len(records)
        mode = "causal" if is_causal else "noncausal"
        sm = f"{config.num_comp_sm}:{config.num_comm_sm}"
        print(
            f"{method:<28} {mode:<10} {sm:>8} "
            f"{f'{len(records)}/{total_cases}':>8} "
            f"{min(times):>10.3f} {sum(times) / len(times):>10.3f} "
            f"{median(times):>10.3f} {max(times):>10.3f} "
            f"{mean_tflops:>14.1f} {weighted_tflops:>18.1f} "
            f"{weighted_tflops / world_size:>14.1f}"
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    workload_cases: Sequence[HybridBenchmarkCase] | None = None,
    skip_incompatible_methods: bool = False,
) -> None:
    if workload_cases is None:
        _main_single(
            argv,
            skip_incompatible_methods=skip_incompatible_methods,
        )
        return
    if not workload_cases:
        raise SystemExit("workload_cases must not be empty")
    if argv is None:
        raise SystemExit("workload_cases require forwarded benchmark arguments")

    args = parse_args(argv)
    methods = parse_methods(args.methods)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("SM90 Hopper CUDA device is required")
    if args.headdim != 128 or args.kvhead * args.headdim != 1024:
        raise SystemExit("hierarchical communication requires D=128 and KVH * D == 1024")
    if args.qhead % args.kvhead:
        raise SystemExit("qhead must be divisible by kvhead")
    if (
        args.allgather_overlapping_heads_k_stride <= 0
        or args.kvhead % args.allgather_overlapping_heads_k_stride
    ):
        raise SystemExit(
            "--allgather-overlapping-heads-k-stride must be a positive divisor "
            "of --kvhead, "
            f"got stride={args.allgather_overlapping_heads_k_stride}, "
            f"kvhead={args.kvhead}"
        )
    for workload_case in workload_cases:
        if any(method != "zepplin" for method in methods):
            validate_metadata(
                list(workload_case.global_lengths),
                list(workload_case.ring_sizes),
                list(workload_case.ring_starts),
                int(os.environ["LOCAL_WORLD_SIZE"]),
                args.mode,
            )
        elif not (
            len(workload_case.global_lengths)
            == len(workload_case.ring_sizes)
            == len(workload_case.ring_starts)
        ):
            raise SystemExit(
                "global lengths, ring sizes, and ring starts must have the same length"
            )
    rank, world_size = init_distributed()
    pools: ForwardParallelPools | None = None
    try:
        all_cp_capacity = (
            max_all_cp_rank_capacity(workload_cases, world_size)
            if "mega_ring_all_cp" in methods
            else None
        )
        hybrid_capacity = (
            max_hybrid_rank_capacity(workload_cases, world_size)
            if "mega_ring_hybrid" in methods
            else None
        )
        pools = ForwardParallelPools(
            all_cp=(
                make_mega_parallel_tensors(
                    rank,
                    world_size,
                    all_cp_capacity,
                    args.kvhead,
                    args.headdim,
                )
                if all_cp_capacity is not None
                else None
            ),
            hybrid=(
                make_mega_parallel_tensors(
                    rank,
                    world_size,
                    hybrid_capacity,
                    args.kvhead,
                    args.headdim,
                )
                if hybrid_capacity is not None
                else None
            ),
        )
        if rank == 0:
            print(
                "Reusable forward IPC pools: "
                f"cases={len(workload_cases)}, "
                f"all_cp_rank_capacity={all_cp_capacity}, "
                f"hybrid_rank_capacity={hybrid_capacity}",
                flush=True,
            )

        all_samples: list[ForwardSummarySample] = []
        for workload_case in workload_cases:
            all_samples.extend(
                _main_single(
                    _argv_for_case(argv, workload_case),
                    skip_incompatible_methods=skip_incompatible_methods,
                    parallel_pools=pools,
                    case_label=workload_case.label,
                    case_index=workload_case.case_index,
                    manage_process_group=False,
                )
            )
        if rank == 0:
            _print_forward_summary(all_samples, len(workload_cases), world_size)
    finally:
        if dist.is_initialized():
            cuda_barrier()
        if pools is not None:
            pools.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
