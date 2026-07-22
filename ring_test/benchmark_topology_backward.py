"""Distributed hierarchical causal mega-ring backward benchmark."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

import torch
import torch.distributed as dist

THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
for path in (THIS_DIR, DEMO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import min_fa3_op
from baseline.magi_attention import (
    MagiAttentionBaseline,
    MagiAttentionConfig,
    probe_magi_attention_all_ranks,
)
from baseline.megatron_hybrid_cp import (
    MegatronHybridCPAttention,
    backward_reference as megatron_hybrid_cp_backward_reference,
    build_hybrid_cp_plan,
    create_hybrid_cp_process_groups,
    hybrid_cp_incompatibility,
    make_packed_hybrid_cp_inputs,
)
from allgather_attention import (
    Llama3AllGatherAttention,
    repartition_sequence_shards_to_llama3,
    select_fa3_backend,
)
from hybrid_backward_baselines import (
    VarlenAllGatherBackward,
    VarlenFa3RingBackward,
    ZepplinBackward,
)
from ring_test.utils import (
    MEGA_RING_ALL_CP_ALIGNMENT,
    HybridBenchmarkCase,
    SENTINEL,
    align_mega_ring_all_cp_lengths,
    assert_all_ranks,
    hierarchical_reference,
    init_distributed,
    local_lengths_for_rank,
    make_cu_seqlens,
    make_local_qkv,
    parse_int_list,
)
from zepplin import (
    DEFAULT_ZEPPLIN_THRESHOLD,
    make_zepplin_plan,
    zepplin_incompatibility,
)


METHOD_ORDER = [
    "allgather_attention",
    "llama3_allgather_attention",
    "fa3_ring",
    "megatron_hybrid_cp",
    "magi_attention",
    "zepplin",
    "mega_ring_all_cp",
    "mega_ring_hybrid",
]
BLOCK_BASELINE_METHODS = {
    "allgather_attention",
    "llama3_allgather_attention",
    "fa3_ring",
    "zepplin",
}
ALL_CP_METHODS = BLOCK_BASELINE_METHODS - {"zepplin"} | {"mega_ring_all_cp"}
BLOCK_ALL_CP_METHODS = ALL_CP_METHODS - {"mega_ring_all_cp"}
SM_SWEEP_METHODS = {"mega_ring_all_cp", "mega_ring_hybrid"}
FUSED_MEGA_RING_METHODS = {"mega_ring_all_cp", "mega_ring_hybrid"}
OVERLAPPED_ALLGATHER_METHODS = {
    "allgather_attention",
    "llama3_allgather_attention",
}


@dataclass(frozen=True)
class SmConfig:
    num_comp_sm: int
    num_comm_sm: int


@dataclass(frozen=True)
class TimingResult:
    max_ms: float
    rank_times_ms: list[float] | None


@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    config: SmConfig | None
    timing: TimingResult
    aggregate_tflops: float
    check: str
    note: str


@dataclass(frozen=True)
class BackwardSummarySample:
    case_index: int
    result: BenchmarkResult


@dataclass
class BackwardMegaPool:
    remote_k: min_fa3_op.TKParallelTensor
    remote_v: min_fa3_op.TKParallelTensor
    remote_dk: min_fa3_op.TKParallelTensor
    remote_dv: min_fa3_op.TKParallelTensor
    completion: min_fa3_op.TKParallelTensor
    rank_capacity: int
    rank: int

    def populate(self, local_k: torch.Tensor, local_v: torch.Tensor) -> None:
        if local_k.size(0) > self.rank_capacity:
            raise RuntimeError(
                f"local K/V rows {local_k.size(0)} exceed pooled rank capacity "
                f"{self.rank_capacity}"
            )
        self.remote_k.data_.fill_(SENTINEL)
        self.remote_v.data_.fill_(SENTINEL)
        owner_begin = self.rank * self.rank_capacity
        owner_end = owner_begin + local_k.size(0)
        self.remote_k.data_[owner_begin:owner_end].copy_(local_k)
        self.remote_v.data_[owner_begin:owner_end].copy_(local_v)

@dataclass
class BackwardParallelPools:
    all_cp: BackwardMegaPool | None
    hybrid: BackwardMegaPool | None

    def close(self) -> None:
        self.all_cp = None
        self.hybrid = None


@dataclass
class MethodRun:
    prepare: Callable[[], object]
    launch: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    note: str
    aligned_global_lengths: tuple[int, ...] | None = None


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
                f"unknown method '{token}', expected one of {METHOD_ORDER} or all"
            )
    deduped: list[str] = []
    for method in methods:
        if method not in deduped:
            deduped.append(method)
    if not deduped:
        raise SystemExit("--methods must provide at least one method")
    return deduped


def requests_all_methods(spec: str) -> bool:
    return any(token.strip() == "all" for token in spec.split(","))


def resolve_magi_attention_availability(
    methods: list[str], method_spec: str
) -> tuple[list[str], str | None]:
    if "magi_attention" not in methods:
        return methods, None
    available, reason = probe_magi_attention_all_ranks(dist.group.WORLD)
    if available:
        return methods, None
    detail = reason or "unknown import failure"
    if requests_all_methods(method_spec):
        return [method for method in methods if method != "magi_attention"], detail
    raise SystemExit(f"magi_attention is unavailable: {detail}")


def method_incompatibility(
    method: str,
    global_lengths: list[int],
    world_size: int,
    zepplin_threshold: int = DEFAULT_ZEPPLIN_THRESHOLD,
    megatron_max_seqlen_per_rank: int = 8192,
) -> str | None:
    if method == "megatron_hybrid_cp":
        return hybrid_cp_incompatibility(
            global_lengths,
            world_size,
            True,
            max_seqlen_per_rank=megatron_max_seqlen_per_rank,
        )
    if method == "zepplin":
        return zepplin_incompatibility(
            global_lengths, world_size, True, zepplin_threshold
        )
    if method == "mega_ring_all_cp":
        return None
    if method not in ALL_CP_METHODS:
        return None
    for batch_idx, global_len in enumerate(global_lengths):
        if global_len % world_size:
            return (
                "all-CP baselines require every global length to be divisible by "
                f"world_size: batch={batch_idx}, global_len={global_len}, "
                f"world_size={world_size}"
            )
        local_len = global_len // world_size
        if local_len % 2:
            return (
                "causal all-CP baselines require even local lengths: "
                f"batch={batch_idx}, local_len={local_len}"
            )
    if method == "llama3_allgather_attention" and sum(global_lengths) % (2 * world_size):
        return (
            "Llama3 whole-packed all-gather requires total tokens divisible by "
            f"2 * world_size, got total={sum(global_lengths)}"
        )
    return None


def compatible_methods(
    methods: list[str],
    global_lengths: list[int],
    world_size: int,
    *,
    skip_incompatible: bool,
    zepplin_threshold: int = DEFAULT_ZEPPLIN_THRESHOLD,
    megatron_max_seqlen_per_rank: int = 8192,
) -> tuple[list[str], list[tuple[str, str]]]:
    active: list[str] = []
    skipped: list[tuple[str, str]] = []
    for method in methods:
        reason = method_incompatibility(
            method,
            global_lengths,
            world_size,
            zepplin_threshold,
            megatron_max_seqlen_per_rank,
        )
        if reason is None:
            active.append(method)
        elif skip_incompatible:
            skipped.append((method, reason))
        else:
            raise SystemExit(f"method '{method}' is incompatible: {reason}")
    if not active:
        raise SystemExit("no compatible backward methods remain")
    return active, skipped


def parse_sm_configs(spec: str) -> list[SmConfig]:
    configs: list[SmConfig] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        fields = token.split(":")
        if len(fields) != 2:
            raise SystemExit(f"invalid SM config '{token}', expected COMP:COMM")
        try:
            configs.append(SmConfig(int(fields[0]), int(fields[1])))
        except ValueError as exc:
            raise SystemExit(f"invalid SM config '{token}', expected two integers") from exc
    if not configs:
        raise SystemExit("--sm-configs must provide at least one COMP:COMM pair")
    return configs


def resolve_sm_configs(args: argparse.Namespace) -> list[SmConfig]:
    if args.sm_configs is not None:
        if args.num_comp_sm is not None or args.num_comm_sm is not None:
            raise SystemExit("use either --sm-configs or --num-comp-sm/--num-comm-sm")
        return parse_sm_configs(args.sm_configs)
    if (args.num_comp_sm is None) != (args.num_comm_sm is None):
        raise SystemExit("--num-comp-sm and --num-comm-sm must be provided together")
    if args.num_comp_sm is not None:
        return [SmConfig(args.num_comp_sm, args.num_comm_sm)]
    return [SmConfig(100, 16)]


def make_cases(args: argparse.Namespace) -> list[tuple[int, int]]:
    batches = parse_int_list(args.b, "--b")
    seqlens = parse_int_list(args.seqlen, "--seqlen")
    if len(batches) != len(seqlens):
        raise SystemExit("--b and --seqlen must contain the same number of cases")
    if any(batch <= 0 for batch in batches):
        raise SystemExit("--b values must be positive")
    if any(seqlen <= 0 or seqlen % 256 for seqlen in seqlens):
        raise SystemExit("causal local --seqlen values must be positive and divisible by 256")
    return list(zip(batches, seqlens))


def make_topology(
    batch_size: int, local_seqlen: int, size_spec: str, start_spec: str
) -> tuple[list[int], list[int], list[int]]:
    size_pattern = parse_int_list(size_spec, "--ring-sizes")
    start_pattern = parse_int_list(start_spec, "--ring-starts")
    if len(size_pattern) != len(start_pattern):
        raise SystemExit("--ring-sizes and --ring-starts must have the same length")
    pairs = [
        (size_pattern[idx % len(size_pattern)], start_pattern[idx % len(start_pattern)])
        for idx in range(batch_size)
    ]
    pairs.sort(key=lambda pair: pair[0], reverse=True)
    ring_sizes = [pair[0] for pair in pairs]
    ring_starts = [pair[1] for pair in pairs]
    global_lengths = [local_seqlen * ring_size for ring_size in ring_sizes]
    return global_lengths, ring_sizes, ring_starts


def validate_backward_metadata(
    global_lengths: list[int],
    ring_sizes: list[int],
    ring_starts: list[int],
    world_size: int,
) -> None:
    if not global_lengths:
        raise SystemExit("the benchmark requires at least one sequence")
    if not (len(global_lengths) == len(ring_sizes) == len(ring_starts)):
        raise SystemExit("global lengths, ring sizes, and ring starts must have the same length")
    previous_ring_size = 8
    for batch_idx, (global_len, ring_size, ring_start) in enumerate(
        zip(global_lengths, ring_sizes, ring_starts)
    ):
        if ring_size not in (1, 2, 4, 8) or ring_size > previous_ring_size:
            raise SystemExit(f"invalid ring size/order at batch {batch_idx}")
        if ring_size > world_size:
            raise SystemExit(f"ring size exceeds world size at batch {batch_idx}")
        if ring_start < 0 or ring_start % ring_size or ring_start + ring_size > world_size:
            raise SystemExit(f"invalid aligned ring start at batch {batch_idx}")
        if global_len <= 0 or global_len % ring_size:
            raise SystemExit(f"invalid global length at batch {batch_idx}")
        local_len = global_len // ring_size
        if local_len % 128:
            raise SystemExit(f"local length is not 128-row aligned at batch {batch_idx}")
        if ring_size > 1 and (local_len // 2) % 128:
            raise SystemExit(f"causal local half length is not 128-row aligned at batch {batch_idx}")
        previous_ring_size = ring_size


def make_remote_kv(
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    rank: int,
    world_size: int,
    rank_capacity: int,
) -> tuple[min_fa3_op.TKParallelTensor, min_fa3_op.TKParallelTensor]:
    arena_shape = [world_size * rank_capacity, local_k.size(1), local_k.size(2)]
    remote_k = min_fa3_op.TKParallelTensor(
        arena_shape, torch.bfloat16, rank, world_size, False
    )
    remote_v = min_fa3_op.TKParallelTensor(
        arena_shape, torch.bfloat16, rank, world_size, False
    )
    remote_k.data_.fill_(SENTINEL)
    remote_v.data_.fill_(SENTINEL)
    owner_begin = rank * rank_capacity
    remote_k.data_[owner_begin : owner_begin + local_k.size(0)].copy_(local_k)
    remote_v.data_[owner_begin : owner_begin + local_v.size(0)].copy_(local_v)
    return remote_k, remote_v


def make_backward_pool(
    rank: int,
    world_size: int,
    rank_capacity: int,
    accum_numel: int,
    kv_heads: int,
    head_dim: int,
) -> BackwardMegaPool:
    arena_shape = [world_size * rank_capacity, kv_heads, head_dim]
    return BackwardMegaPool(
        remote_k=min_fa3_op.TKParallelTensor(
            arena_shape, torch.bfloat16, rank, world_size, False
        ),
        remote_v=min_fa3_op.TKParallelTensor(
            arena_shape, torch.bfloat16, rank, world_size, False
        ),
        remote_dk=min_fa3_op.TKParallelTensor(
            [accum_numel], torch.float32, rank, world_size, False
        ),
        remote_dv=min_fa3_op.TKParallelTensor(
            [accum_numel], torch.float32, rank, world_size, False
        ),
        completion=min_fa3_op.TKParallelTensor(
            [1], torch.int32, rank, world_size, False
        ),
        rank_capacity=rank_capacity,
        rank=rank,
    )


def _hybrid_case_rank_capacity(
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    world_size: int,
) -> int:
    return max(
        sum(
            local_lengths_for_rank(
                list(global_lengths),
                list(ring_sizes),
                list(ring_starts),
                rank,
            )
        )
        for rank in range(world_size)
    )


def make_backward_parallel_pools(
    workloads: Sequence[tuple[str, list[int], list[int], list[int]]],
    methods: Sequence[str],
    rank: int,
    world_size: int,
    kv_heads: int,
    head_dim: int,
) -> BackwardParallelPools:
    all_cp_pool = None
    if "mega_ring_all_cp" in methods:
        all_cp_cases = [
            (
                sum(align_mega_ring_all_cp_lengths(global_lengths)) // world_size,
                len(global_lengths),
            )
            for _label, global_lengths, _ring_sizes, _ring_starts in workloads
        ]
        all_cp_rank_capacity = max(capacity for capacity, _batch in all_cp_cases)
        all_cp_accum_numel = max(
            kv_heads
            * (((capacity + batch * 128 + 127) // 128) * 128)
            * head_dim
            for capacity, batch in all_cp_cases
        )
        all_cp_pool = make_backward_pool(
            rank,
            world_size,
            all_cp_rank_capacity,
            all_cp_accum_numel,
            kv_heads,
            head_dim,
        )

    hybrid_pool = None
    if "mega_ring_hybrid" in methods:
        hybrid_cases = [
            (
                _hybrid_case_rank_capacity(
                    global_lengths, ring_sizes, ring_starts, world_size
                ),
                len(global_lengths),
            )
            for _label, global_lengths, ring_sizes, ring_starts in workloads
        ]
        hybrid_rank_capacity = max(capacity for capacity, _batch in hybrid_cases)
        hybrid_rank_capacity = ((hybrid_rank_capacity + 127) // 128) * 128
        hybrid_accum_numel = max(
            kv_heads
            * (((capacity + batch * 128 + 127) // 128) * 128)
            * head_dim
            for capacity, batch in hybrid_cases
        )
        hybrid_pool = make_backward_pool(
            rank,
            world_size,
            hybrid_rank_capacity,
            hybrid_accum_numel,
            kv_heads,
            head_dim,
        )
    return BackwardParallelPools(all_cp=all_cp_pool, hybrid=hybrid_pool)


def gather_padded_rank_tensor(tensor: torch.Tensor, rank_capacity: int) -> torch.Tensor:
    padded = torch.zeros(
        (rank_capacity, tensor.size(1), tensor.size(2)),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    padded[: tensor.size(0)].copy_(tensor)
    gathered = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)
    return torch.stack(gathered)


def make_reference(
    q: torch.Tensor,
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    dout: torch.Tensor,
    all_rank_lengths: list[list[int]],
    cu_host: torch.Tensor,
    global_lengths: list[int],
    ring_sizes: list[int],
    ring_starts: list[int],
    rank_capacity: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gathered_k = gather_padded_rank_tensor(local_k, rank_capacity)
    gathered_v = gather_padded_rank_tensor(local_v, rank_capacity)
    q_ref = q.detach().clone().requires_grad_(True)
    gathered_k_ref = gathered_k.detach().clone().requires_grad_(True)
    gathered_v_ref = gathered_v.detach().clone().requires_grad_(True)
    out_ref, lse_ref = hierarchical_reference(
        q_ref,
        gathered_k_ref,
        gathered_v_ref,
        all_rank_lengths,
        cu_host,
        global_lengths,
        ring_sizes,
        ring_starts,
        rank,
        True,
    )
    if q.size(0) > 0:
        dq_ref, dk_ref_all, dv_ref_all = torch.autograd.grad(
            out_ref, (q_ref, gathered_k_ref, gathered_v_ref), dout
        )
    else:
        dq_ref = torch.zeros_like(q_ref)
        dk_ref_all = torch.zeros_like(gathered_k_ref)
        dv_ref_all = torch.zeros_like(gathered_v_ref)
    dist.all_reduce(dk_ref_all)
    dist.all_reduce(dv_ref_all)
    local_total = local_k.size(0)
    return (
        out_ref.detach(),
        lse_ref.detach(),
        dq_ref.detach(),
        dk_ref_all[rank, :local_total].detach(),
        dv_ref_all[rank, :local_total].detach(),
    )


def measure_backward_ms(
    prepare: Callable[[], None],
    launch: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    warmup_iters: int,
    num_iters: int,
    rank: int,
) -> TimingResult:
    """Average per-rank times and the per-iteration maximum across ranks."""
    for _ in range(warmup_iters):
        prepare()
        torch.cuda.synchronize()
        dist.barrier()
        launch()
        torch.cuda.synchronize()

    local_samples: list[float] = []
    max_samples: list[float] = []
    for _ in range(num_iters):
        prepare()
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        launch()
        torch.cuda.synchronize()
        local_ms = (time.perf_counter() - start) * 1e3
        max_ms = torch.tensor([local_ms], device="cuda", dtype=torch.float64)
        dist.all_reduce(max_ms, op=dist.ReduceOp.MAX)
        local_samples.append(local_ms)
        max_samples.append(float(max_ms.item()))

    local_avg = sum(local_samples) / len(local_samples)
    max_avg = sum(max_samples) / len(max_samples)
    local_time = torch.tensor([local_avg], device="cuda", dtype=torch.float64)
    gathered = [torch.empty_like(local_time) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_time)
    rank_times = [value.item() for value in gathered] if rank == 0 else None
    return TimingResult(max_avg, rank_times)


def aggregate_backward_tflops(
    global_lengths: Sequence[int], q_heads: int, head_dim: int, time_ms: float
) -> float:
    # FA backward is approximately 2.5x forward. Forward performs four FLOPs
    # per causal QK/PV score and value pair, hence ten backward FLOPs per pair.
    score_count = sum(length * (length + 1) // 2 for length in global_lengths)
    flops = 10 * score_count * q_heads * head_dim
    return flops / time_ms / 1e9


def check_gradients(
    label: str,
    prepare: Callable[[], None],
    launch: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    references: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dq_atol: float,
    dkv_atol: float,
    rtol: float,
) -> None:
    prepare()
    torch.cuda.synchronize()
    dist.barrier()
    actual = launch()
    torch.cuda.synchronize()
    local_error = None
    for name, gradient, reference, atol in zip(
        ("dQ", "dK", "dV"),
        actual,
        references,
        (dq_atol, dkv_atol, dkv_atol),
    ):
        try:
            torch.testing.assert_close(
                gradient.float(), reference.float(), atol=atol, rtol=rtol
            )
        except AssertionError as exc:
            local_error = f"{label} {name} check failed: {exc}"
            break
    assert_all_ranks(local_error)


def print_results(
    results: list[BenchmarkResult],
    world_size: int,
) -> None:
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for result in results:
        rank_times = result.timing.rank_times_ms
        if rank_times is None:
            time_text = f"max_across_ranks={result.timing.max_ms:.3f}"
        else:
            per_rank = ", ".join(
                f"t{rank}={time_ms:.3f}" for rank, time_ms in enumerate(rank_times)
            )
            time_text = f"{per_rank} | max_across_ranks={result.timing.max_ms:.3f}"
        rows.append(
            (
                result.method,
                (
                    "-"
                    if result.config is None
                    else f"{result.config.num_comp_sm}:{result.config.num_comm_sm}"
                ),
                time_text,
                f"{result.aggregate_tflops:.2f}",
                f"{result.aggregate_tflops / world_size:.2f}",
                result.check,
                result.note,
            )
        )
    method_width = max(24, *(len(row[0]) for row in rows))
    time_width = max(64, *(len(row[2]) for row in rows))
    print(
        f"{'Method':<{method_width}} {'Comp:Comm':>10} "
        f"{'Time ms':<{time_width}} {'Agg TFLOPS':>12} "
        f"{'Avg/GPU':>10} {'Check':>8}  Note"
    )
    for method, sm_config, time_text, aggregate, average, check, note in rows:
        print(
            f"{method:<{method_width}} {sm_config:>10} "
            f"{time_text:<{time_width}} {aggregate:>12} "
            f"{average:>10} {check:>8}  {note}"
        )


def benchmark_topology(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    global_lengths: list[int],
    ring_sizes: list[int],
    ring_starts: list[int],
    sm_configs: list[SmConfig],
    label: str,
    methods: list[str],
    allgather_backend: str | None,
    parallel_pools: BackwardParallelPools | None = None,
) -> list[BenchmarkResult]:
    if any(method not in {"zepplin", "magi_attention"} for method in methods):
        validate_backward_metadata(global_lengths, ring_sizes, ring_starts, world_size)
    elif "zepplin" in methods and not (
        len(global_lengths) == len(ring_sizes) == len(ring_starts)
    ):
        raise SystemExit(
            "global lengths, ring sizes, and ring starts must have the same length"
        )
    device = torch.device("cuda", rank)
    mega_ring_all_cp_global_lengths = align_mega_ring_all_cp_lengths(global_lengths)
    zepplin_plan = (
        make_zepplin_plan(
            global_lengths,
            world_size,
            True,
            args.zepplin_threshold,
        )
        if "zepplin" in methods
        else None
    )
    if rank == 0:
        print(
            f"\nWorkload: {label}, B={len(global_lengths)}, "
            f"global_tokens={sum(global_lengths)}, global_seqlens={global_lengths}"
        )
        if "mega_ring_all_cp" in methods:
            print(
                "Mega-ring all-CP workload: "
                f"alignment={MEGA_RING_ALL_CP_ALIGNMENT}, "
                f"global_tokens={sum(mega_ring_all_cp_global_lengths)}, "
                f"global_seqlens={mega_ring_all_cp_global_lengths}"
            )
        print(f"Hybrid rings: sizes={ring_sizes}, starts={ring_starts}")
        if zepplin_plan is not None:
            print(
                f"Zepplin placement: threshold={zepplin_plan.threshold}, "
                f"G1={len(zepplin_plan.short_indices)}, "
                f"Gworld={len(zepplin_plan.long_indices)}, "
                f"G1_rank_loads={list(zepplin_plan.short_loads)}"
            )
        print(
            "Timing excludes forward preparation, owner-accumulator reset, and the "
            "pre-launch distributed barrier; method-internal phase barriers are included; "
            "reported time is the average of the per-iteration max-rank end-to-end "
            "backward op times.",
            flush=True,
        )

    baseline_runs: dict[str, MethodRun] = {}
    all_cp_mega_runs: dict[SmConfig, MethodRun] = {}
    if "magi_attention" in methods:
        magi_runner = MagiAttentionBaseline(
            dist.group.WORLD,
            global_lengths,
            args.qhead,
            args.kvhead,
            args.headdim,
            True,
            device,
            config=MagiAttentionConfig(
                overlap_degree=args.magi_overlap_degree,
                seed=args.seed,
            ),
            enable_backward=True,
        )
        baseline_runs["magi_attention"] = MethodRun(
            magi_runner.prepare_backward,
            magi_runner.backward,
            None,
            magi_runner.note
            + "; forward preparation excluded from backward timing",
        )
    if "megatron_hybrid_cp" in methods:
        if allgather_backend is None:
            raise RuntimeError(
                "Megatron hybrid-CP baseline requires a block backend"
            )
        megatron_plan = build_hybrid_cp_plan(
            global_lengths,
            world_size,
            args.megatron_max_seqlen_per_rank,
        )
        megatron_groups = create_hybrid_cp_process_groups(dist.group.WORLD)
        megatron_inputs = make_packed_hybrid_cp_inputs(
            megatron_plan,
            rank,
            args.qhead,
            args.kvhead,
            args.headdim,
            device,
            is_causal=True,
            seed=args.seed + 151,
            with_dout=True,
        )
        if megatron_inputs.dout is None:
            raise AssertionError("backward input preparation did not create dO")
        megatron_runner = MegatronHybridCPAttention(
            megatron_plan,
            megatron_groups,
            megatron_inputs.q,
            megatron_inputs.k,
            megatron_inputs.v,
            True,
            allgather_backend,
            dout=megatron_inputs.dout,
        )
        megatron_reference = (
            megatron_hybrid_cp_backward_reference(megatron_runner)
            if args.check
            else None
        )
        baseline_runs["megatron_hybrid_cp"] = MethodRun(
            megatron_runner.forward_all,
            megatron_runner.backward_all,
            megatron_reference,
            megatron_runner.note
            + "; complete forward preparation excluded from backward timing",
        )
    if any(method in BLOCK_ALL_CP_METHODS for method in methods):
        if allgather_backend is None:
            raise RuntimeError("block baselines require a selected block backend")
        all_cp_lengths = [length // world_size for length in global_lengths]
        all_cp_total = sum(all_cp_lengths)
        all_cp_cu_host = torch.tensor(
            [0, *accumulate(all_cp_lengths)], dtype=torch.int32
        )
        all_cp_cu = all_cp_cu_host.to(device=device)
        all_cp_q, all_cp_k, all_cp_v = make_local_qkv(
            all_cp_total,
            args.qhead,
            args.kvhead,
            args.headdim,
            rank,
            True,
            device,
            base_seed=args.seed + 101,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + 20_260_817 + rank)
        all_cp_dout = torch.randn(
            all_cp_q.shape, device=device, generator=generator
        ).to(torch.bfloat16)

        all_cp_reference = None
        if args.check:
            all_cp_reference = make_reference(
                all_cp_q,
                all_cp_k,
                all_cp_v,
                all_cp_dout,
                [all_cp_lengths for _ in range(world_size)],
                all_cp_cu_host,
                global_lengths,
                [world_size] * len(global_lengths),
                [0] * len(global_lengths),
                all_cp_total,
                rank,
            )

        if "allgather_attention" in methods:
            allgather_runner = VarlenAllGatherBackward(
                dist.group.WORLD,
                all_cp_q,
                all_cp_k,
                all_cp_v,
                all_cp_lengths,
                allgather_backend,
                heads_k_stride=args.allgather_overlapping_heads_k_stride,
            )
            baseline_runs["allgather_attention"] = MethodRun(
                allgather_runner.forward,
                lambda: allgather_runner.backward(all_cp_dout),
                None if all_cp_reference is None else all_cp_reference[2:],
                allgather_runner.note,
            )

        if "llama3_allgather_attention" in methods:
            llama_q = repartition_sequence_shards_to_llama3(
                dist.group.WORLD, all_cp_q, global_lengths, True
            )
            llama_k = repartition_sequence_shards_to_llama3(
                dist.group.WORLD, all_cp_k, global_lengths, True
            )
            llama_v = repartition_sequence_shards_to_llama3(
                dist.group.WORLD, all_cp_v, global_lengths, True
            )
            llama_dout = repartition_sequence_shards_to_llama3(
                dist.group.WORLD, all_cp_dout, global_lengths, True
            )
            llama_runner = Llama3AllGatherAttention(
                dist.group.WORLD,
                llama_q,
                llama_k,
                llama_v,
                global_lengths,
                True,
                allgather_backend,
                heads_k_stride=args.allgather_overlapping_heads_k_stride,
                enable_backward=True,
            )
            llama_reference = None
            if all_cp_reference is not None:
                llama_reference = tuple(
                    repartition_sequence_shards_to_llama3(
                        dist.group.WORLD, gradient, global_lengths, True
                    )
                    for gradient in all_cp_reference[2:]
                )
            baseline_runs["llama3_allgather_attention"] = MethodRun(
                llama_runner.forward,
                lambda: llama_runner.backward(llama_dout),
                llama_reference,
                llama_runner.note,
            )

        if "fa3_ring" in methods:
            fa3_ring_runner = VarlenFa3RingBackward(
                dist.group.WORLD,
                all_cp_q,
                all_cp_k,
                all_cp_v,
                all_cp_dout,
                all_cp_lengths,
                allgather_backend,
            )
            baseline_runs["fa3_ring"] = MethodRun(
                fa3_ring_runner.forward,
                fa3_ring_runner.backward,
                None if all_cp_reference is None else all_cp_reference[2:],
                fa3_ring_runner.note,
            )

    if "mega_ring_all_cp" in methods:
        mega_all_cp_lengths = [
            length // world_size for length in mega_ring_all_cp_global_lengths
        ]
        mega_all_cp_total = sum(mega_all_cp_lengths)
        mega_all_cp_cu_host = torch.tensor(
            [0, *accumulate(mega_all_cp_lengths)], dtype=torch.int32
        )
        mega_all_cp_cu = mega_all_cp_cu_host.to(device=device)
        mega_all_cp_q, mega_all_cp_k, mega_all_cp_v = make_local_qkv(
            mega_all_cp_total,
            args.qhead,
            args.kvhead,
            args.headdim,
            rank,
            True,
            device,
            base_seed=args.seed + 101,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + 20_260_817 + rank)
        mega_all_cp_dout = torch.randn(
            mega_all_cp_q.shape, device=device, generator=generator
        ).to(torch.bfloat16)

        mega_all_cp_reference = None
        if args.check:
            mega_all_cp_reference = make_reference(
                mega_all_cp_q,
                mega_all_cp_k,
                mega_all_cp_v,
                mega_all_cp_dout,
                [mega_all_cp_lengths for _ in range(world_size)],
                mega_all_cp_cu_host,
                mega_ring_all_cp_global_lengths,
                [world_size] * len(global_lengths),
                [0] * len(global_lengths),
                mega_all_cp_total,
                rank,
            )

        mega_all_cp_pool = (
            None if parallel_pools is None else parallel_pools.all_cp
        )
        if mega_all_cp_pool is None:
            mega_all_cp_remote_k, mega_all_cp_remote_v = make_remote_kv(
                mega_all_cp_k,
                mega_all_cp_v,
                rank,
                world_size,
                mega_all_cp_total,
            )
        else:
            mega_all_cp_pool.populate(mega_all_cp_k, mega_all_cp_v)
            mega_all_cp_remote_k = mega_all_cp_pool.remote_k
            mega_all_cp_remote_v = mega_all_cp_pool.remote_v
        mega_all_cp_k_arena = mega_all_cp_remote_k.data_
        mega_all_cp_v_arena = mega_all_cp_remote_v.data_
        mega_all_cp_global_host = torch.tensor(
            mega_ring_all_cp_global_lengths, dtype=torch.int32
        )
        mega_all_cp_ring_sizes_host = torch.full(
            (len(global_lengths),), world_size, dtype=torch.int32
        )
        mega_all_cp_ring_starts_host = torch.zeros(
            len(global_lengths), dtype=torch.int32
        )
        mega_all_cp_max_local_len = max(mega_all_cp_lengths)

        torch.cuda.synchronize()
        dist.barrier()
        forward_config = sm_configs[0]
        mega_all_cp_out, mega_all_cp_lse = min_fa3_op.forward_varlen_mega_ring(
            mega_all_cp_q,
            mega_all_cp_k_arena,
            mega_all_cp_v_arena,
            mega_all_cp_cu,
            mega_all_cp_cu,
            mega_all_cp_max_local_len,
            mega_all_cp_max_local_len,
            True,
            cu_seqlens_q_host=mega_all_cp_cu_host,
            cu_seqlens_k_host=mega_all_cp_cu_host,
            remote_k=mega_all_cp_remote_k,
            remote_v=mega_all_cp_remote_v,
            num_comp_sm=forward_config.num_comp_sm,
            num_comm_sm=forward_config.num_comm_sm,
            global_seqlens_host=mega_all_cp_global_host,
            ring_sizes_host=mega_all_cp_ring_sizes_host,
            ring_starts_host=mega_all_cp_ring_starts_host,
            return_lse=True,
        )
        torch.cuda.synchronize()
        dist.barrier()

        if mega_all_cp_reference is not None:
            local_error = None
            try:
                torch.testing.assert_close(
                    mega_all_cp_out.float(),
                    mega_all_cp_reference[0].float(),
                    atol=0.2,
                    rtol=args.rtol,
                )
                torch.testing.assert_close(
                    mega_all_cp_lse,
                    mega_all_cp_reference[1],
                    atol=0.2,
                    rtol=args.rtol,
                )
            except AssertionError as exc:
                local_error = (
                    f"all-CP mega-ring forward preparation check failed: {exc}"
                )
            assert_all_ranks(local_error)

        mega_all_cp_padded_capacity = (
            (mega_all_cp_total + len(global_lengths) * 128 + 127) // 128
        ) * 128
        mega_all_cp_accum_numel = (
            args.kvhead * mega_all_cp_padded_capacity * args.headdim
        )
        if mega_all_cp_pool is None:
            mega_all_cp_remote_dk = min_fa3_op.TKParallelTensor(
                [mega_all_cp_accum_numel], torch.float32, rank, world_size, False
            )
            mega_all_cp_remote_dv = min_fa3_op.TKParallelTensor(
                [mega_all_cp_accum_numel], torch.float32, rank, world_size, False
            )
            mega_all_cp_completion = min_fa3_op.TKParallelTensor(
                [1], torch.int32, rank, world_size, False
            )
        else:
            mega_all_cp_remote_dk = mega_all_cp_pool.remote_dk
            mega_all_cp_remote_dv = mega_all_cp_pool.remote_dv
            mega_all_cp_completion = mega_all_cp_pool.completion

        def prepare_all_cp_mega() -> None:
            mega_all_cp_remote_dk.data_.zero_()
            mega_all_cp_remote_dv.data_.zero_()
            mega_all_cp_completion.data_.zero_()

        for config in sm_configs:
            def launch_all_cp_mega(
                config: SmConfig = config,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return min_fa3_op.backward_varlen_mega_ring(
                    mega_all_cp_dout,
                    mega_all_cp_q,
                    mega_all_cp_k_arena,
                    mega_all_cp_v_arena,
                    mega_all_cp_out,
                    mega_all_cp_lse,
                    mega_all_cp_cu,
                    mega_all_cp_cu,
                    mega_all_cp_max_local_len,
                    mega_all_cp_max_local_len,
                    cu_seqlens_q_host=mega_all_cp_cu_host,
                    cu_seqlens_k_host=mega_all_cp_cu_host,
                    remote_k=mega_all_cp_remote_k,
                    remote_v=mega_all_cp_remote_v,
                    remote_dk_accum=mega_all_cp_remote_dk,
                    remote_dv_accum=mega_all_cp_remote_dv,
                    remote_dkv_completion=mega_all_cp_completion,
                    num_comp_sm=config.num_comp_sm,
                    num_comm_sm=config.num_comm_sm,
                    global_seqlens_host=mega_all_cp_global_host,
                    ring_sizes_host=mega_all_cp_ring_sizes_host,
                    ring_starts_host=mega_all_cp_ring_starts_host,
                )

            all_cp_mega_runs[config] = MethodRun(
                prepare_all_cp_mega,
                launch_all_cp_mega,
                None
                if mega_all_cp_reference is None
                else mega_all_cp_reference[2:],
                "all-CP fused mega-ring; remote reset excluded",
                tuple(mega_ring_all_cp_global_lengths),
            )

    if zepplin_plan is not None:
        if allgather_backend is None:
            raise RuntimeError("zepplin baseline requires a selected block backend")
        zepplin_rank_lengths = [
            zepplin_plan.topology_lengths_for_rank(source_rank)
            for source_rank in range(world_size)
        ]
        zepplin_local_lengths = zepplin_plan.packed_lengths_for_rank(rank)
        zepplin_local_total = sum(zepplin_local_lengths)
        zepplin_q, zepplin_k, zepplin_v = make_local_qkv(
            zepplin_local_total,
            args.qhead,
            args.kvhead,
            args.headdim,
            rank,
            True,
            device,
            base_seed=args.seed + 137,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + 20_260_917 + rank)
        zepplin_dout = torch.randn(
            zepplin_q.shape, device=device, generator=generator
        ).to(torch.bfloat16)
        zepplin_reference = None
        if args.check:
            zepplin_cu_host = torch.tensor(
                [0, *accumulate(zepplin_plan.topology_lengths_for_rank(rank))],
                dtype=torch.int32,
            )
            zepplin_rank_capacity = max(
                sum(lengths) for lengths in zepplin_rank_lengths
            )
            zepplin_reference = make_reference(
                zepplin_q,
                zepplin_k,
                zepplin_v,
                zepplin_dout,
                zepplin_rank_lengths,
                zepplin_cu_host,
                zepplin_plan.packed_global_lengths,
                zepplin_plan.ring_sizes,
                zepplin_plan.ring_starts,
                zepplin_rank_capacity,
                rank,
            )
        zepplin_runner = ZepplinBackward(
            dist.group.WORLD,
            zepplin_q,
            zepplin_k,
            zepplin_v,
            zepplin_dout,
            zepplin_plan,
            allgather_backend,
        )
        baseline_runs["zepplin"] = MethodRun(
            zepplin_runner.forward,
            zepplin_runner.backward,
            None if zepplin_reference is None else zepplin_reference[2:],
            zepplin_runner.note,
        )

    hybrid_runs: dict[SmConfig, MethodRun] = {}
    if "mega_ring_hybrid" in methods:
        all_rank_lengths = [
            local_lengths_for_rank(global_lengths, ring_sizes, ring_starts, source_rank)
            for source_rank in range(world_size)
        ]
        local_lengths = all_rank_lengths[rank]
        local_total = sum(local_lengths)
        cu, cu_host = make_cu_seqlens(local_lengths, device)
        q, local_k, local_v = make_local_qkv(
            local_total,
            args.qhead,
            args.kvhead,
            args.headdim,
            rank,
            True,
            device,
            base_seed=args.seed + 17,
        )
        capacity = torch.tensor([local_total], device=device, dtype=torch.int32)
        dist.all_reduce(capacity, op=dist.ReduceOp.MAX)
        rank_capacity = ((int(capacity.item()) + 127) // 128) * 128
        hybrid_pool = None if parallel_pools is None else parallel_pools.hybrid
        if hybrid_pool is None:
            remote_k, remote_v = make_remote_kv(
                local_k, local_v, rank, world_size, rank_capacity
            )
        else:
            hybrid_pool.populate(local_k, local_v)
            remote_k = hybrid_pool.remote_k
            remote_v = hybrid_pool.remote_v
        k, v = remote_k.data_, remote_v.data_
        global_host = torch.tensor(global_lengths, dtype=torch.int32)
        ring_sizes_host = torch.tensor(ring_sizes, dtype=torch.int32)
        ring_starts_host = torch.tensor(ring_starts, dtype=torch.int32)
        max_local_len = max(max(lengths) for lengths in all_rank_lengths)

        torch.cuda.synchronize()
        dist.barrier()
        forward_config = sm_configs[0]
        out, lse = min_fa3_op.forward_varlen_mega_ring(
            q,
            k,
            v,
            cu,
            cu,
            max_local_len,
            max_local_len,
            True,
            cu_seqlens_q_host=cu_host,
            cu_seqlens_k_host=cu_host,
            remote_k=remote_k,
            remote_v=remote_v,
            num_comp_sm=forward_config.num_comp_sm,
            num_comm_sm=forward_config.num_comm_sm,
            global_seqlens_host=global_host,
            ring_sizes_host=ring_sizes_host,
            ring_starts_host=ring_starts_host,
            return_lse=True,
        )
        torch.cuda.synchronize()
        dist.barrier()

        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + 20_260_716 + rank)
        dout = torch.randn(q.shape, device=device, generator=generator).to(torch.bfloat16)
        hybrid_reference = None
        if args.check:
            hybrid_reference = make_reference(
                q,
                local_k,
                local_v,
                dout,
                all_rank_lengths,
                cu_host,
                global_lengths,
                ring_sizes,
                ring_starts,
                rank_capacity,
                rank,
            )
            local_error = None
            try:
                torch.testing.assert_close(
                    out.float(), hybrid_reference[0].float(), atol=0.2, rtol=args.rtol
                )
                torch.testing.assert_close(
                    lse, hybrid_reference[1], atol=0.2, rtol=args.rtol
                )
            except AssertionError as exc:
                local_error = f"hybrid forward preparation check failed: {exc}"
            assert_all_ranks(local_error)

        padded_capacity = (
            (rank_capacity + len(global_lengths) * 128 + 127) // 128
        ) * 128
        accum_numel = args.kvhead * padded_capacity * args.headdim
        if hybrid_pool is None:
            remote_dk = min_fa3_op.TKParallelTensor(
                [accum_numel], torch.float32, rank, world_size, False
            )
            remote_dv = min_fa3_op.TKParallelTensor(
                [accum_numel], torch.float32, rank, world_size, False
            )
            completion = min_fa3_op.TKParallelTensor(
                [1], torch.int32, rank, world_size, False
            )
        else:
            remote_dk = hybrid_pool.remote_dk
            remote_dv = hybrid_pool.remote_dv
            completion = hybrid_pool.completion

        def prepare_hybrid() -> None:
            remote_dk.data_.zero_()
            remote_dv.data_.zero_()
            completion.data_.zero_()

        for config in sm_configs:
            def launch_hybrid(
                config: SmConfig = config,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return min_fa3_op.backward_varlen_mega_ring(
                    dout,
                    q,
                    k,
                    v,
                    out,
                    lse,
                    cu,
                    cu,
                    max_local_len,
                    max_local_len,
                    cu_seqlens_q_host=cu_host,
                    cu_seqlens_k_host=cu_host,
                    remote_k=remote_k,
                    remote_v=remote_v,
                    remote_dk_accum=remote_dk,
                    remote_dv_accum=remote_dv,
                    remote_dkv_completion=completion,
                    num_comp_sm=config.num_comp_sm,
                    num_comm_sm=config.num_comm_sm,
                    global_seqlens_host=global_host,
                    ring_sizes_host=ring_sizes_host,
                    ring_starts_host=ring_starts_host,
                )

            hybrid_runs[config] = MethodRun(
                prepare_hybrid,
                launch_hybrid,
                None if hybrid_reference is None else hybrid_reference[2:],
                "hierarchical hybrid fused mega-ring; remote reset excluded",
            )

    results: list[BenchmarkResult] = []
    for method in methods:
        if method in baseline_runs:
            method_runs = [(None, baseline_runs[method])]
        elif method in SM_SWEEP_METHODS:
            runs_by_config = (
                all_cp_mega_runs
                if method == "mega_ring_all_cp"
                else hybrid_runs
            )
            method_runs = [
                (config, runs_by_config[config]) for config in sm_configs
            ]
        else:
            raise RuntimeError(f"unhandled backward method {method}")
        for config, run in method_runs:
            timing = measure_backward_ms(
                run.prepare, run.launch, args.warmup_iters, args.num_iters, rank
            )
            check_status = "skip"
            if run.reference is not None:
                config_label = (
                    ""
                    if config is None
                    else f" SM {config.num_comp_sm}:{config.num_comm_sm}"
                )
                check_gradients(
                    f"{method}{config_label}",
                    run.prepare,
                    run.launch,
                    run.reference,
                    args.dq_atol,
                    args.dkv_atol,
                    args.rtol,
                )
                check_status = "ok"
            aggregate_tflops = aggregate_backward_tflops(
                global_lengths, args.qhead, args.headdim, timing.max_ms
            )
            if rank == 0:
                note = run.note
                if run.aligned_global_lengths is not None:
                    aligned_aggregate_tflops = aggregate_backward_tflops(
                        run.aligned_global_lengths,
                        args.qhead,
                        args.headdim,
                        timing.max_ms,
                    )
                    note = (
                        f"{note}; {MEGA_RING_ALL_CP_ALIGNMENT}-aligned "
                        f"Agg TFLOPS={aligned_aggregate_tflops:.2f}, "
                        f"Avg/GPU={aligned_aggregate_tflops / world_size:.2f}"
                    )
                results.append(
                    BenchmarkResult(
                        method,
                        config,
                        timing,
                        aggregate_tflops,
                        check_status,
                        note,
                    )
                )
            dist.barrier()

    if rank == 0:
        print_results(results, world_size)
    dist.barrier()
    return results


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def magi_overlap_degree(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 8:
        raise argparse.ArgumentTypeError("value must be an integer in [1, 8]")
    return parsed


def _print_backward_summary(
    samples: Sequence[BackwardSummarySample], total_cases: int, world_size: int
) -> None:
    grouped: dict[
        tuple[str, SmConfig | None], list[BackwardSummarySample]
    ] = defaultdict(list)
    for sample in samples:
        grouped[(sample.result.method, sample.result.config)].append(sample)

    print("\nCross-case backward summary")
    print(
        f"{'Method':<28} {'SM':>8} {'Cases':>8} "
        f"{'Min ms':>10} {'Mean ms':>10} {'P50 ms':>10} {'Max ms':>10} "
        f"{'Mean TFLOPS':>14} {'Weighted TFLOPS':>18} {'Weighted/GPU':>14}"
    )
    for (method, config), records in grouped.items():
        times = [record.result.timing.max_ms for record in records]
        weighted_tflops = sum(
            record.result.aggregate_tflops * record.result.timing.max_ms
            for record in records
        ) / sum(times)
        mean_tflops = sum(
            record.result.aggregate_tflops for record in records
        ) / len(records)
        sm = "-" if config is None else f"{config.num_comp_sm}:{config.num_comm_sm}"
        print(
            f"{method:<28} {sm:>8} {f'{len(records)}/{total_cases}':>8} "
            f"{min(times):>10.3f} {sum(times) / len(times):>10.3f} "
            f"{median(times):>10.3f} {max(times):>10.3f} "
            f"{mean_tflops:>14.2f} {weighted_tflops:>18.2f} "
            f"{weighted_tflops / world_size:>14.2f}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark explicit-topology causal mega-ring backward"
    )
    parser.add_argument(
        "--global-seqlens",
        help="Explicit comma-separated global lengths; overrides --b/--seqlen generation",
    )
    parser.add_argument("--b", default="4", help="Comma-separated synthetic batch-size cases")
    parser.add_argument(
        "--seqlen", default="256", help="Comma-separated synthetic member-rank lengths"
    )
    parser.add_argument("--ring-sizes", default="8,4,2,1")
    parser.add_argument("--ring-starts", default="0,4,2,7")
    parser.add_argument("--qhead", type=int, default=16)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument(
        "--allgather-overlapping-heads-k-stride",
        type=int,
        default=4,
        help="KV heads per all-gather/attention overlap pipeline chunk",
    )
    parser.add_argument("--mode", choices=("causal",), default="causal")
    parser.add_argument(
        "--methods",
        default="all",
        help=f"Comma-separated methods from {METHOD_ORDER}, or all",
    )
    parser.add_argument("--sm-configs")
    parser.add_argument(
        "--zepplin-threshold",
        type=positive_int,
        default=DEFAULT_ZEPPLIN_THRESHOLD,
    )
    parser.add_argument(
        "--megatron-max-seqlen-per-rank",
        type=positive_int,
        default=8192,
    )
    parser.add_argument(
        "--magi-overlap-degree",
        type=magi_overlap_degree,
        default=2,
        help="Static MagiAttention overlap degree (1-8)",
    )
    parser.add_argument("--num-comp-sm", type=int)
    parser.add_argument("--num-comm-sm", type=int)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dq-atol", type=float, default=1.0)
    parser.add_argument("--dkv-atol", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=0.2)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    workload_cases: Sequence[HybridBenchmarkCase] | None = None,
    skip_incompatible_methods: bool = False,
) -> None:
    args = parse_args(argv)
    requested_methods = parse_methods(args.methods)
    if args.headdim != 128:
        raise SystemExit("this benchmark requires D=128")
    if (
        any(method in FUSED_MEGA_RING_METHODS for method in requested_methods)
        and args.kvhead * args.headdim != 1024
    ):
        raise SystemExit("fused mega-ring methods require KVH * D == 1024")
    if args.qhead % args.kvhead:
        raise SystemExit("qhead must be divisible by kvhead")
    if any(
        method in OVERLAPPED_ALLGATHER_METHODS for method in requested_methods
    ) and (
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
        raise SystemExit("warmup iterations must be non-negative and measured iterations positive")
    sm_configs = resolve_sm_configs(args)

    rank, world_size = init_distributed()
    parallel_pools: BackwardParallelPools | None = None
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(rank) != (9, 0):
            raise SystemExit("SM90 Hopper CUDA device is required")
        requested_methods, magi_skip_reason = resolve_magi_attention_availability(
            requested_methods, args.methods
        )
        sm_count = torch.cuda.get_device_properties(rank).multi_processor_count
        for config in sm_configs:
            if config.num_comp_sm <= 0 or config.num_comm_sm <= 0:
                raise SystemExit("backward requires positive compute and communication SM counts")
            if config.num_comp_sm + config.num_comm_sm > sm_count:
                raise SystemExit(
                    f"SM config {config.num_comp_sm}:{config.num_comm_sm} "
                    f"exceeds device SM count {sm_count}"
                )

        allgather_backend = (
            select_fa3_backend(dist.group.WORLD, require_backward=True)
            if any(
                method in BLOCK_BASELINE_METHODS
                or method == "megatron_hybrid_cp"
                for method in requested_methods
            )
            else None
        )

        if rank == 0:
            configs = ",".join(
                f"{config.num_comp_sm}:{config.num_comm_sm}" for config in sm_configs
            )
            print(
                f"Explicit-topology mega-ring causal backward: world_size={world_size}, "
                f"methods={requested_methods}, "
                f"QH={args.qhead}, KVH={args.kvhead}, D={args.headdim}, "
                "allgather_overlapping_heads_k_stride="
                f"{args.allgather_overlapping_heads_k_stride}, "
                f"sm_configs={configs}, "
                f"zepplin_threshold={args.zepplin_threshold}, "
                "megatron_max_seqlen_per_rank="
                f"{args.megatron_max_seqlen_per_rank}, "
                f"magi_overlap_degree={args.magi_overlap_degree}, "
                f"warmup={args.warmup_iters}, "
                f"iters={args.num_iters}, check={args.check}",
                flush=True,
            )
            if allgather_backend is not None:
                backend_name = (
                    "external FA3"
                    if allgather_backend == "external_fa3"
                    else "in-repo min_fa3 fallback"
                )
                print(f"Block baseline backend: {backend_name}", flush=True)
            if magi_skip_reason is not None:
                print(
                    "Skipped method magi_attention: "
                    f"{magi_skip_reason}",
                    flush=True,
                )

        if workload_cases is not None:
            if not workload_cases:
                raise SystemExit("workload_cases must not be empty")
            workloads = [
                (
                    workload_case.label,
                    list(workload_case.global_lengths),
                    list(workload_case.ring_sizes),
                    list(workload_case.ring_starts),
                )
                for workload_case in workload_cases
            ]
        elif args.global_seqlens is not None:
            workloads = [
                (
                    "explicit topology",
                    parse_int_list(args.global_seqlens, "--global-seqlens"),
                    parse_int_list(args.ring_sizes, "--ring-sizes"),
                    parse_int_list(args.ring_starts, "--ring-starts"),
                )
            ]
        else:
            workloads = []
            for batch_size, local_seqlen in make_cases(args):
                global_lengths, ring_sizes, ring_starts = make_topology(
                    batch_size, local_seqlen, args.ring_sizes, args.ring_starts
                )
                workloads.append(
                    (
                        f"synthetic B={batch_size}, local_S={local_seqlen}",
                        global_lengths,
                        ring_sizes,
                        ring_starts,
                    )
                    )

        for _label, global_lengths, ring_sizes, ring_starts in workloads:
            if any(
                method not in {"zepplin", "magi_attention"}
                for method in requested_methods
            ):
                validate_backward_metadata(
                    global_lengths, ring_sizes, ring_starts, world_size
                )
            elif "zepplin" in requested_methods and not (
                len(global_lengths) == len(ring_sizes) == len(ring_starts)
            ):
                raise SystemExit(
                    "global lengths, ring sizes, and ring starts must have the same length"
                )

        parallel_pools = make_backward_parallel_pools(
            workloads,
            requested_methods,
            rank,
            world_size,
            args.kvhead,
            args.headdim,
        )
        if rank == 0:
            all_cp_pool = parallel_pools.all_cp
            hybrid_pool = parallel_pools.hybrid
            print(
                "Reusable backward IPC pools: "
                f"cases={len(workloads)}, "
                f"all_cp_rank_capacity="
                f"{None if all_cp_pool is None else all_cp_pool.rank_capacity}, "
                f"all_cp_accum_numel="
                f"{None if all_cp_pool is None else all_cp_pool.remote_dk.data_.numel()}, "
                f"hybrid_rank_capacity="
                f"{None if hybrid_pool is None else hybrid_pool.rank_capacity}, "
                f"hybrid_accum_numel="
                f"{None if hybrid_pool is None else hybrid_pool.remote_dk.data_.numel()}",
                flush=True,
            )
            del all_cp_pool, hybrid_pool

        summary_samples: list[BackwardSummarySample] = []
        for case_index, (label, global_lengths, ring_sizes, ring_starts) in enumerate(workloads):
            active_methods, skipped_methods = compatible_methods(
                requested_methods,
                global_lengths,
                world_size,
                skip_incompatible=skip_incompatible_methods,
                zepplin_threshold=args.zepplin_threshold,
                megatron_max_seqlen_per_rank=args.megatron_max_seqlen_per_rank,
            )
            if rank == 0 and skipped_methods:
                print(f"\nSkipped methods for {label}:")
                for method, reason in skipped_methods:
                    print(f"  {method}: {reason}")
            case_results = benchmark_topology(
                args,
                rank,
                world_size,
                global_lengths,
                ring_sizes,
                ring_starts,
                sm_configs,
                label,
                active_methods,
                allgather_backend,
                parallel_pools,
            )
            if rank == 0:
                summary_samples.extend(
                    BackwardSummarySample(case_index, result)
                    for result in case_results
                )
        if rank == 0:
            _print_backward_summary(summary_samples, len(workloads), world_size)
    finally:
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        if parallel_pools is not None:
            parallel_pools.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
