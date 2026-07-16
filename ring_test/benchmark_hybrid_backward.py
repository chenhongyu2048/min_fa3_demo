"""Distributed hierarchical causal mega-ring backward benchmark."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.distributed as dist

THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
for path in (THIS_DIR, DEMO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import min_fa3_op
from allgather_attention import (
    Llama3AllGatherAttention,
    repartition_sequence_shards_to_llama3,
    select_allgather_backend,
)
from hybrid_backward_baselines import (
    VarlenAllGatherBackward,
    VarlenFa3RingBackward,
)
from mega_ring_test_min_fa3_varlen_hybrid_multi_rank import (
    SENTINEL,
    assert_all_ranks,
    hierarchical_reference,
    init_distributed,
    local_lengths_for_rank,
    make_cu_seqlens,
    make_local_qkv,
    parse_int_list,
)


METHOD_ORDER = [
    "allgather_attention",
    "llama3_allgather_attention",
    "fa3_ring",
    "mega_ring_hybrid",
]
ALL_CP_METHODS = set(METHOD_ORDER[:-1])


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


@dataclass
class MethodRun:
    prepare: Callable[[], object]
    launch: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    note: str


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


def method_incompatibility(
    method: str,
    global_lengths: list[int],
    world_size: int,
) -> str | None:
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
) -> tuple[list[str], list[tuple[str, str]]]:
    active: list[str] = []
    skipped: list[tuple[str, str]] = []
    for method in methods:
        reason = method_incompatibility(method, global_lengths, world_size)
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

    local_median = statistics.median(local_samples)
    max_median = statistics.median(max_samples)
    local_time = torch.tensor([local_median], device="cuda", dtype=torch.float64)
    gathered = [torch.empty_like(local_time) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_time)
    rank_times = [value.item() for value in gathered] if rank == 0 else None
    return TimingResult(max_median, rank_times)


def aggregate_backward_tflops(
    global_lengths: list[int], q_heads: int, head_dim: int, time_ms: float
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
) -> list[BenchmarkResult]:
    validate_backward_metadata(global_lengths, ring_sizes, ring_starts, world_size)
    device = torch.device("cuda", rank)
    if rank == 0:
        print(
            f"\nWorkload: {label}, B={len(global_lengths)}, "
            f"global_tokens={sum(global_lengths)}, global_seqlens={global_lengths}"
        )
        print(f"Hybrid rings: sizes={ring_sizes}, starts={ring_starts}")
        print(
            "Timing excludes forward preparation, owner-accumulator reset, and the "
            "distributed barrier; "
            "reported time is the median max-rank end-to-end backward op time.",
            flush=True,
        )

    baseline_runs: dict[str, MethodRun] = {}
    if any(method in ALL_CP_METHODS for method in methods):
        if allgather_backend is None:
            raise RuntimeError("all-CP baselines require a selected block backend")
        all_cp_lengths = [length // world_size for length in global_lengths]
        all_cp_total = sum(all_cp_lengths)
        all_cp_cu_host = torch.tensor(
            [0, *accumulate(all_cp_lengths)], dtype=torch.int32
        )
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
        remote_k, remote_v = make_remote_kv(
            local_k, local_v, rank, world_size, rank_capacity
        )
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
        remote_dk = min_fa3_op.TKParallelTensor(
            [accum_numel], torch.float32, rank, world_size, False
        )
        remote_dv = min_fa3_op.TKParallelTensor(
            [accum_numel], torch.float32, rank, world_size, False
        )
        completion = min_fa3_op.TKParallelTensor(
            [1], torch.int32, rank, world_size, False
        )

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
        method_runs = (
            [(None, baseline_runs[method])]
            if method in baseline_runs
            else [(config, hybrid_runs[config]) for config in sm_configs]
        )
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
                results.append(
                    BenchmarkResult(
                        method,
                        config,
                        timing,
                        aggregate_tflops,
                        check_status,
                        run.note,
                    )
                )
            dist.barrier()

    if rank == 0:
        print_results(results, world_size)
    dist.barrier()
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark hierarchical causal mega-ring backward"
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
    parser.add_argument("--mode", choices=("causal",), default="causal")
    parser.add_argument(
        "--methods",
        default="all",
        help=f"Comma-separated methods from {METHOD_ORDER}, or all",
    )
    parser.add_argument("--sm-configs")
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
    skip_incompatible_methods: bool = False,
) -> None:
    args = parse_args(argv)
    requested_methods = parse_methods(args.methods)
    if args.headdim != 128 or args.kvhead * args.headdim != 1024:
        raise SystemExit("This path requires D=128 and KVH * D == 1024")
    if args.qhead % args.kvhead:
        raise SystemExit("qhead must be divisible by kvhead")
    if args.warmup_iters < 0 or args.num_iters <= 0:
        raise SystemExit("warmup iterations must be non-negative and measured iterations positive")
    sm_configs = resolve_sm_configs(args)

    rank, world_size = init_distributed()
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(rank) != (9, 0):
            raise SystemExit("SM90 Hopper CUDA device is required")
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
            select_allgather_backend(dist.group.WORLD)
            if any(method in ALL_CP_METHODS for method in requested_methods)
            else None
        )

        if rank == 0:
            configs = ",".join(
                f"{config.num_comp_sm}:{config.num_comm_sm}" for config in sm_configs
            )
            print(
                f"Hierarchical mega-ring causal backward: world_size={world_size}, "
                f"methods={requested_methods}, "
                f"QH={args.qhead}, KVH={args.kvhead}, D={args.headdim}, "
                f"sm_configs={configs}, warmup={args.warmup_iters}, "
                f"iters={args.num_iters}, check={args.check}",
                flush=True,
            )
            if allgather_backend is not None:
                backend_name = (
                    "external FA3"
                    if allgather_backend == "external_fa3"
                    else "local min_fa3 fallback"
                )
                print(f"All-CP block backend: {backend_name}", flush=True)

        if args.global_seqlens is not None:
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

        for label, global_lengths, ring_sizes, ring_starts in workloads:
            active_methods, skipped_methods = compatible_methods(
                requested_methods,
                global_lengths,
                world_size,
                skip_incompatible=skip_incompatible_methods,
            )
            if rank == 0 and skipped_methods:
                print(f"\nSkipped methods for {label}:")
                for method, reason in skipped_methods:
                    print(f"  {method}: {reason}")
            benchmark_topology(
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
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
