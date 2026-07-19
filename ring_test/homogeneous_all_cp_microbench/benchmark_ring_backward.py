"""Torchrun benchmark for causal all-CP varlen backward.

The script compares two all-gather baselines, a complete Python/NCCL zigzag
ring built from the local min_fa3 varlen backward, and the fused mega-ring
backward kernel. Forward preparation, allocations, and fused remote-workspace
resets are outside the CUDA-event timing interval.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import min_fa3_op

from allgather_attention import (
    AllGatherAttention,
    Llama3AllGatherAttention,
    repartition_sequence_shards_to_llama3,
    select_fa3_backend,
)
from hybrid_backward_baselines import VarlenFa3RingBackward
from ring_common import raise_if_any_rank_failed


METHOD_ORDER = [
    "allgather_attention",
    "llama3_allgather_attention",
    "min_varlen_python_ring",
    "min_varlen_mega_ring",
]
SM_SWEEP_METHODS = {"min_varlen_mega_ring"}


@dataclass(frozen=True)
class Case:
    batch_size: int
    seqlen: int
    q_heads: int
    kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class SmConfig:
    num_comp_sm: int
    num_comm_sm: int


@dataclass
class MethodRun:
    name: str
    prepare_fn: Callable[[], None]
    timing_fn: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    note: str = ""


@dataclass
class TimingResult:
    max_time_ms: float
    rank_times_ms: list[float] | None


@dataclass
class Result:
    time_ms: float | None
    aggregate_tflops: float | None
    avg_gpu_tflops: float | None
    check: str
    note: str = ""
    rank_times_ms: list[float] | None = None


def parse_seqlen_spec(spec: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise SystemExit("--seqlen must provide at least one case")
    return values


def parse_batch_spec(spec: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise SystemExit("--b must provide at least one case")
    return values


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


def parse_sm_config_spec(spec: str) -> list[SmConfig]:
    configs: list[SmConfig] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise SystemExit(f"invalid SM config '{token}', expected num_comp_sm:num_comm_sm")
        try:
            configs.append(SmConfig(int(parts[0]), int(parts[1])))
        except ValueError as exc:
            raise SystemExit(f"invalid SM config '{token}', expected two integers") from exc
    if not configs:
        raise SystemExit("--sm-configs must provide at least one configuration")
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed causal mega-ring varlen backward benchmark")
    parser.add_argument(
        "--b", type=str, default="1,1,1",
        help="Comma-separated batch sizes per rank, paired one-to-one with --seqlen",
    )
    parser.add_argument(
        "--seqlen", "--seqlens", dest="seqlen", type=str, default="256,512,1024",
        help="Comma-separated local sequence lengths",
    )
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument(
        "--allgather-overlapping-heads-k-stride",
        type=int,
        default=1,
        help="KV heads per all-gather/attention overlap pipeline chunk",
    )
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--num-comp-sm", type=int, default=64)
    parser.add_argument("--num-comm-sm", type=int, default=8)
    parser.add_argument(
        "--sm-configs",
        type=str,
        default=None,
        help="Comma-separated compute:communication SM pairs; overrides the individual SM arguments",
    )
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float, default=0.3)
    parser.add_argument("--rtol", type=float, default=0.3)
    return parser.parse_args()


def init_distributed() -> tuple[int, int]:
    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit("Run with torchrun so LOCAL_RANK and LOCAL_WORLD_SIZE are set")
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    if dist.get_world_size() != local_world_size:
        raise SystemExit(
            "This benchmark is single-node only because TKParallelTensor uses local IPC: "
            f"world_size={dist.get_world_size()}, local_world_size={local_world_size}"
        )
    return local_rank, local_world_size


def cuda_barrier() -> None:
    dist.barrier(device_ids=[torch.cuda.current_device()])


def make_cu_seqlens(case: Case, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.arange(
        0,
        (case.batch_size + 1) * case.seqlen,
        case.seqlen,
        dtype=torch.int32,
    )
    return host.to(device=device), host


def make_inputs(
    case: Case,
    local_rank: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + local_rank * 1009 + case.seqlen)
    total_tokens = case.batch_size * case.seqlen

    def randn(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator) * 0.25
        ).to(torch.bfloat16).contiguous()

    q = randn((total_tokens, case.q_heads, case.head_dim))
    k = randn((total_tokens, case.kv_heads, case.head_dim))
    v = randn((total_tokens, case.kv_heads, case.head_dim))
    dout = randn((total_tokens, case.q_heads, case.head_dim))
    return q, k, v, dout


def make_mega_parallel_tensors(
    k: torch.Tensor,
    v: torch.Tensor,
    local_rank: int,
    local_world_size: int,
) -> tuple[min_fa3_op.TKParallelTensor, min_fa3_op.TKParallelTensor]:
    full_shape = [local_world_size * k.size(0), k.size(1), k.size(2)]
    remote_k = min_fa3_op.TKParallelTensor(full_shape, torch.bfloat16, local_rank, local_world_size, False)
    remote_v = min_fa3_op.TKParallelTensor(full_shape, torch.bfloat16, local_rank, local_world_size, False)
    remote_k.data_.zero_()
    remote_v.data_.zero_()
    start = local_rank * k.size(0)
    remote_k.data_[start : start + k.size(0)].copy_(k)
    remote_v.data_[start : start + v.size(0)].copy_(v)
    return remote_k, remote_v


def build_method_runs(
    case: Case,
    local_rank: int,
    local_world_size: int,
    sm_config: SmConfig,
    seed: int,
    allgather_backend: str,
    methods: list[str],
    overlapping_heads_k_stride: int = 1,
) -> dict[str, MethodRun]:
    q, local_k, local_v, dout = make_inputs(case, local_rank, seed)
    cu, cu_host = make_cu_seqlens(case, q.device)
    global_seqlens_host = torch.full(
        (case.batch_size,), case.seqlen * local_world_size, dtype=torch.int32
    )
    ring_sizes_host = torch.full(
        (case.batch_size,), local_world_size, dtype=torch.int32
    )
    ring_starts_host = torch.zeros(case.batch_size, dtype=torch.int32)
    allgather_attention = AllGatherAttention(
        dist.group.WORLD,
        q,
        local_k,
        local_v,
        case.batch_size,
        case.seqlen,
        True,
        allgather_backend,
        heads_k_stride=overlapping_heads_k_stride,
        enable_backward=True,
    )
    allgather_run = MethodRun(
        "allgather_attention",
        allgather_attention.forward,
        lambda: allgather_attention.backward(dout),
        allgather_attention.note,
    )
    llama3_run = None
    if "llama3_allgather_attention" in methods:
        global_seqlens = [case.seqlen * local_world_size] * case.batch_size
        llama3_q = repartition_sequence_shards_to_llama3(
            dist.group.WORLD, q, global_seqlens, True
        )
        llama3_k = repartition_sequence_shards_to_llama3(
            dist.group.WORLD, local_k, global_seqlens, True
        )
        llama3_v = repartition_sequence_shards_to_llama3(
            dist.group.WORLD, local_v, global_seqlens, True
        )
        llama3_dout = repartition_sequence_shards_to_llama3(
            dist.group.WORLD, dout, global_seqlens, True
        )
        llama3_attention = Llama3AllGatherAttention(
            dist.group.WORLD,
            llama3_q,
            llama3_k,
            llama3_v,
            global_seqlens,
            True,
            allgather_backend,
            heads_k_stride=overlapping_heads_k_stride,
            enable_backward=True,
        )
        llama3_run = MethodRun(
            "llama3_allgather_attention",
            llama3_attention.forward,
            lambda: llama3_attention.backward(llama3_dout),
            llama3_attention.note,
        )
    remote_k, remote_v = make_mega_parallel_tensors(local_k, local_v, local_rank, local_world_size)
    torch.cuda.synchronize()
    cuda_barrier()

    out, lse = min_fa3_op.forward_varlen_mega_ring(
        q,
        remote_k.data_,
        remote_v.data_,
        cu,
        cu,
        case.seqlen,
        case.seqlen,
        True,
        cu_seqlens_q_host=cu_host,
        cu_seqlens_k_host=cu_host,
        remote_k=remote_k,
        remote_v=remote_v,
        num_comp_sm=sm_config.num_comp_sm,
        num_comm_sm=sm_config.num_comm_sm,
        global_seqlens_host=global_seqlens_host,
        ring_sizes_host=ring_sizes_host,
        ring_starts_host=ring_starts_host,
        return_lse=True,
    )
    torch.cuda.synchronize()
    cuda_barrier()

    python_ring = VarlenFa3RingBackward(
        dist.group.WORLD,
        q,
        local_k,
        local_v,
        dout,
        [case.seqlen] * case.batch_size,
        "min_fa3",
    )
    total_tokens = case.batch_size * case.seqlen
    total_k_padded = ((total_tokens + case.batch_size * 128 + 127) // 128) * 128
    accum_numel = case.kv_heads * total_k_padded * case.head_dim
    remote_dk = min_fa3_op.TKParallelTensor(
        [accum_numel], torch.float32, local_rank, local_world_size, False
    )
    remote_dv = min_fa3_op.TKParallelTensor(
        [accum_numel], torch.float32, local_rank, local_world_size, False
    )
    remote_completion = min_fa3_op.TKParallelTensor(
        [1], torch.int32, local_rank, local_world_size, False
    )

    def prepare_mega() -> None:
        remote_dk.data_.zero_()
        remote_dv.data_.zero_()
        remote_completion.data_.zero_()

    def run_mega() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return min_fa3_op.backward_varlen_mega_ring(
            dout,
            q,
            remote_k.data_,
            remote_v.data_,
            out,
            lse,
            cu,
            cu,
            case.seqlen,
            case.seqlen,
            cu_seqlens_q_host=cu_host,
            cu_seqlens_k_host=cu_host,
            remote_k=remote_k,
            remote_v=remote_v,
            remote_dk_accum=remote_dk,
            remote_dv_accum=remote_dv,
            remote_dkv_completion=remote_completion,
            num_comp_sm=sm_config.num_comp_sm,
            num_comm_sm=sm_config.num_comm_sm,
            global_seqlens_host=global_seqlens_host,
            ring_sizes_host=ring_sizes_host,
            ring_starts_host=ring_starts_host,
        )

    runs = {
        "allgather_attention": allgather_run,
        "min_varlen_python_ring": MethodRun(
            "min_varlen_python_ring",
            python_ring.forward,
            python_ring.backward,
            "min_fa3 block kernels + NCCL P2P",
        ),
        "min_varlen_mega_ring": MethodRun(
            "min_varlen_mega_ring",
            prepare_mega,
            run_mega,
            "fused; remote workspace reset excluded",
        ),
    }
    if llama3_run is not None:
        runs["llama3_allgather_attention"] = llama3_run
    return runs


def prepare_method(run: MethodRun) -> None:
    run.prepare_fn()
    torch.cuda.synchronize()
    cuda_barrier()


def measure_distributed_ms(
    run: MethodRun,
    warmup_iters: int,
    num_iters: int,
) -> TimingResult:
    for _ in range(warmup_iters):
        prepare_method(run)
        run.timing_fn()
    torch.cuda.synchronize()
    cuda_barrier()

    local_samples: list[float] = []
    max_samples: list[float] = []
    for _ in range(num_iters):
        prepare_method(run)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run.timing_fn()
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)

        max_elapsed = torch.tensor([elapsed_ms], device="cuda", dtype=torch.float64)
        dist.all_reduce(max_elapsed, op=dist.ReduceOp.MAX)
        local_samples.append(elapsed_ms)
        max_samples.append(max_elapsed.item())
    cuda_barrier()

    local_avg = sum(local_samples) / len(local_samples)
    max_avg = sum(max_samples) / len(max_samples)
    local_avg_tensor = torch.tensor([local_avg], device="cuda", dtype=torch.float64)
    gathered = [torch.empty_like(local_avg_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_avg_tensor)
    rank_times = [value.item() for value in gathered] if dist.get_rank() == 0 else None
    return TimingResult(max_avg, rank_times)


def aggregate_backward_tflops(case: Case, world_size: int, time_ms: float) -> float:
    global_seqlen = world_size * case.seqlen
    flops = (
        5.0
        * case.batch_size
        * global_seqlen
        * global_seqlen
        * case.q_heads
        * case.head_dim
    )
    return flops / (time_ms * 1.0e-3) / 1.0e12


def check_gradients(
    method: str,
    run: MethodRun,
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    atol: float,
    rtol: float,
) -> str:
    local_error = None
    try:
        prepare_method(run)
        actual = run.timing_fn()
        torch.cuda.synchronize()
        for name, actual_grad, reference_grad in zip(("dQ", "dK", "dV"), actual, reference):
            torch.testing.assert_close(
                actual_grad.float(), reference_grad.float(), atol=atol, rtol=rtol
            )
    except Exception as exc:
        local_error = f"{method}: {exc}"
    raise_if_any_rank_failed(local_error, dist.group.WORLD)
    return "ok"


def run_case(
    case: Case,
    methods: list[str],
    local_rank: int,
    local_world_size: int,
    sm_config: SmConfig,
    args: argparse.Namespace,
) -> dict[str, Result]:
    runs = build_method_runs(
        case,
        local_rank,
        local_world_size,
        sm_config,
        args.seed,
        args.allgather_backend,
        methods,
        overlapping_heads_k_stride=args.allgather_overlapping_heads_k_stride,
    )
    reference = None
    llama3_reference = None
    if args.check:
        reference_run = runs["min_varlen_python_ring"]
        prepare_method(reference_run)
        reference = reference_run.timing_fn()
        torch.cuda.synchronize()
        cuda_barrier()
        if "llama3_allgather_attention" in methods:
            global_seqlens = [case.seqlen * local_world_size] * case.batch_size
            llama3_reference = tuple(
                repartition_sequence_shards_to_llama3(
                    dist.group.WORLD, gradient, global_seqlens, True
                )
                for gradient in reference
            )

    results: dict[str, Result] = {}
    for method in methods:
        run = runs[method]
        try:
            timing = measure_distributed_ms(run, args.warmup_iters, args.num_iters)
            aggregate_tflops = aggregate_backward_tflops(case, local_world_size, timing.max_time_ms)
            check = "skip"
            if args.check and method == "min_varlen_python_ring":
                check = "reference"
            elif args.check and reference is not None:
                expected = (
                    llama3_reference
                    if method == "llama3_allgather_attention"
                    else reference
                )
                check = check_gradients(method, run, expected, args.atol, args.rtol)
            results[method] = Result(
                timing.max_time_ms,
                aggregate_tflops,
                aggregate_tflops / local_world_size,
                check,
                run.note,
                timing.rank_times_ms,
            )
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            results[method] = Result(None, None, None, "oom", str(exc).splitlines()[0])
        except Exception as exc:
            results[method] = Result(None, None, None, "error", str(exc))
        cuda_barrier()
    return results


def print_results(case: Case, methods: list[str], results: dict[str, Result]) -> None:
    baseline = results.get("min_varlen_python_ring")
    print(
        f"\nB={case.batch_size}, local_S={case.seqlen}, QH={case.q_heads}, "
        f"KVH={case.kv_heads}, D={case.head_dim}, mode=causal"
    )
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for method in methods:
        result = results[method]
        if result.time_ms is None:
            time_s = "N/A"
        elif result.rank_times_ms is None:
            time_s = f"max_across_ranks={result.time_ms:.3f}"
        else:
            rank_s = ", ".join(
                f"t{rank}={time_ms:.3f}" for rank, time_ms in enumerate(result.rank_times_ms)
            )
            time_s = f"{rank_s} | max_across_ranks={result.time_ms:.3f}"
        aggregate_s = "N/A" if result.aggregate_tflops is None else f"{result.aggregate_tflops:.1f}"
        per_gpu_s = "N/A" if result.avg_gpu_tflops is None else f"{result.avg_gpu_tflops:.1f}"
        if result.time_ms is None or baseline is None or baseline.time_ms is None:
            speedup_s = "N/A"
        else:
            speedup_s = f"{baseline.time_ms / result.time_ms:.3f}x"
        rows.append((method, time_s, aggregate_s, per_gpu_s, speedup_s, result.check, result.note))

    method_width = max((25, *(len(row[0]) for row in rows)))
    time_width = max((64, *(len(row[1]) for row in rows)))
    print(
        f"{'Method':<{method_width}} {'Time ms':<{time_width}} {'Agg TFLOPS':>11} "
        f"{'Avg/GPU':>9} {'vs Python':>10} {'Check':>10}  Note"
    )
    for method, time_s, aggregate_s, per_gpu_s, speedup_s, check, note in rows:
        print(
            f"{method:<{method_width}} {time_s:<{time_width}} {aggregate_s:>11} "
            f"{per_gpu_s:>9} {speedup_s:>10} {check:>10}  {note}"
        )


def validate_args(
    args: argparse.Namespace,
    cases: list[Case],
    sm_configs: list[SmConfig],
    local_world_size: int,
) -> None:
    invalid_batches = [case.batch_size for case in cases if case.batch_size <= 0]
    if invalid_batches:
        raise SystemExit(f"--b values must be positive, got {invalid_batches}")
    seqlens = [case.seqlen for case in cases]
    invalid = [seqlen for seqlen in seqlens if seqlen <= 0 or seqlen % 256 != 0]
    if invalid:
        raise SystemExit(
            "causal mega-ring backward requires positive local sequence lengths divisible by 256; "
            f"invalid lengths: {invalid}"
        )
    if args.headdim != 128:
        raise SystemExit(f"This benchmark requires D=128, got {args.headdim}")
    if args.qhead % args.kvhead != 0:
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
    if args.kvhead * args.headdim != 1024:
        raise SystemExit("mega-ring communication requires kvhead * headdim == 1024")
    if not 1 <= local_world_size <= 8:
        raise SystemExit(f"mega-ring backward requires world_size in [1, 8], got {local_world_size}")
    if args.warmup_iters < 0 or args.num_iters <= 0:
        raise SystemExit("--warmup-iters must be non-negative and --num-iters must be positive")
    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    for config in sm_configs:
        if config.num_comp_sm <= 0 or config.num_comm_sm <= 0:
            raise SystemExit("mega-ring backward requires positive compute and communication SM counts")
        if config.num_comp_sm + config.num_comm_sm > sm_count:
            raise SystemExit(
                f"SM config {config.num_comp_sm}:{config.num_comm_sm} exceeds device SM count {sm_count}"
            )


def make_cases(args: argparse.Namespace) -> list[Case]:
    batches = parse_batch_spec(args.b)
    seqlens = parse_seqlen_spec(args.seqlen)
    if len(batches) != len(seqlens):
        raise SystemExit(
            f"--b and --seqlen must contain the same number of cases, got {len(batches)} and {len(seqlens)}"
        )
    return [
        Case(batch, seqlen, args.qhead, args.kvhead, args.headdim)
        for batch, seqlen in zip(batches, seqlens)
    ]


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    cases = make_cases(args)
    sm_configs = (
        parse_sm_config_spec(args.sm_configs)
        if args.sm_configs is not None
        else [SmConfig(args.num_comp_sm, args.num_comm_sm)]
    )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    local_rank, local_world_size = init_distributed()
    try:
        if torch.cuda.get_device_capability() != (9, 0):
            raise SystemExit(f"This benchmark requires SM90, got {torch.cuda.get_device_capability()}")
        args.allgather_backend = select_fa3_backend(
            dist.group.WORLD, require_backward=True
        )
        validate_args(args, cases, sm_configs, local_world_size)
        if local_rank == 0:
            configs = ",".join(f"{cfg.num_comp_sm}:{cfg.num_comm_sm}" for cfg in sm_configs)
            print(
                f"Config: world_size={local_world_size}, methods={methods}, B={args.b}, "
                f"seqlen={args.seqlen}, QH={args.qhead}, KVH={args.kvhead}, D={args.headdim}, "
                "allgather_overlapping_heads_k_stride="
                f"{args.allgather_overlapping_heads_k_stride}, "
                f"sm_configs={configs}, "
                f"warmup={args.warmup_iters}, iters={args.num_iters}, "
                f"check={args.check}"
            )
            print("Timing excludes forward preparation, allocations, and fused remote-workspace reset.")
            print("Agg TFLOPS uses 10 FLOPs per visible score for causal backward.")

        for config_index, sm_config in enumerate(sm_configs):
            config_methods = [
                method
                for method in methods
                if config_index == 0 or method in SM_SWEEP_METHODS
            ]
            if not config_methods:
                continue
            if local_rank == 0:
                print(
                    f"\nSM config: num_comp_sm={sm_config.num_comp_sm}, "
                    f"num_comm_sm={sm_config.num_comm_sm}",
                    flush=True,
                )
            for case in cases:
                if local_rank == 0:
                    print(
                        f"\nRunning B={case.batch_size}, local_S={case.seqlen}, causal=True",
                        flush=True,
                    )
                results = run_case(
                    case,
                    config_methods,
                    local_rank,
                    local_world_size,
                    sm_config,
                    args,
                )
                if local_rank == 0:
                    print_results(case, config_methods, results)
                cuda_barrier()
                torch.cuda.empty_cache()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
