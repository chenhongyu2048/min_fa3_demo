"""
Benchmark: min_fa3 parallel_remote_load payload time and effective bandwidth

Examples:
    torchrun --nproc_per_node=2 benchmark_parallel_remote_load.py
    torchrun --nproc_per_node=2 benchmark_parallel_remote_load.py --shape 4096x4096,8192x4096 --src-rank 0
    torchrun --nproc_per_node=4 benchmark_parallel_remote_load.py --shape 4096x4096 --src-rank 1 --num-blocks 64
"""

from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist

import min_fa3_op


@dataclass(frozen=True)
class Case:
    rows: int
    cols: int


@dataclass(frozen=True)
class Result:
    payload_mib: float
    time_ms: float
    bandwidth_gbps: float
    num_blocks: int


def parse_shape_spec(spec: str) -> list[Case]:
    cases: list[Case] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise SystemExit("--shape must provide comma-separated ROWSxCOLS cases, for example 4096x4096,8192x4096")
        rows_str, cols_str = token.split("x", 1)
        cases.append(Case(rows=int(rows_str), cols=int(cols_str)))
    if not cases:
        raise SystemExit("--shape must provide at least one ROWSxCOLS case")
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark min_fa3 parallel_remote_load time and effective payload bandwidth"
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="4096x4096",
        help="Comma-separated ROWSxCOLS cases. Both dimensions must be multiples of 128.",
    )
    parser.add_argument("--src-rank", type=int, default=0, help="Source rank to read from.")
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Fixed thread-block count for the remote-load kernel. Defaults to the current device SM count.",
    )
    parser.add_argument("--num-iters", type=int, default=100, help="Timing iterations.")
    parser.add_argument("--warmup-iters", type=int, default=20, help="Warmup iterations.")
    return parser.parse_args()


def init_distributed() -> tuple[int, int]:
    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit("Run this benchmark with torchrun so LOCAL_RANK and LOCAL_WORLD_SIZE are set")

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    if dist.get_world_size() != local_world_size:
        raise SystemExit(
            "ThunderKittens remote-load demo is single-node only: "
            f"world_size={dist.get_world_size()}, local_world_size={local_world_size}"
        )
    if local_world_size < 2:
        raise SystemExit("parallel_remote_load benchmark requires at least 2 local ranks")

    return local_rank, local_world_size


def make_rank_local_tensor(rows: int, cols: int, local_rank: int) -> torch.Tensor:
    tensor = torch.empty((rows, cols), device="cuda", dtype=torch.bfloat16)
    tensor.fill_(float(local_rank))
    return tensor.contiguous()


def prepare_case_tensors(
    case: Case,
    num_blocks: int | None,
    local_rank: int,
    local_world_size: int,
) -> tuple[min_fa3_op.TKParallelTensor, torch.Tensor, int]:
    local_tensor = make_rank_local_tensor(case.rows, case.cols, local_rank)
    output = torch.empty_like(local_tensor)
    input_tk = min_fa3_op.create_parallel_tensor(
        local_tensor,
        local_rank=local_rank,
        local_world_size=local_world_size,
    )
    resolved_num_blocks = num_blocks
    if resolved_num_blocks is None:
        resolved_num_blocks = torch.cuda.get_device_properties(local_rank).multi_processor_count
    return input_tk, output, resolved_num_blocks


def median_time_ms(fn: Callable[[], None], warmup_iters: int, num_iters: int) -> float:
    for _ in range(warmup_iters):
        dist.barrier()
        fn()
        torch.cuda.synchronize()
    dist.barrier()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for i in range(num_iters):
        dist.barrier()
        start_events[i].record()
        fn()
        end_events[i].record()
        torch.cuda.synchronize()
    dist.barrier()
    return statistics.median(s.elapsed_time(e) for s, e in zip(start_events, end_events))


def gather_time_ms(local_time_ms: float, local_world_size: int) -> list[float]:
    local_time = torch.tensor([local_time_ms], device="cuda", dtype=torch.float64)
    gathered = [torch.empty_like(local_time) for _ in range(local_world_size)]
    dist.all_gather(gathered, local_time)
    return [tensor.item() for tensor in gathered]


def summarize_case(
    case: Case,
    local_time_ms: float,
    num_blocks: int,
    src_rank: int,
    local_rank: int,
    local_world_size: int,
) -> Result | None:
    times_by_rank = gather_time_ms(local_time_ms, local_world_size)
    if local_rank != 0:
        return None

    receiver_ranks = [rank for rank in range(local_world_size) if rank != src_rank]
    measured_ranks = receiver_ranks if receiver_ranks else list(range(local_world_size))
    time_ms = max(times_by_rank[rank] for rank in measured_ranks)
    payload_bytes = case.rows * case.cols * torch.tensor([], dtype=torch.bfloat16).element_size()
    bandwidth_gbps = payload_bytes / (time_ms * 1e-3) / 1e9
    payload_mib = payload_bytes / (1024**2)
    return Result(
        payload_mib=payload_mib,
        time_ms=time_ms,
        bandwidth_gbps=bandwidth_gbps,
        num_blocks=num_blocks,
    )


def print_results(results: list[tuple[Case, Result]], src_rank: int, local_world_size: int) -> None:
    print()
    print("=" * 88)
    print(
        f"parallel_remote_load benchmark "
        f"(world_size={local_world_size}, src_rank={src_rank}, time_ms=slowest non-source-rank median)"
    )
    print("=" * 88)
    print(f"{'Shape':>16} {'Payload(MiB)':>14} {'Blocks':>8} {'Time(ms)':>12} {'Bandwidth(GB/s)':>18}")
    print("-" * 88)
    for case, result in results:
        print(
            f"{case.rows}x{case.cols: <7} "
            f"{result.payload_mib:14.2f} "
            f"{result.num_blocks:8d} "
            f"{result.time_ms:12.4f} "
            f"{result.bandwidth_gbps:18.2f}"
        )
    print("-" * 88)


if __name__ == "__main__":
    args = parse_args()
    cases = parse_shape_spec(args.shape)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (9, 0):
        raise SystemExit(f"This demo requires SM90 Hopper, got {(major, minor)}")
    if args.num_blocks is not None and args.num_blocks <= 0:
        raise SystemExit(f"--num-blocks must be positive when provided, got num_blocks={args.num_blocks}")
    if args.num_iters <= 0:
        raise SystemExit(f"--num-iters must be positive, got num_iters={args.num_iters}")
    if args.warmup_iters < 0:
        raise SystemExit(f"--warmup-iters must be non-negative, got warmup_iters={args.warmup_iters}")

    local_rank, local_world_size = init_distributed()
    if args.src_rank < 0 or args.src_rank >= local_world_size:
        raise SystemExit(
            f"--src-rank must be in [0, {local_world_size}), got src_rank={args.src_rank}"
        )

    results: list[tuple[Case, Result]] = []
    try:
        for case in cases:
            if case.rows <= 0 or case.cols <= 0:
                raise SystemExit(f"Each shape dimension must be positive, got {case.rows}x{case.cols}")
            if case.rows % 128 != 0 or case.cols % 128 != 0:
                raise SystemExit(
                    f"This demo requires rows and cols to be multiples of 128, got {case.rows}x{case.cols}"
                )

            input_tk, output, resolved_num_blocks = prepare_case_tensors(
                case,
                args.num_blocks,
                local_rank,
                local_world_size,
            )
            local_time_ms = median_time_ms(
                lambda: min_fa3_op.parallel_remote_load(
                    input_tensor=input_tk,
                    src_rank=args.src_rank,
                    output=output,
                    num_blocks=resolved_num_blocks,
                ),
                args.warmup_iters,
                args.num_iters,
            )
            summary = summarize_case(
                case,
                local_time_ms,
                resolved_num_blocks,
                args.src_rank,
                local_rank,
                local_world_size,
            )
            if summary is not None:
                results.append((case, summary))

            dist.barrier()
            input_tk = None
            output = None
            torch.cuda.empty_cache()

        if local_rank == 0:
            print_results(results, args.src_rank, local_world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
