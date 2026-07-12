"""Distributed hierarchical hybrid mega-ring forward benchmark."""

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
from mega_ring_test_min_fa3_varlen_hybrid_multi_rank import (
    hierarchical_reference,
    local_lengths_for_rank,
    make_cu_seqlens,
    make_local_qkv,
    parse_int_list,
)


@dataclass(frozen=True)
class SmConfig:
    num_comp_sm: int
    num_comm_sm: int


@dataclass(frozen=True)
class TimingResult:
    local_ms: float
    max_ms: float
    rank_times_ms: list[float] | None


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


def init_distributed() -> tuple[int, int]:
    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit("Run this benchmark with torchrun")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size not in (2, 4, 8):
        raise SystemExit(f"hierarchical mega ring requires 2, 4, or 8 ranks, got {world_size}")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", rank))
    if dist.get_world_size() != world_size:
        raise SystemExit("This benchmark requires a single-node torchrun process group")
    return rank, world_size


def cuda_barrier() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def validate_metadata(
    global_lengths: list[int], ring_sizes: list[int], ring_starts: list[int], world_size: int, mode: str
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


def aggregate_score_count(global_lengths: list[int], is_causal: bool) -> int:
    if is_causal:
        return sum(length * (length + 1) // 2 for length in global_lengths)
    return sum(length * length for length in global_lengths)


def aggregate_tflops(
    global_lengths: list[int], q_heads: int, head_dim: int, is_causal: bool, time_ms: float
) -> float:
    flops = 4 * aggregate_score_count(global_lengths, is_causal) * q_heads * head_dim
    return float(flops) / (time_ms * 1e-3) / 1e12


def raise_if_any_rank_failed(local_error: str | None) -> None:
    failed = torch.tensor([local_error is not None], device="cuda", dtype=torch.int32)
    dist.all_reduce(failed)
    if failed.item() == 0:
        return
    if local_error is not None:
        raise AssertionError(local_error)
    raise AssertionError("another rank failed hierarchical output validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark hierarchical hybrid mega-ring forward")
    parser.add_argument("--global-seqlens", required=True)
    parser.add_argument("--ring-sizes", required=True)
    parser.add_argument("--ring-starts", required=True)
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--mode", choices=("noncausal", "causal", "both"), default="causal")
    parser.add_argument("--sm-configs", default="128:4,124:8,120:12,116:16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=40)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("--rtol", type=float, default=2e-1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("SM90 Hopper CUDA device is required")
    if args.headdim != 128 or args.kvhead * args.headdim != 1024:
        raise SystemExit("hierarchical communication requires D=128 and KVH * D == 1024")
    if args.qhead % args.kvhead:
        raise SystemExit("qhead must be divisible by kvhead")
    if args.warmup_iters < 0 or args.num_iters <= 0:
        raise SystemExit("warmup iterations must be non-negative and measured iterations must be positive")

    rank, world_size = init_distributed()
    try:
        global_lengths = parse_int_list(args.global_seqlens, "--global-seqlens")
        ring_sizes = parse_int_list(args.ring_sizes, "--ring-sizes")
        ring_starts = parse_int_list(args.ring_starts, "--ring-starts")
        validate_metadata(global_lengths, ring_sizes, ring_starts, world_size, args.mode)
        sm_configs = parse_sm_configs(args.sm_configs)
        sm_count = torch.cuda.get_device_properties(rank).multi_processor_count
        for config in sm_configs:
            if config.num_comp_sm <= 0 or config.num_comm_sm <= 0:
                raise SystemExit("hierarchical benchmark requires positive compute and communication SM counts")
            if config.num_comp_sm + config.num_comm_sm > sm_count:
                raise SystemExit(
                    f"SM config {config.num_comp_sm}:{config.num_comm_sm} exceeds device SM count {sm_count}"
                )

        torch.manual_seed(args.seed + rank)
        all_rank_lengths = [
            local_lengths_for_rank(global_lengths, ring_sizes, ring_starts, source_rank)
            for source_rank in range(world_size)
        ]
        local_lengths = all_rank_lengths[rank]
        local_total = sum(local_lengths)
        device = torch.device("cuda", rank)
        cu_seqlens, cu_seqlens_host = make_cu_seqlens(local_lengths, device)
        q, local_k, local_v = make_local_qkv(
            local_total,
            args.qhead,
            args.kvhead,
            args.headdim,
            rank,
            False,
            device,
            base_seed=args.seed,
        )

        capacity = torch.tensor([local_total], device=device, dtype=torch.int32)
        dist.all_reduce(capacity, op=dist.ReduceOp.MAX)
        rank_capacity = int(capacity.item())
        arena_shape = [world_size * rank_capacity, args.kvhead, args.headdim]
        remote_k = min_fa3_op.TKParallelTensor(arena_shape, torch.bfloat16, rank, world_size, False)
        remote_v = min_fa3_op.TKParallelTensor(arena_shape, torch.bfloat16, rank, world_size, False)
        k, v = remote_k.data_, remote_v.data_
        k.zero_()
        v.zero_()
        owner_begin = rank * rank_capacity
        k[owner_begin:owner_begin + local_total].copy_(local_k)
        v[owner_begin:owner_begin + local_total].copy_(local_v)

        gathered_k = None
        gathered_v = None
        if args.check:
            padded_k = torch.zeros(
                (rank_capacity, args.kvhead, args.headdim), device=device, dtype=torch.bfloat16
            )
            padded_v = torch.zeros_like(padded_k)
            padded_k[:local_total].copy_(local_k)
            padded_v[:local_total].copy_(local_v)
            gathered_k_parts = [torch.empty_like(padded_k) for _ in range(world_size)]
            gathered_v_parts = [torch.empty_like(padded_v) for _ in range(world_size)]
            dist.all_gather(gathered_k_parts, padded_k)
            dist.all_gather(gathered_v_parts, padded_v)
            gathered_k = torch.stack(gathered_k_parts)
            gathered_v = torch.stack(gathered_v_parts)

        cuda_barrier()

        global_host = torch.tensor(global_lengths, dtype=torch.int32)
        ring_sizes_host = torch.tensor(ring_sizes, dtype=torch.int32)
        ring_starts_host = torch.tensor(ring_starts, dtype=torch.int32)
        max_local_len = max(max(lengths) for lengths in all_rank_lengths)
        modes = {
            "noncausal": [False],
            "causal": [True],
            "both": [False, True],
        }[args.mode]

        if rank == 0:
            print(
                f"world_size={world_size}, global_seqlens={global_lengths}, "
                f"ring_sizes={ring_sizes}, ring_starts={ring_starts}, "
                f"QH={args.qhead}, KVH={args.kvhead}, D={args.headdim}",
                flush=True,
            )

        for is_causal in modes:
            for config in sm_configs:
                def launch() -> tuple[torch.Tensor, torch.Tensor]:
                    return min_fa3_op.forward_varlen_mega_ring(
                        q,
                        k,
                        v,
                        cu_seqlens,
                        cu_seqlens,
                        max_local_len,
                        max_local_len,
                        is_causal,
                        cu_seqlens_q_host=cu_seqlens_host,
                        cu_seqlens_k_host=cu_seqlens_host,
                        remote_k=remote_k,
                        remote_v=remote_v,
                        num_comp_sm=config.num_comp_sm,
                        num_comm_sm=config.num_comm_sm,
                        global_seqlens_host=global_host,
                        ring_sizes_host=ring_sizes_host,
                        ring_starts_host=ring_starts_host,
                        return_lse=True,
                    )

                timing = measure_distributed_ms(
                    launch, args.warmup_iters, args.num_iters, rank
                )
                agg_tflops = aggregate_tflops(
                    global_lengths, args.qhead, args.headdim, is_causal, timing.max_ms
                )
                check_status = "skip"
                if args.check:
                    out, lse = launch()
                    torch.cuda.synchronize()
                    expected_out, expected_lse = hierarchical_reference(
                        q,
                        gathered_k,
                        gathered_v,
                        all_rank_lengths,
                        cu_seqlens_host,
                        global_lengths,
                        ring_sizes,
                        ring_starts,
                        rank,
                        is_causal,
                    )
                    local_error = None
                    try:
                        torch.testing.assert_close(
                            out.float(), expected_out.float(), atol=args.atol, rtol=args.rtol
                        )
                        torch.testing.assert_close(
                            lse, expected_lse, atol=args.atol, rtol=args.rtol
                        )
                    except AssertionError as exc:
                        local_error = f"SM {config.num_comp_sm}:{config.num_comm_sm}: {exc}"
                    raise_if_any_rank_failed(local_error)
                    check_status = "ok"
                cuda_barrier()

                if rank == 0:
                    mode = "causal" if is_causal else "noncausal"
                    rank_times = ",".join(f"{value:.4f}" for value in timing.rank_times_ms)
                    print(
                        f"mode={mode:<9} SM={config.num_comp_sm}:{config.num_comm_sm:<2} "
                        f"max_ms={timing.max_ms:.4f} agg_TFLOPS={agg_tflops:.1f} "
                        f"avg_gpu_TFLOPS={agg_tflops / world_size:.1f} check={check_status} "
                        f"rank_ms=[{rank_times}]",
                        flush=True,
                    )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
