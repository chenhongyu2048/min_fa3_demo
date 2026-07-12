"""Distributed hierarchical hybrid forward benchmark with all-CP baselines."""

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
from ring_common import (
    flash_varlen_block_attention,
    min_fa3_varlen_block_attention,
    ring_varlen_forward,
    zigzag_ring_varlen_forward,
)

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func3
except ImportError:
    flash_attn_varlen_func3 = None


METHOD_ORDER = [
    "allgather_attention",
    "fa3_ring",
    "mega_ring_all_cp",
    "mega_ring_hybrid",
]


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
    global_lengths: list[int],
    ring_sizes: list[int],
    ring_starts: list[int],
    world_size: int,
    mode: str,
    methods: list[str],
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

    all_cp_methods = {"allgather_attention", "fa3_ring", "mega_ring_all_cp"}
    if all_cp_methods.intersection(methods):
        for idx, global_len in enumerate(global_lengths):
            if global_len % world_size:
                raise SystemExit(
                    "all-CP baselines require every global length to be divisible by world_size: "
                    f"batch={idx}, global_len={global_len}, world_size={world_size}"
                )
            local_len = global_len // world_size
            if mode in ("causal", "both") and local_len % 2:
                raise SystemExit(
                    f"causal all-CP baseline requires even local length at batch {idx}, got {local_len}"
                )
            if (
                "mega_ring_all_cp" in methods
                and mode in ("causal", "both")
                and (local_len // 2) % 128
            ):
                raise SystemExit(
                    "causal all-CP mega-ring requires local_len / 2 to be 128-aligned: "
                    f"batch={idx}, local_len={local_len}"
                )


def fa3_or_min_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_q_host: torch.Tensor,
    cu_seqlens_k_host: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if flash_attn_varlen_func3 is not None:
        return flash_varlen_block_attention(
            "fa3",
            flash_attn_varlen_func3,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            is_causal,
        )
    return min_fa3_varlen_block_attention(
        min_fa3_op.forward_varlen,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_q_host,
        cu_seqlens_k_host,
        max_seqlen_q,
        max_seqlen_k,
        is_causal,
    )


class VarlenAllGatherAttention:
    """All-gather K/V and run one batched varlen attention per visible Q half."""

    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        local_lengths: list[int],
        rank: int,
        world_size: int,
        is_causal: bool,
    ) -> None:
        self.q = q
        self.k = k
        self.v = v
        self.local_lengths = local_lengths
        self.rank = rank
        self.world_size = world_size
        self.is_causal = is_causal
        self.local_total = sum(local_lengths)
        self.global_lengths = [length * world_size for length in local_lengths]
        self.gathered_k = torch.empty(
            (world_size * self.local_total, k.size(1), k.size(2)), dtype=k.dtype, device=k.device
        )
        self.gathered_v = torch.empty_like(self.gathered_k)
        self.ordered_k = torch.empty_like(self.gathered_k)
        self.ordered_v = torch.empty_like(self.gathered_v)

        global_order: list[int] = []
        q_front_indices: list[int] = []
        q_back_indices: list[int] = []
        k_front_indices: list[int] = []
        k_back_indices: list[int] = []
        local_offset = 0
        global_offset = 0
        half_lengths: list[int] = []
        front_k_lengths: list[int] = []
        back_k_lengths: list[int] = []
        for local_len in local_lengths:
            if is_causal:
                half_len = local_len // 2
                half_lengths.append(half_len)
                front_k_len = (rank + 1) * half_len
                back_k_len = (2 * world_size - rank) * half_len
                front_k_lengths.append(front_k_len)
                back_k_lengths.append(back_k_len)
                q_front_indices.extend(range(local_offset, local_offset + half_len))
                q_back_indices.extend(range(local_offset + half_len, local_offset + local_len))
                for source_rank in range(world_size):
                    source = source_rank * self.local_total + local_offset
                    global_order.extend(range(source, source + half_len))
                for source_rank in reversed(range(world_size)):
                    source = source_rank * self.local_total + local_offset + half_len
                    global_order.extend(range(source, source + half_len))
                k_front_indices.extend(range(global_offset, global_offset + front_k_len))
                k_back_indices.extend(range(global_offset, global_offset + back_k_len))
            else:
                for source_rank in range(world_size):
                    source = source_rank * self.local_total + local_offset
                    global_order.extend(range(source, source + local_len))
            local_offset += local_len
            global_offset += local_len * world_size

        self.global_order = torch.tensor(global_order, device=q.device, dtype=torch.int64)
        self.local_cu, self.local_cu_host = make_cu_seqlens(local_lengths, q.device)
        self.global_cu, self.global_cu_host = make_cu_seqlens(self.global_lengths, q.device)
        self.max_local = max(local_lengths)
        self.max_global = max(self.global_lengths)

        if is_causal:
            self.q_front_indices = torch.tensor(q_front_indices, device=q.device, dtype=torch.int64)
            self.q_back_indices = torch.tensor(q_back_indices, device=q.device, dtype=torch.int64)
            self.k_front_indices = torch.tensor(k_front_indices, device=q.device, dtype=torch.int64)
            self.k_back_indices = torch.tensor(k_back_indices, device=q.device, dtype=torch.int64)
            self.q_front = q.index_select(0, self.q_front_indices).contiguous()
            self.q_back = q.index_select(0, self.q_back_indices).contiguous()
            self.k_front = torch.empty(
                (len(k_front_indices), k.size(1), k.size(2)), dtype=k.dtype, device=k.device
            )
            self.v_front = torch.empty_like(self.k_front)
            self.k_back = torch.empty(
                (len(k_back_indices), k.size(1), k.size(2)), dtype=k.dtype, device=k.device
            )
            self.v_back = torch.empty_like(self.k_back)
            self.half_cu, self.half_cu_host = make_cu_seqlens(half_lengths, q.device)
            self.front_k_cu, self.front_k_cu_host = make_cu_seqlens(front_k_lengths, q.device)
            self.back_k_cu, self.back_k_cu_host = make_cu_seqlens(back_k_lengths, q.device)
            self.out = torch.empty_like(q)

    def forward(self) -> torch.Tensor:
        dist.all_gather_into_tensor(self.gathered_k, self.k)
        dist.all_gather_into_tensor(self.gathered_v, self.v)
        torch.index_select(self.gathered_k, 0, self.global_order, out=self.ordered_k)
        torch.index_select(self.gathered_v, 0, self.global_order, out=self.ordered_v)
        if not self.is_causal:
            out, _ = fa3_or_min_block_attention(
                self.q,
                self.ordered_k,
                self.ordered_v,
                self.local_cu,
                self.global_cu,
                self.local_cu_host,
                self.global_cu_host,
                self.max_local,
                self.max_global,
                False,
            )
            return out

        torch.index_select(self.ordered_k, 0, self.k_front_indices, out=self.k_front)
        torch.index_select(self.ordered_v, 0, self.k_front_indices, out=self.v_front)
        torch.index_select(self.ordered_k, 0, self.k_back_indices, out=self.k_back)
        torch.index_select(self.ordered_v, 0, self.k_back_indices, out=self.v_back)
        out_front, _ = fa3_or_min_block_attention(
            self.q_front,
            self.k_front,
            self.v_front,
            self.half_cu,
            self.front_k_cu,
            self.half_cu_host,
            self.front_k_cu_host,
            max(self.local_lengths) // 2,
            max(length * (self.rank + 1) // 2 for length in self.local_lengths),
            True,
        )
        out_back, _ = fa3_or_min_block_attention(
            self.q_back,
            self.k_back,
            self.v_back,
            self.half_cu,
            self.back_k_cu,
            self.half_cu_host,
            self.back_k_cu_host,
            max(self.local_lengths) // 2,
            max(length * (2 * self.world_size - self.rank) // 2 for length in self.local_lengths),
            True,
        )
        self.out.index_copy_(0, self.q_front_indices, out_front)
        self.out.index_copy_(0, self.q_back_indices, out_back)
        return self.out


def fa3_ring_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    local_lengths: list[int],
    is_causal: bool,
) -> torch.Tensor:
    max_local_len = max(local_lengths)
    if is_causal:
        return zigzag_ring_varlen_forward(
            dist.group.WORLD,
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens_host,
            max_local_len,
            fa3_or_min_block_attention,
        )
    return ring_varlen_forward(
        dist.group.WORLD,
        q,
        k,
        v,
        False,
        lambda q_, k_, v_, causal_: fa3_or_min_block_attention(
            q_,
            k_,
            v_,
            cu_seqlens,
            cu_seqlens,
            cu_seqlens_host,
            cu_seqlens_host,
            max_local_len,
            max_local_len,
            causal_,
        ),
    )


def make_mega_parallel_tensors(
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    rank: int,
    world_size: int,
    rank_capacity: int,
) -> tuple[min_fa3_op.TKParallelTensor, min_fa3_op.TKParallelTensor]:
    arena_shape = [world_size * rank_capacity, local_k.size(1), local_k.size(2)]
    remote_k = min_fa3_op.TKParallelTensor(arena_shape, torch.bfloat16, rank, world_size, False)
    remote_v = min_fa3_op.TKParallelTensor(arena_shape, torch.bfloat16, rank, world_size, False)
    remote_k.data_.zero_()
    remote_v.data_.zero_()
    owner_begin = rank * rank_capacity
    remote_k.data_[owner_begin:owner_begin + local_k.size(0)].copy_(local_k)
    remote_v.data_[owner_begin:owner_begin + local_v.size(0)].copy_(local_v)
    return remote_k, remote_v


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
    parser = argparse.ArgumentParser(description="Benchmark hierarchical hybrid forward with all-CP baselines")
    parser.add_argument("--global-seqlens", required=True)
    parser.add_argument("--ring-sizes", required=True)
    parser.add_argument("--ring-starts", required=True)
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--mode", choices=("noncausal", "causal", "both"), default="causal")
    parser.add_argument(
        "--methods",
        default="all",
        help=f"Comma-separated methods from {METHOD_ORDER}, or all",
    )
    parser.add_argument("--sm-configs", default="128:4,124:8,120:12,116:16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=40)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("--rtol", type=float, default=2e-1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    global flash_attn_varlen_func3

    args = parse_args()
    methods = parse_methods(args.methods)
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
        validate_metadata(global_lengths, ring_sizes, ring_starts, world_size, args.mode, methods)
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
        fa_available = torch.tensor(
            [flash_attn_varlen_func3 is not None], device=device, dtype=torch.int32
        )
        dist.all_reduce(fa_available, op=dist.ReduceOp.MIN)
        if not fa_available.item():
            flash_attn_varlen_func3 = None
        backend_note = "external FA3" if flash_attn_varlen_func3 is not None else "local min_fa3 fallback"

        modes = {
            "noncausal": [False],
            "causal": [True],
            "both": [False, True],
        }[args.mode]

        if rank == 0:
            print(
                f"world_size={world_size}, methods={methods}, global_seqlens={global_lengths}, "
                f"ring_sizes={ring_sizes}, ring_starts={ring_starts}, "
                f"QH={args.qhead}, KVH={args.kvhead}, D={args.headdim}, "
                f"FA backend={backend_note}",
                flush=True,
            )

        for is_causal in modes:
            all_cp_runs: dict[str, tuple[Callable[[], object], str]] = {}
            expected_all_cp_out = None
            expected_all_cp_lse = None
            if any(method != "mega_ring_hybrid" for method in methods):
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
                all_cp_global_host = torch.tensor(global_lengths, dtype=torch.int32)
                all_cp_ring_sizes_host = torch.full(
                    (len(global_lengths),), world_size, dtype=torch.int32
                )
                all_cp_ring_starts_host = torch.zeros(len(global_lengths), dtype=torch.int32)

                if "allgather_attention" in methods:
                    allgather_runner = VarlenAllGatherAttention(
                        all_cp_q,
                        all_cp_k,
                        all_cp_v,
                        all_cp_lengths,
                        rank,
                        world_size,
                        is_causal,
                    )
                    all_cp_runs["allgather_attention"] = (
                        allgather_runner.forward,
                        f"all-CP all-gather; {backend_note}",
                    )
                if "fa3_ring" in methods:
                    all_cp_runs["fa3_ring"] = (
                        lambda: fa3_ring_forward(
                            all_cp_q,
                            all_cp_k,
                            all_cp_v,
                            all_cp_cu,
                            all_cp_cu_host,
                            all_cp_lengths,
                            is_causal,
                        ),
                        f"all-CP NCCL ring; {backend_note}",
                    )
                if "mega_ring_all_cp" in methods:
                    all_cp_remote_k, all_cp_remote_v = make_mega_parallel_tensors(
                        all_cp_k, all_cp_v, rank, world_size, all_cp_total
                    )

                if args.check:
                    gathered_all_cp_k = gather_padded_rank_tensor(all_cp_k, all_cp_total)
                    gathered_all_cp_v = gather_padded_rank_tensor(all_cp_v, all_cp_total)
                    expected_all_cp_out, expected_all_cp_lse = hierarchical_reference(
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

            hybrid_run_data = None
            if "mega_ring_hybrid" in methods:
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
                hybrid_remote_k, hybrid_remote_v = make_mega_parallel_tensors(
                    hybrid_local_k,
                    hybrid_local_v,
                    rank,
                    world_size,
                    hybrid_rank_capacity,
                )
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
            for config in sm_configs:
                runs: list[MethodRun] = []
                for method in methods:
                    if method in all_cp_runs:
                        launch, note = all_cp_runs[method]
                        runs.append(
                            MethodRun(
                                method,
                                launch,
                                expected_all_cp_out,
                                None,
                                note,
                            )
                        )
                    elif method == "mega_ring_all_cp":
                        def launch_all_cp_mega() -> tuple[torch.Tensor, torch.Tensor]:
                            return min_fa3_op.forward_varlen_mega_ring(
                                all_cp_q,
                                all_cp_remote_k.data_,
                                all_cp_remote_v.data_,
                                all_cp_cu,
                                all_cp_cu,
                                max(all_cp_lengths),
                                max(all_cp_lengths),
                                is_causal,
                                cu_seqlens_q_host=all_cp_cu_host,
                                cu_seqlens_k_host=all_cp_cu_host,
                                remote_k=all_cp_remote_k,
                                remote_v=all_cp_remote_v,
                                num_comp_sm=config.num_comp_sm,
                                num_comm_sm=config.num_comm_sm,
                                global_seqlens_host=all_cp_global_host,
                                ring_sizes_host=all_cp_ring_sizes_host,
                                ring_starts_host=all_cp_ring_starts_host,
                                return_lse=True,
                            )

                        runs.append(
                            MethodRun(
                                method,
                                launch_all_cp_mega,
                                expected_all_cp_out,
                                expected_all_cp_lse,
                                "all-CP fused mega-ring",
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

                for run in runs:
                    timing = measure_distributed_ms(
                        run.launch, args.warmup_iters, args.num_iters, rank
                    )
                    agg_tflops = aggregate_tflops(
                        global_lengths, args.qhead, args.headdim, is_causal, timing.max_ms
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
                        mode = "causal" if is_causal else "noncausal"
                        rank_times = ",".join(f"{value:.4f}" for value in timing.rank_times_ms)
                        print(
                            f"method={run.name:<20} mode={mode:<9} "
                            f"SM={config.num_comp_sm}:{config.num_comm_sm:<2} "
                            f"max_ms={timing.max_ms:.4f} agg_TFLOPS={agg_tflops:.1f} "
                            f"avg_gpu_TFLOPS={agg_tflops / world_size:.1f} "
                            f"check={check_status} rank_ms=[{rank_times}] note={run.note}",
                            flush=True,
                        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
