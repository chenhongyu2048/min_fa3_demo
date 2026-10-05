"""Six-level causal Mega Ring ablation on 2/4/8 GPUs."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Sequence

import torch
import torch.distributed as dist

THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import balancer
import min_fa3_op
from ring_test.forward_ablation import (
    ForwardAblationPlan,
    canonicalize_lengths,
    local_lengths_for_rank,
    make_cu_seqlens,
    profiles_for_world_size,
)
from ring_test.load_balance_bench.topology import PlannerControls, make_br_pbs_topology
from ring_test.utils import aligned_length_note
from scripts.test_mega_ring.mega_ring_test_min_fa3_varlen_hybrid_multi_rank import (
    hierarchical_reference,
)


DEFAULT_DATASET = "arxiv"
DEFAULT_TARGET_TOKENS = 128 * 1024
DEFAULT_NUM_CASES = 20
DEFAULT_SM_CONFIGS = "128:4,124:8,120:12,116:16"
DEFAULT_TOKEN_BALANCE_TOLERANCE = 0.05
DEFAULT_Q_HEADS = 32
DEFAULT_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128


@dataclass
class DistributedInputs:
    q: torch.Tensor
    local_k: torch.Tensor
    local_v: torch.Tensor
    cu: torch.Tensor
    cu_host: torch.Tensor
    remote_k: object
    remote_v: object
    rank_capacity: int
    max_local_length: int
    global_lengths: tuple[int, ...]
    ring_sizes: tuple[int, ...]
    ring_starts: tuple[int, ...]
    sample_ids: tuple[int, ...]


@dataclass(frozen=True)
class AblationSummarySample:
    case_index: int
    profile: str
    sm_config: "SmConfig"
    time_ms: float
    aggregate_tflops: float


@dataclass(frozen=True)
class TimingResult:
    local_ms: float
    max_ms: float
    rank_times_ms: tuple[float, ...] | None


@dataclass(frozen=True)
class SmConfig:
    num_comp_sm: int
    num_comm_sm: int

    @property
    def label(self) -> str:
        return f"{self.num_comp_sm}:{self.num_comm_sm}"


def parse_lengths(spec: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in spec.split(",") if token.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--seqlen must contain positive integers")
    return values


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_sm_configs(spec: str) -> tuple[SmConfig, ...]:
    configs: list[SmConfig] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        fields = token.split(":")
        if len(fields) != 2:
            raise ValueError(f"invalid SM config {token!r}, expected COMP:COMM")
        try:
            num_comp_sm, num_comm_sm = (int(field) for field in fields)
        except ValueError as exc:
            raise ValueError(
                f"invalid SM config {token!r}, expected integer COMP:COMM"
            ) from exc
        if num_comp_sm <= 0 or num_comm_sm <= 0:
            raise ValueError(
                "forward ablation requires positive compute and communication "
                f"SM counts, got {token!r}"
            )
        configs.append(SmConfig(num_comp_sm, num_comm_sm))
    if not configs:
        raise ValueError("--sm-configs must provide at least one COMP:COMM pair")
    return tuple(configs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interleave-comm-windows", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dataset", choices=(DEFAULT_DATASET,), default=DEFAULT_DATASET
    )
    parser.add_argument(
        "--target-tokens", type=positive_int, default=DEFAULT_TARGET_TOKENS
    )
    parser.add_argument("--num-cases", type=positive_int, default=DEFAULT_NUM_CASES)
    parser.add_argument(
        "--b",
        type=positive_int,
        help="Batch size for the explicit one-case --seqlen override",
    )
    parser.add_argument(
        "--seqlen",
        help=(
            "Explicit one-case override retained for focused debugging; "
            "otherwise ArXiv cases are sampled"
        ),
    )
    parser.add_argument("--qhead", type=positive_int, default=DEFAULT_Q_HEADS)
    parser.add_argument("--kvhead", type=positive_int, default=DEFAULT_KV_HEADS)
    parser.add_argument("--headdim", type=positive_int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--mode", choices=("causal",), default="causal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--token-balance-tolerance",
        type=float,
        default=DEFAULT_TOKEN_BALANCE_TOLERANCE,
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=40)
    parser.add_argument(
        "--sm-configs",
        default=DEFAULT_SM_CONFIGS,
        help="Comma-separated compute:communication SM allocations",
    )
    parser.add_argument(
        "--check", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--atol", type=float, default=0.2)
    parser.add_argument("--rtol", type=float, default=0.2)
    args = parser.parse_args()
    try:
        args.sm_configs = parse_sm_configs(args.sm_configs)
    except ValueError as exc:
        parser.error(str(exc))
    if args.seqlen is not None:
        raw_lengths = parse_lengths(args.seqlen)
        if args.b is None:
            args.b = len(raw_lengths)
        if args.b > len(raw_lengths):
            parser.error(f"--b must be in [1, {len(raw_lengths)}]")
        args.raw_length_cases = (raw_lengths[: args.b],)
        args.num_cases = 1
        args.case_source = "explicit --seqlen"
    elif args.b is not None:
        parser.error("--b requires the explicit --seqlen override")
    else:
        try:
            args.raw_length_cases = tuple(
                tuple(lengths)
                for lengths in balancer.generate_dataset_length_cases(
                    args.dataset,
                    args.target_tokens,
                    args.seed,
                    args.num_cases,
                )
            )
        except ValueError as exc:
            parser.error(str(exc))
        args.case_source = f"dataset={args.dataset}"
    if args.headdim != 128:
        parser.error("forward ablation requires D=128")
    if args.kvhead * args.headdim != 1024:
        parser.error("forward ablation requires KVH * D == 1024")
    if args.qhead % args.kvhead:
        parser.error("qhead must be divisible by kvhead")
    if args.token_balance_tolerance < 0:
        parser.error("token balance tolerance must be non-negative")
    if args.warmup_iters < 0 or args.num_iters <= 0:
        parser.error(
            "warmup iterations must be non-negative and measured iterations positive"
        )
    return args


def init_distributed() -> tuple[int, int, torch.device]:
    import os

    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit("run with torchrun --nproc_per_node=2, 4, or 8")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size not in (2, 4, 8):
        raise SystemExit(f"forward ablation requires 2, 4, or 8 ranks, got {world_size}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise SystemExit("strict forward ablation requires Hopper SM90")
    dist.init_process_group("nccl", device_id=device)
    return rank, world_size, device


def validate_sm_configs(
    sm_configs: Sequence[SmConfig], device: torch.device
) -> None:
    device_sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    for config in sm_configs:
        requested = config.num_comp_sm + config.num_comm_sm
        if requested > device_sm_count:
            raise SystemExit(
                f"SM config {config.label} requests {requested} SMs, but "
                f"device {device.index} has {device_sm_count}"
            )


def cuda_barrier() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def _zigzag_shard(tensor: torch.Tensor, ring_size: int, local_rank: int) -> torch.Tensor:
    if ring_size == 1:
        return tensor
    half = tensor.size(0) // (2 * ring_size)
    front = tensor[local_rank * half : (local_rank + 1) * half]
    back_index = 2 * ring_size - 1 - local_rank
    back = tensor[back_index * half : (back_index + 1) * half]
    return torch.cat((front, back), dim=0)


def make_global_consistent_local_qkv(
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    sample_ids: Sequence[int],
    rank: int,
    device: torch.device,
    seed: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_parts: list[torch.Tensor] = []
    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    for global_length, ring_size, ring_start, sample_id in zip(
        global_lengths, ring_sizes, ring_starts, sample_ids
    ):
        if not ring_start <= rank < ring_start + ring_size:
            continue
        local_rank = rank - ring_start
        shapes = (
            (global_length, q_heads, head_dim),
            (global_length, kv_heads, head_dim),
            (global_length, kv_heads, head_dim),
        )
        tensors: list[torch.Tensor] = []
        for kind, shape in enumerate(shapes):
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + sample_id * 1009 + kind * 1_000_003)
            value = torch.randn(
                shape, device=device, dtype=torch.float32, generator=generator
            )
            if kind == 2:
                value.mul_(0.5)
            tensors.append(value.to(torch.bfloat16))
        q_parts.append(_zigzag_shard(tensors[0], ring_size, local_rank))
        k_parts.append(_zigzag_shard(tensors[1], ring_size, local_rank))
        v_parts.append(_zigzag_shard(tensors[2], ring_size, local_rank))
    if not q_parts:
        raise RuntimeError(f"rank {rank} received no local samples")
    return (
        torch.cat(q_parts).contiguous(),
        torch.cat(k_parts).contiguous(),
        torch.cat(v_parts).contiguous(),
    )


def make_inputs(
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    sample_ids: Sequence[int],
    rank: int,
    device: torch.device,
    seed: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> DistributedInputs:
    world_size = dist.get_world_size()
    rank_lengths = tuple(
        local_lengths_for_rank(global_lengths, ring_sizes, ring_starts, source_rank)
        for source_rank in range(world_size)
    )
    local_lengths = rank_lengths[rank]
    q, local_k, local_v = make_global_consistent_local_qkv(
        global_lengths,
        ring_sizes,
        ring_starts,
        sample_ids,
        rank,
        device,
        seed,
        q_heads,
        kv_heads,
        head_dim,
    )
    if q.size(0) != sum(local_lengths):
        raise RuntimeError("packed local input does not match topology lengths")
    cu, cu_host = make_cu_seqlens(local_lengths, device)
    rank_capacity = max(sum(lengths) for lengths in rank_lengths)
    rank_capacity = (rank_capacity + 127) // 128 * 128
    arena_shape = (world_size * rank_capacity, kv_heads, head_dim)
    remote_k = min_fa3_op.TKParallelTensor(
        arena_shape, torch.bfloat16, rank, world_size, False
    )
    remote_v = min_fa3_op.TKParallelTensor(
        arena_shape, torch.bfloat16, rank, world_size, False
    )
    remote_k.data_.zero_()
    remote_v.data_.zero_()
    owner = rank * rank_capacity
    remote_k.data_[owner : owner + local_k.size(0)].copy_(local_k)
    remote_v.data_[owner : owner + local_v.size(0)].copy_(local_v)
    max_local_length = max(max(lengths) for lengths in rank_lengths)
    return DistributedInputs(
        q,
        local_k,
        local_v,
        cu,
        cu_host,
        remote_k,
        remote_v,
        rank_capacity,
        max_local_length,
        tuple(global_lengths),
        tuple(ring_sizes),
        tuple(ring_starts),
        tuple(sample_ids),
    )


def make_plan(
    inputs: DistributedInputs,
    profile: str,
    sm_config: SmConfig,
    *,
    collect_stats: bool = False,
    interleave_comm_windows: bool = True,
) -> ForwardAblationPlan:
    return ForwardAblationPlan(
        inputs.q,
        inputs.remote_k,
        inputs.remote_v,
        inputs.cu,
        inputs.cu_host,
        inputs.max_local_length,
        inputs.global_lengths,
        inputs.ring_sizes,
        inputs.ring_starts,
        profile,
        num_comp_sm=sm_config.num_comp_sm,
        num_comm_sm=sm_config.num_comm_sm,
        collect_stats=collect_stats,
        world_size=dist.get_world_size(),
        interleave_comm_windows=interleave_comm_windows,
    )


def gather_local_kv(inputs: DistributedInputs) -> tuple[torch.Tensor, torch.Tensor]:
    def gather(tensor: torch.Tensor) -> torch.Tensor:
        padded = torch.zeros(
            (inputs.rank_capacity, tensor.size(1), tensor.size(2)),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        padded[: tensor.size(0)].copy_(tensor)
        parts = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, padded)
        return torch.stack(parts)

    return gather(inputs.local_k), gather(inputs.local_v)


def assert_distributed_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    error: str | None = None
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        error = f"{name}: {exc}"
    failed = torch.tensor([error is not None], device=actual.device, dtype=torch.int32)
    dist.all_reduce(failed)
    if failed.item():
        raise AssertionError(error or f"{name}: another rank failed")


def reference(inputs: DistributedInputs, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    gathered_k, gathered_v = gather_local_kv(inputs)
    all_rank_lengths = [
        list(
            local_lengths_for_rank(
                inputs.global_lengths,
                inputs.ring_sizes,
                inputs.ring_starts,
                source_rank,
            )
        )
        for source_rank in range(dist.get_world_size())
    ]
    return hierarchical_reference(
        inputs.q,
        gathered_k,
        gathered_v,
        all_rank_lengths,
        inputs.cu_host,
        list(inputs.global_lengths),
        list(inputs.ring_sizes),
        list(inputs.ring_starts),
        rank,
        True,
    )


def check_ready_state(plan: ForwardAblationPlan, inputs: DistributedInputs, rank: int) -> None:
    """One final snapshot in --check mode, using the prepared Q ordering."""
    batch = len(inputs.ring_sizes)
    rounded = (batch + 3) // 4 * 4
    metadata = plan.scheduler_metadata.cpu().tolist()
    local_lengths = [int(inputs.cu_host[b + 1] - inputs.cu_host[b]) for b in range(batch)]
    ready = [
        (rank % g + 2 * (g - 1 - rank % g)) * (length // 256)
        if g > 1 and length else 0
        for g, length in zip(inputs.ring_sizes, local_lengths)
    ]
    totals: list[int] = []
    for g in (8, 4, 2):
        for virtual_b in range(batch):
            b = metadata[2 * rounded + virtual_b]
            if inputs.ring_sizes[b] != g or not local_lengths[b]:
                continue
            blocks = metadata[rounded + virtual_b]
            heads_in_l2 = metadata[3 * rounded + virtual_b]
            h = blocks // 2
            for first_head in range(0, inputs.q.size(1), heads_in_l2):
                heads = min(heads_in_l2, inputs.q.size(1) - first_head)
                for m in range(blocks - 1, -1, -1):
                    remote = (rank % g) * h if m < h else ready[b]
                    totals.extend([m + 1 + remote] * heads)
    count = plan.hierarchy["reduction_tiles"]
    assert len(totals) == count
    actual = torch.cat((plan.ready_prefix, plan.q_state[:, :count].reshape(-1), plan.completed_tiles[:1]))
    expected = torch.tensor(ready + totals + totals + [2] * count + [count],
                            device=actual.device, dtype=torch.int32)
    assert_distributed_close("final R/A/C/output_state/completed", actual, expected, 0, 0)
    mapping = plan.kv_map.cpu().tolist()
    offsets = plan.kv_map_offsets.cpu().tolist()
    sorted_actual: list[int] = []
    sorted_expected: list[int] = []
    for b, (g, length) in enumerate(zip(inputs.ring_sizes, local_lengths)):
        if g == 1 or length == 0:
            continue
        h, r = length // 256, rank % g
        expected_a: list[int] = []
        expected_b: list[int] = []
        for step in range(1, g):
            source_rank = rank - r + (r - step + g) % g
            base = (source_rank * inputs.rank_capacity + int(inputs.cu_host[b])) // 128
            target = expected_a if step <= r else expected_b
            target.extend(range(base, base + (h if step <= r else 2 * h)))
        begin, end = offsets[b], offsets[b + 1]
        split = begin + r * h
        sorted_actual.extend(sorted(value - 1 for value in mapping[begin:split]))
        sorted_actual.extend(sorted(value - 1 for value in mapping[split:end]))
        sorted_expected.extend(sorted(expected_a))
        sorted_expected.extend(sorted(expected_b))
    assert_distributed_close(
        "final map A/B physical tile sets",
        torch.tensor(sorted_actual, device=actual.device, dtype=torch.int32),
        torch.tensor(sorted_expected, device=actual.device, dtype=torch.int32), 0, 0,
    )


def clear_remote_kv(inputs: DistributedInputs, rank: int) -> None:
    """Clear cached remote shards in this rank's arena, preserving its source shard."""
    owner_begin = rank * inputs.rank_capacity
    owner_end = owner_begin + inputs.rank_capacity
    for arena in (inputs.remote_k.data_, inputs.remote_v.data_):
        arena[:owner_begin].zero_()
        arena[owner_end:].zero_()
    cuda_barrier()


def correctness(
    rank: int,
    device: torch.device,
    seed: int,
    atol: float,
    rtol: float,
    sm_configs: Sequence[SmConfig],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    interleave_comm_windows: bool = True,
) -> None:
    world_size = dist.get_world_size()
    profiles = profiles_for_world_size(world_size)
    all_cp_lengths = (8192, 4096)
    all_cp = make_inputs(
        all_cp_lengths,
        (world_size, world_size),
        (0, 0),
        (0, 1),
        rank,
        device,
        seed,
        q_heads,
        kv_heads,
        head_dim,
    )
    expected_o, expected_lse = reference(all_cp, rank)
    mixed_rings = tuple(g for g in (8, 4, 2, 1) if g <= world_size)
    mixed = make_inputs(
        tuple(256 * g for g in mixed_rings),
        mixed_rings,
        (0,) * len(mixed_rings),
        tuple(range(len(mixed_rings))),
        rank,
        device,
        seed + 17,
        q_heads,
        kv_heads,
        head_dim,
    )
    mixed_expected_o, mixed_expected_lse = reference(mixed, rank)
    for sm_config in sm_configs:
        for profile in profiles[:-1]:
            plan = make_plan(all_cp, profile.name, sm_config,
                             interleave_comm_windows=interleave_comm_windows)
            clear_remote_kv(all_cp, rank)
            for _ in range(3 if plan.profile.dynamic_segments else 1):
                if plan.profile.dynamic_segments:
                    plan.out.fill_(float("nan"))
                    plan.lse.fill_(float("nan"))
                out, lse = plan.run()
                torch.cuda.synchronize(device)
                assert_distributed_close(
                    f"{profile.name} SM {sm_config.label} O",
                    out.float(),
                    expected_o.float(),
                    atol,
                    rtol,
                )
                assert_distributed_close(
                    f"{profile.name} SM {sm_config.label} LSE",
                    lse,
                    expected_lse,
                    atol,
                    rtol,
                )
            if plan.profile.dynamic_segments:
                check_ready_state(plan, all_cp, rank)

        mixed_plan = make_plan(mixed, "hybrid_br_pbs", sm_config,
                               interleave_comm_windows=interleave_comm_windows)
        clear_remote_kv(mixed, rank)
        for _ in range(3):
            mixed_plan.out.fill_(float("nan"))
            mixed_plan.lse.fill_(float("nan"))
            mixed_o, mixed_lse = mixed_plan.run()
            torch.cuda.synchronize(device)
            assert_distributed_close(
                f"L6 mixed SM {sm_config.label} O",
                mixed_o.float(),
                mixed_expected_o.float(),
                atol,
                rtol,
            )
            assert_distributed_close(
                f"L6 mixed SM {sm_config.label} LSE",
                mixed_lse,
                mixed_expected_lse,
                atol,
                rtol,
            )
        check_ready_state(mixed_plan, mixed, rank)
        for profile in profiles:
            inputs = mixed if profile.id == 6 else all_cp
            expected_out = mixed_expected_o if profile.id == 6 else expected_o
            expected_lse_value = mixed_expected_lse if profile.id == 6 else expected_lse
            stats_plan = make_plan(inputs, profile.name, sm_config, collect_stats=True,
                                   interleave_comm_windows=interleave_comm_windows)
            clear_remote_kv(inputs, rank)
            stats_plan.out.fill_(float("nan"))
            stats_plan.lse.fill_(float("nan"))
            probe = stats_plan.probe()
            assert_distributed_close(
                f"{profile.name} stats O", stats_plan.out.float(), expected_out.float(),
                atol, rtol,
            )
            assert_distributed_close(
                f"{profile.name} stats LSE", stats_plan.lse, expected_lse_value, atol, rtol,
            )
            blocks = [
                (g, int(inputs.cu_host[b + 1] - inputs.cu_host[b]) // 128)
                for b, g in enumerate(inputs.ring_sizes)
            ]
            # Local triangular work plus both halves' remote work, per Q head.
            cp_reads = sum(n * (g * n + 1) // 2 for g, n in blocks if g > 1) * q_heads
            local_blocks = [n for g, n in blocks if g == 1]
            local_visits = sum(local_blocks) * q_heads
            local_reads = sum(n * (n + 1) // 2 for n in local_blocks) * q_heads
            if profile.dynamic_segments:
                check_ready_state(stats_plan, inputs, rank)
                expected_span = cp_reads
                expected_visits = probe["segment_claims"] + local_visits
            else:
                expected_span = stats_plan.hierarchy["total_work_tiles"]
                expected_visits = expected_span
            actual_stats = torch.tensor(
                [probe["segment_span_sum"], probe["qo_visits"], probe["kv_tile_reads"]],
                device=device, dtype=torch.int64,
            )
            expected_stats = torch.tensor(
                [expected_span, expected_visits, cp_reads + local_reads],
                device=device, dtype=torch.int64,
            )
            assert_distributed_close(
                f"{profile.name} stats tile counts", actual_stats, expected_stats, 0, 0,
            )
        cuda_barrier()
        if rank == 0:
            print(
                "correctness: PASS "
                f"(W={world_size}, SM={sm_config.label}, "
                "L1-L5 all-CP and L6 mixed hierarchy, "
                "including stats probes)"
            )


def causal_tflops(
    lengths: Sequence[int], q_heads: int, head_dim: int, latency_ms: float
) -> float:
    scores = sum(length * (length + 1) // 2 for length in lengths)
    flops = 4 * scores * q_heads * head_dim
    return flops / (latency_ms * 1e-3) / 1e12


def measure_distributed_ms(
    plan: ForwardAblationPlan,
    warmup_iters: int,
    num_iters: int,
    rank: int,
) -> TimingResult:
    for _ in range(warmup_iters):
        plan.run()
    cuda_barrier()

    local_samples: list[float] = []
    max_samples: list[float] = []
    for _ in range(num_iters):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        plan.run()
        end.record()
        end.synchronize()
        elapsed_ms = begin.elapsed_time(end)
        elapsed = torch.tensor([elapsed_ms], device="cuda", dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        local_samples.append(elapsed_ms)
        max_samples.append(float(elapsed.item()))
    cuda_barrier()

    local_avg = sum(local_samples) / len(local_samples)
    max_avg = sum(max_samples) / len(max_samples)
    local_tensor = torch.tensor([local_avg], device="cuda", dtype=torch.float64)
    gathered = [torch.empty_like(local_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_tensor)
    rank_times = (
        tuple(float(value.item()) for value in gathered) if rank == 0 else None
    )
    return TimingResult(local_avg, max_avg, rank_times)


def performance_case(
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
    case_index: int,
    raw_lengths: tuple[int, ...],
) -> list[AblationSummarySample]:
    world_size = dist.get_world_size()
    profiles = profiles_for_world_size(world_size)
    controls = PlannerControls(
        token_balance_tolerance=args.token_balance_tolerance
    )
    br_pbs = make_br_pbs_topology(raw_lengths, world_size, True, controls)
    if tuple(sorted(br_pbs.sample_ids)) != tuple(range(len(raw_lengths))):
        raise RuntimeError("BR-PBS did not preserve every sampled ArXiv sequence")
    workload_lengths = br_pbs.global_lengths
    all_cp_alignment = 256 * world_size
    all_cp_lengths = canonicalize_lengths(workload_lengths, all_cp_alignment)
    all_cp = make_inputs(
        all_cp_lengths,
        (world_size,) * len(all_cp_lengths),
        (0,) * len(all_cp_lengths),
        br_pbs.sample_ids,
        rank,
        device,
        args.seed,
        args.qhead,
        args.kvhead,
        args.headdim,
    )
    hybrid = make_inputs(
        workload_lengths,
        br_pbs.ring_sizes,
        br_pbs.ring_starts,
        br_pbs.sample_ids,
        rank,
        device,
        args.seed,
        args.qhead,
        args.kvhead,
        args.headdim,
    )
    summary_samples: list[AblationSummarySample] = []
    if rank == 0:
        print(
            f"\nForward ablation case {case_index + 1}/{args.num_cases}: "
            f"{args.case_source}, B={len(raw_lengths)}, "
            f"raw_tokens={sum(raw_lengths)}, QH={args.qhead}, "
            f"KVH={args.kvhead}, D={args.headdim}, raw_lengths={raw_lengths}"
        )
        print(
            "BR-PBS workload: "
            f"token_tolerance={args.token_balance_tolerance}, "
            f"execution_tokens={sum(workload_lengths)}, "
            f"global_seqlens={workload_lengths}, rings={br_pbs.ring_sizes}, "
            f"starts={br_pbs.ring_starts}"
        )
        print(
            "All-CP L1-L5: "
            f"alignment={all_cp_alignment}, execution_tokens={sum(all_cp_lengths)}, "
            f"global_seqlens={all_cp_lengths}"
        )

    for sm_config in args.sm_configs:
        plans = {
            profile.name: make_plan(
                hybrid if profile.id == 6 else all_cp,
                profile.name,
                sm_config,
                interleave_comm_windows=args.interleave_comm_windows,
            )
            for profile in profiles
        }
        if rank == 0:
            print(
                "level\tprofile\tsm_config\tmean_ms\tagg_tflops"
                "\tavg_gpu_tflops\tcheck\tkernels\tnote"
            )
        for profile in profiles:
            timing = measure_distributed_ms(
                plans[profile.name],
                args.warmup_iters,
                args.num_iters,
                rank,
            )
            if rank == 0:
                aggregate_tflops = causal_tflops(
                    workload_lengths, args.qhead, args.headdim, timing.max_ms
                )
                note = "hierarchical hybrid fused mega-ring"
                if profile.id <= 5:
                    aligned_tflops = causal_tflops(
                        all_cp_lengths, args.qhead, args.headdim, timing.max_ms
                    )
                    note = (
                        f"all-CP G{world_size}; "
                        f"{aligned_length_note(workload_lengths, all_cp_lengths)}; "
                        f"aligned-length Agg TFLOPS={aligned_tflops:.3f}, "
                        f"Avg/GPU={aligned_tflops / world_size:.3f}"
                    )
                check_status = "ok" if args.check else "skip"
                print(
                    f"L{profile.id}\t{profile.name}\t{sm_config.label}\t"
                    f"{timing.max_ms:.6f}\t{aggregate_tflops:.3f}\t"
                    f"{aggregate_tflops / world_size:.3f}\t{check_status}\t"
                    f"{profile.kernel_launches}\t{note}"
                )
                summary_samples.append(
                    AblationSummarySample(
                        case_index,
                        profile.name,
                        sm_config,
                        timing.max_ms,
                        aggregate_tflops,
                    )
                )
                rank_times = ", ".join(
                    f"t{rank_index}={time_ms:.3f}"
                    for rank_index, time_ms in enumerate(timing.rank_times_ms or ())
                )
                print(
                    f"rank_time L{profile.id} SM={sm_config.label}: "
                    f"{rank_times} | max_across_ranks={timing.max_ms:.3f}"
                )
        del plans
    return summary_samples


def print_performance_summary(
    samples: Sequence[AblationSummarySample],
    total_cases: int,
    sm_configs: Sequence[SmConfig],
    world_size: int = 8,
) -> None:
    grouped: dict[tuple[str, SmConfig], list[AblationSummarySample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.profile, sample.sm_config)].append(sample)

    print("\nCross-case forward ablation summary")
    print(
        "Agg TFLOPS uses the original BR-PBS workload lengths; all-CP "
        "aligned-length TFLOPS are reported in each case Note."
    )
    print(
        f"{'Level':<7} {'Profile':<28} {'SM':>8} {'Cases':>8} "
        f"{'Min ms':>10} {'Mean ms':>10} {'P50 ms':>10} {'Max ms':>10} "
        f"{'Mean TFLOPS':>14} {'Weighted TFLOPS':>18} {'Weighted/GPU':>14}"
    )
    for profile in profiles_for_world_size(world_size):
        for sm_config in sm_configs:
            records = grouped[(profile.name, sm_config)]
            if not records:
                continue
            times = [record.time_ms for record in records]
            weighted_tflops = sum(
                record.aggregate_tflops * record.time_ms for record in records
            ) / sum(times)
            mean_tflops = sum(
                record.aggregate_tflops for record in records
            ) / len(records)
            print(
                f"L{profile.id:<6} {profile.name:<28} {sm_config.label:>8} "
                f"{f'{len(records)}/{total_cases}':>8} "
                f"{min(times):>10.3f} {sum(times) / len(times):>10.3f} "
                f"{median(times):>10.3f} {max(times):>10.3f} "
                f"{mean_tflops:>14.1f} {weighted_tflops:>18.1f} "
                f"{weighted_tflops / world_size:>14.1f}"
            )


def performance(
    args: argparse.Namespace, rank: int, device: torch.device
) -> None:
    summary_samples: list[AblationSummarySample] = []
    if rank == 0:
        sm_configs = ",".join(config.label for config in args.sm_configs)
        print(
            "Forward ablation config: "
            f"world_size={dist.get_world_size()}, "
            f"dataset={args.dataset}, target_tokens={args.target_tokens}, "
            f"cases={args.num_cases}, seed={args.seed}, "
            f"token_balance_tolerance={args.token_balance_tolerance}, "
            f"QH={args.qhead}, KVH={args.kvhead}, D={args.headdim}, "
            f"mode={args.mode}, sm_configs={sm_configs}, "
            f"warmup={args.warmup_iters}, iters={args.num_iters}, "
            f"check={args.check}, stats_probe=False"
        )
    for case_index, raw_lengths in enumerate(args.raw_length_cases):
        summary_samples.extend(
            performance_case(args, rank, device, case_index, raw_lengths)
        )
    if rank == 0:
        print_performance_summary(
            summary_samples, args.num_cases, args.sm_configs, dist.get_world_size()
        )


def main() -> None:
    args = parse_args()
    rank, _, device = init_distributed()
    try:
        validate_sm_configs(args.sm_configs, device)
        if args.check:
            correctness(
                rank,
                device,
                args.seed,
                args.atol,
                args.rtol,
                args.sm_configs,
                args.qhead,
                args.kvhead,
                args.headdim,
                args.interleave_comm_windows,
            )
        performance(args, rank, device)
    finally:
        if dist.is_initialized():
            cuda_barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
