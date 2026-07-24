"""Strict six-level causal W8 Mega Ring forward ablation on eight H100s."""

from __future__ import annotations

import argparse
import random
import sys
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

import min_fa3_op
from ring_test.forward_ablation import (
    ForwardAblationPlan,
    PROFILES,
    canonicalize_lengths,
    local_lengths_for_rank,
    make_cu_seqlens,
)
from ring_test.load_balance_bench.topology import make_br_pbs_topology
from scripts.test_mega_ring.mega_ring_test_min_fa3_varlen_hybrid_multi_rank import (
    hierarchical_reference,
)


DEFAULT_LENGTHS = "16384,12288,8192,6144,4096,2048,2048,2048"
Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128


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


def parse_lengths(spec: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in spec.split(",") if token.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--seqlen must contain positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=int, default=8)
    parser.add_argument("--seqlen", default=DEFAULT_LENGTHS)
    parser.add_argument("--qhead", type=int, default=Q_HEADS)
    parser.add_argument("--kvhead", type=int, default=KV_HEADS)
    parser.add_argument("--headdim", type=int, default=HEAD_DIM)
    parser.add_argument("--mode", choices=("causal",), default="causal")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--atol", type=float, default=0.2)
    parser.add_argument("--rtol", type=float, default=0.2)
    args = parser.parse_args()
    raw_lengths = parse_lengths(args.seqlen)
    if args.b <= 0 or args.b > len(raw_lengths):
        parser.error(f"--b must be in [1, {len(raw_lengths)}]")
    args.raw_lengths = raw_lengths[: args.b]
    if (args.qhead, args.kvhead, args.headdim) != (Q_HEADS, KV_HEADS, HEAD_DIM):
        parser.error("strict ablation fixes QH=16, KVH=8, D=128")
    if args.warmup < 0 or args.iters <= 0 or args.rounds <= 0 or args.repeat != 5:
        parser.error("warmup must be nonnegative, iters/rounds positive, and repeat=5")
    return args


def init_distributed() -> tuple[int, int, torch.device]:
    import os

    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit("run with torchrun --nproc_per_node=8")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size != 8:
        raise SystemExit(f"strict forward ablation requires 8 ranks, got {world_size}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise SystemExit("strict forward ablation requires Hopper SM90")
    dist.init_process_group("nccl", device_id=device)
    return rank, world_size, device


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
            (global_length, Q_HEADS, HEAD_DIM),
            (global_length, KV_HEADS, HEAD_DIM),
            (global_length, KV_HEADS, HEAD_DIM),
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
) -> DistributedInputs:
    rank_lengths = tuple(
        local_lengths_for_rank(global_lengths, ring_sizes, ring_starts, source_rank)
        for source_rank in range(8)
    )
    local_lengths = rank_lengths[rank]
    q, local_k, local_v = make_global_consistent_local_qkv(
        global_lengths, ring_sizes, ring_starts, sample_ids, rank, device, seed
    )
    if q.size(0) != sum(local_lengths):
        raise RuntimeError("packed local input does not match topology lengths")
    cu, cu_host = make_cu_seqlens(local_lengths, device)
    rank_capacity = max(sum(lengths) for lengths in rank_lengths)
    rank_capacity = (rank_capacity + 127) // 128 * 128
    arena_shape = (8 * rank_capacity, KV_HEADS, HEAD_DIM)
    remote_k = min_fa3_op.TKParallelTensor(
        arena_shape, torch.bfloat16, rank, 8, False
    )
    remote_v = min_fa3_op.TKParallelTensor(
        arena_shape, torch.bfloat16, rank, 8, False
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
    inputs: DistributedInputs, profile: str, *, collect_stats: bool = False
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
        collect_stats=collect_stats,
    )


def gather_local_kv(inputs: DistributedInputs) -> tuple[torch.Tensor, torch.Tensor]:
    def gather(tensor: torch.Tensor) -> torch.Tensor:
        padded = torch.zeros(
            (inputs.rank_capacity, tensor.size(1), tensor.size(2)),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        padded[: tensor.size(0)].copy_(tensor)
        parts = [torch.empty_like(padded) for _ in range(8)]
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
        for source_rank in range(8)
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


def correctness_and_probes(
    rank: int, device: torch.device, seed: int, repeat: int, atol: float, rtol: float
) -> None:
    all_cp_lengths = (8192, 4096)
    all_cp = make_inputs(
        all_cp_lengths,
        (8, 8),
        (0, 0),
        (0, 1),
        rank,
        device,
        seed,
    )
    expected_o, expected_lse = reference(all_cp, rank)
    for profile in PROFILES[:5]:
        plan = make_plan(all_cp, profile.name)
        for _ in range(repeat):
            out, lse = plan.run()
        torch.cuda.synchronize(device)
        assert_distributed_close(
            f"{profile.name} O", out.float(), expected_o.float(), atol, rtol
        )
        assert_distributed_close(
            f"{profile.name} LSE", lse, expected_lse, atol, rtol
        )

    probe_results: dict[str, dict[str, object]] = {}
    for profile in PROFILES[:5]:
        probe_results[profile.name] = make_plan(
            all_cp, profile.name, collect_stats=True
        ).probe()
    if probe_results["step_external_reduce"]["attention_launches"] != 8:
        raise AssertionError("L1 did not report eight attention launches")
    if probe_results["step_external_reduce"]["reduction_launches"] != 8:
        raise AssertionError("L1 did not report eight reduction launches")
    if probe_results["step_fused_reduce"]["reduction_launches"] != 0:
        raise AssertionError("L2 unexpectedly reported an external reduction")
    if probe_results["linear_queue_no_recycle"]["recycled_cta_work"] != 0:
        raise AssertionError("L3 communication CTAs consumed compute work")
    if probe_results["linear_queue_recycle"]["segment_span_max"] != 1:
        raise AssertionError("L4 formed a multi-step segment")
    if probe_results["dynamic_segment_recycle"]["segment_span_max"] <= 1:
        raise AssertionError("L5 did not form a multi-step ready segment")

    mixed = make_inputs(
        (2048, 1024, 512, 256),
        (8, 4, 2, 1),
        (0, 0, 0, 0),
        (0, 1, 2, 3),
        rank,
        device,
        seed + 17,
    )
    mixed_expected_o, mixed_expected_lse = reference(mixed, rank)
    mixed_plan = make_plan(mixed, "hybrid_br_pbs")
    for _ in range(repeat):
        mixed_o, mixed_lse = mixed_plan.run()
    torch.cuda.synchronize(device)
    assert_distributed_close(
        "L6 mixed O", mixed_o.float(), mixed_expected_o.float(), atol, rtol
    )
    assert_distributed_close(
        "L6 mixed LSE", mixed_lse, mixed_expected_lse, atol, rtol
    )
    mixed_probe = make_plan(mixed, "hybrid_br_pbs", collect_stats=True).probe()
    if len(mixed_probe["ring_sizes"]) < 2:
        raise AssertionError("L6 did not consume multiple ring sizes")
    cuda_barrier()
    if rank == 0:
        print("correctness: PASS (L1-L5 all-CP and L6 mixed hierarchy, repeat=5)")
        for name, counters in probe_results.items():
            print(f"probe {name}: {counters}")
        print(f"probe hybrid_br_pbs: {mixed_probe}")


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def original_tflops(raw_lengths: Sequence[int], latency_ms: float) -> float:
    scores = sum(length * (length + 1) // 2 for length in raw_lengths)
    flops = 4 * scores * Q_HEADS * HEAD_DIM
    return flops / (latency_ms * 1e-3) / 1e12


def time_profiles(
    plans: dict[str, ForwardAblationPlan],
    warmup: int,
    iterations: int,
    rounds: int,
    seed: int,
) -> dict[str, list[float]]:
    samples = {profile.name: [] for profile in PROFILES}
    for round_index in range(rounds):
        order = [profile.name for profile in PROFILES]
        random.Random(seed + round_index).shuffle(order)
        for _ in range(warmup):
            for name in order:
                plans[name].run()
        cuda_barrier()
        for iteration in range(iterations):
            iteration_order = list(order)
            random.Random(seed + round_index * iterations + iteration).shuffle(
                iteration_order
            )
            for name in iteration_order:
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                plans[name].run()
                end.record()
                end.synchronize()
                local_ms = begin.elapsed_time(end)
                max_ms = torch.tensor([local_ms], device="cuda", dtype=torch.float64)
                dist.all_reduce(max_ms, op=dist.ReduceOp.MAX)
                samples[name].append(float(max_ms.item()))
        cuda_barrier()
    return samples


def gather_probe(
    plan: ForwardAblationPlan, rank: int
) -> tuple[dict[str, object], list[list[int]] | None]:
    counters = plan.probe()
    local = torch.tensor(
        (
            int(counters["qo_visits"]),
            int(counters["kv_tile_reads"]),
            int(counters["recycled_cta_work"]),
            int(counters["segment_span_max"]),
            int(counters["segment_claims"]),
        ),
        device="cuda",
        dtype=torch.int64,
    )
    gathered = [torch.empty_like(local) for _ in range(8)]
    dist.all_gather(gathered, local)
    rows = [tensor.cpu().tolist() for tensor in gathered] if rank == 0 else None
    return counters, rows


def performance(
    args: argparse.Namespace, rank: int, device: torch.device
) -> None:
    canonical = canonicalize_lengths(args.raw_lengths)
    all_cp = make_inputs(
        canonical,
        (8,) * len(canonical),
        (0,) * len(canonical),
        tuple(range(len(canonical))),
        rank,
        device,
        args.seed,
    )
    br_pbs = make_br_pbs_topology(canonical, 8, True)
    if br_pbs.padding_tokens != 0 or tuple(sorted(br_pbs.sample_ids)) != tuple(
        range(len(canonical))
    ):
        raise RuntimeError("BR-PBS changed the once-canonicalized workload")
    if len(set(br_pbs.ring_sizes)) < 2:
        raise RuntimeError("default performance workload must produce multiple ring sizes")
    hybrid = make_inputs(
        br_pbs.global_lengths,
        br_pbs.ring_sizes,
        br_pbs.ring_starts,
        br_pbs.sample_ids,
        rank,
        device,
        args.seed,
    )
    plans = {
        profile.name: make_plan(
            hybrid if profile.id == 6 else all_cp,
            profile.name,
        )
        for profile in PROFILES
    }
    probe_plans = {
        profile.name: make_plan(
            hybrid if profile.id == 6 else all_cp,
            profile.name,
            collect_stats=True,
        )
        for profile in PROFILES
    }
    cuda_barrier()
    samples = time_profiles(
        plans, args.warmup, args.iters, args.rounds, args.seed
    )
    probes: dict[str, tuple[dict[str, object], list[list[int]] | None]] = {}
    for profile in PROFILES:
        probes[profile.name] = gather_probe(probe_plans[profile.name], rank)
    cuda_barrier()

    if rank != 0:
        return
    print(
        "performance workload: "
        f"raw={args.raw_lengths}, canonical={canonical}, "
        f"br_pbs_lengths={br_pbs.global_lengths}, "
        f"br_pbs_rings={br_pbs.ring_sizes}, starts={br_pbs.ring_starts}"
    )
    print(
        "level\tprofile\tmedian_ms\tp10_ms\tp90_ms\toriginal_token_tflops"
        "\tkernels\tspan_max\tqo_visits\tkv_tile_reads"
    )
    for profile in PROFILES:
        values = samples[profile.name]
        med = median(values)
        counters, _ = probes[profile.name]
        print(
            f"L{profile.id}\t{profile.name}\t{med:.6f}\t"
            f"{quantile(values, 0.10):.6f}\t{quantile(values, 0.90):.6f}\t"
            f"{original_tflops(args.raw_lengths, med):.3f}\t"
            f"{profile.kernel_launches}\t{counters['segment_span_max']}\t"
            f"{counters['qo_visits']}\t{counters['kv_tile_reads']}"
        )
        _, rank_rows = probes[profile.name]
        print(
            f"rank_load L{profile.id} "
            "[rank,qo_visits,kv_tile_reads,recycled_work,span_max,claims]="
            f"{[[rank_index, *row] for rank_index, row in enumerate(rank_rows or [])]}"
        )


def main() -> None:
    args = parse_args()
    rank, _, device = init_distributed()
    try:
        if not args.skip_correctness:
            correctness_and_probes(
                rank, device, args.seed, args.repeat, args.atol, args.rtol
            )
        if not args.correctness_only:
            performance(args, rank, device)
    finally:
        if dist.is_initialized():
            cuda_barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
