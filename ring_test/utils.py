"""Shared hybrid benchmark helpers.

Copied and trimmed from
``scripts/test_mega_ring/mega_ring_test_min_fa3_varlen_hybrid_multi_rank.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from baseline.megatron_hybrid_cp import BalancedCPScheduler, HybridCPPlan


SENTINEL = -123.0
BASE_SEED = 20260713
MEGA_RING_ALL_CP_ALIGNMENT = 2048


@dataclass(frozen=True)
class HybridBenchmarkCase:
    label: str
    case_index: int
    num_cases: int
    global_lengths: tuple[int, ...]
    ring_sizes: tuple[int, ...]
    ring_starts: tuple[int, ...]


def align_mega_ring_all_cp_lengths(global_lengths: list[int]) -> list[int]:
    """Round global lengths for the all-CP mega-ring's eight-rank alignment."""
    alignment = MEGA_RING_ALL_CP_ALIGNMENT
    return [
        ((length + alignment - 1) // alignment) * alignment
        for length in global_lengths
    ]


def aligned_length_note(
    original_lengths: tuple[int, ...] | list[int],
    aligned_lengths: tuple[int, ...] | list[int],
) -> str:
    """Describe effective and physically executed token counts."""

    original_tokens = sum(original_lengths)
    aligned_tokens = sum(aligned_lengths)
    return (
        f"tokens(original/aligned)={original_tokens}/{aligned_tokens}, "
        f"padding={aligned_tokens - original_tokens} tokens"
    )


def hybrid_cp_saturation_note(
    original_global_lengths: tuple[int, ...] | list[int],
    execution_plan: HybridCPPlan,
) -> str:
    """Describe samples whose initial CP demand was capped to physical WORLD."""

    original_lengths = tuple(int(length) for length in original_global_lengths)
    if len(original_lengths) != len(execution_plan.assignments):
        raise ValueError("original lengths and execution plan must have equal size")
    scheduler = BalancedCPScheduler(
        execution_plan.max_seqlen_per_rank, execution_plan.world_size
    )
    details: list[str] = []
    for sample_id, original_length in enumerate(original_lengths):
        uncapped_cp_size = scheduler.gpus_needed(original_length)
        if uncapped_cp_size <= execution_plan.world_size:
            continue
        assignment = execution_plan.assignment(sample_id)
        local_length = assignment.global_length // assignment.cp_size
        details.append(
            f"sample {sample_id} required_cp(original/capped)="
            f"{uncapped_cp_size}/{execution_plan.world_size}, "
            f"local_length={local_length}"
        )
    return "CP saturation: " + "; ".join(details) if details else ""


def parse_int_list(spec: str, name: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise SystemExit(f"{name} must provide at least one integer")
    return values


def make_uniform_workload_cases(
    context_lengths_spec: str,
    batch_sizes_spec: str,
    world_size: int,
) -> list[HybridBenchmarkCase]:
    """Build fixed-total-token uniform cases for one reusable process group."""

    context_lengths = parse_int_list(context_lengths_spec, "--context-lengths")
    batch_sizes = parse_int_list(batch_sizes_spec, "--batch-sizes")
    if any(context_length <= 0 for context_length in context_lengths):
        raise SystemExit("--context-lengths values must be positive")
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise SystemExit("--batch-sizes values must be positive")

    case_specs: list[tuple[int, int, int, int]] = []
    for context_length in context_lengths:
        for batch_size in batch_sizes:
            if context_length % batch_size:
                raise SystemExit(
                    f"context length {context_length} is not divisible by "
                    f"batch size {batch_size}"
                )
            if batch_size <= world_size:
                if world_size % batch_size:
                    raise SystemExit(
                        f"batch size {batch_size} must divide world size {world_size}"
                    )
                ring_size = world_size // batch_size
            else:
                if batch_size % world_size:
                    raise SystemExit(
                        f"batch size {batch_size} must be a multiple of world size "
                        f"{world_size} when it exceeds the world size"
                    )
                ring_size = 1
            if ring_size not in (1, 2, 4, 8):
                raise SystemExit(
                    f"uniform workload produced unsupported ring size {ring_size}"
                )

            sequence_length = context_length // batch_size
            if sequence_length % (ring_size * 256):
                raise SystemExit(
                    f"sequence length {sequence_length} does not satisfy "
                    f"G{ring_size} alignment"
                )
            case_specs.append(
                (context_length, batch_size, sequence_length, ring_size)
            )

    num_cases = len(case_specs)
    workload_cases: list[HybridBenchmarkCase] = []
    for case_index, (
        context_length,
        batch_size,
        sequence_length,
        ring_size,
    ) in enumerate(case_specs):
        ring_sizes = (ring_size,) * batch_size
        ring_starts = tuple(
            index % world_size if ring_size == 1 else index * ring_size
            for index in range(batch_size)
        )
        workload_cases.append(
            HybridBenchmarkCase(
                label=(
                    f"uniform context={context_length}, batch={batch_size}, "
                    f"seqlen={sequence_length}, hybrid=G{ring_size}"
                ),
                case_index=case_index,
                num_cases=num_cases,
                global_lengths=(sequence_length,) * batch_size,
                ring_sizes=ring_sizes,
                ring_starts=ring_starts,
            )
        )
    return workload_cases


def init_distributed() -> tuple[int, int]:
    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit("Run this benchmark with torchrun")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", device_id=torch.device("cuda", rank)
        )
    if dist.get_world_size() != world_size or world_size not in (2, 4, 8):
        raise SystemExit(
            "hierarchical mega ring requires one node with 2, 4, or 8 ranks, "
            f"got {world_size}"
        )
    return rank, world_size


def make_cu_seqlens(
    lengths: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.zeros((len(lengths) + 1,), dtype=torch.int32)
    for idx, length in enumerate(lengths):
        host[idx + 1] = host[idx] + length
    return host.to(device=device), host


def sequence_shards_to_global_order(
    global_seqlens: list[int], world_size: int, causal: bool
) -> list[int]:
    """Map rank-major per-sequence shards to original packed sequence order."""
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if any(length <= 0 or length % world_size for length in global_seqlens):
        raise ValueError(
            "every global sequence length must be positive and divisible by world_size"
        )

    local_lengths = [length // world_size for length in global_seqlens]
    local_total = sum(local_lengths)
    order: list[int] = []
    local_offset = 0
    for local_len in local_lengths:
        if causal:
            if local_len % 2:
                raise ValueError(
                    "causal per-sequence shards require even local sequence lengths"
                )
            half = local_len // 2
            for source_rank in range(world_size):
                source = source_rank * local_total + local_offset
                order.extend(range(source, source + half))
            for source_rank in reversed(range(world_size)):
                source = source_rank * local_total + local_offset + half
                order.extend(range(source, source + half))
        else:
            for source_rank in range(world_size):
                source = source_rank * local_total + local_offset
                order.extend(range(source, source + local_len))
        local_offset += local_len
    return order


def local_lengths_for_rank(
    global_lengths: list[int],
    ring_sizes: list[int],
    ring_starts: list[int],
    rank: int,
) -> list[int]:
    return [
        global_len // ring_size
        if ring_start <= rank < ring_start + ring_size
        else 0
        for global_len, ring_size, ring_start in zip(
            global_lengths, ring_sizes, ring_starts
        )
    ]


def make_local_qkv(
    total_tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    rank: int,
    is_causal: bool,
    device: torch.device,
    base_seed: int = BASE_SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Keep each mode independent of execution order. Rank-wise V offsets make
    # layout mistakes visible without creating nearly tied large logits.
    seed = base_seed + rank * 1009 + int(is_causal) * 1_000_003
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(
        (total_tokens, q_heads, head_dim),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).to(torch.bfloat16)
    k = torch.randn(
        (total_tokens, kv_heads, head_dim),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).to(torch.bfloat16)
    v = (
        torch.randn(
            (total_tokens, kv_heads, head_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(0.5)
        .add_(rank * 0.125)
        .to(torch.bfloat16)
    )
    return q.contiguous(), k.contiguous(), v.contiguous()


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_positions: torch.Tensor | None,
    key_positions: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    qf, kf, vf = q.float(), k.float(), v.float()
    repeat = q.size(1) // k.size(1)
    if repeat != 1:
        kf = kf.repeat_interleave(repeat, dim=1)
        vf = vf.repeat_interleave(repeat, dim=1)
    scores = torch.einsum("qhd,khd->hqk", qf, kf) * (q.size(-1) ** -0.5)
    if query_positions is not None:
        if key_positions is None:
            raise ValueError("causal reference requires key positions")
        mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", probs, vf).to(torch.bfloat16)
    return out, lse


def hierarchical_reference(
    q: torch.Tensor,
    gathered_k: torch.Tensor,
    gathered_v: torch.Tensor,
    all_rank_lengths: list[list[int]],
    local_cu: torch.Tensor,
    global_lengths: list[int],
    ring_sizes: list[int],
    ring_starts: list[int],
    rank: int,
    is_causal: bool,
    *,
    causal_layout: str = "zigzag",
) -> tuple[torch.Tensor, torch.Tensor]:
    if causal_layout not in ("zigzag", "contiguous"):
        raise ValueError(f"unsupported causal layout {causal_layout!r}")
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for batch_idx, (global_len, ring_size, ring_start) in enumerate(
        zip(global_lengths, ring_sizes, ring_starts)
    ):
        q_begin = int(local_cu[batch_idx])
        q_end = int(local_cu[batch_idx + 1])
        if q_begin == q_end:
            continue
        q_batch = q[q_begin:q_end]
        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        key_positions: list[torch.Tensor] = []
        local_len = global_len // ring_size
        half_len = local_len // 2
        for source_rank in range(ring_start, ring_start + ring_size):
            source_offset = sum(all_rank_lengths[source_rank][:batch_idx])
            source_end = source_offset + local_len
            k_parts.append(gathered_k[source_rank, source_offset:source_end])
            v_parts.append(gathered_v[source_rank, source_offset:source_end])
            if is_causal and ring_size > 1 and causal_layout == "zigzag":
                subgroup_rank = source_rank - ring_start
                front = (
                    torch.arange(half_len, device=q.device)
                    + subgroup_rank * half_len
                )
                back = (
                    torch.arange(half_len, device=q.device)
                    + (2 * ring_size - 1 - subgroup_rank) * half_len
                )
                key_positions.append(torch.cat((front, back)))
            elif is_causal and ring_size > 1:
                subgroup_rank = source_rank - ring_start
                key_positions.append(
                    torch.arange(local_len, device=q.device)
                    + subgroup_rank * local_len
                )
        k_batch = torch.cat(k_parts)
        v_batch = torch.cat(v_parts)
        if is_causal and ring_size > 1 and causal_layout == "zigzag":
            subgroup_rank = rank - ring_start
            query_front = (
                torch.arange(half_len, device=q.device) + subgroup_rank * half_len
            )
            query_back = (
                torch.arange(half_len, device=q.device)
                + (2 * ring_size - 1 - subgroup_rank) * half_len
            )
            query_positions = torch.cat((query_front, query_back))
            key_position_tensor = torch.cat(key_positions)
        elif is_causal and ring_size > 1:
            subgroup_rank = rank - ring_start
            query_positions = (
                torch.arange(local_len, device=q.device)
                + subgroup_rank * local_len
            )
            key_position_tensor = torch.cat(key_positions)
        elif is_causal:
            query_positions = torch.arange(local_len, device=q.device)
            key_position_tensor = torch.arange(local_len, device=q.device)
        else:
            query_positions = None
            key_position_tensor = None
        out, lse = attention_reference(
            q_batch,
            k_batch,
            v_batch,
            query_positions,
            key_position_tensor,
        )
        outputs.append(out)
        lses.append(lse)
    if not outputs:
        return q.new_empty(q.shape), torch.empty(
            (q.size(1), 0), device=q.device, dtype=torch.float32
        )
    return torch.cat(outputs), torch.cat(lses, dim=1)


def zeppelin_reference(
    q: torch.Tensor,
    gathered_k: torch.Tensor,
    gathered_v: torch.Tensor,
    plan: object,
    rank: int,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore raw sequences using explicit ordered Zeppelin group members."""

    packed_by_rank = [
        plan.packed_assignments_for_rank(source_rank)  # type: ignore[attr-defined]
        for source_rank in range(plan.world_size)  # type: ignore[attr-defined]
    ]
    source_offsets: list[dict[int, int]] = []
    for assignments in packed_by_rank:
        offset = 0
        offsets: dict[int, int] = {}
        for assignment in assignments:
            offsets[assignment.sample_id] = offset
            offset += assignment.local_length
        source_offsets.append(offsets)

    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    q_offset = 0
    for assignment in packed_by_rank[rank]:
        local_length = assignment.local_length
        q_batch = q[q_offset : q_offset + local_length]
        q_offset += local_length
        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        key_positions: list[torch.Tensor] = []
        half_length = local_length // 2
        for group_rank, source_rank in enumerate(assignment.group_members):
            source_begin = source_offsets[source_rank][assignment.sample_id]
            source_end = source_begin + local_length
            k_parts.append(gathered_k[source_rank, source_begin:source_end])
            v_parts.append(gathered_v[source_rank, source_begin:source_end])
            if is_causal and assignment.group_size > 1:
                front = (
                    torch.arange(half_length, device=q.device)
                    + group_rank * half_length
                )
                back = (
                    torch.arange(half_length, device=q.device)
                    + (2 * assignment.group_size - 1 - group_rank) * half_length
                )
                key_positions.append(torch.cat((front, back)))

        if is_causal and assignment.group_size > 1:
            local_group_rank = assignment.group_members.index(rank)
            query_front = (
                torch.arange(half_length, device=q.device)
                + local_group_rank * half_length
            )
            query_back = (
                torch.arange(half_length, device=q.device)
                + (2 * assignment.group_size - 1 - local_group_rank) * half_length
            )
            query_positions = torch.cat((query_front, query_back))
            key_position_tensor = torch.cat(key_positions)
        elif is_causal:
            query_positions = torch.arange(local_length, device=q.device)
            key_position_tensor = query_positions
        else:
            query_positions = None
            key_position_tensor = None

        out, lse = attention_reference(
            q_batch,
            torch.cat(k_parts),
            torch.cat(v_parts),
            query_positions,
            key_position_tensor,
        )
        outputs.append(out)
        lses.append(lse)

    if not outputs:
        return q.new_empty(q.shape), torch.empty(
            (q.size(1), 0), device=q.device, dtype=torch.float32
        )
    return torch.cat(outputs), torch.cat(lses, dim=1)


def assert_all_ranks(local_error: str | None) -> None:
    failed = torch.tensor(
        [local_error is not None], device="cuda", dtype=torch.int32
    )
    dist.all_reduce(failed)
    if failed.item() == 0:
        return
    if local_error is not None:
        raise AssertionError(local_error)
    raise AssertionError("another rank failed")
