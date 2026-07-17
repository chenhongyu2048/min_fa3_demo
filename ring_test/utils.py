"""Shared hybrid benchmark helpers.

Copied and trimmed from
``scripts/test_mega_ring/mega_ring_test_min_fa3_varlen_hybrid_multi_rank.py``.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


SENTINEL = -123.0
BASE_SEED = 20260713
MEGA_RING_ALL_CP_ALIGNMENT = 2048


def align_mega_ring_all_cp_lengths(global_lengths: list[int]) -> list[int]:
    """Round global lengths for the all-CP mega-ring's eight-rank alignment."""
    alignment = MEGA_RING_ALL_CP_ALIGNMENT
    return [
        ((length + alignment - 1) // alignment) * alignment
        for length in global_lengths
    ]


def parse_int_list(spec: str, name: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise SystemExit(f"{name} must provide at least one integer")
    return values


def init_distributed() -> tuple[int, int]:
    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit("Run this benchmark with torchrun")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(rank)
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
            if is_causal and ring_size > 1:
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
        k_batch = torch.cat(k_parts)
        v_batch = torch.cat(v_parts)
        if is_causal and ring_size > 1:
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
