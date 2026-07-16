"""Negative host-validation tests for hierarchical mega-ring backward."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

import min_fa3_op
from mega_ring_test_min_fa3_varlen_hybrid_multi_rank import (
    assert_all_ranks,
    init_distributed,
    make_cu_seqlens,
)


Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
BASE_RANK_CAPACITY = 512


def int32(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


def noncontiguous_int32(values: list[int]) -> torch.Tensor:
    storage = torch.empty((2 * len(values),), dtype=torch.int32)
    view = storage[::2]
    view.copy_(int32(values))
    assert not view.is_contiguous()
    return view


def make_parallel_pair(
    shape: list[int],
    dtype: torch.dtype,
    rank: int,
    world_size: int,
) -> tuple[min_fa3_op.TKParallelTensor, min_fa3_op.TKParallelTensor]:
    return (
        min_fa3_op.TKParallelTensor(shape, dtype, rank, world_size, False),
        min_fa3_op.TKParallelTensor(shape, dtype, rank, world_size, False),
    )


def make_cases(rank: int, world_size: int, device: torch.device) -> list[dict[str, Any]]:
    valid_global = int32([2048])
    valid_rings = int32([8])
    valid_starts = int32([0])

    nonmember_lengths = [0] * world_size
    nonmember_lengths[rank] = 128
    nonmember_lengths[(rank + 1) % world_size] = 128

    return [
        {
            "name": "host_cu_seqlens_dtype",
            "expected": "host cu_seqlens tensors must have dtype int32",
            "q_lengths": [0],
            "q_host_override": torch.tensor([0, 0], dtype=torch.int64),
            "global": valid_global,
            "rings": valid_rings,
            "starts": valid_starts,
        },
        {
            "name": "topology_metadata_device",
            "expected": "global_seqlens_host must be a CPU tensor",
            "q_lengths": [0],
            "global": valid_global.to(device=device),
            "rings": valid_rings,
            "starts": valid_starts,
        },
        {
            "name": "topology_metadata_dtype",
            "expected": "global_seqlens_host must have dtype int32",
            "q_lengths": [0],
            "global": torch.tensor([2048], dtype=torch.int64),
            "rings": valid_rings,
            "starts": valid_starts,
        },
        {
            "name": "topology_metadata_shape",
            "expected": "ring_starts_host must have shape [B]",
            "q_lengths": [0],
            "global": valid_global,
            "rings": valid_rings,
            "starts": int32([0, 0]),
        },
        {
            "name": "topology_metadata_contiguity",
            "expected": "ring_sizes_host must be contiguous",
            "q_lengths": [0, 0],
            "global": int32([2048, 1024]),
            "rings": noncontiguous_int32([8, 4]),
            "starts": int32([0, 4]),
        },
        {
            "name": "unordered_ring_sizes",
            "expected": "batch must be ordered by non-increasing ring size",
            "q_lengths": [256 if rank < 2 else 0, 256 if rank < 4 else 0],
            "global": int32([512, 1024]),
            "rings": int32([2, 4]),
            "starts": int32([0, 0]),
        },
        {
            "name": "unsupported_ring_size",
            "expected": "ring_sizes_host must contain only 1, 2, 4, or 8",
            "q_lengths": [0],
            "global": int32([384]),
            "rings": int32([3]),
            "starts": int32([0]),
        },
        {
            "name": "misaligned_ring_start",
            "expected": "invalid aligned ring range",
            "q_lengths": [0],
            "global": int32([512]),
            "rings": int32([2]),
            "starts": int32([1]),
        },
        {
            "name": "ring_range_crosses_world",
            "expected": "invalid aligned ring range",
            "q_lengths": [0],
            "global": int32([512]),
            "rings": int32([2]),
            "starts": int32([world_size]),
        },
        {
            "name": "global_length_not_divisible",
            "expected": "global length must be divisible by ring size",
            "q_lengths": [0],
            "global": int32([1025]),
            "rings": int32([8]),
            "starts": int32([0]),
        },
        {
            "name": "nonmember_has_local_rows",
            "expected": "local q/k length does not match ring metadata",
            "q_lengths": nonmember_lengths,
            "global": int32([128] * world_size),
            "rings": int32([1] * world_size),
            "starts": int32(list(range(world_size))),
        },
        {
            "name": "member_length_mismatch",
            "expected": "local q/k length does not match ring metadata",
            "q_lengths": [128],
            "global": valid_global,
            "rings": valid_rings,
            "starts": valid_starts,
        },
        {
            "name": "self_attention_length_mismatch",
            "expected": "self-attention q_len == k_len",
            "q_lengths": [128],
            "k_lengths": [256],
            "global": valid_global,
            "rings": valid_rings,
            "starts": valid_starts,
        },
        {
            "name": "local_length_alignment",
            "expected": "every local q/k length must be 128-row aligned",
            "q_lengths": [130],
            "global": int32([1040]),
            "rings": valid_rings,
            "starts": valid_starts,
        },
        {
            "name": "arena_capacity_insufficient",
            "expected": "local_total_k must fit in rank_kv_capacity",
            "q_lengths": [256],
            "global": valid_global,
            "rings": valid_rings,
            "starts": valid_starts,
            "arena": "small",
        },
        {
            "name": "arena_capacity_unaligned",
            "expected": "rank_kv_capacity must be 128-row aligned",
            "q_lengths": [128],
            "global": int32([1024]),
            "rings": valid_rings,
            "starts": valid_starts,
            "arena": "unaligned",
        },
        {
            "name": "causal_half_unaligned",
            "expected": "G2/G4/G8 local half length must be 128-row aligned",
            "q_lengths": [128],
            "global": int32([1024]),
            "rings": valid_rings,
            "starts": valid_starts,
        },
        {
            "name": "accumulator_dtype",
            "expected": "remote dK/dV accumulators must have dtype float32",
            "q_lengths": [256],
            "global": valid_global,
            "rings": valid_rings,
            "starts": valid_starts,
            "accum": "dtype",
        },
        {
            "name": "accumulator_capacity",
            "expected": "remote dK/dV accumulators must each contain",
            "q_lengths": [256],
            "global": valid_global,
            "rings": valid_rings,
            "starts": valid_starts,
        },
        {
            "name": "completion_shape",
            "expected": "remote_dkv_completion must be a one-element int32 parallel tensor",
            "q_lengths": [256],
            "global": valid_global,
            "rings": valid_rings,
            "starts": valid_starts,
            "completion": "shape",
        },
    ]


def run_validation(rank: int, world_size: int) -> None:
    device = torch.device("cuda", rank)
    guard_padded_capacity = BASE_RANK_CAPACITY + (world_size + 1) * 128
    guard_accum_numel = KV_HEADS * guard_padded_capacity * HEAD_DIM
    arenas = {
        "base": make_parallel_pair(
            [world_size * BASE_RANK_CAPACITY, KV_HEADS, HEAD_DIM],
            torch.bfloat16,
            rank,
            world_size,
        ),
        "small": make_parallel_pair(
            [world_size * 128, KV_HEADS, HEAD_DIM],
            torch.bfloat16,
            rank,
            world_size,
        ),
        "unaligned": make_parallel_pair(
            [world_size * 129, KV_HEADS, HEAD_DIM],
            torch.bfloat16,
            rank,
            world_size,
        ),
    }
    accumulators = {
        # This is intentionally larger than every case but has the wrong exact
        # numel. It is a no-launch safeguard for checks that run earlier.
        "shape": make_parallel_pair(
            [guard_accum_numel], torch.float32, rank, world_size
        ),
        "dtype": make_parallel_pair([1], torch.bfloat16, rank, world_size),
    }
    completions = {
        "valid": min_fa3_op.TKParallelTensor([1], torch.int32, rank, world_size, False),
        "shape": min_fa3_op.TKParallelTensor([2], torch.int32, rank, world_size, False),
    }

    cases = make_cases(rank, world_size, device)
    for case in cases:
        q_lengths = case["q_lengths"]
        k_lengths = case.get("k_lengths", q_lengths)
        cu_q, cu_q_host = make_cu_seqlens(q_lengths, device)
        cu_k, cu_k_host = make_cu_seqlens(k_lengths, device)
        q_host_arg = case.get("q_host_override", cu_q_host)
        k_host_arg = case.get("k_host_override", cu_k_host)

        total_q = int(cu_q_host[-1])
        q = torch.zeros(
            (total_q, Q_HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16
        )
        out = torch.zeros_like(q)
        dout = torch.zeros_like(q)
        lse = torch.zeros((Q_HEADS, total_q), device=device, dtype=torch.float32)

        remote_k, remote_v = arenas[case.get("arena", "base")]
        remote_dk, remote_dv = accumulators[case.get("accum", "shape")]
        completion = completions[case.get("completion", "valid")]

        local_error = None
        try:
            min_fa3_op.backward_varlen_mega_ring(
                dout,
                q,
                remote_k.data_,
                remote_v.data_,
                out,
                lse,
                cu_q,
                cu_k,
                256,
                256,
                cu_seqlens_q_host=q_host_arg,
                cu_seqlens_k_host=k_host_arg,
                remote_k=remote_k,
                remote_v=remote_v,
                remote_dk_accum=remote_dk,
                remote_dv_accum=remote_dv,
                remote_dkv_completion=completion,
                num_comp_sm=1,
                num_comm_sm=1,
                global_seqlens_host=case["global"],
                ring_sizes_host=case["rings"],
                ring_starts_host=case["starts"],
            )
        except RuntimeError as exc:
            message = str(exc)
            if case["expected"] not in message:
                local_error = (
                    f"{case['name']}: expected error containing {case['expected']!r}, "
                    f"got {message!r}"
                )
        else:
            local_error = f"{case['name']}: invalid input was accepted"

        assert_all_ranks(local_error)
        if rank == 0:
            print(f"validation {case['name']}: ok", flush=True)
        dist.barrier()

    if rank == 0:
        print(f"hierarchical mega-ring backward validation: {len(cases)} cases passed")


if __name__ == "__main__":
    rank, world_size = init_distributed()
    if world_size != 8:
        raise SystemExit("Run validation coverage with eight local ranks")
    try:
        run_validation(rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
