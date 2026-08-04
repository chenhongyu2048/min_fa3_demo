"""Head-sharded decode context parallel attention for the minimal FA3 demo.

The LSE correction is copied and trimmed from vLLM commit
``a89015c6df8eeb37a843b717c97a5be1355de83d``:
``vllm/v1/attention/ops/common.py::cp_lse_ag_out_rs``.  The two-state merge is
copied and trimmed from the same commit's
``vllm/v1/attention/ops/triton_merge_attn_states.py``.  This module has no
runtime dependency on vLLM or SGLang.

The communication-stream/event ordering follows the side-stream design used
by SGLang commit ``8d6549bc4039d33635844495d86684677a4f0df8``.  The runner's
formal capture APIs include the local kernel, post-processing, NCCL
collectives, and the optional communication-stream fork/join in one CUDA
graph.  Its workspace must not be used concurrently.

The module also contains standalone, copied-and-trimmed vLLM ``ag_rs`` and
``a2a`` runners plus an SGLang MHA runner.  All runners invoke the same local
dense or packed KV-cache kernel; only orchestration differs.

The TP/DCP topology validation is copied and trimmed from the GQA/MQA
constraints in the pinned vLLM commit and the contiguous DCP group
construction in the pinned SGLang commit.
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import min_fa3_op
from dcp_mega_metadata import (
    METADATA_HEADER_INTS,
    build_dcp_mega_metadata,
    pack_dcp_mega_metadata,
)


@dataclass(frozen=True)
class TopologyIssue:
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DCPTopology:
    q_heads: int
    kv_heads: int
    tp_size: int
    dcp_size: int

    @property
    def q_heads_local(self) -> int:
        return self.q_heads // self.tp_size

    @property
    def q_heads_per_kv(self) -> int:
        return self.q_heads // self.kv_heads

    @property
    def kv_replicas(self) -> int:
        return self.tp_size // self.kv_heads

    def kv_head_for_rank(self, tp_rank: int) -> int:
        self._check_rank(tp_rank)
        return tp_rank // self.kv_replicas

    def kv_replica_ranks(self, tp_rank: int) -> tuple[int, ...]:
        kv_head = self.kv_head_for_rank(tp_rank)
        start = kv_head * self.kv_replicas
        return tuple(range(start, start + self.kv_replicas))

    def dcp_group_ranks(self, tp_rank: int) -> tuple[int, ...]:
        replica_ranks = self.kv_replica_ranks(tp_rank)
        offset = tp_rank - replica_ranks[0]
        start = replica_ranks[0] + (offset // self.dcp_size) * self.dcp_size
        return tuple(range(start, start + self.dcp_size))

    def dcp_rank(self, tp_rank: int) -> int:
        group = self.dcp_group_ranks(tp_rank)
        return tp_rank - group[0]

    def q_head_range(self, tp_rank: int) -> tuple[int, int]:
        self._check_rank(tp_rank)
        start = tp_rank * self.q_heads_local
        return start, start + self.q_heads_local

    def all_dcp_groups(self) -> tuple[tuple[int, ...], ...]:
        groups: list[tuple[int, ...]] = []
        for rank in range(self.tp_size):
            group = self.dcp_group_ranks(rank)
            if not groups or groups[-1] != group:
                groups.append(group)
        return tuple(groups)

    def to_dict(self) -> dict[str, object]:
        return {
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "tp_size": self.tp_size,
            "dcp_size": self.dcp_size,
            "q_heads_local": self.q_heads_local,
            "q_heads_per_kv": self.q_heads_per_kv,
            "kv_replicas": self.kv_replicas,
            "dcp_groups": [list(group) for group in self.all_dcp_groups()],
        }

    def _check_rank(self, tp_rank: int) -> None:
        if not 0 <= tp_rank < self.tp_size:
            raise ValueError(
                f"tp_rank must be in [0, {self.tp_size}), got {tp_rank}"
            )


def validate_topology(
    q_heads: int,
    kv_heads: int,
    tp_size: int,
    dcp_size: int,
) -> tuple[TopologyIssue, ...]:
    issues: list[TopologyIssue] = []
    values = {
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "tp_size": tp_size,
        "dcp_size": dcp_size,
    }
    for name, value in values.items():
        if value <= 0:
            issues.append(
                TopologyIssue("nonpositive_value", f"{name} must be positive, got {value}")
            )
    if issues:
        return tuple(issues)

    if q_heads % tp_size:
        issues.append(
            TopologyIssue(
                "q_heads_not_divisible_by_tp",
                f"global Q heads {q_heads} must be divisible by TP size {tp_size}",
            )
        )
    if q_heads % kv_heads:
        issues.append(
            TopologyIssue(
                "q_heads_not_divisible_by_kv_heads",
                f"global Q heads {q_heads} must be divisible by global KV heads {kv_heads}",
            )
        )
    if tp_size <= kv_heads:
        issues.append(
            TopologyIssue(
                "tp_not_greater_than_kv_heads",
                f"DCP GQA/MQA requires TP size {tp_size} > global KV heads {kv_heads}",
            )
        )
    if tp_size % kv_heads:
        issues.append(
            TopologyIssue(
                "kv_heads_not_divisible_into_tp",
                f"TP size {tp_size} must be divisible by global KV heads {kv_heads}",
            )
        )
    if tp_size % dcp_size:
        issues.append(
            TopologyIssue(
                "dcp_not_divisible_into_tp",
                f"DCP size {dcp_size} must divide TP size {tp_size}",
            )
        )

    if tp_size % kv_heads == 0:
        replicas = tp_size // kv_heads
        if dcp_size > replicas:
            issues.append(
                TopologyIssue(
                    "dcp_exceeds_kv_replicas",
                    f"DCP size {dcp_size} exceeds KV replica count {replicas}",
                )
            )
        if replicas % dcp_size:
            issues.append(
                TopologyIssue(
                    "kv_replicas_not_divisible_by_dcp",
                    f"KV replica count {replicas} must be divisible by DCP size {dcp_size}",
                )
            )

    if q_heads % kv_heads == 0:
        q_per_kv = q_heads // kv_heads
        if q_per_kv % dcp_size:
            issues.append(
                TopologyIssue(
                    "q_per_kv_not_divisible_by_dcp",
                    f"Q heads per KV head {q_per_kv} must be divisible by DCP size {dcp_size}",
                )
            )
    return tuple(issues)


def make_topology(
    q_heads: int,
    kv_heads: int,
    tp_size: int,
    dcp_size: int,
) -> DCPTopology:
    issues = validate_topology(q_heads, kv_heads, tp_size, dcp_size)
    if issues:
        details = "; ".join(issue.detail for issue in issues)
        raise ValueError(f"invalid DCP topology: {details}")
    return DCPTopology(q_heads, kv_heads, tp_size, dcp_size)


def validate_group_ranks(
    topology: DCPTopology,
    ranks: Iterable[int],
) -> tuple[TopologyIssue, ...]:
    group = tuple(ranks)
    issues: list[TopologyIssue] = []
    if len(group) != topology.dcp_size:
        issues.append(
            TopologyIssue(
                "dcp_group_wrong_size",
                f"DCP group has {len(group)} ranks, expected {topology.dcp_size}",
            )
        )
    if any(rank < 0 or rank >= topology.tp_size for rank in group):
        issues.append(
            TopologyIssue(
                "dcp_group_rank_out_of_range",
                f"DCP group ranks must be in [0, {topology.tp_size}), got {group}",
            )
        )
        return tuple(issues)
    kv_heads = {topology.kv_head_for_rank(rank) for rank in group}
    if len(kv_heads) != 1:
        issues.append(
            TopologyIssue(
                "dcp_group_crosses_kv_replica_boundary",
                f"DCP group {group} spans global KV heads {sorted(kv_heads)}",
            )
        )
    if group and group != tuple(range(group[0], group[0] + len(group))):
        issues.append(
            TopologyIssue(
                "dcp_group_not_contiguous",
                f"DCP group must contain contiguous TP ranks, got {group}",
            )
        )
    return tuple(issues)


_DCPResult = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


# Copied and trimmed from vLLM commit
# a89015c6df8eeb37a843b717c97a5be1355de83d,
# vllm/v1/attention/ops/dcp_alltoall.py.  The packed A2A combine arrived in
# vLLM PR #41160; PR #45487 made the per-call allocations CUDA-Graph safe and
# PR #47801 restored the required FP32 LSE bit-cast contract.
def _dcp_a2a_lse_weighted_combine_reference(
    outputs: torch.Tensor,
    lses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Torch base-e reference for the copied A2A combine."""
    if outputs.ndim != 4 or lses.shape != outputs.shape[:3]:
        raise ValueError("outputs must be [N, T, H, D] and lses must be [N, T, H]")
    valid_lses = torch.where(
        torch.isnan(lses) | torch.isinf(lses),
        torch.full_like(lses, -float("inf")),
        lses,
    )
    lse_max = valid_lses.max(dim=0).values
    finite_max = torch.where(
        lse_max == -float("inf"), torch.zeros_like(lse_max), lse_max
    )
    weights = torch.exp(valid_lses - finite_max.unsqueeze(0))
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
    weight_sum = weights.sum(dim=0)
    normalized = weights / weight_sum.clamp(min=1.0e-10).unsqueeze(0)
    combined = (outputs * normalized.unsqueeze(-1)).sum(dim=0)
    global_lse = torch.log(weight_sum) + finite_max
    return combined, global_lse


def _dcp_a2a_lse_pack_dim(output_dtype: torch.dtype) -> int:
    bits = torch.finfo(output_dtype).bits
    if bits == 16:
        return 2
    if bits == 32:
        return 1
    raise ValueError(f"Cannot pack fp32 LSE into output dtype {output_dtype}.")


def _dcp_a2a_head_owner_ranges(
    h_group: int, world_size: int
) -> tuple[tuple[int, int], ...]:
    if h_group <= 0 or world_size <= 0 or h_group % world_size:
        raise ValueError(
            f"H_group={h_group} must be positive and divisible by DCP={world_size}"
        )
    h_local = h_group // world_size
    return tuple(
        (rank * h_local, (rank + 1) * h_local) for rank in range(world_size)
    )


def _dcp_a2a_payload_bytes(
    total_tokens: int,
    h_local: int,
    head_dim: int,
    world_size: int,
    element_size: int = 2,
) -> tuple[int, int]:
    values = (total_tokens, h_local, head_dim, world_size, element_size)
    if any(value <= 0 for value in values):
        raise ValueError("A2A payload dimensions and element_size must be positive")
    per_owner = total_tokens * h_local * (head_dim + 2) * element_size
    return world_size * per_owner, (world_size - 1) * per_owner


@triton.jit
def _dcp_a2a_pack_send_kernel(
    out_ptr,
    lse_ptr,
    send_ptr,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    lse_stride_B,
    lse_stride_H,
    send_stride_N,
    send_stride_B,
    send_stride_H,
    send_stride_D,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H_PER_RANK: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    local_head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    for rank_idx in tl.static_range(N):
        src_head_idx = rank_idx * H_PER_RANK + local_head_idx
        send_base = (
            rank_idx * send_stride_N
            + batch_idx * send_stride_B
            + local_head_idx * send_stride_H
        )
        out_offsets = (
            batch_idx * out_stride_B
            + src_head_idx * out_stride_H
            + d_offsets * out_stride_D
        )
        tl.store(
            send_ptr + send_base + d_offsets * send_stride_D,
            tl.load(out_ptr + out_offsets),
        )

        lse_val = tl.load(
            lse_ptr + batch_idx * lse_stride_B + src_head_idx * lse_stride_H
        )
        if LSE_PACK_DIM == 1:
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lse_val.to(send_ptr.dtype.element_ty),
            )
        else:
            lse_bits = lse_val.to(tl.uint32, bitcast=True)
            lo = (lse_bits & 0xFFFF).to(tl.uint16)
            hi = ((lse_bits >> 16) & 0xFFFF).to(tl.uint16)
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lo.to(send_ptr.dtype.element_ty, bitcast=True),
            )
            tl.store(
                send_ptr + send_base + (HEAD_DIM + 1) * send_stride_D,
                hi.to(send_ptr.dtype.element_ty, bitcast=True),
            )


@triton.jit
def _dcp_a2a_unpack_combine_kernel(
    recv_ptr,
    out_ptr,
    out_lse_ptr,
    recv_stride_N,
    recv_stride_B,
    recv_stride_H,
    recv_stride_D,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    out_lse_stride_B,
    out_lse_stride_H,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    lse_max = -float("inf")
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(
                recv_ptr + recv_base + HEAD_DIM * recv_stride_D
            ).to(tl.float32)
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(
                recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D
            )
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        lse_max = tl.maximum(lse_max, lse_val)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    lse_sum = 0.0
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(
                recv_ptr + recv_base + HEAD_DIM * recv_stride_D
            ).to(tl.float32)
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(
                recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D
            )
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            lse_sum += tl.exp(lse_val - lse_max)
        else:
            lse_sum += tl.exp2(lse_val - lse_max)

    if IS_BASE_E:
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(
                recv_ptr + recv_base + HEAD_DIM * recv_stride_D
            ).to(tl.float32)
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(
                recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D
            )
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            weight = tl.exp(lse_val - global_lse)
        else:
            weight = tl.exp2(lse_val - global_lse)
        weight = tl.where(weight != weight, 0.0, weight)
        acc += (
            tl.load(recv_ptr + recv_base + d_offsets * recv_stride_D).to(
                tl.float32
            )
            * weight
        )

    final_offsets = (
        batch_idx * out_stride_B
        + head_idx * out_stride_H
        + d_offsets * out_stride_D
    )
    tl.store(out_ptr + final_offsets, acc)
    if RETURN_LSE:
        out_lse_offset = (
            batch_idx * out_lse_stride_B + head_idx * out_lse_stride_H
        )
        tl.store(out_lse_ptr + out_lse_offset, global_lse)


def _dcp_a2a_pack_send(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    send_buffer: torch.Tensor,
    world_size: int,
    h_per_rank: int,
    head_dim: int,
    lse_pack_dim: int,
) -> None:
    grid = (partial_out.shape[0], h_per_rank, 1)
    _dcp_a2a_pack_send_kernel[grid](
        partial_out,
        partial_lse,
        send_buffer,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        send_buffer.stride(0),
        send_buffer.stride(1),
        send_buffer.stride(2),
        send_buffer.stride(3),
        N=world_size,
        HEAD_DIM=head_dim,
        H_PER_RANK=h_per_rank,
        LSE_PACK_DIM=lse_pack_dim,
    )


def _dcp_a2a_unpack_combine(
    recv_buffer: torch.Tensor,
    head_dim: int,
    lse_pack_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_size, total_tokens, h_per_rank, _ = recv_buffer.shape
    output = torch.empty(
        (total_tokens, h_per_rank, head_dim),
        device=recv_buffer.device,
        dtype=recv_buffer.dtype,
    )
    output_lse = torch.empty(
        (total_tokens, h_per_rank),
        device=recv_buffer.device,
        dtype=torch.float32,
    )
    grid = (total_tokens, h_per_rank, 1)
    _dcp_a2a_unpack_combine_kernel[grid](
        recv_buffer,
        output,
        output_lse,
        recv_buffer.stride(0),
        recv_buffer.stride(1),
        recv_buffer.stride(2),
        recv_buffer.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output_lse.stride(0),
        output_lse.stride(1),
        N=world_size,
        HEAD_DIM=head_dim,
        IS_BASE_E=True,
        RETURN_LSE=True,
        LSE_PACK_DIM=lse_pack_dim,
    )
    return output, output_lse


@triton.jit
def _vllm_correct_attn_cp_out_kernel(
    partial_out_ptr,
    gathered_lse_ptr,
    corrected_out_ptr,
    global_lse_ptr,
    out_stride_t,
    out_stride_h,
    lse_stride_n,
    lse_stride_t,
    lse_stride_h,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    WORLD_SIZE_ROUNDED: tl.constexpr,
    RANK: tl.constexpr,
):
    """Trimmed from vLLM ``_correct_attn_cp_out_kernel`` at the pinned commit."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    rank_offsets = tl.arange(0, WORLD_SIZE_ROUNDED)
    rank_mask = rank_offsets < WORLD_SIZE
    lse_offsets = (
        rank_offsets * lse_stride_n
        + token_idx * lse_stride_t
        + head_idx * lse_stride_h
    )
    partial_lses = tl.load(
        gathered_lse_ptr + lse_offsets, mask=rank_mask, other=-float("inf")
    )
    partial_lses = tl.where(
        (partial_lses != partial_lses) | (partial_lses == float("inf")),
        -float("inf"),
        partial_lses,
    )
    lse_max = tl.max(partial_lses, axis=0)
    finite_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    lse_sum = tl.sum(tl.exp(partial_lses - finite_max), axis=0)
    global_lse = tl.log(lse_sum) + finite_max
    tl.store(global_lse_ptr + token_idx * H + head_idx, global_lse)

    local_lse = tl.load(
        gathered_lse_ptr
        + RANK * lse_stride_n
        + token_idx * lse_stride_t
        + head_idx * lse_stride_h
    )
    log_weight = local_lse - global_lse
    log_weight = tl.where(
        (log_weight != log_weight) | (log_weight == float("inf")),
        -float("inf"),
        log_weight,
    )
    weight = tl.exp(log_weight)
    dim_offsets = tl.arange(0, D)
    out_offsets = token_idx * out_stride_t + head_idx * out_stride_h + dim_offsets
    values = tl.load(partial_out_ptr + out_offsets).to(tl.float32) * weight
    tl.store(corrected_out_ptr + out_offsets, tl.where(weight == 0.0, 0.0, values))


@triton.jit
def _correct_and_pack_kernel(
    partial_out_ptr,
    gathered_lse_ptr,
    packed_out_ptr,
    local_lse_ptr,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    lse_stride_n,
    lse_stride_b,
    lse_stride_h,
    lse_stride_s,
    local_lse_stride_b,
    local_lse_stride_h,
    local_lse_stride_s,
    B: tl.constexpr,
    SQ: tl.constexpr,
    H_GROUP: tl.constexpr,
    H_LOCAL: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    WORLD_SIZE_ROUNDED: tl.constexpr,
    RANK: tl.constexpr,
    STORE_LOCAL_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = token_idx // SQ
    query_idx = token_idx - batch_idx * SQ

    rank_offsets = tl.arange(0, WORLD_SIZE_ROUNDED)
    rank_mask = rank_offsets < WORLD_SIZE
    lse_offsets = (
        rank_offsets * lse_stride_n
        + batch_idx * lse_stride_b
        + head_idx * lse_stride_h
        + query_idx * lse_stride_s
    )
    partial_lses = tl.load(gathered_lse_ptr + lse_offsets, mask=rank_mask, other=-float("inf"))
    partial_lses = tl.where(
        (partial_lses != partial_lses) | (partial_lses == float("inf")),
        -float("inf"),
        partial_lses,
    )
    lse_max = tl.max(partial_lses, axis=0)
    finite_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    lse_sum = tl.sum(tl.exp(partial_lses - finite_max), axis=0)
    global_lse = tl.log(lse_sum) + finite_max

    local_lse = tl.load(
        gathered_lse_ptr
        + RANK * lse_stride_n
        + batch_idx * lse_stride_b
        + head_idx * lse_stride_h
        + query_idx * lse_stride_s
    )
    log_weight = local_lse - global_lse
    log_weight = tl.where(
        (log_weight != log_weight) | (log_weight == float("inf")),
        -float("inf"),
        log_weight,
    )
    weight = tl.exp(log_weight)

    dim_offsets = tl.arange(0, D)
    partial_offsets = (
        batch_idx * out_stride_b
        + query_idx * out_stride_s
        + head_idx * out_stride_h
        + dim_offsets
    )
    values = tl.load(partial_out_ptr + partial_offsets).to(tl.float32) * weight
    values = tl.where(weight == 0.0, 0.0, values)

    heads_per_group_kv = H_GROUP // H_KV
    heads_per_local_kv = H_LOCAL // H_KV
    kv_head = head_idx // heads_per_group_kv
    head_within_group_kv = head_idx - kv_head * heads_per_group_kv
    owner = head_within_group_kv // heads_per_local_kv
    local_head = (
        kv_head * heads_per_local_kv
        + head_within_group_kv
        - owner * heads_per_local_kv
    )
    packed_offsets = (
        ((((owner * B + batch_idx) * SQ + query_idx) * H_LOCAL + local_head) * D)
        + dim_offsets
    )
    tl.store(packed_out_ptr + packed_offsets, values)

    if STORE_LOCAL_LSE:
        if owner == RANK:
            local_lse_offset = (
                batch_idx * local_lse_stride_b
                + local_head * local_lse_stride_h
                + query_idx * local_lse_stride_s
            )
            tl.store(local_lse_ptr + local_lse_offset, global_lse)


@triton.jit
def _merge_attn_states_kernel(
    context_out_ptr,
    context_lse_ptr,
    chunk_out_ptr,
    chunk_lse_ptr,
    merged_out_ptr,
    merged_lse_ptr,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    lse_stride_b,
    lse_stride_h,
    lse_stride_s,
    B: tl.constexpr,
    SQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = token_idx // SQ
    query_idx = token_idx - batch_idx * SQ
    lse_offset = (
        batch_idx * lse_stride_b
        + head_idx * lse_stride_h
        + query_idx * lse_stride_s
    )
    context_lse = tl.load(context_lse_ptr + lse_offset).to(tl.float32)
    chunk_lse = tl.load(chunk_lse_ptr + lse_offset).to(tl.float32)
    context_lse = tl.where(
        (context_lse != context_lse) | (context_lse == float("inf")),
        -float("inf"),
        context_lse,
    )
    chunk_lse = tl.where(
        (chunk_lse != chunk_lse) | (chunk_lse == float("inf")),
        -float("inf"),
        chunk_lse,
    )
    lse_max = tl.maximum(context_lse, chunk_lse)
    finite_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    context_weight = tl.exp(context_lse - finite_max)
    chunk_weight = tl.exp(chunk_lse - finite_max)
    weight_sum = context_weight + chunk_weight
    merged_lse = tl.log(weight_sum) + finite_max
    inverse_weight_sum = tl.where(weight_sum == 0.0, 0.0, 1.0 / weight_sum)

    dim_offsets = tl.arange(0, D)
    out_offset = (
        batch_idx * out_stride_b
        + query_idx * out_stride_s
        + head_idx * out_stride_h
        + dim_offsets
    )
    context_out = tl.load(context_out_ptr + out_offset).to(tl.float32)
    chunk_out = tl.load(chunk_out_ptr + out_offset).to(tl.float32)
    merged_out = (
        context_out * context_weight + chunk_out * chunk_weight
    ) * inverse_weight_sum
    merged_out = tl.where(weight_sum == 0.0, 0.0, merged_out)
    tl.store(merged_out_ptr + out_offset, merged_out)
    if STORE_LSE:
        tl.store(merged_lse_ptr + lse_offset, merged_lse)


def _nvtx_range(name: str):
    return torch.cuda.nvtx.range(name) if hasattr(torch.cuda, "nvtx") else nullcontext()


class DCPAttentionCUDAGraph:
    """One fixed-signature CUDA Graph captured from a DCP attention runner.

    The input and output tensors are capture-bound.  Callers may update input
    tensor contents in place before :meth:`replay`, but must close and
    recapture when an address, shape, stride, dtype, or captured scalar changes.
    """

    def __init__(
        self,
        runner: "DCPAttentionRunner",
        graph: torch.cuda.CUDAGraph,
        output: _DCPResult,
        capture_stream: torch.cuda.Stream,
        static_references: tuple[object, ...],
        retained_works: tuple[dist.Work, ...],
        signature: dict[str, object],
        *,
        record_timing: bool,
        overlap_q_allgather: bool,
    ) -> None:
        self._runner: Optional[DCPAttentionRunner] = runner
        self._graph: Optional[torch.cuda.CUDAGraph] = graph
        self._output: Optional[_DCPResult] = output
        self._capture_stream: Optional[torch.cuda.Stream] = capture_stream
        self._static_references = static_references
        self._retained_works = retained_works
        self.signature = signature
        self.record_timing = record_timing
        self.overlap_q_allgather = overlap_q_allgather
        self._closed = False

    @property
    def output(self) -> _DCPResult:
        """Return the static output whose contents are replaced by each replay."""
        if self._closed or self._output is None:
            raise RuntimeError("DCPAttentionCUDAGraph is closed")
        return self._output

    def replay(self) -> _DCPResult:
        """Asynchronously enqueue one replay and return the static output."""
        if self._closed or self._graph is None or self._runner is None:
            raise RuntimeError("DCPAttentionCUDAGraph is closed")
        capture_stream = self._capture_stream
        assert capture_stream is not None
        current_stream = torch.cuda.current_stream(self._runner.device)
        if current_stream.cuda_stream != capture_stream.cuda_stream:
            capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(capture_stream):
                self._graph.replay()
            current_stream.wait_stream(capture_stream)
        else:
            self._graph.replay()
        return self.output

    def close(self) -> None:
        """Synchronize replay, reset the graph, and release runner ownership."""
        if self._closed:
            return
        runner = self._runner
        graph = self._graph
        try:
            if runner is not None:
                torch.cuda.synchronize(runner.device)
            if graph is not None:
                graph.reset()
        finally:
            if runner is not None:
                runner._release_graph(self)
            self._retained_works = ()
            self._static_references = ()
            self._capture_stream = None
            self._output = None
            self._graph = None
            self._runner = None
            self._closed = True

    def __enter__(self) -> "DCPAttentionCUDAGraph":
        if self._closed:
            raise RuntimeError("DCPAttentionCUDAGraph is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


class DCPAttentionRunner:
    """Reusable head-sharded DCP workspace backed by one NCCL process group.

    Calls may be enqueued back-to-back, including from different current CUDA
    streams.  Calling the same runner concurrently from multiple host threads
    is unsupported and raises ``RuntimeError``.
    """

    method_name = "ours_no_overlap"
    overlap_method_name = "ours_overlap"
    varlen_method_name = "ours_no_overlap_varlen"
    varlen_overlap_method_name = "ours_overlap_varlen"
    output_collective_kind = "bf16_reduce_scatter"
    workspace_policy = "persistent_grow_only"
    supports_overlap_q_allgather = True

    def __init__(self, process_group: Optional[dist.ProcessGroup]):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before DCPAttentionRunner")
        if not torch.cuda.is_available():
            raise RuntimeError("DCPAttentionRunner requires CUDA")
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)
        if self.world_size <= 0:
            raise RuntimeError("DCP process group must contain at least one rank")
        if self.world_size > 8:
            raise ValueError(f"This minimal DCP runner supports at most 8 ranks, got {self.world_size}")
        backend = str(dist.get_backend(process_group)).lower()
        if "nccl" not in backend:
            raise RuntimeError(f"DCPAttentionRunner requires an NCCL process group, got {backend}")

        self.device = torch.device("cuda", torch.cuda.current_device())
        self.communication_stream = torch.cuda.Stream(device=self.device)
        self._input_ready = torch.cuda.Event()
        self._q_group_ready = torch.cuda.Event()
        self._history_ready = torch.cuda.Event()
        self._context_ready = torch.cuda.Event()
        self._buffers: dict[str, torch.Tensor] = {}
        self._works: list[dist.Work] = []
        self._inflight_tensors: list[
            tuple[torch.cuda.Event, tuple[torch.Tensor, ...]]
        ] = []
        self._workspace_completion: Optional[tuple[torch.cuda.Event, int]] = None
        self._enqueue_lock = threading.Lock()
        self._active_graph: Optional[DCPAttentionCUDAGraph] = None
        self._capture_owner_thread: Optional[int] = None
        self._capture_in_progress = False
        self._capture_tensors: list[torch.Tensor] = []
        self._timing_events = {
            # External event nodes remain host-queryable after graph replay.
            # Dependency-only events above intentionally retain the default
            # internal capture semantics.
            name: torch.cuda.Event(enable_timing=True, external=True)
            for name in (
                "attention_start",
                "attention_end",
                "q_ag_start",
                "q_ag_end",
                "chunk_start",
                "chunk_end",
                "ag_chunk_end",
                "history_start",
                "history_end",
                "lse_correct_start",
                "lse_correct_end",
                "reduce_scatter_start",
                "reduce_scatter_end",
                "a2a_pack_start",
                "a2a_pack_end",
                "a2a_all_to_all_start",
                "a2a_all_to_all_end",
                "a2a_unpack_combine_start",
                "a2a_unpack_combine_end",
                "merge_start",
                "merge_end",
            )
        }
        self._last_timing_kind: Optional[str] = None

    @staticmethod
    def _tensor_signature(tensor: torch.Tensor) -> dict[str, object]:
        signature: dict[str, object] = {
            "address": tensor.data_ptr(),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
        if tensor.device.type == "cpu":
            signature["contents"] = tensor.tolist()
        return signature

    def _graph_signature(
        self,
        operation: str,
        bindings: dict[str, object],
    ) -> dict[str, object]:
        values: dict[str, object] = {}
        for name, value in bindings.items():
            values[name] = (
                self._tensor_signature(value)
                if isinstance(value, torch.Tensor)
                else value
            )
        return {
            "operation": operation,
            "runner": type(self).__name__,
            "world_size": self.world_size,
            "rank": self.rank,
            "bindings": values,
        }

    def _check_forward_state(self) -> None:
        owner = self._capture_owner_thread
        current_thread = threading.get_ident()
        if self._active_graph is not None:
            raise RuntimeError(
                "This runner has an active CUDA Graph; replay or close it before "
                "calling eager forward"
            )
        if owner is not None and owner != current_thread:
            raise RuntimeError("This runner is currently being captured by another thread")
        if torch.cuda.is_current_stream_capturing() and not self._capture_in_progress:
            raise RuntimeError(
                "Direct CUDA Graph capture is unsupported; use capture_decode(), "
                "capture_chunk_prefill(), capture_decode_varlen(), or "
                "capture_chunk_prefill_varlen()"
            )

    def _clear_completed_bookkeeping(self) -> None:
        torch.cuda.synchronize(self.device)
        self._works.clear()
        self._inflight_tensors.clear()
        self._workspace_completion = None

    def _capture_forward(
        self,
        operation: str,
        forward_call: Callable[[], _DCPResult],
        bindings: dict[str, object],
        *,
        overlap_q_allgather: bool,
        record_timing: bool,
        capture_warmup: int,
    ) -> DCPAttentionCUDAGraph:
        if not isinstance(capture_warmup, int) or capture_warmup < 0:
            raise ValueError("capture_warmup must be a nonnegative integer")
        if overlap_q_allgather and not self.supports_overlap_q_allgather:
            raise ValueError(
                f"{type(self).__name__} only supports single-stream execution; "
                "overlap_q_allgather must be False"
            )
        if self._active_graph is not None:
            raise RuntimeError("This runner already has an active CUDA Graph")
        if self._capture_owner_thread is not None:
            raise RuntimeError("This runner is already preparing a CUDA Graph capture")

        self._capture_owner_thread = threading.get_ident()
        capture_stream = torch.cuda.Stream(device=self.device)
        graph: Optional[torch.cuda.CUDAGraph] = None
        try:
            caller_stream = torch.cuda.current_stream(self.device)
            capture_stream.wait_stream(caller_stream)
            with torch.cuda.stream(capture_stream):
                for _ in range(capture_warmup):
                    forward_call()
            caller_stream.wait_stream(capture_stream)

            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.process_group)
            torch.cuda.synchronize(self.device)
            self._clear_completed_bookkeeping()
            self._capture_tensors = []

            graph = torch.cuda.CUDAGraph()
            self._capture_in_progress = True
            with torch.cuda.graph(
                graph,
                stream=capture_stream,
                capture_error_mode="global",
            ):
                output = forward_call()
            self._capture_in_progress = False
            caller_stream.wait_stream(capture_stream)

            static_references: tuple[object, ...] = (
                tuple(bindings.values()),
                output,
                tuple(self._buffers.values()),
                tuple(self._capture_tensors),
                tuple(self._inflight_tensors),
            )
            captured = DCPAttentionCUDAGraph(
                self,
                graph,
                output,
                capture_stream,
                static_references,
                tuple(self._works),
                self._graph_signature(operation, bindings),
                record_timing=record_timing,
                overlap_q_allgather=overlap_q_allgather,
            )
            self._active_graph = captured
            return captured
        except Exception:
            self._capture_in_progress = False
            torch.cuda.synchronize(self.device)
            if graph is not None:
                graph.reset()
            self._clear_completed_bookkeeping()
            raise
        finally:
            self._capture_owner_thread = None

    def _release_graph(self, graph: DCPAttentionCUDAGraph) -> None:
        if self._active_graph is graph:
            self._active_graph = None
        self._works.clear()
        self._inflight_tensors.clear()
        self._workspace_completion = None
        self._capture_tensors.clear()

    def capture_decode(
        self,
        q_local: torch.Tensor,
        k_cache_local: torch.Tensor,
        v_cache_local: torch.Tensor,
        cache_seqlens_local: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
        *,
        record_timing: bool = False,
        capture_warmup: int = 3,
    ) -> DCPAttentionCUDAGraph:
        bindings = dict(
            q_local=q_local,
            k_cache_local=k_cache_local,
            v_cache_local=v_cache_local,
            cache_seqlens_local=cache_seqlens_local,
            num_splits=num_splits,
            return_lse=return_lse,
            overlap_q_allgather=overlap_q_allgather,
        )
        return self._capture_forward(
            "decode",
            lambda: self.forward_decode(
                q_local,
                k_cache_local,
                v_cache_local,
                cache_seqlens_local,
                num_splits=num_splits,
                return_lse=return_lse,
                overlap_q_allgather=overlap_q_allgather,
                _record_timing=record_timing,
            ),
            bindings,
            overlap_q_allgather=overlap_q_allgather,
            record_timing=record_timing,
            capture_warmup=capture_warmup,
        )

    def capture_chunk_prefill(
        self,
        q_local: torch.Tensor,
        k_history_local: torch.Tensor,
        v_history_local: torch.Tensor,
        history_seqlens_local: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
        *,
        record_timing: bool = False,
        capture_warmup: int = 3,
    ) -> DCPAttentionCUDAGraph:
        bindings = dict(
            q_local=q_local,
            k_history_local=k_history_local,
            v_history_local=v_history_local,
            history_seqlens_local=history_seqlens_local,
            k_chunk=k_chunk,
            v_chunk=v_chunk,
            num_splits=num_splits,
            return_lse=return_lse,
            overlap_q_allgather=overlap_q_allgather,
        )
        return self._capture_forward(
            "chunk_prefill",
            lambda: self.forward_chunk_prefill(
                q_local,
                k_history_local,
                v_history_local,
                history_seqlens_local,
                k_chunk,
                v_chunk,
                num_splits=num_splits,
                return_lse=return_lse,
                overlap_q_allgather=overlap_q_allgather,
                _record_timing=record_timing,
            ),
            bindings,
            overlap_q_allgather=overlap_q_allgather,
            record_timing=record_timing,
            capture_warmup=capture_warmup,
        )

    def capture_decode_varlen(
        self,
        q_local: torch.Tensor,
        k_cache_local: torch.Tensor,
        v_cache_local: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_local: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k_local: int,
        *,
        cu_seqlens_q_host: torch.Tensor,
        cu_seqlens_k_local_host: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
        record_timing: bool = False,
        capture_warmup: int = 3,
    ) -> DCPAttentionCUDAGraph:
        bindings = dict(
            q_local=q_local,
            k_cache_local=k_cache_local,
            v_cache_local=v_cache_local,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_local=cu_seqlens_k_local,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k_local=max_seqlen_k_local,
            cu_seqlens_q_host=cu_seqlens_q_host,
            cu_seqlens_k_local_host=cu_seqlens_k_local_host,
            num_splits=num_splits,
            return_lse=return_lse,
            overlap_q_allgather=overlap_q_allgather,
        )
        return self._capture_forward(
            "decode_varlen",
            lambda: self.forward_decode_varlen(
                q_local,
                k_cache_local,
                v_cache_local,
                cu_seqlens_q,
                cu_seqlens_k_local,
                max_seqlen_q,
                max_seqlen_k_local,
                cu_seqlens_q_host=cu_seqlens_q_host,
                cu_seqlens_k_local_host=cu_seqlens_k_local_host,
                num_splits=num_splits,
                return_lse=return_lse,
                overlap_q_allgather=overlap_q_allgather,
                _record_timing=record_timing,
            ),
            bindings,
            overlap_q_allgather=overlap_q_allgather,
            record_timing=record_timing,
            capture_warmup=capture_warmup,
        )

    def capture_chunk_prefill_varlen(
        self,
        q_local: torch.Tensor,
        k_history_local: torch.Tensor,
        v_history_local: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_history_local: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_history_local: int,
        *,
        cu_seqlens_q_host: torch.Tensor,
        cu_seqlens_history_local_host: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
        record_timing: bool = False,
        capture_warmup: int = 3,
    ) -> DCPAttentionCUDAGraph:
        bindings = dict(
            q_local=q_local,
            k_history_local=k_history_local,
            v_history_local=v_history_local,
            k_chunk=k_chunk,
            v_chunk=v_chunk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_history_local=cu_seqlens_history_local,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_history_local=max_seqlen_history_local,
            cu_seqlens_q_host=cu_seqlens_q_host,
            cu_seqlens_history_local_host=cu_seqlens_history_local_host,
            num_splits=num_splits,
            return_lse=return_lse,
            overlap_q_allgather=overlap_q_allgather,
        )
        return self._capture_forward(
            "chunk_prefill_varlen",
            lambda: self.forward_chunk_prefill_varlen(
                q_local,
                k_history_local,
                v_history_local,
                k_chunk,
                v_chunk,
                cu_seqlens_q,
                cu_seqlens_history_local,
                max_seqlen_q,
                max_seqlen_history_local,
                cu_seqlens_q_host=cu_seqlens_q_host,
                cu_seqlens_history_local_host=cu_seqlens_history_local_host,
                num_splits=num_splits,
                return_lse=return_lse,
                overlap_q_allgather=overlap_q_allgather,
                _record_timing=record_timing,
            ),
            bindings,
            overlap_q_allgather=overlap_q_allgather,
            record_timing=record_timing,
            capture_warmup=capture_warmup,
        )

    def _buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        numel = 1
        for extent in shape:
            numel *= extent
        storage = self._buffers.get(name)
        if (
            storage is None
            or storage.device != self.device
            or storage.dtype != dtype
            or storage.numel() < numel
        ):
            storage = torch.empty(numel, device=self.device, dtype=dtype)
            self._buffers[name] = storage
        result = storage[:numel].view(shape)
        if self._capture_in_progress:
            self._capture_tensors.append(result)
        return result

    def _record_timing(self, name: str, stream: torch.cuda.Stream) -> None:
        self._timing_events[name].record(stream)

    def _retain_work(self, work: Optional[dist.Work]) -> None:
        if work is not None:
            if not self._capture_in_progress:
                self._works = [
                    pending for pending in self._works if not pending.is_completed()
                ]
            self._works.append(work)
            # This inserts a completion dependency only on the currently active
            # G stream and is guaranteed to return immediately.  Work.wait()
            # here can delay host submission of local chunk attention on C.
            work.block_current_stream()

    def _reap_inflight_tensors(self) -> None:
        self._check_forward_state()
        if self._capture_in_progress:
            return
        self._inflight_tensors = [
            item for item in self._inflight_tensors if not item[0].query()
        ]

    def _retain_merge_inputs(
        self,
        compute_stream: torch.cuda.Stream,
        *tensors: torch.Tensor,
    ) -> torch.cuda.Event:
        # Triton launches with raw pointers rather than through the PyTorch
        # dispatcher.  Keep its inputs alive until the merge kernel completes,
        # including when the next call uses a different current stream.
        completion = torch.cuda.Event()
        completion.record(compute_stream)
        self._inflight_tensors.append((completion, tensors))
        if self._capture_in_progress:
            self._capture_tensors.extend(tensors)
        return completion

    def _prepare_persistent_workspace(
        self, compute_stream: torch.cuda.Stream
    ) -> None:
        pending = self._workspace_completion
        if pending is None:
            return
        completion, stream_id = pending
        if self._capture_in_progress:
            if compute_stream.cuda_stream != stream_id:
                compute_stream.wait_event(completion)
        elif not completion.query() and compute_stream.cuda_stream != stream_id:
            compute_stream.wait_event(completion)
        self._workspace_completion = None

    def _check_common(
        self,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        seqlens_local: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        self._check_forward_state()
        if q_local.device != self.device:
            raise ValueError(f"q_local must be on runner device {self.device}, got {q_local.device}")
        for tensor, name in ((q_local, "q_local"), (k_local, "k_local"), (v_local, "v_local")):
            if not tensor.is_cuda or tensor.dtype != torch.bfloat16 or tensor.ndim != 4:
                raise ValueError(f"{name} must be a CUDA BF16 tensor with shape [B, S, H, 128]")
            if tensor.shape[-1] != 128 or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous BSHD with head_dim 128")
            if tensor.device != self.device:
                raise ValueError(f"{name} must be on runner device {self.device}")
        if k_local.shape != v_local.shape:
            raise ValueError("k_local and v_local must have identical shapes")
        b, sq, h_local, d = q_local.shape
        if b <= 0 or sq <= 0 or h_local <= 0:
            raise ValueError("q_local requires positive B, Sq, and Hq_local")
        if k_local.shape[0] != b or k_local.shape[2] <= 0:
            raise ValueError("local K/V batch must match q_local and Hkv_group must be positive")
        h_kv = k_local.shape[2]
        if h_local % h_kv:
            raise ValueError(
                f"Hq_local must be divisible by Hkv_group, got {h_local} and {h_kv}"
            )
        if (
            not seqlens_local.is_cuda
            or seqlens_local.device != self.device
            or seqlens_local.dtype != torch.int32
            or seqlens_local.shape != (b,)
            or not seqlens_local.is_contiguous()
        ):
            raise ValueError("local sequence lengths must be contiguous CUDA int32 with shape [B]")
        return b, sq, h_local, d

    @staticmethod
    def _check_num_splits(num_splits: int) -> None:
        if not isinstance(num_splits, int) or not 0 <= num_splits <= 128:
            raise ValueError(
                "num_splits must be 0 (auto), 1 (NoSplit), or in [2, 128]"
            )

    @staticmethod
    def _validate_host_cu_seqlens(
        cu_seqlens_host: torch.Tensor,
        total_tokens: int,
        max_seqlen: int,
        name: str,
    ) -> list[int]:
        values = cu_seqlens_host.tolist()
        if values[0] != 0:
            raise ValueError(f"{name} must start with 0")
        lengths = [end - start for start, end in zip(values, values[1:])]
        if any(length <= 0 for length in lengths):
            raise ValueError(f"{name} must be strictly increasing")
        if values[-1] != total_tokens:
            raise ValueError(
                f"{name}[-1] must equal total token count {total_tokens}, "
                f"got {values[-1]}"
            )
        actual_max = max(lengths)
        if max_seqlen != actual_max:
            raise ValueError(
                f"max length for {name} must equal {actual_max}, got {max_seqlen}"
            )
        return lengths

    def _check_packed_common(
        self,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_local: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k_local: int,
        cu_seqlens_q_host: torch.Tensor,
        cu_seqlens_k_local_host: torch.Tensor,
        num_splits: int,
    ) -> tuple[int, int, int, int, int, list[int], list[int]]:
        """Validate packed inputs using CPU mirrors without synchronizing CUDA."""
        self._check_forward_state()
        self._check_num_splits(num_splits)
        for tensor, name in (
            (q_local, "q_local"),
            (k_local, "k_local"),
            (v_local, "v_local"),
        ):
            if not tensor.is_cuda or tensor.dtype != torch.bfloat16 or tensor.ndim != 3:
                raise ValueError(
                    f"{name} must be a CUDA BF16 tensor with shape [total_tokens, H, 128]"
                )
            if tensor.shape[-1] != 128 or not tensor.is_contiguous():
                raise ValueError(
                    f"{name} must be contiguous [total_tokens, H, 128] with head_dim 128"
                )
            if tensor.device != self.device:
                raise ValueError(f"{name} must be on runner device {self.device}")
        if k_local.shape != v_local.shape:
            raise ValueError("k_local and v_local must have identical shapes")

        total_q, h_local, d = q_local.shape
        total_k_local, h_kv, _ = k_local.shape
        if total_q <= 0 or h_local <= 0:
            raise ValueError("q_local requires positive total_q and Hq_local")
        if total_k_local <= 0 or h_kv <= 0:
            raise ValueError("local K/V requires positive total_k_local and Hkv_group")
        if h_local % h_kv:
            raise ValueError(
                f"Hq_local must be divisible by Hkv_group, got {h_local} and {h_kv}"
            )

        for tensor, name in (
            (cu_seqlens_q, "cu_seqlens_q"),
            (cu_seqlens_k_local, "cu_seqlens_k_local"),
        ):
            if (
                not tensor.is_cuda
                or tensor.device != self.device
                or tensor.dtype != torch.int32
                or tensor.ndim != 1
                or tensor.numel() < 2
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous CUDA int32 with shape [B + 1], B >= 1"
                )
        if cu_seqlens_q.shape != cu_seqlens_k_local.shape:
            raise ValueError("Q and local K cumulative lengths must have the same batch size")

        for host, device, name in (
            (cu_seqlens_q_host, cu_seqlens_q, "cu_seqlens_q_host"),
            (
                cu_seqlens_k_local_host,
                cu_seqlens_k_local,
                "cu_seqlens_k_local_host",
            ),
        ):
            if (
                host.device.type != "cpu"
                or host.dtype != torch.int32
                or host.ndim != 1
                or not host.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous CPU int32 with shape [B + 1]")
            if host.shape != device.shape:
                raise ValueError(f"{name} must match its CUDA cumulative-length shape")

        if max_seqlen_q <= 0 or max_seqlen_k_local <= 0:
            raise ValueError("max_seqlen_q and max_seqlen_k_local must be positive")
        q_lengths = self._validate_host_cu_seqlens(
            cu_seqlens_q_host,
            total_q,
            max_seqlen_q,
            "cu_seqlens_q_host",
        )
        k_lengths = self._validate_host_cu_seqlens(
            cu_seqlens_k_local_host,
            total_k_local,
            max_seqlen_k_local,
            "cu_seqlens_k_local_host",
        )
        batch_size = cu_seqlens_q.numel() - 1
        return total_q, h_local, d, h_kv, batch_size, q_lengths, k_lengths

    def _check_packed_chunk_inputs(
        self,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        total_q: int,
        h_kv: int,
        d: int,
    ) -> None:
        expected = (total_q, h_kv, d)
        if k_chunk.shape != expected or v_chunk.shape != expected:
            raise ValueError(
                "k_chunk and v_chunk must have shape "
                f"[total_q, Hkv_group, 128]={expected}"
            )
        for tensor, name in ((k_chunk, "k_chunk"), (v_chunk, "v_chunk")):
            if (
                not tensor.is_cuda
                or tensor.device != self.device
                or tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous CUDA BF16 on the runner device"
                )

    def _check_packed_runner_topology(self, h_kv: int) -> None:
        del h_kv

    def _start_q_allgather(
        self,
        q_local: torch.Tensor,
        h_kv: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> torch.Tensor:
        self._prepare_persistent_workspace(compute_stream)
        self._input_ready.record(compute_stream)
        b, sq, h_local, d = q_local.shape
        q_rank_major = self._buffer(
            "q_rank_major", (self.world_size * b, sq, h_local, d), q_local.dtype
        )
        q_group = self._buffer(
            "q_group", (b, sq, self.world_size * h_local, d), q_local.dtype
        )
        q_local.record_stream(self.communication_stream)
        q_rank_major.record_stream(self.communication_stream)
        q_group.record_stream(self.communication_stream)
        with torch.cuda.stream(self.communication_stream):
            self.communication_stream.wait_event(self._input_ready)
            if timing:
                self._record_timing("q_ag_start", self.communication_stream)
            with _nvtx_range("dcp_q_allgather"):
                work = dist.all_gather_into_tensor(
                    q_rank_major,
                    q_local,
                    group=self.process_group,
                    async_op=True,
                )
                self._retain_work(work)
            with _nvtx_range("dcp_q_rank_major_to_bshd"):
                heads_per_local_kv = h_local // h_kv
                rank_major_view = q_rank_major.view(
                    self.world_size,
                    b,
                    sq,
                    h_kv,
                    heads_per_local_kv,
                    d,
                ).permute(1, 2, 3, 0, 4, 5)
                q_group.view(
                    b,
                    sq,
                    h_kv,
                    self.world_size,
                    heads_per_local_kv,
                    d,
                ).copy_(rank_major_view)
            if timing:
                self._record_timing("q_ag_end", self.communication_stream)
            self._q_group_ready.record(self.communication_stream)
        q_group.record_stream(compute_stream)
        return q_group

    def _gather_q_single_stream(
        self,
        q_local: torch.Tensor,
        h_kv: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> torch.Tensor:
        self._prepare_persistent_workspace(compute_stream)
        b, sq, h_local, d = q_local.shape
        q_rank_major = self._buffer(
            "q_rank_major", (self.world_size * b, sq, h_local, d), q_local.dtype
        )
        q_group = self._buffer(
            "q_group", (b, sq, self.world_size * h_local, d), q_local.dtype
        )
        if timing:
            self._record_timing("q_ag_start", compute_stream)
        with _nvtx_range("dcp_single_stream_q_allgather"):
            dist.all_gather_into_tensor(
                q_rank_major,
                q_local,
                group=self.process_group,
            )
        with _nvtx_range("dcp_single_stream_q_rank_major_to_bshd"):
            heads_per_local_kv = h_local // h_kv
            rank_major_view = q_rank_major.view(
                self.world_size,
                b,
                sq,
                h_kv,
                heads_per_local_kv,
                d,
            ).permute(1, 2, 3, 0, 4, 5)
            q_group.view(
                b,
                sq,
                h_kv,
                self.world_size,
                heads_per_local_kv,
                d,
            ).copy_(rank_major_view)
        if timing:
            self._record_timing("q_ag_end", compute_stream)
        return q_group

    def _start_q_allgather_varlen(
        self,
        q_local: torch.Tensor,
        h_kv: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> torch.Tensor:
        return self._start_q_allgather(
            q_local.unsqueeze(0), h_kv, compute_stream, timing=timing
        ).squeeze(0)

    def _gather_q_single_stream_varlen(
        self,
        q_local: torch.Tensor,
        h_kv: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> torch.Tensor:
        return self._gather_q_single_stream(
            q_local.unsqueeze(0), h_kv, compute_stream, timing=timing
        ).squeeze(0)

    def _finish_context(
        self,
        history_out: torch.Tensor,
        history_lse: torch.Tensor,
        output: torch.Tensor,
        local_lse: Optional[torch.Tensor],
        h_kv: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> None:
        b, sq, h_group, d = history_out.shape
        h_local = h_group // self.world_size
        gathered_lse = self._buffer(
            "gathered_lse",
            (self.world_size * b, h_group, sq),
            torch.float32,
        )
        packed_out = self._buffer(
            "packed_corrected_out",
            (self.world_size * b, sq, h_local, d),
            torch.bfloat16,
        )
        for tensor in (history_out, history_lse, output):
            tensor.record_stream(self.communication_stream)
        if local_lse is not None:
            local_lse.record_stream(self.communication_stream)
        gathered_lse.record_stream(self.communication_stream)
        packed_out.record_stream(self.communication_stream)

        with torch.cuda.stream(self.communication_stream):
            self.communication_stream.wait_event(self._history_ready)
            if timing:
                self._record_timing("lse_correct_start", self.communication_stream)
            with _nvtx_range("dcp_lse_allgather"):
                work = dist.all_gather_into_tensor(
                    gathered_lse,
                    history_lse,
                    group=self.process_group,
                    async_op=True,
                )
                self._retain_work(work)
            with _nvtx_range("dcp_lse_correct_and_head_pack"):
                local_lse_arg = local_lse if local_lse is not None else gathered_lse
                _correct_and_pack_kernel[(b * sq, h_group)](
                    history_out,
                    gathered_lse,
                    packed_out,
                    local_lse_arg,
                    *history_out.stride()[:3],
                    *gathered_lse.view(self.world_size, b, h_group, sq).stride(),
                    *(local_lse.stride() if local_lse is not None else (0, 0, 0)),
                    B=b,
                    SQ=sq,
                    H_GROUP=h_group,
                    H_LOCAL=h_local,
                    H_KV=h_kv,
                    D=d,
                    WORLD_SIZE=self.world_size,
                    WORLD_SIZE_ROUNDED=triton.next_power_of_2(self.world_size),
                    RANK=self.rank,
                    STORE_LOCAL_LSE=local_lse is not None,
                    num_warps=4,
                )
            if timing:
                self._record_timing("lse_correct_end", self.communication_stream)
                self._record_timing("reduce_scatter_start", self.communication_stream)
            with _nvtx_range("dcp_output_reduce_scatter"):
                work = dist.reduce_scatter_tensor(
                    output,
                    packed_out,
                    op=dist.ReduceOp.SUM,
                    group=self.process_group,
                    async_op=True,
                )
                self._retain_work(work)
            if timing:
                self._record_timing("reduce_scatter_end", self.communication_stream)
            self._context_ready.record(self.communication_stream)

        output.record_stream(compute_stream)
        if local_lse is not None:
            local_lse.record_stream(compute_stream)

    def _finish_context_single_stream(
        self,
        history_out: torch.Tensor,
        history_lse: torch.Tensor,
        output: torch.Tensor,
        local_lse: Optional[torch.Tensor],
        h_kv: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> None:
        b, sq, h_group, d = history_out.shape
        h_local = h_group // self.world_size
        gathered_lse = self._buffer(
            "gathered_lse",
            (self.world_size * b, h_group, sq),
            torch.float32,
        )
        packed_out = self._buffer(
            "packed_corrected_out",
            (self.world_size * b, sq, h_local, d),
            torch.bfloat16,
        )
        if timing:
            self._record_timing("lse_correct_start", compute_stream)
        with _nvtx_range("dcp_single_stream_lse_allgather"):
            dist.all_gather_into_tensor(
                gathered_lse,
                history_lse,
                group=self.process_group,
            )
        with _nvtx_range("dcp_single_stream_lse_correct_and_head_pack"):
            local_lse_arg = local_lse if local_lse is not None else gathered_lse
            _correct_and_pack_kernel[(b * sq, h_group)](
                history_out,
                gathered_lse,
                packed_out,
                local_lse_arg,
                *history_out.stride()[:3],
                *gathered_lse.view(self.world_size, b, h_group, sq).stride(),
                *(local_lse.stride() if local_lse is not None else (0, 0, 0)),
                B=b,
                SQ=sq,
                H_GROUP=h_group,
                H_LOCAL=h_local,
                H_KV=h_kv,
                D=d,
                WORLD_SIZE=self.world_size,
                WORLD_SIZE_ROUNDED=triton.next_power_of_2(self.world_size),
                RANK=self.rank,
                STORE_LOCAL_LSE=local_lse is not None,
                num_warps=4,
            )
        if timing:
            self._record_timing("lse_correct_end", compute_stream)
            self._record_timing("reduce_scatter_start", compute_stream)
        with _nvtx_range("dcp_single_stream_output_reduce_scatter"):
            dist.reduce_scatter_tensor(
                output,
                packed_out,
                op=dist.ReduceOp.SUM,
                group=self.process_group,
            )
        if timing:
            self._record_timing("reduce_scatter_end", compute_stream)

    def _run_context_attention(
        self,
        q_group: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        seqlens_local: torch.Tensor,
        num_splits: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
        use_dependency_events: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_dependency_events:
            compute_stream.wait_event(self._q_group_ready)
        if timing:
            self._record_timing("ag_chunk_end", compute_stream)
            self._record_timing("history_start", compute_stream)
        with _nvtx_range("dcp_local_history_attention"):
            history_out, history_lse = min_fa3_op.forward_kvcache(
                q_group,
                k_local,
                v_local,
                seqlens_local,
                num_splits=num_splits,
                return_lse=True,
                is_causal=False,
            )
        if timing:
            self._record_timing("history_end", compute_stream)
        if use_dependency_events:
            self._history_ready.record(compute_stream)
        return history_out, history_lse

    def _run_context_attention_varlen(
        self,
        q_group: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_local: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k_local: int,
        cu_seqlens_q_host: torch.Tensor,
        cu_seqlens_k_local_host: torch.Tensor,
        num_splits: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
        use_dependency_events: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_dependency_events:
            compute_stream.wait_event(self._q_group_ready)
        if timing:
            self._record_timing("ag_chunk_end", compute_stream)
            self._record_timing("history_start", compute_stream)
        with _nvtx_range("dcp_varlen_local_history_attention"):
            history_out, history_lse = min_fa3_op.forward_kvcache_varlen(
                q_group,
                k_local,
                v_local,
                cu_seqlens_q,
                cu_seqlens_k_local,
                max_seqlen_q,
                max_seqlen_k_local,
                cu_seqlens_q_host=cu_seqlens_q_host,
                cu_seqlens_k_host=cu_seqlens_k_local_host,
                num_splits=num_splits,
                return_lse=True,
                is_causal=False,
            )
        if timing:
            self._record_timing("history_end", compute_stream)
        if use_dependency_events:
            self._history_ready.record(compute_stream)
        return history_out, history_lse

    def _finish_context_varlen(
        self,
        history_out: torch.Tensor,
        history_lse: torch.Tensor,
        output: torch.Tensor,
        local_lse: Optional[torch.Tensor],
        h_kv: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
        use_side_stream: bool,
    ) -> None:
        finish = self._finish_context if use_side_stream else self._finish_context_single_stream
        finish(
            history_out.unsqueeze(0),
            history_lse.unsqueeze(0),
            output.unsqueeze(0),
            local_lse.unsqueeze(0) if local_lse is not None else None,
            h_kv,
            compute_stream,
            timing=timing,
        )

    def forward_decode(
        self,
        q_local: torch.Tensor,
        k_cache_local: torch.Tensor,
        v_cache_local: torch.Tensor,
        cache_seqlens_local: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
        *,
        _record_timing: bool = False,
    ) -> _DCPResult:
        """Run DCP decode for ``q_local`` with shape ``[B, 1, Hq_local, 128]``."""
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError("DCPAttentionRunner does not support concurrent forward calls")
        try:
            self._reap_inflight_tensors()
            b, sq, h_local, _ = self._check_common(
                q_local, k_cache_local, v_cache_local, cache_seqlens_local
            )
            if sq != 1:
                raise ValueError(f"forward_decode requires Sq=1, got {sq}")
            compute_stream = torch.cuda.current_stream(self.device)
            if _record_timing:
                self._last_timing_kind = "decode"
                self._record_timing("attention_start", compute_stream)
            use_side_stream = self.world_size > 1 and overlap_q_allgather
            if self.world_size == 1:
                if _record_timing:
                    self._record_timing("q_ag_start", compute_stream)
                    self._record_timing("q_ag_end", compute_stream)
                    self._record_timing("ag_chunk_end", compute_stream)
                    self._record_timing("history_start", compute_stream)
                with _nvtx_range("dcp_local_history_attention"):
                    result = min_fa3_op.forward_kvcache(
                        q_local,
                        k_cache_local,
                        v_cache_local,
                        cache_seqlens_local,
                        num_splits=num_splits,
                        return_lse=return_lse,
                    )
                if _record_timing:
                    self._record_timing("history_end", compute_stream)
                    self._record_timing("attention_end", compute_stream)
                return result

            if use_side_stream:
                q_group = self._start_q_allgather(
                    q_local,
                    k_cache_local.shape[2],
                    compute_stream,
                    timing=_record_timing,
                )
            else:
                q_group = self._gather_q_single_stream(
                    q_local,
                    k_cache_local.shape[2],
                    compute_stream,
                    timing=_record_timing,
                )
            history_out, history_lse = self._run_context_attention(
                q_group,
                k_cache_local,
                v_cache_local,
                cache_seqlens_local,
                num_splits,
                compute_stream,
                timing=_record_timing,
                use_dependency_events=use_side_stream,
            )
            output = torch.empty_like(q_local)
            local_lse = (
                torch.empty((b, h_local, sq), device=self.device, dtype=torch.float32)
                if return_lse
                else None
            )
            if use_side_stream:
                self._finish_context(
                    history_out,
                    history_lse,
                    output,
                    local_lse,
                    k_cache_local.shape[2],
                    compute_stream,
                    timing=_record_timing,
                )
                compute_stream.wait_event(self._context_ready)
                completion = self._context_ready
            else:
                self._finish_context_single_stream(
                    history_out,
                    history_lse,
                    output,
                    local_lse,
                    k_cache_local.shape[2],
                    compute_stream,
                    timing=_record_timing,
                )
                completion = torch.cuda.Event()
                completion.record(compute_stream)
            self._workspace_completion = (completion, compute_stream.cuda_stream)
            if _record_timing:
                self._record_timing("attention_end", compute_stream)
            return (output, local_lse) if return_lse else output
        finally:
            self._enqueue_lock.release()

    def forward_chunk_prefill(
        self,
        q_local: torch.Tensor,
        k_history_local: torch.Tensor,
        v_history_local: torch.Tensor,
        history_seqlens_local: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
        *,
        _record_timing: bool = False,
    ) -> _DCPResult:
        """Run split context/chunk prefill and stably merge both states."""
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError("DCPAttentionRunner does not support concurrent forward calls")
        try:
            self._reap_inflight_tensors()
            b, sq, h_local, d = self._check_common(
                q_local, k_history_local, v_history_local, history_seqlens_local
            )
            if k_chunk.shape != (b, sq, k_history_local.shape[2], d):
                raise ValueError(
                    "k_chunk must have shape [B, Sq, Hkv_group, 128] matching local history"
                )
            if v_chunk.shape != k_chunk.shape:
                raise ValueError("v_chunk must have the same shape as k_chunk")
            for tensor, name in ((k_chunk, "k_chunk"), (v_chunk, "v_chunk")):
                if (
                    not tensor.is_cuda
                    or tensor.device != self.device
                    or tensor.dtype != torch.bfloat16
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(f"{name} must be contiguous CUDA BF16 on the runner device")

            compute_stream = torch.cuda.current_stream(self.device)
            if _record_timing:
                self._last_timing_kind = "chunk"
                self._record_timing("attention_start", compute_stream)
            use_side_stream = self.world_size > 1 and overlap_q_allgather

            if use_side_stream:
                q_group = self._start_q_allgather(
                    q_local,
                    k_history_local.shape[2],
                    compute_stream,
                    timing=_record_timing,
                )
            elif self.world_size > 1:
                q_group = self._gather_q_single_stream(
                    q_local,
                    k_history_local.shape[2],
                    compute_stream,
                    timing=_record_timing,
                )
            else:
                q_group = q_local
                if _record_timing:
                    self._record_timing("q_ag_start", compute_stream)
                    self._record_timing("q_ag_end", compute_stream)

            if _record_timing:
                self._record_timing("chunk_start", compute_stream)
            chunk_lengths = torch.full(
                (b,), sq, device=self.device, dtype=torch.int32
            )
            with _nvtx_range("dcp_local_chunk_attention"):
                chunk_out, chunk_lse = min_fa3_op.forward_kvcache(
                    q_local,
                    k_chunk,
                    v_chunk,
                    chunk_lengths,
                    num_splits=num_splits,
                    return_lse=True,
                    is_causal=True,
                )
            if _record_timing:
                self._record_timing("chunk_end", compute_stream)

            history_out, history_lse = self._run_context_attention(
                q_group,
                k_history_local,
                v_history_local,
                history_seqlens_local,
                num_splits,
                compute_stream,
                timing=_record_timing,
                use_dependency_events=use_side_stream,
            )

            context_out = torch.empty_like(q_local)
            context_lse = torch.empty(
                (b, h_local, sq), device=self.device, dtype=torch.float32
            )
            if self.world_size == 1:
                context_out.copy_(history_out)
                context_lse.copy_(history_lse)
            elif use_side_stream:
                self._finish_context(
                    history_out,
                    history_lse,
                    context_out,
                    context_lse,
                    k_history_local.shape[2],
                    compute_stream,
                    timing=_record_timing,
                )
                compute_stream.wait_event(self._context_ready)
            else:
                self._finish_context_single_stream(
                    history_out,
                    history_lse,
                    context_out,
                    context_lse,
                    k_history_local.shape[2],
                    compute_stream,
                    timing=_record_timing,
                )

            merged_out = torch.empty_like(q_local)
            merged_lse = (
                torch.empty((b, h_local, sq), device=self.device, dtype=torch.float32)
                if return_lse
                else None
            )
            if _record_timing:
                self._record_timing("merge_start", compute_stream)
            with _nvtx_range("dcp_merge_context_chunk_states"):
                merged_lse_arg = merged_lse if merged_lse is not None else context_lse
                _merge_attn_states_kernel[(b * sq, h_local)](
                    context_out,
                    context_lse,
                    chunk_out,
                    chunk_lse,
                    merged_out,
                    merged_lse_arg,
                    *merged_out.stride()[:3],
                    *context_lse.stride(),
                    B=b,
                    SQ=sq,
                    H=h_local,
                    D=d,
                    STORE_LSE=merged_lse is not None,
                    num_warps=4,
                )
            if _record_timing:
                self._record_timing("merge_end", compute_stream)
                self._record_timing("attention_end", compute_stream)
            completion = self._retain_merge_inputs(
                compute_stream,
                context_out,
                context_lse,
                chunk_out,
                chunk_lse,
            )
            if self.world_size > 1:
                self._workspace_completion = (
                    completion,
                    compute_stream.cuda_stream,
                )
            return (merged_out, merged_lse) if return_lse else merged_out
        finally:
            self._enqueue_lock.release()

    def forward_decode_varlen(
        self,
        q_local: torch.Tensor,
        k_cache_local: torch.Tensor,
        v_cache_local: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_local: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k_local: int,
        *,
        cu_seqlens_q_host: torch.Tensor,
        cu_seqlens_k_local_host: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
        _record_timing: bool = False,
    ) -> _DCPResult:
        """Run packed DCP decode with one query token per sequence."""
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError("DCPAttentionRunner does not support concurrent forward calls")
        try:
            self._reap_inflight_tensors()
            (
                _,
                h_local,
                _,
                h_kv,
                _,
                q_lengths,
                _,
            ) = self._check_packed_common(
                q_local,
                k_cache_local,
                v_cache_local,
                cu_seqlens_q,
                cu_seqlens_k_local,
                max_seqlen_q,
                max_seqlen_k_local,
                cu_seqlens_q_host,
                cu_seqlens_k_local_host,
                num_splits,
            )
            self._check_packed_runner_topology(h_kv)
            if any(length != 1 for length in q_lengths):
                raise ValueError("forward_decode_varlen requires every q_len == 1")

            compute_stream = torch.cuda.current_stream(self.device)
            if _record_timing:
                self._last_timing_kind = "decode"
                self._record_timing("attention_start", compute_stream)
            use_side_stream = self.world_size > 1 and overlap_q_allgather
            if self.world_size == 1:
                if _record_timing:
                    self._record_timing("q_ag_start", compute_stream)
                    self._record_timing("q_ag_end", compute_stream)
                    self._record_timing("ag_chunk_end", compute_stream)
                    self._record_timing("history_start", compute_stream)
                with _nvtx_range("dcp_varlen_local_history_attention"):
                    result = min_fa3_op.forward_kvcache_varlen(
                        q_local,
                        k_cache_local,
                        v_cache_local,
                        cu_seqlens_q,
                        cu_seqlens_k_local,
                        max_seqlen_q,
                        max_seqlen_k_local,
                        cu_seqlens_q_host=cu_seqlens_q_host,
                        cu_seqlens_k_host=cu_seqlens_k_local_host,
                        num_splits=num_splits,
                        return_lse=return_lse,
                        is_causal=False,
                    )
                if _record_timing:
                    self._record_timing("history_end", compute_stream)
                    self._record_timing("attention_end", compute_stream)
                return result

            if use_side_stream:
                q_group = self._start_q_allgather_varlen(
                    q_local, h_kv, compute_stream, timing=_record_timing
                )
            else:
                q_group = self._gather_q_single_stream_varlen(
                    q_local, h_kv, compute_stream, timing=_record_timing
                )
            history_out, history_lse = self._run_context_attention_varlen(
                q_group,
                k_cache_local,
                v_cache_local,
                cu_seqlens_q,
                cu_seqlens_k_local,
                max_seqlen_q,
                max_seqlen_k_local,
                cu_seqlens_q_host,
                cu_seqlens_k_local_host,
                num_splits,
                compute_stream,
                timing=_record_timing,
                use_dependency_events=use_side_stream,
            )
            output = torch.empty_like(q_local)
            local_lse = (
                torch.empty(
                    (h_local, q_local.shape[0]),
                    device=self.device,
                    dtype=torch.float32,
                )
                if return_lse
                else None
            )
            self._finish_context_varlen(
                history_out,
                history_lse,
                output,
                local_lse,
                h_kv,
                compute_stream,
                timing=_record_timing,
                use_side_stream=use_side_stream,
            )
            if use_side_stream:
                compute_stream.wait_event(self._context_ready)
                completion = self._context_ready
            else:
                completion = torch.cuda.Event()
                completion.record(compute_stream)
            self._workspace_completion = (completion, compute_stream.cuda_stream)
            if _record_timing:
                self._record_timing("attention_end", compute_stream)
            return (output, local_lse) if return_lse else output
        finally:
            self._enqueue_lock.release()

    def forward_chunk_prefill_varlen(
        self,
        q_local: torch.Tensor,
        k_history_local: torch.Tensor,
        v_history_local: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_history_local: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_history_local: int,
        *,
        cu_seqlens_q_host: torch.Tensor,
        cu_seqlens_history_local_host: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
        _record_timing: bool = False,
    ) -> _DCPResult:
        """Run packed split history/chunk prefill and merge both states."""
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError("DCPAttentionRunner does not support concurrent forward calls")
        try:
            self._reap_inflight_tensors()
            (
                total_q,
                h_local,
                d,
                h_kv,
                _,
                _,
                _,
            ) = self._check_packed_common(
                q_local,
                k_history_local,
                v_history_local,
                cu_seqlens_q,
                cu_seqlens_history_local,
                max_seqlen_q,
                max_seqlen_history_local,
                cu_seqlens_q_host,
                cu_seqlens_history_local_host,
                num_splits,
            )
            self._check_packed_runner_topology(h_kv)
            self._check_packed_chunk_inputs(k_chunk, v_chunk, total_q, h_kv, d)

            compute_stream = torch.cuda.current_stream(self.device)
            if _record_timing:
                self._last_timing_kind = "chunk"
                self._record_timing("attention_start", compute_stream)
            use_side_stream = self.world_size > 1 and overlap_q_allgather

            if use_side_stream:
                q_group = self._start_q_allgather_varlen(
                    q_local, h_kv, compute_stream, timing=_record_timing
                )
            elif self.world_size > 1:
                q_group = self._gather_q_single_stream_varlen(
                    q_local, h_kv, compute_stream, timing=_record_timing
                )
            else:
                q_group = q_local
                if _record_timing:
                    self._record_timing("q_ag_start", compute_stream)
                    self._record_timing("q_ag_end", compute_stream)

            if _record_timing:
                self._record_timing("chunk_start", compute_stream)
            with _nvtx_range("dcp_varlen_local_chunk_attention"):
                chunk_out, chunk_lse = min_fa3_op.forward_kvcache_varlen(
                    q_local,
                    k_chunk,
                    v_chunk,
                    cu_seqlens_q,
                    cu_seqlens_q,
                    max_seqlen_q,
                    max_seqlen_q,
                    cu_seqlens_q_host=cu_seqlens_q_host,
                    cu_seqlens_k_host=cu_seqlens_q_host,
                    num_splits=num_splits,
                    return_lse=True,
                    is_causal=True,
                )
            if _record_timing:
                self._record_timing("chunk_end", compute_stream)

            history_out, history_lse = self._run_context_attention_varlen(
                q_group,
                k_history_local,
                v_history_local,
                cu_seqlens_q,
                cu_seqlens_history_local,
                max_seqlen_q,
                max_seqlen_history_local,
                cu_seqlens_q_host,
                cu_seqlens_history_local_host,
                num_splits,
                compute_stream,
                timing=_record_timing,
                use_dependency_events=use_side_stream,
            )

            context_out = torch.empty_like(q_local)
            context_lse = torch.empty(
                (h_local, total_q), device=self.device, dtype=torch.float32
            )
            if self.world_size == 1:
                context_out.copy_(history_out)
                context_lse.copy_(history_lse)
            else:
                self._finish_context_varlen(
                    history_out,
                    history_lse,
                    context_out,
                    context_lse,
                    h_kv,
                    compute_stream,
                    timing=_record_timing,
                    use_side_stream=use_side_stream,
                )
                if use_side_stream:
                    compute_stream.wait_event(self._context_ready)

            merged_out = torch.empty_like(q_local)
            merged_lse = torch.empty_like(context_lse) if return_lse else None
            if _record_timing:
                self._record_timing("merge_start", compute_stream)
            with _nvtx_range("dcp_varlen_merge_context_chunk_states"):
                merged_lse_arg = merged_lse if merged_lse is not None else context_lse
                _merge_attn_states_kernel[(total_q, h_local)](
                    context_out,
                    context_lse,
                    chunk_out,
                    chunk_lse,
                    merged_out,
                    merged_lse_arg,
                    *merged_out.unsqueeze(0).stride()[:3],
                    *context_lse.unsqueeze(0).stride(),
                    B=1,
                    SQ=total_q,
                    H=h_local,
                    D=d,
                    STORE_LSE=merged_lse is not None,
                    num_warps=4,
                )
            if _record_timing:
                self._record_timing("merge_end", compute_stream)
                self._record_timing("attention_end", compute_stream)
            completion = self._retain_merge_inputs(
                compute_stream,
                context_out,
                context_lse,
                chunk_out,
                chunk_lse,
            )
            if self.world_size > 1:
                self._workspace_completion = (
                    completion,
                    compute_stream.cuda_stream,
                )
            return (merged_out, merged_lse) if return_lse else merged_out
        finally:
            self._enqueue_lock.release()

    def last_timing_ms(self, synchronize: bool = True) -> dict[str, float]:
        """Return timings for the most recent call made with ``_record_timing=True``."""
        if self._last_timing_kind is None:
            raise RuntimeError("No timed DCP forward has been recorded")
        if synchronize:
            self._timing_events["attention_end"].synchronize()
        events = self._timing_events
        values = {
            "q_allgather_and_reorder_ms": events["q_ag_start"].elapsed_time(events["q_ag_end"]),
            "local_history_attention_ms": events["history_start"].elapsed_time(events["history_end"]),
            "attention_end_to_end_ms": events["attention_start"].elapsed_time(events["attention_end"]),
        }
        if self._last_timing_kind == "chunk":
            values.update(
                local_chunk_attention_ms=events["chunk_start"].elapsed_time(events["chunk_end"]),
                overlapped_ag_chunk_window_ms=events["q_ag_start"].elapsed_time(
                    events["ag_chunk_end"]
                ),
                state_merge_ms=events["merge_start"].elapsed_time(events["merge_end"]),
            )
        else:
            values.update(
                local_chunk_attention_ms=0.0,
                overlapped_ag_chunk_window_ms=values["q_allgather_and_reorder_ms"],
                state_merge_ms=0.0,
            )
        is_a2a = self.output_collective_kind == "bf16_packed_all_to_all"
        if self.world_size > 1 and not is_a2a:
            values.update(
                lse_allgather_correct_ms=events["lse_correct_start"].elapsed_time(
                    events["lse_correct_end"]
                ),
                output_reduce_scatter_ms=events["reduce_scatter_start"].elapsed_time(
                    events["reduce_scatter_end"]
                ),
            )
        else:
            values.update(lse_allgather_correct_ms=0.0, output_reduce_scatter_ms=0.0)
        if self.world_size > 1 and is_a2a:
            values.update(
                a2a_pack_ms=events["a2a_pack_start"].elapsed_time(
                    events["a2a_pack_end"]
                ),
                a2a_all_to_all_ms=events["a2a_all_to_all_start"].elapsed_time(
                    events["a2a_all_to_all_end"]
                ),
                a2a_unpack_combine_ms=events[
                    "a2a_unpack_combine_start"
                ].elapsed_time(events["a2a_unpack_combine_end"]),
            )
        else:
            values.update(
                a2a_pack_ms=0.0,
                a2a_all_to_all_ms=0.0,
                a2a_unpack_combine_ms=0.0,
            )
        values["output_collective_ms"] = (
            values["a2a_all_to_all_ms"]
            if is_a2a
            else values["output_reduce_scatter_ms"]
        )
        values["sequential_ag_plus_chunk_ms"] = (
            values["q_allgather_and_reorder_ms"] + values["local_chunk_attention_ms"]
        )
        return values


class _SequentialDCPAttentionRunnerBase(DCPAttentionRunner):
    """Shared local-kernel plumbing for pinned production orchestration paths."""

    chunk_before_context = False
    workspace_policy = "framework_style_per_call"
    supports_overlap_q_allgather = False

    def _reject_overlap(self, overlap_q_allgather: bool) -> None:
        if overlap_q_allgather:
            raise ValueError(
                f"{type(self).__name__} only supports single-stream execution; "
                "overlap_q_allgather must be False"
            )

    def _check_packed_runner_topology(self, h_kv: int) -> None:
        if h_kv != 1:
            raise ValueError(
                f"{type(self).__name__} models TP > global Hkv, so each rank must "
                f"hold exactly one KV head; got {h_kv}"
            )

    def _check_reference_common(
        self,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        seqlens_local: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        shape = self._check_common(q_local, k_local, v_local, seqlens_local)
        if k_local.shape[2] != 1:
            raise ValueError(
                f"{type(self).__name__} models TP > global Hkv, so each rank must "
                f"hold exactly one KV head; got {k_local.shape[2]}"
            )
        return shape

    def _check_chunk_inputs(
        self,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        b: int,
        sq: int,
        d: int,
    ) -> None:
        if k_chunk.shape != (b, sq, 1, d) or v_chunk.shape != k_chunk.shape:
            raise ValueError("k_chunk and v_chunk must have shape [B, Sq, 1, 128]")
        for tensor, name in ((k_chunk, "k_chunk"), (v_chunk, "v_chunk")):
            if (
                not tensor.is_cuda
                or tensor.device != self.device
                or tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous CUDA BF16 on the runner device"
                )

    def _all_gather_q(
        self,
        q_local: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> torch.Tensor:
        if timing:
            self._record_timing("q_ag_start", compute_stream)
        if self.world_size == 1:
            q_group = q_local
        else:
            b, sq, h_local, d = q_local.shape
            q_rank_major = torch.empty(
                (self.world_size * b, sq, h_local, d),
                device=self.device,
                dtype=q_local.dtype,
            )
            with _nvtx_range(f"{self.method_name}_q_allgather"):
                dist.all_gather_into_tensor(
                    q_rank_major, q_local, group=self.process_group
                )
                q_group = (
                    q_rank_major.view(self.world_size, b, sq, h_local, d)
                    .permute(1, 2, 0, 3, 4)
                    .reshape(b, sq, self.world_size * h_local, d)
                    .contiguous()
                )
        if timing:
            self._record_timing("q_ag_end", compute_stream)
        return q_group

    def _all_gather_q_varlen_sequential(
        self,
        q_local: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> torch.Tensor:
        if timing:
            self._record_timing("q_ag_start", compute_stream)
        if self.world_size == 1:
            q_group = q_local
        else:
            total_q, h_local, d = q_local.shape
            q_rank_major = torch.empty(
                (self.world_size * total_q, h_local, d),
                device=self.device,
                dtype=q_local.dtype,
            )
            with _nvtx_range(f"{self.method_name}_varlen_q_allgather"):
                dist.all_gather_into_tensor(
                    q_rank_major, q_local, group=self.process_group
                )
                q_group = (
                    q_rank_major.view(self.world_size, total_q, h_local, d)
                    .permute(1, 0, 2, 3)
                    .reshape(total_q, self.world_size * h_local, d)
                    .contiguous()
                )
        if timing:
            self._record_timing("q_ag_end", compute_stream)
        return q_group

    def _run_history_attention_sequential(
        self,
        q_group: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        seqlens_local: torch.Tensor,
        num_splits: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timing:
            self._record_timing("history_start", compute_stream)
        with _nvtx_range(f"{self.method_name}_local_history_attention"):
            history_out, history_lse = min_fa3_op.forward_kvcache(
                q_group,
                k_local,
                v_local,
                seqlens_local,
                num_splits=num_splits,
                return_lse=True,
                is_causal=False,
            )
        if timing:
            self._record_timing("history_end", compute_stream)
        return history_out, history_lse

    def _run_chunk_attention_sequential(
        self,
        q_local: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        num_splits: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timing:
            self._record_timing("chunk_start", compute_stream)
        chunk_lengths = torch.full(
            (q_local.shape[0],),
            q_local.shape[1],
            device=self.device,
            dtype=torch.int32,
        )
        with _nvtx_range(f"{self.method_name}_local_chunk_attention"):
            chunk_out, chunk_lse = min_fa3_op.forward_kvcache(
                q_local,
                k_chunk,
                v_chunk,
                chunk_lengths,
                num_splits=num_splits,
                return_lse=True,
                is_causal=True,
            )
        if timing:
            self._record_timing("chunk_end", compute_stream)
        return chunk_out, chunk_lse

    def _run_history_attention_varlen_sequential(
        self,
        q_group: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_local: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k_local: int,
        cu_seqlens_q_host: torch.Tensor,
        cu_seqlens_k_local_host: torch.Tensor,
        num_splits: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timing:
            self._record_timing("history_start", compute_stream)
        with _nvtx_range(f"{self.method_name}_varlen_local_history_attention"):
            history_out, history_lse = min_fa3_op.forward_kvcache_varlen(
                q_group,
                k_local,
                v_local,
                cu_seqlens_q,
                cu_seqlens_k_local,
                max_seqlen_q,
                max_seqlen_k_local,
                cu_seqlens_q_host=cu_seqlens_q_host,
                cu_seqlens_k_host=cu_seqlens_k_local_host,
                num_splits=num_splits,
                return_lse=True,
                is_causal=False,
            )
        if timing:
            self._record_timing("history_end", compute_stream)
        return history_out, history_lse

    def _run_chunk_attention_varlen_sequential(
        self,
        q_local: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        cu_seqlens_q_host: torch.Tensor,
        num_splits: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timing:
            self._record_timing("chunk_start", compute_stream)
        with _nvtx_range(f"{self.method_name}_varlen_local_chunk_attention"):
            chunk_out, chunk_lse = min_fa3_op.forward_kvcache_varlen(
                q_local,
                k_chunk,
                v_chunk,
                cu_seqlens_q,
                cu_seqlens_q,
                max_seqlen_q,
                max_seqlen_q,
                cu_seqlens_q_host=cu_seqlens_q_host,
                cu_seqlens_k_host=cu_seqlens_q_host,
                num_splits=num_splits,
                return_lse=True,
                is_causal=True,
            )
        if timing:
            self._record_timing("chunk_end", compute_stream)
        return chunk_out, chunk_lse

    def _combine_context(
        self,
        history_out: torch.Tensor,
        history_lse: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _merge_context_and_chunk(
        self,
        context_out: torch.Tensor,
        context_lse: torch.Tensor,
        chunk_out: torch.Tensor,
        chunk_lse: torch.Tensor,
        return_lse: bool,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> _DCPResult:
        raise NotImplementedError

    def _combine_context_varlen_sequential(
        self,
        history_out: torch.Tensor,
        history_lse: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, lse = self._combine_context(
            history_out.unsqueeze(0),
            history_lse.unsqueeze(0),
            compute_stream,
            timing=timing,
        )
        return output.squeeze(0), lse.squeeze(0)

    def _merge_context_and_chunk_varlen_sequential(
        self,
        context_out: torch.Tensor,
        context_lse: torch.Tensor,
        chunk_out: torch.Tensor,
        chunk_lse: torch.Tensor,
        return_lse: bool,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> _DCPResult:
        result = self._merge_context_and_chunk(
            context_out.unsqueeze(0),
            context_lse.unsqueeze(0),
            chunk_out.unsqueeze(0),
            chunk_lse.unsqueeze(0),
            return_lse,
            compute_stream,
            timing=timing,
        )
        if return_lse:
            output, lse = result
            return output.squeeze(0), lse.squeeze(0)
        return result.squeeze(0)

    def forward_decode(
        self,
        q_local: torch.Tensor,
        k_cache_local: torch.Tensor,
        v_cache_local: torch.Tensor,
        cache_seqlens_local: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
        *,
        _record_timing: bool = False,
    ) -> _DCPResult:
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError(f"{type(self).__name__} does not support concurrent calls")
        try:
            self._reap_inflight_tensors()
            self._reject_overlap(overlap_q_allgather)
            _, sq, _, _ = self._check_reference_common(
                q_local, k_cache_local, v_cache_local, cache_seqlens_local
            )
            if sq != 1:
                raise ValueError(f"forward_decode requires Sq=1, got {sq}")
            compute_stream = torch.cuda.current_stream(self.device)
            if _record_timing:
                self._last_timing_kind = "decode"
                self._record_timing("attention_start", compute_stream)
            q_group = self._all_gather_q(
                q_local, compute_stream, timing=_record_timing
            )
            if _record_timing:
                self._record_timing("ag_chunk_end", compute_stream)
            history_out, history_lse = self._run_history_attention_sequential(
                q_group,
                k_cache_local,
                v_cache_local,
                cache_seqlens_local,
                num_splits,
                compute_stream,
                timing=_record_timing,
            )
            output, lse = self._combine_context(
                history_out,
                history_lse,
                compute_stream,
                timing=_record_timing,
            )
            if _record_timing:
                self._record_timing("attention_end", compute_stream)
            return (output, lse) if return_lse else output
        finally:
            self._enqueue_lock.release()

    def forward_chunk_prefill(
        self,
        q_local: torch.Tensor,
        k_history_local: torch.Tensor,
        v_history_local: torch.Tensor,
        history_seqlens_local: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
        *,
        _record_timing: bool = False,
    ) -> _DCPResult:
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError(f"{type(self).__name__} does not support concurrent calls")
        try:
            self._reap_inflight_tensors()
            self._reject_overlap(overlap_q_allgather)
            b, sq, _, d = self._check_reference_common(
                q_local,
                k_history_local,
                v_history_local,
                history_seqlens_local,
            )
            self._check_chunk_inputs(k_chunk, v_chunk, b, sq, d)
            compute_stream = torch.cuda.current_stream(self.device)
            if _record_timing:
                self._last_timing_kind = "chunk"
                self._record_timing("attention_start", compute_stream)

            chunk_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
            if self.chunk_before_context:
                chunk_state = self._run_chunk_attention_sequential(
                    q_local,
                    k_chunk,
                    v_chunk,
                    num_splits,
                    compute_stream,
                    timing=_record_timing,
                )

            q_group = self._all_gather_q(
                q_local, compute_stream, timing=_record_timing
            )
            if _record_timing and self.chunk_before_context:
                self._record_timing("ag_chunk_end", compute_stream)
            history_out, history_lse = self._run_history_attention_sequential(
                q_group,
                k_history_local,
                v_history_local,
                history_seqlens_local,
                num_splits,
                compute_stream,
                timing=_record_timing,
            )
            context_out, context_lse = self._combine_context(
                history_out,
                history_lse,
                compute_stream,
                timing=_record_timing,
            )

            if chunk_state is None:
                chunk_state = self._run_chunk_attention_sequential(
                    q_local,
                    k_chunk,
                    v_chunk,
                    num_splits,
                    compute_stream,
                    timing=_record_timing,
                )
                if _record_timing:
                    self._record_timing("ag_chunk_end", compute_stream)
            result = self._merge_context_and_chunk(
                context_out,
                context_lse,
                *chunk_state,
                return_lse,
                compute_stream,
                timing=_record_timing,
            )
            if _record_timing:
                self._record_timing("attention_end", compute_stream)
            return result
        finally:
            self._enqueue_lock.release()

    def forward_decode_varlen(
        self,
        q_local: torch.Tensor,
        k_cache_local: torch.Tensor,
        v_cache_local: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_local: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k_local: int,
        *,
        cu_seqlens_q_host: torch.Tensor,
        cu_seqlens_k_local_host: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
        _record_timing: bool = False,
    ) -> _DCPResult:
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError(f"{type(self).__name__} does not support concurrent calls")
        try:
            self._reap_inflight_tensors()
            self._reject_overlap(overlap_q_allgather)
            (
                _,
                _,
                _,
                h_kv,
                _,
                q_lengths,
                _,
            ) = self._check_packed_common(
                q_local,
                k_cache_local,
                v_cache_local,
                cu_seqlens_q,
                cu_seqlens_k_local,
                max_seqlen_q,
                max_seqlen_k_local,
                cu_seqlens_q_host,
                cu_seqlens_k_local_host,
                num_splits,
            )
            self._check_packed_runner_topology(h_kv)
            if any(length != 1 for length in q_lengths):
                raise ValueError("forward_decode_varlen requires every q_len == 1")

            compute_stream = torch.cuda.current_stream(self.device)
            if _record_timing:
                self._last_timing_kind = "decode"
                self._record_timing("attention_start", compute_stream)
            q_group = self._all_gather_q_varlen_sequential(
                q_local, compute_stream, timing=_record_timing
            )
            if _record_timing:
                self._record_timing("ag_chunk_end", compute_stream)
            history_out, history_lse = self._run_history_attention_varlen_sequential(
                q_group,
                k_cache_local,
                v_cache_local,
                cu_seqlens_q,
                cu_seqlens_k_local,
                max_seqlen_q,
                max_seqlen_k_local,
                cu_seqlens_q_host,
                cu_seqlens_k_local_host,
                num_splits,
                compute_stream,
                timing=_record_timing,
            )
            output, lse = self._combine_context_varlen_sequential(
                history_out,
                history_lse,
                compute_stream,
                timing=_record_timing,
            )
            if _record_timing:
                self._record_timing("attention_end", compute_stream)
            return (output, lse) if return_lse else output
        finally:
            self._enqueue_lock.release()

    def forward_chunk_prefill_varlen(
        self,
        q_local: torch.Tensor,
        k_history_local: torch.Tensor,
        v_history_local: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_history_local: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_history_local: int,
        *,
        cu_seqlens_q_host: torch.Tensor,
        cu_seqlens_history_local_host: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
        _record_timing: bool = False,
    ) -> _DCPResult:
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError(f"{type(self).__name__} does not support concurrent calls")
        try:
            self._reap_inflight_tensors()
            self._reject_overlap(overlap_q_allgather)
            (
                total_q,
                _,
                d,
                h_kv,
                _,
                _,
                _,
            ) = self._check_packed_common(
                q_local,
                k_history_local,
                v_history_local,
                cu_seqlens_q,
                cu_seqlens_history_local,
                max_seqlen_q,
                max_seqlen_history_local,
                cu_seqlens_q_host,
                cu_seqlens_history_local_host,
                num_splits,
            )
            self._check_packed_runner_topology(h_kv)
            self._check_packed_chunk_inputs(k_chunk, v_chunk, total_q, h_kv, d)

            compute_stream = torch.cuda.current_stream(self.device)
            if _record_timing:
                self._last_timing_kind = "chunk"
                self._record_timing("attention_start", compute_stream)

            chunk_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
            if self.chunk_before_context:
                chunk_state = self._run_chunk_attention_varlen_sequential(
                    q_local,
                    k_chunk,
                    v_chunk,
                    cu_seqlens_q,
                    max_seqlen_q,
                    cu_seqlens_q_host,
                    num_splits,
                    compute_stream,
                    timing=_record_timing,
                )

            q_group = self._all_gather_q_varlen_sequential(
                q_local, compute_stream, timing=_record_timing
            )
            if _record_timing and self.chunk_before_context:
                self._record_timing("ag_chunk_end", compute_stream)
            history_out, history_lse = self._run_history_attention_varlen_sequential(
                q_group,
                k_history_local,
                v_history_local,
                cu_seqlens_q,
                cu_seqlens_history_local,
                max_seqlen_q,
                max_seqlen_history_local,
                cu_seqlens_q_host,
                cu_seqlens_history_local_host,
                num_splits,
                compute_stream,
                timing=_record_timing,
            )
            context_out, context_lse = self._combine_context_varlen_sequential(
                history_out,
                history_lse,
                compute_stream,
                timing=_record_timing,
            )

            if chunk_state is None:
                chunk_state = self._run_chunk_attention_varlen_sequential(
                    q_local,
                    k_chunk,
                    v_chunk,
                    cu_seqlens_q,
                    max_seqlen_q,
                    cu_seqlens_q_host,
                    num_splits,
                    compute_stream,
                    timing=_record_timing,
                )
                if _record_timing:
                    self._record_timing("ag_chunk_end", compute_stream)
            result = self._merge_context_and_chunk_varlen_sequential(
                context_out,
                context_lse,
                *chunk_state,
                return_lse,
                compute_stream,
                timing=_record_timing,
            )
            if _record_timing:
                self._record_timing("attention_end", compute_stream)
            return result
        finally:
            self._enqueue_lock.release()


class VLLMDCPAttentionRunner(_SequentialDCPAttentionRunnerBase):
    """Pinned vLLM default AG+RS orchestration using the local min FA3 op."""

    method_name = "vllm_ag_rs_min_fa3"
    varlen_method_name = "vllm_ag_rs_min_fa3_varlen"
    output_collective_kind = "bf16_reduce_scatter"

    def _combine_context(
        self,
        history_out: torch.Tensor,
        history_lse: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, sq, h_group, d = history_out.shape
        t = b * sq
        h_local = h_group // self.world_size
        if self.world_size == 1:
            return history_out, history_lse

        if timing:
            self._record_timing("lse_correct_start", compute_stream)
        with _nvtx_range("vllm_lse_allgather_and_triton_correction"):
            lse_flat = history_lse.permute(0, 2, 1).contiguous().view(t, h_group)
            gathered_lse = torch.empty(
                (self.world_size * t, h_group),
                device=self.device,
                dtype=torch.float32,
            )
            dist.all_gather_into_tensor(
                gathered_lse, lse_flat, group=self.process_group
            )
            gathered_lse_view = gathered_lse.view(self.world_size, t, h_group)
            corrected = torch.empty_like(history_out)
            global_lse_flat = torch.empty(
                (t, h_group), device=self.device, dtype=torch.float32
            )
            _vllm_correct_attn_cp_out_kernel[(t, h_group)](
                history_out.view(t, h_group, d),
                gathered_lse_view,
                corrected.view(t, h_group, d),
                global_lse_flat,
                *history_out.view(t, h_group, d).stride()[:2],
                *gathered_lse_view.stride(),
                T=t,
                H=h_group,
                D=d,
                WORLD_SIZE=self.world_size,
                WORLD_SIZE_ROUNDED=triton.next_power_of_2(self.world_size),
                RANK=self.rank,
                num_warps=4,
            )
        if timing:
            self._record_timing("lse_correct_end", compute_stream)
            self._record_timing("reduce_scatter_start", compute_stream)
        with _nvtx_range("vllm_bf16_output_reduce_scatter"):
            packed_head_major = corrected.view(t, h_group, d).movedim(0, 1).contiguous()
            output_head_major = torch.empty(
                (h_local, t, d), device=self.device, dtype=torch.bfloat16
            )
            dist.reduce_scatter_tensor(
                output_head_major,
                packed_head_major,
                op=dist.ReduceOp.SUM,
                group=self.process_group,
            )
            output = (
                output_head_major.movedim(0, 1)
                .contiguous()
                .view(b, sq, h_local, d)
            )
        if timing:
            self._record_timing("reduce_scatter_end", compute_stream)
        head_start = self.rank * h_local
        local_lse = (
            global_lse_flat.view(b, sq, h_group)[:, :, head_start : head_start + h_local]
            .permute(0, 2, 1)
            .contiguous()
        )
        return output, local_lse

    def _merge_context_and_chunk(
        self,
        context_out: torch.Tensor,
        context_lse: torch.Tensor,
        chunk_out: torch.Tensor,
        chunk_lse: torch.Tensor,
        return_lse: bool,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> _DCPResult:
        b, sq, h_local, d = context_out.shape
        merged_out = torch.empty_like(context_out)
        merged_lse = (
            torch.empty_like(context_lse) if return_lse else context_lse
        )
        if timing:
            self._record_timing("merge_start", compute_stream)
        with _nvtx_range("vllm_triton_merge_attn_states"):
            _merge_attn_states_kernel[(b * sq, h_local)](
                context_out,
                context_lse,
                chunk_out,
                chunk_lse,
                merged_out,
                merged_lse,
                *merged_out.stride()[:3],
                *context_lse.stride(),
                B=b,
                SQ=sq,
                H=h_local,
                D=d,
                STORE_LSE=return_lse,
                num_warps=4,
            )
        if timing:
            self._record_timing("merge_end", compute_stream)
        self._retain_merge_inputs(
            compute_stream, context_out, context_lse, chunk_out, chunk_lse
        )
        return (merged_out, merged_lse) if return_lse else merged_out


class VLLMA2ADCPAttentionRunner(VLLMDCPAttentionRunner):
    """Pinned vLLM A2A orchestration using the local min FA3 op.

    The data flow matches the ordinary GQA backend integration at
    ``vllm/v1/attention/backends/flash_attn.py`` in pinned commit
    ``a89015c6df8eeb37a843b717c97a5be1355de83d``.  Only the partial-state
    combine differs from :class:`VLLMDCPAttentionRunner`.
    """

    method_name = "vllm_a2a_min_fa3"
    varlen_method_name = "vllm_a2a_min_fa3_varlen"
    output_collective_kind = "bf16_packed_all_to_all"

    def _combine_context(
        self,
        history_out: torch.Tensor,
        history_lse: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_out.ndim != 4:
            raise ValueError("A2A history output must have shape [B, S, H_group, D]")
        b, sq, h_group, d = history_out.shape
        if history_out.dtype != torch.bfloat16 or d != 128:
            raise ValueError("A2A history output must be BF16 with head_dim 128")
        if history_lse.shape != (b, h_group, sq):
            raise ValueError(
                "A2A history LSE must have shape "
                f"[{b}, {h_group}, {sq}], got {tuple(history_lse.shape)}"
            )
        _dcp_a2a_head_owner_ranges(h_group, self.world_size)
        if history_lse.dtype != torch.float32:
            # vLLM PR #47801: the pack kernel bit-casts one FP32 LSE into two
            # BF16 lanes even when an attention backend returned activation dtype.
            history_lse = history_lse.to(torch.float32)
        if self.world_size == 1:
            return history_out, history_lse

        total_tokens = b * sq
        h_local = h_group // self.world_size
        output_flat = history_out.view(total_tokens, h_group, d)
        lse_flat = (
            history_lse.permute(0, 2, 1)
            .contiguous()
            .view(total_tokens, h_group)
        )
        lse_pack_dim = _dcp_a2a_lse_pack_dim(history_out.dtype)
        buffer_shape = (
            self.world_size,
            total_tokens,
            h_local,
            d + lse_pack_dim,
        )
        # vLLM PR #45487: graph-captured A2A buffers must be per-call tensors
        # owned by the graph private pool, never slices of a growable workspace.
        send_buffer = torch.empty(
            buffer_shape, device=self.device, dtype=history_out.dtype
        )
        recv_buffer = torch.empty_like(send_buffer)
        if self._capture_in_progress:
            self._capture_tensors.extend((send_buffer, recv_buffer))

        if timing:
            self._record_timing("a2a_pack_start", compute_stream)
        with _nvtx_range("vllm_a2a_pack_partial_states"):
            _dcp_a2a_pack_send(
                output_flat,
                lse_flat,
                send_buffer,
                self.world_size,
                h_local,
                d,
                lse_pack_dim,
            )
        if timing:
            self._record_timing("a2a_pack_end", compute_stream)
            self._record_timing("a2a_all_to_all_start", compute_stream)
        with _nvtx_range("vllm_packed_output_lse_all_to_all"):
            work = dist.all_to_all_single(
                recv_buffer.view(-1),
                send_buffer.view(-1),
                group=self.process_group,
                async_op=True,
            )
            work.wait()
        if self._capture_in_progress:
            self._works.append(work)
        if timing:
            self._record_timing("a2a_all_to_all_end", compute_stream)
            self._record_timing("a2a_unpack_combine_start", compute_stream)
        with _nvtx_range("vllm_a2a_unpack_lse_weighted_combine"):
            output, output_lse = _dcp_a2a_unpack_combine(
                recv_buffer, d, lse_pack_dim
            )
        if timing:
            self._record_timing("a2a_unpack_combine_end", compute_stream)
        self._retain_merge_inputs(
            compute_stream,
            history_out,
            history_lse,
            lse_flat,
            send_buffer,
            recv_buffer,
        )
        return (
            output.view(b, sq, h_local, d),
            output_lse.view(b, sq, h_local).permute(0, 2, 1).contiguous(),
        )


class SGLangDCPAttentionRunner(_SequentialDCPAttentionRunnerBase):
    """Pinned SGLang MHA AG+FP32-AR orchestration using the local min FA3 op."""

    method_name = "sglang_mha_ag_ar_min_fa3"
    varlen_method_name = "sglang_mha_ag_ar_min_fa3_varlen"
    output_collective_kind = "fp32_all_reduce"
    chunk_before_context = True

    def _combine_context(
        self,
        history_out: torch.Tensor,
        history_lse: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, sq, h_group, d = history_out.shape
        t = b * sq
        h_local = h_group // self.world_size
        if self.world_size == 1:
            return history_out, history_lse

        if timing:
            self._record_timing("lse_correct_start", compute_stream)
        with _nvtx_range("sglang_lse_allgather_torch_correction"):
            local_lse = history_lse.permute(0, 2, 1).contiguous().view(t, h_group)
            gathered_lse = torch.empty(
                (self.world_size * t, h_group),
                device=self.device,
                dtype=torch.float32,
            )
            dist.all_gather_into_tensor(
                gathered_lse, local_lse, group=self.process_group
            )
            global_lse = torch.logsumexp(
                gathered_lse.view(self.world_size, t, h_group), dim=0
            )
            scale = torch.exp(local_lse - global_lse).unsqueeze(-1)
            scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
            partial = torch.nan_to_num(
                history_out.view(t, h_group, d).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ) * scale
        if timing:
            self._record_timing("lse_correct_end", compute_stream)
            self._record_timing("reduce_scatter_start", compute_stream)
        with _nvtx_range("sglang_fp32_output_allreduce"):
            dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=self.process_group)
            head_start = self.rank * h_local
            output = (
                partial[:, head_start : head_start + h_local]
                .contiguous()
                .to(torch.bfloat16)
                .view(b, sq, h_local, d)
            )
        if timing:
            self._record_timing("reduce_scatter_end", compute_stream)
        local_global_lse = (
            global_lse.view(b, sq, h_group)[:, :, head_start : head_start + h_local]
            .permute(0, 2, 1)
            .contiguous()
        )
        return output, local_global_lse

    def _merge_context_and_chunk(
        self,
        context_out: torch.Tensor,
        context_lse: torch.Tensor,
        chunk_out: torch.Tensor,
        chunk_lse: torch.Tensor,
        return_lse: bool,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> _DCPResult:
        if timing:
            self._record_timing("merge_start", compute_stream)
        with _nvtx_range("sglang_torch_fp32_merge_attn_states"):
            merged_lse = torch.logaddexp(context_lse, chunk_lse)
            context_scale = (
                torch.exp(context_lse - merged_lse).permute(0, 2, 1).unsqueeze(-1)
            )
            chunk_scale = (
                torch.exp(chunk_lse - merged_lse).permute(0, 2, 1).unsqueeze(-1)
            )
            context_scale = torch.nan_to_num(
                context_scale, nan=0.0, posinf=0.0, neginf=0.0
            )
            chunk_scale = torch.nan_to_num(
                chunk_scale, nan=0.0, posinf=0.0, neginf=0.0
            )
            merged_out = (
                context_out.float() * context_scale + chunk_out.float() * chunk_scale
            ).to(torch.bfloat16)
        if timing:
            self._record_timing("merge_end", compute_stream)
        return (merged_out, merged_lse) if return_lse else merged_out

    def last_timing_ms(self, synchronize: bool = True) -> dict[str, float]:
        values = super().last_timing_ms(synchronize=synchronize)
        values["output_allreduce_ms"] = values["output_collective_ms"]
        values["output_reduce_scatter_ms"] = 0.0
        return values


@dataclass(frozen=True)
class _DCPMegaReplay:
    backend: Callable[..., object]
    backend_args: tuple[object, ...]
    result: object
    q_ready_count: int
    attention_count: int
    publish_count: int
    receive_count: int


@dataclass(frozen=True)
class _DCPMegaReplayPending:
    replay: _DCPMegaReplay
    pre_phase: int
    post_phase: int
    stream: torch.cuda.Stream


class DCPMegaAttentionRunner:
    """Persistent single-node workspace for batched varlen DCP mega forward.

    Construction is collective over ``node_process_group`` because the IPC
    :class:`TKParallelTensor` arenas exchange handles exactly once.  The
    workspace is intentionally eager-only and may not be used concurrently.
    """

    method_name = "dcp_mega_varlen"
    supports_cuda_graph = False
    supports_decode = False
    output_collective_kind = "bf16_ipc_a2a"
    workspace_policy = "runner_preallocated"

    _METADATA_HEADER_INTS = METADATA_HEADER_INTS
    _METADATA_HOST_SLOTS = 2
    PHASE_TIMESTAMP_NAMES = (
        "kernel_start",
        "q_allgather_done",
        "attention_done",
        "history_combine_done",
        "publish_done",
        "receive_done",
        "final_combine_done",
        "kernel_done",
    )

    def __init__(
        self,
        process_group: dist.ProcessGroup,
        node_process_group: dist.ProcessGroup,
        *,
        max_total_q: int,
        max_batch: int,
        Hq_local: int,
        max_num_splits: int = 128,
        num_comm_sm: int = 8,
        block_n_override: Optional[int] = None,
        record_phase_timestamps: bool = False,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before DCPMegaAttentionRunner"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("DCPMegaAttentionRunner requires CUDA")
        if block_n_override not in (None, 128, 176):
            raise ValueError("block_n_override must be None, 128, or 176")
        if Hq_local not in (4, 8):
            raise ValueError("DCP mega requires PackGQA and Hq_local in {4, 8}")
        values = {
            "max_total_q": max_total_q,
            "max_batch": max_batch,
            "Hq_local": Hq_local,
            "max_num_splits": max_num_splits,
            "num_comm_sm": num_comm_sm,
        }
        if any(not isinstance(value, int) or value <= 0 for value in values.values()):
            raise ValueError(f"runner capacities must be positive integers, got {values}")
        if max_batch > max_total_q:
            raise ValueError("positive-length varlen batches require max_batch <= max_total_q")
        if max_num_splits > 128:
            raise ValueError("max_num_splits must be in [1, 128]")

        self.process_group = process_group
        self.node_process_group = node_process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)
        self.node_world_size = dist.get_world_size(node_process_group)
        self.node_rank = dist.get_rank(node_process_group)
        if self.world_size not in (2, 4, 8):
            raise ValueError(
                f"DCP mega only supports DCP size 2, 4, or 8, got {self.world_size}"
            )
        if self.node_world_size not in (2, 4, 8):
            raise ValueError(
                "node-wide TK IPC currently requires node TP size in {2, 4, 8}; "
                f"got {self.node_world_size}"
            )
        if self.node_world_size < self.world_size:
            raise ValueError("node TP group cannot be smaller than the DCP group")
        for group, name in (
            (process_group, "DCP"),
            (node_process_group, "node TP"),
        ):
            backend = str(dist.get_backend(group)).lower()
            if "nccl" not in backend:
                raise RuntimeError(f"{name} process group must use NCCL, got {backend}")

        self.device = torch.device("cuda", torch.cuda.current_device())
        if self.node_rank != self.device.index:
            raise ValueError(
                "node TP group rank must equal the CUDA device index required by "
                f"TKParallelTensor; got node_rank={self.node_rank}, device={self.device.index}"
            )
        node_global_ranks = tuple(dist.get_process_group_ranks(node_process_group))
        dcp_global_ranks = tuple(dist.get_process_group_ranks(process_group))
        node_index = {global_rank: idx for idx, global_rank in enumerate(node_global_ranks)}
        try:
            self.dcp_node_ranks = tuple(node_index[rank] for rank in dcp_global_ranks)
        except KeyError as error:
            raise ValueError("every DCP rank must belong to node_process_group") from error
        if self.dcp_node_ranks[self.rank] != self.node_rank:
            raise ValueError(
                "DCP and node TP groups disagree about the current rank mapping: "
                f"dcp_node_ranks={self.dcp_node_ranks}, dcp_rank={self.rank}, "
                f"node_rank={self.node_rank}"
            )

        props = torch.cuda.get_device_properties(self.device)
        if props.major != 9 or props.minor != 0:
            raise RuntimeError(
                "DCP mega only supports Hopper SM90; current capability is "
                f"{props.major}.{props.minor}"
            )
        if num_comm_sm > props.multi_processor_count - 1:
            raise ValueError(
                "num_comm_sm must leave at least one compute SM; "
                f"got num_comm_sm={num_comm_sm}, num_sms={props.multi_processor_count}"
            )

        self.max_total_q = max_total_q
        self.max_batch = max_batch
        self.Hq_local = Hq_local
        self.max_num_splits = max_num_splits
        self.num_comm_sm = num_comm_sm
        self.block_n_override = block_n_override
        self.record_phase_timestamps = bool(record_phase_timestamps)
        self.num_sms = props.multi_processor_count
        self._padded_total_q = ((max_total_q + 15) // 16) * 16
        self._max_token_blocks = self._padded_total_q // 16

        # DCP_MEGA: VMM-backed bases are page aligned. The logical Q view keeps
        # max_total_q while the IPC allocation contains a 16-row tail pad.
        self._ipc_q = min_fa3_op.TKParallelTensor(
            [self._padded_total_q, Hq_local, 128],
            torch.bfloat16,
            self.node_rank,
            self.node_world_size,
            False,
        )
        self._ipc_history_send_o = min_fa3_op.TKParallelTensor(
            [self.world_size, self._padded_total_q, Hq_local, 128],
            torch.bfloat16,
            self.node_rank,
            self.node_world_size,
            False,
        )
        self._ipc_history_send_lse = min_fa3_op.TKParallelTensor(
            [self.world_size, self._padded_total_q, Hq_local],
            torch.float32,
            self.node_rank,
            self.node_world_size,
            False,
        )
        self._ipc_tile_ready = min_fa3_op.TKParallelTensor(
            [self.world_size, self._max_token_blocks],
            torch.int32,
            self.node_rank,
            self.node_world_size,
            False,
        )
        self._ipc_barrier = min_fa3_op.TKParallelTensor(
            [1],
            torch.int32,
            self.node_rank,
            self.node_world_size,
            False,
        )

        cuda = dict(device=self.device)
        group_heads = self.world_size * Hq_local
        self._q_group = torch.empty(
            (self._padded_total_q, group_heads, 128),
            dtype=torch.bfloat16,
            **cuda,
        )
        self._chunk_o = torch.empty(
            (max_total_q, Hq_local, 128), dtype=torch.bfloat16, **cuda
        )
        self._chunk_lse = torch.empty(
            (Hq_local, max_total_q), dtype=torch.float32, **cuda
        )
        self._history_o = torch.empty(
            (max_total_q, group_heads, 128), dtype=torch.bfloat16, **cuda
        )
        self._history_lse = torch.empty(
            (group_heads, max_total_q), dtype=torch.float32, **cuda
        )
        self._history_receive_o = torch.empty(
            (self.world_size, self._padded_total_q, Hq_local, 128),
            dtype=torch.bfloat16,
            **cuda,
        )
        self._history_receive_lse = torch.empty(
            (self.world_size, self._padded_total_q, Hq_local),
            dtype=torch.float32,
            **cuda,
        )
        self._chunk_o_partial = torch.empty(
            (max_num_splits, Hq_local, max_total_q, 128),
            dtype=torch.float32,
            **cuda,
        )
        self._chunk_lse_partial = torch.empty(
            (max_num_splits, Hq_local, max_total_q),
            dtype=torch.float32,
            **cuda,
        )
        self._history_o_partial = torch.empty(
            (max_num_splits, group_heads, max_total_q, 128),
            dtype=torch.float32,
            **cuda,
        )
        self._history_lse_partial = torch.empty(
            (max_num_splits, group_heads, max_total_q),
            dtype=torch.float32,
            **cuda,
        )
        self._output = torch.empty(
            (max_total_q, Hq_local, 128), dtype=torch.bfloat16, **cuda
        )
        self._output_lse = torch.empty(
            (Hq_local, max_total_q), dtype=torch.float32, **cuda
        )

        max_group_heads = group_heads
        max_pack_tiles = (
            (max_total_q * max_group_heads + 127) // 128 + max_batch - 1
        )
        max_history_base_tiles = max_pack_tiles
        max_chunk_base_tiles = (
            (max_total_q * Hq_local + 127) // 128 + max_batch - 1
        )
        max_attention = (
            max_history_base_tiles + max_chunk_base_tiles
        ) * max_num_splits
        max_q_tasks = self.world_size * self._max_token_blocks
        max_q_ready = self._max_token_blocks
        max_publish = self.world_size * self._max_token_blocks
        max_final = self._max_token_blocks
        vectors_per_work = 16 * Hq_local
        max_q_dependencies = max_history_base_tiles * min(128, max_q_ready)
        max_publish_dependencies = (
            max_publish * vectors_per_work * max_num_splits
        )
        max_final_dependencies = max_final * vectors_per_work * max_num_splits
        self._metadata_capacity = (
            self._METADATA_HEADER_INTS
            + max_attention * 8
            + max_q_tasks * 4
            + max_q_dependencies
            + max_publish * 8
            + max_publish_dependencies
            + max_final * 8
            + max_final_dependencies
            + 2 * max_batch
        )
        self._metadata_hosts = tuple(
            torch.empty(self._metadata_capacity, dtype=torch.int32, pin_memory=True)
            for _ in range(self._METADATA_HOST_SLOTS)
        )
        self._metadata_host_arrays = tuple(
            tensor.numpy() for tensor in self._metadata_hosts
        )
        self._metadata_device = torch.empty(
            self._metadata_capacity, dtype=torch.int32, **cuda
        )
        self._q_ready = torch.empty(max_q_ready, dtype=torch.int32, **cuda)
        self._attention_done = torch.empty(
            max_attention, dtype=torch.int32, **cuda
        )
        self._publish_ready = torch.empty(
            max_publish, dtype=torch.int32, **cuda
        )
        self._receive_ready = torch.empty(
            max_final * (self.world_size - 1),
            dtype=torch.int32,
            **cuda,
        )
        self._queue_state = torch.empty(9, dtype=torch.int32, **cuda)
        self._phase_timestamps = torch.empty(
            len(self.PHASE_TIMESTAMP_NAMES), dtype=torch.int64, **cuda
        )

        self._enqueue_lock = threading.Lock()
        self._completion_event = torch.cuda.Event()
        self._metadata_slot_events = tuple(
            torch.cuda.Event() for _ in range(self._METADATA_HOST_SLOTS)
        )
        self._metadata_slot_used = [False] * self._METADATA_HOST_SLOTS
        self._next_metadata_slot = 0
        self._has_completion = False
        self._phase = 0
        self._closed = False
        self._last_dispatch = None
        self._last_queue_counts: dict[str, object] | None = None
        self._last_replay: _DCPMegaReplay | None = None
        self._replay_pending: _DCPMegaReplayPending | None = None

        # The arenas must be visibly initialized before any subgroup starts a
        # forward. This is construction-time synchronization, not hot-path work.
        stream = torch.cuda.current_stream(self.device)
        self._ipc_tile_ready.data_.zero_()
        self._ipc_barrier.data_.zero_()
        stream.synchronize()
        dist.barrier(group=self.node_process_group)

    @property
    def q_backing(self) -> torch.Tensor:
        """Logical Q arena; pass only a prefix view to forward."""
        return self._ipc_q.data_[: self.max_total_q]

    def q_local(self, total_q: int) -> torch.Tensor:
        if not isinstance(total_q, int) or not 0 < total_q <= self.max_total_q:
            raise ValueError(f"total_q must be in [1, {self.max_total_q}]")
        return self._ipc_q.data_[:total_q]

    @property
    def last_dispatch(self):
        return self._last_dispatch

    @property
    def last_queue_counts(self) -> dict[str, object] | None:
        return self._last_queue_counts

    def copy_last_phase_timestamps(self, destination: torch.Tensor) -> None:
        """Copy the last raw ``%globaltimer`` milestones without synchronizing."""
        if not self.record_phase_timestamps:
            raise RuntimeError("phase timestamp recording is disabled for this runner")
        if (
            destination.device != self.device
            or destination.dtype != torch.int64
            or destination.shape != self._phase_timestamps.shape
            or not destination.is_contiguous()
        ):
            raise ValueError(
                "destination must be a contiguous CUDA int64 tensor with shape "
                f"{tuple(self._phase_timestamps.shape)} on {self.device}"
            )
        destination.copy_(self._phase_timestamps)

    def _next_phases(self) -> tuple[int, int]:
        if self._phase >= (1 << 31) - 4:
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.node_process_group)
            self._ipc_barrier.data_.zero_()
            self._ipc_tile_ready.data_.zero_()
            torch.cuda.current_stream(self.device).synchronize()
            dist.barrier(group=self.node_process_group)
            self._phase = 0
        self._phase += 2
        return self._phase - 1, self._phase

    def _pack_metadata(
        self,
        metadata,
        pre_phase: int,
        post_phase: int,
        host_array,
    ) -> int:
        try:
            payload = pack_dcp_mega_metadata(
                metadata,
                pre_phase=pre_phase,
                post_phase=post_phase,
                capacity=self._metadata_capacity,
            )
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        used = len(payload)
        host_array[:used] = payload
        return used

    @staticmethod
    def _check_packed_bf16(tensor: torch.Tensor, name: str) -> None:
        if (
            not tensor.is_cuda
            or tensor.dtype != torch.bfloat16
            or tensor.ndim != 3
            or tensor.shape[2] != 128
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA BF16 [total_tokens, heads, 128]"
            )

    @staticmethod
    def _host_offsets(tensor: torch.Tensor, name: str) -> tuple[int, ...]:
        if tensor.is_cuda or tensor.dtype != torch.int32 or tensor.ndim != 1:
            raise ValueError(f"{name} must be a contiguous CPU int32 tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        return tuple(int(value) for value in tensor.tolist())

    def forward_chunk_prefill_varlen(
        self,
        q_local: torch.Tensor,
        k_history_local: torch.Tensor,
        v_history_local: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_history_local: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_history_local: int,
        *,
        cu_seqlens_q_host: torch.Tensor,
        cu_seqlens_history_local_host: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
    ) -> _DCPResult:
        if self._closed:
            raise RuntimeError("DCPMegaAttentionRunner is closed")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DCP mega does not support CUDA Graph capture")
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError("DCP mega workspace does not support concurrent forward")
        try:
            self._check_packed_bf16(q_local, "q_local")
            self._check_packed_bf16(k_history_local, "k_history_local")
            self._check_packed_bf16(v_history_local, "v_history_local")
            self._check_packed_bf16(k_chunk, "k_chunk")
            self._check_packed_bf16(v_chunk, "v_chunk")
            total_q = q_local.shape[0]
            if not 0 < total_q <= self.max_total_q:
                raise ValueError(
                    f"q_local total_q must be in [1, {self.max_total_q}], got {total_q}"
                )
            if q_local.shape[1] != self.Hq_local:
                raise ValueError(
                    f"q_local must have Hq_local={self.Hq_local}, got {q_local.shape[1]}"
                )
            if q_local.data_ptr() != self._ipc_q.data_.data_ptr():
                raise ValueError(
                    "q_local must be a prefix view of runner.q_backing; implicit Q copies "
                    "are intentionally unsupported"
                )
            if any(tensor.device != self.device for tensor in (
                q_local,
                k_history_local,
                v_history_local,
                k_chunk,
                v_chunk,
                cu_seqlens_q,
                cu_seqlens_history_local,
            )):
                raise ValueError("all CUDA inputs must be on the runner device")
            for tensor, name in (
                (k_history_local, "k_history_local"),
                (v_history_local, "v_history_local"),
                (k_chunk, "k_chunk"),
                (v_chunk, "v_chunk"),
            ):
                if tensor.shape[1] != 1:
                    raise ValueError(f"{name} must have Hkv_group == 1")
            if k_history_local.shape != v_history_local.shape:
                raise ValueError("history K and V must have identical shapes")
            if k_chunk.shape != v_chunk.shape or k_chunk.shape[0] != total_q:
                raise ValueError("chunk K/V must both have shape [total_q, 1, 128]")
            for tensor, name in (
                (cu_seqlens_q, "cu_seqlens_q"),
                (cu_seqlens_history_local, "cu_seqlens_history_local"),
            ):
                if (
                    not tensor.is_cuda
                    or tensor.dtype != torch.int32
                    or tensor.ndim != 1
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(f"{name} must be a contiguous CUDA int32 tensor")

            q_offsets = self._host_offsets(cu_seqlens_q_host, "cu_seqlens_q_host")
            history_offsets = self._host_offsets(
                cu_seqlens_history_local_host,
                "cu_seqlens_history_local_host",
            )
            if len(q_offsets) - 1 > self.max_batch:
                raise ValueError(
                    f"batch size exceeds runner max_batch={self.max_batch}"
                )
            if q_offsets[-1] != total_q:
                raise ValueError("cu_seqlens_q_host[-1] must equal q_local.size(0)")
            if history_offsets[-1] != k_history_local.shape[0]:
                raise ValueError(
                    "cu_seqlens_history_local_host[-1] must equal history token count"
                )
            actual_max_q = max(b - a for a, b in zip(q_offsets, q_offsets[1:]))
            actual_max_history = max(
                b - a for a, b in zip(history_offsets, history_offsets[1:])
            )
            if max_seqlen_q != actual_max_q:
                raise ValueError(
                    f"max_seqlen_q={max_seqlen_q} does not match host mirror {actual_max_q}"
                )
            if max_seqlen_history_local != actual_max_history:
                raise ValueError(
                    "max_seqlen_history_local does not match its host mirror: "
                    f"{max_seqlen_history_local} vs {actual_max_history}"
                )
            if num_splits < 0 or num_splits > self.max_num_splits:
                raise ValueError(
                    f"num_splits must be 0 or in [1, {self.max_num_splits}]"
                )

            metadata = build_dcp_mega_metadata(
                q_offsets,
                history_offsets,
                hq_local=self.Hq_local,
                dcp_size=self.world_size,
                num_sms=self.num_sms,
                requested_num_splits=num_splits,
                block_n_override=self.block_n_override,
            )
            if metadata.dispatch.effective_num_splits > self.max_num_splits:
                raise ValueError(
                    "automatic split heuristic exceeds runner max_num_splits: "
                    f"{metadata.dispatch.effective_num_splits} > {self.max_num_splits}"
                )
            self._last_dispatch = metadata.dispatch
            token_blocks = metadata.token_block_count
            actual_counts = {
                "q_transfer_tasks": len(metadata.q_tasks),
                "publish_tasks": len(metadata.publish),
                "receive_tasks": metadata.receive_count,
                "final_tasks": len(metadata.final),
                "system_ready_signals": metadata.tile_ready_count,
            }
            self._last_queue_counts = {
                "token_blocks": token_blocks,
                "q_ready_counters": metadata.q_ready_count,
                "actual": actual_counts,
            }
            pre_phase, post_phase = self._next_phases()
            metadata_slot = self._next_metadata_slot
            if self._metadata_slot_used[metadata_slot]:
                self._metadata_slot_events[metadata_slot].synchronize()
            metadata_host = self._metadata_hosts[metadata_slot]
            metadata_used = self._pack_metadata(
                metadata,
                pre_phase,
                post_phase,
                self._metadata_host_arrays[metadata_slot],
            )

            stream = torch.cuda.current_stream(self.device)
            if self._has_completion:
                stream.wait_event(self._completion_event)
            output = self._output[:total_q]
            output_lse = self._output_lse[:, :total_q]
            backend = getattr(
                min_fa3_op, "forward_chunk_prefill_varlen_dcp_mega", None
            )
            if backend is None:
                raise RuntimeError(
                    "the installed _min_fa3_op extension does not contain the DCP mega "
                    "backend; rebuild the extension"
                )
            backend_args = (
                q_local,
                k_history_local,
                v_history_local,
                k_chunk,
                v_chunk,
                cu_seqlens_q,
                cu_seqlens_history_local,
                int(max_seqlen_q),
                int(max_seqlen_history_local),
                self._ipc_q,
                self._ipc_history_send_o,
                self._ipc_history_send_lse,
                self._ipc_tile_ready,
                self._ipc_barrier,
                self._q_group,
                self._chunk_o,
                self._chunk_lse,
                self._history_o,
                self._history_lse,
                self._history_receive_o,
                self._history_receive_lse,
                self._chunk_o_partial,
                self._chunk_lse_partial,
                self._history_o_partial,
                self._history_lse_partial,
                output,
                output_lse,
                metadata_host,
                self._metadata_device,
                metadata_used,
                self._q_ready,
                self._attention_done,
                self._publish_ready,
                self._receive_ready,
                self._queue_state,
                self._phase_timestamps,
                self.record_phase_timestamps,
                list(self.dcp_node_ranks),
                self.rank,
                self.num_comm_sm,
                bool(return_lse),
            )
            try:
                backend(*backend_args)
            finally:
                self._metadata_slot_events[metadata_slot].record(stream)
                self._metadata_slot_used[metadata_slot] = True
                self._next_metadata_slot = (
                    metadata_slot + 1
                ) % self._METADATA_HOST_SLOTS
            self._completion_event.record(stream)
            self._has_completion = True
            result = (output, output_lse) if return_lse else output
            self._last_replay = _DCPMegaReplay(
                backend=backend,
                backend_args=backend_args,
                result=result,
                q_ready_count=metadata.q_ready_count,
                attention_count=len(metadata.attention),
                publish_count=len(metadata.publish),
                receive_count=metadata.receive_count,
            )
            return result
        finally:
            self._enqueue_lock.release()

    forward_chunk_prefill_varlen_dcp_mega = forward_chunk_prefill_varlen

    def prepare_last_forward_replay(self) -> None:
        """Reset reusable state before timing the last prepared forward."""
        if self._closed:
            raise RuntimeError("DCPMegaAttentionRunner is closed")
        if self._last_replay is None:
            raise RuntimeError("no DCP mega forward is available for replay")
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError("DCP mega workspace does not support concurrent forward")
        try:
            if self._replay_pending is not None:
                raise RuntimeError("a prepared DCP mega replay is already pending")
            stream = torch.cuda.current_stream(self.device)
            if self._has_completion:
                stream.wait_event(self._completion_event)
            pre_phase, post_phase = self._next_phases()
            replay = self._last_replay
            self._q_ready[: replay.q_ready_count].zero_()
            self._attention_done[: replay.attention_count].zero_()
            self._publish_ready[: replay.publish_count].zero_()
            self._receive_ready[: replay.receive_count].zero_()
            self._queue_state.zero_()
            if self.record_phase_timestamps:
                self._phase_timestamps.zero_()
            self._replay_pending = _DCPMegaReplayPending(
                replay=replay,
                pre_phase=pre_phase,
                post_phase=post_phase,
                stream=stream,
            )
        except Exception:
            self._enqueue_lock.release()
            raise

    def replay_last_forward(
        self,
        *,
        return_timing_ms: bool = False,
        run_pre_barrier: bool = True,
    ):
        """Run the prepared mega replay, then enqueue the untimed post-barrier."""
        pending = self._replay_pending
        if pending is None:
            raise RuntimeError("prepare_last_forward_replay must be called first")
        stream = torch.cuda.current_stream(self.device)
        if stream != pending.stream:
            self._replay_pending = None
            self._enqueue_lock.release()
            raise RuntimeError("prepared DCP mega replay must use the preparing stream")
        try:
            elapsed_ms = pending.replay.backend(
                *pending.replay.backend_args,
                True,
                pending.pre_phase,
                False,
                run_pre_barrier,
                return_timing_ms,
            )
            min_fa3_op._dcp_mega_varlen_barrier(
                self._ipc_barrier,
                list(self.dcp_node_ranks),
                self.rank,
                pending.post_phase,
            )
            self._completion_event.record(stream)
            self._has_completion = True
            if return_timing_ms:
                return pending.replay.result, float(elapsed_ms)
            return pending.replay.result
        finally:
            self._replay_pending = None
            self._enqueue_lock.release()

    def close(self) -> None:
        if self._closed:
            return
        if self._replay_pending is not None:
            raise RuntimeError("cannot close DCP mega runner with a pending replay")
        if self._has_completion:
            self._completion_event.synchronize()
        for used, event in zip(self._metadata_slot_used, self._metadata_slot_events):
            if used:
                event.synchronize()
        self._closed = True

    def __enter__(self) -> "DCPMegaAttentionRunner":
        if self._closed:
            raise RuntimeError("DCPMegaAttentionRunner is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


__all__ = [
    "DCPTopology",
    "DCPAttentionCUDAGraph",
    "DCPAttentionRunner",
    "DCPMegaAttentionRunner",
    "SGLangDCPAttentionRunner",
    "TopologyIssue",
    "VLLMA2ADCPAttentionRunner",
    "VLLMDCPAttentionRunner",
    "make_topology",
    "validate_group_ranks",
    "validate_topology",
]
