"""Torchrun entry point for global-batch hybrid mega-ring benchmarks.

This benchmark compares the same global batch under four execution strategies:

- ``fa3_all_cp``: Python-side all-CP reference using FA3 block kernels. Every global
  sequence is split across all ranks, matching ``mega_ring_all_cp`` semantics.
- ``fa3_hybrid``: Python-side hybrid FA3 baseline. CP sequences use one batched
  varlen ring; local-only short sequences use one batched local FA call.
- ``mega_ring_all_cp``: legacy fused mega-ring CP. Every global sequence is split
  across all ranks, including sequences below the threshold.
- ``mega_ring_hybrid``: fused mega-ring hybrid mode. Sequences above the
  threshold are split across ranks; shorter sequences are assigned whole to one
  rank.

The all-CP and hybrid layouts differ, but the global workload is the same, so
the reported TFLOPS uses the same global attention FLOP count for every method.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist

THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import min_fa3_op

from ring_common import (
    flash_varlen_block_attention,
    min_fa3_varlen_block_attention,
    raise_if_any_rank_failed,
    ring_varlen_forward,
    zigzag_ring_varlen_forward,
)

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func3
except ImportError:
    print("FA3 flash_attn_varlen_func is not available, using min_fa3 block fallback for FA3 methods", flush=True)
    flash_attn_varlen_func3 = None


METHOD_ORDER = [
    "fa3_all_cp",
    "fa3_hybrid",
    "mega_ring_all_cp",
    "mega_ring_hybrid",
]


@dataclass(frozen=True)
class GlobalCase:
    """One global benchmark problem shape and attention mode."""

    global_lengths: tuple[int, ...]
    q_heads: int
    kv_heads: int
    head_dim: int
    is_causal: bool
    cp_threshold: int


@dataclass(frozen=True)
class Layout:
    """Rank-local layout for one execution strategy."""

    name: str
    local_lengths: tuple[int, ...]
    global_lengths: tuple[int, ...]
    cp_threshold: int


@dataclass(frozen=True)
class SmConfig:
    """One compute/communication SM allocation."""

    num_comp_sm: int
    num_comm_sm: int


@dataclass
class MethodRun:
    """Callable bundle for a benchmark method."""

    name: str
    timing_fn: Callable[[], torch.Tensor]
    ref: Optional[torch.Tensor]
    note: str = ""
    checkable: bool = True


@dataclass
class Result:
    """Printable benchmark result for one method."""

    time_ms: Optional[float]
    agg_tflops: Optional[float]
    avg_gpu_tflops: Optional[float]
    check: str
    note: str = ""
    rank_times_ms: Optional[list[float]] = None


@dataclass
class TimingResult:
    """CUDA-event timing summary for one measured method."""

    local_time_ms: float
    max_time_ms: float
    rank_times_ms: Optional[list[float]]


def parse_lengths(spec: str, name: str) -> list[int]:
    """Parse comma-separated sequence lengths."""
    lengths: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if token:
            lengths.append(int(token))
    if not lengths:
        raise SystemExit(f"{name} must provide at least one length")
    return lengths


def parse_methods(spec: str) -> list[str]:
    """Parse the method list while preserving user order and removing duplicates."""
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
    return deduped


def parse_sm_config_spec(spec: str) -> list[SmConfig]:
    """Parse comma-separated num_comp_sm:num_comm_sm pairs."""
    configs: list[SmConfig] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise SystemExit(f"invalid --sm-configs token '{token}', expected num_comp_sm:num_comm_sm")
        try:
            configs.append(SmConfig(int(parts[0]), int(parts[1])))
        except ValueError as exc:
            raise SystemExit(
                f"invalid --sm-configs token '{token}', expected integer num_comp_sm:num_comm_sm"
            ) from exc
    if not configs:
        raise SystemExit("--sm-configs must provide at least one num_comp_sm:num_comm_sm pair")
    return configs


def parse_args() -> argparse.Namespace:
    """Define the hybrid benchmark command-line interface."""
    parser = argparse.ArgumentParser(
        description="Distributed forward-only hybrid mega-ring benchmark for global varlen batches."
    )
    parser.add_argument(
        "--global-seqlens",
        "--seqlens",
        dest="global_seqlens",
        type=str,
        default="4096,1024,1024",
        help=(
            "Comma-separated global sequence lengths. Entries > threshold are CP sequences; "
            "entries <= threshold are assigned whole to one rank in hybrid mode."
        ),
    )
    parser.add_argument("--qhead", type=int, default=16, help="Number of query/output heads.")
    parser.add_argument("--kvhead", type=int, default=8, help="Number of key/value heads.")
    parser.add_argument("--headdim", type=int, default=128, help="Head dimension D.")
    parser.add_argument(
        "--mode",
        choices=("noncausal", "causal", "both"),
        default="both",
        help="Run noncausal, causal, or both.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help=f"Comma-separated methods from {METHOD_ORDER}, or all.",
    )
    parser.add_argument("--num-comp-sm", type=int, default=1, help="Compute CTAs for mega-ring kernels.")
    parser.add_argument("--num-comm-sm", type=int, default=1, help="Communication CTAs for mega-ring kernels.")
    parser.add_argument(
        "--sm-configs",
        type=str,
        default=None,
        help=(
            "Comma-separated num_comp_sm:num_comm_sm pairs to run in one invocation, "
            "for example 128:4,124:8,116:16. Overrides --num-comp-sm/--num-comm-sm."
        ),
    )
    parser.add_argument("--cp-threshold", type=int, default=2048, help="Global length threshold for hybrid CP.")
    parser.add_argument("--warmup-iters", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--num-iters", type=int, default=20, help="Measured iterations.")
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed.")
    parser.add_argument("--check", dest="check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mega-ring-ready-once",
        dest="mega_ring_ready_once",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the ready-once compact-prefix hybrid mega-ring path.",
    )
    parser.add_argument("--atol", type=float, default=2e-1, help="Correctness absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=2e-1, help="Correctness relative tolerance.")
    return parser.parse_args()


def init_distributed() -> tuple[int, int]:
    """Initialize single-node NCCL distributed execution."""
    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit("Run with torchrun so LOCAL_RANK and LOCAL_WORLD_SIZE are set")

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        dist.init_process_group(backend="nccl", device_id=device)
    except TypeError:
        dist.init_process_group(backend="nccl")

    if dist.get_world_size() != local_world_size:
        raise SystemExit(
            "This ring_test path is single-node only because TKParallelTensor uses local IPC: "
            f"world_size={dist.get_world_size()}, local_world_size={local_world_size}"
        )
    return local_rank, local_world_size


def available_on_all_ranks(local_available: bool) -> bool:
    """Return True only if the optional backend is importable on every rank."""
    available = torch.tensor([1 if local_available else 0], device="cuda", dtype=torch.int32)
    dist.all_reduce(available, op=dist.ReduceOp.MIN)
    return bool(available.item())


def cuda_barrier() -> None:
    """Barrier with an explicit CUDA device to avoid NCCL device-selection warnings."""
    try:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    except TypeError:
        dist.barrier()


def make_cu_seqlens(lengths: list[int] | tuple[int, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create matching device and host cumulative sequence length tensors."""
    host = torch.zeros((len(lengths) + 1,), dtype=torch.int32)
    for idx, length in enumerate(lengths):
        host[idx + 1] = host[idx] + int(length)
    return host.to(device=device), host


def cp_mask_for_global_lengths(global_lengths: tuple[int, ...], cp_threshold: int) -> list[bool]:
    """Return True for global sequences that use context parallelism."""
    return [length > cp_threshold for length in global_lengths]


def assign_local_only_batches(
    global_lengths: tuple[int, ...],
    cp_mask: list[bool],
    local_world_size: int,
) -> tuple[list[int], list[int]]:
    """Assign local-only global sequences to ranks with greedy load balancing."""
    owners = [-1] * len(global_lengths)
    loads = [0] * local_world_size
    local_indices = [idx for idx, is_cp in enumerate(cp_mask) if not is_cp]
    for batch_idx in sorted(local_indices, key=lambda idx: global_lengths[idx], reverse=True):
        owner = min(range(local_world_size), key=lambda rank: (loads[rank], rank))
        owners[batch_idx] = owner
        loads[owner] += global_lengths[batch_idx]
    return owners, loads


def make_all_cp_layout(case: GlobalCase, local_world_size: int) -> Layout:
    """Build the rank-local all-CP layout."""
    local_lengths = tuple(length // local_world_size for length in case.global_lengths)
    return Layout("all_cp", local_lengths, case.global_lengths, 0)


def make_hybrid_layout(case: GlobalCase, local_rank: int, local_world_size: int) -> Layout:
    """Build the rank-local hybrid layout for this rank.

    CP sequences must appear before local-only short sequences in
    ``case.global_lengths``. The CP batch offsets are shared across ranks, while
    local-only batches can have rank-specific zero/full lengths.
    """
    cp_mask = cp_mask_for_global_lengths(case.global_lengths, case.cp_threshold)
    owners, _ = assign_local_only_batches(case.global_lengths, cp_mask, local_world_size)
    local_lengths: list[int] = []
    for batch_idx, global_len in enumerate(case.global_lengths):
        if cp_mask[batch_idx]:
            local_lengths.append(global_len // local_world_size)
        else:
            local_lengths.append(global_len if owners[batch_idx] == local_rank else 0)
    return Layout("hybrid", tuple(local_lengths), case.global_lengths, case.cp_threshold)


def make_inputs(
    local_lengths: tuple[int, ...],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    local_rank: int,
    seed: int,
    salt: int,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate deterministic rank-local Q/K/V inputs for one layout."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + salt * 1000003 + local_rank * 1009 + (1 if is_causal else 0))
    total_tokens = sum(local_lengths)
    q = torch.randn(total_tokens, q_heads, head_dim, dtype=torch.bfloat16, device="cuda", generator=generator)
    k = torch.randn(total_tokens, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda", generator=generator)
    v = torch.randn(total_tokens, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda", generator=generator)
    return q.contiguous(), k.contiguous(), v.contiguous()


def gather_rank_blocks(local_tensor: torch.Tensor, local_rank: int, local_world_size: int) -> torch.Tensor:
    """Gather same-shaped local tensors into one rank-major tensor."""
    blocks = [torch.empty_like(local_tensor) for _ in range(local_world_size)]
    dist.all_gather(blocks, local_tensor)
    blocks[local_rank] = local_tensor
    return torch.cat(blocks, dim=0).contiguous()


def make_mega_parallel_tensors(
    k: torch.Tensor,
    v: torch.Tensor,
    local_rank: int,
    local_world_size: int,
) -> tuple[min_fa3_op.TKParallelTensor, min_fa3_op.TKParallelTensor]:
    """Create full [world_size * local_tokens, KVH, D] TK buffers for mega-ring."""
    full_shape = [local_world_size * k.size(0), k.size(1), k.size(2)]
    remote_k = min_fa3_op.TKParallelTensor(full_shape, torch.bfloat16, local_rank, local_world_size, False)
    remote_v = min_fa3_op.TKParallelTensor(full_shape, torch.bfloat16, local_rank, local_world_size, False)
    remote_k.data_.zero_()
    remote_v.data_.zero_()
    start = local_rank * k.size(0)
    end = start + k.size(0)
    remote_k.data_[start:end].copy_(k)
    remote_v.data_[start:end].copy_(v)
    return remote_k, remote_v


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
    """Run one FA3 varlen block, falling back to the local min_fa3 block."""
    if flash_attn_varlen_func3 is None:
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


def fa3_all_cp_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    layout: Layout,
    is_causal: bool,
) -> torch.Tensor:
    """Run all-CP FA3/min_fa3 over the whole rank-local varlen batch."""
    max_seqlen = max(layout.local_lengths)
    if is_causal:
        return zigzag_ring_varlen_forward(
            dist.group.WORLD,
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens_host,
            max_seqlen,
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
            max_seqlen,
            max_seqlen,
            causal_,
        ),
    )


def fa3_hybrid_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_total: int,
    cp_cu_seqlens: Optional[torch.Tensor],
    cp_cu_seqlens_host: Optional[torch.Tensor],
    cp_max_seqlen: int,
    local_total: int,
    local_cu_seqlens: Optional[torch.Tensor],
    local_cu_seqlens_host: Optional[torch.Tensor],
    local_max_seqlen: int,
    is_causal: bool,
) -> torch.Tensor:
    """Run hybrid FA3/min_fa3 with batched CP and one batched local-only call."""
    outputs: list[torch.Tensor] = []
    if cp_total > 0:
        q_cp = q[:cp_total]
        k_cp = k[:cp_total]
        v_cp = v[:cp_total]
        if is_causal:
            outputs.append(
                zigzag_ring_varlen_forward(
                    dist.group.WORLD,
                    q_cp,
                    k_cp,
                    v_cp,
                    cp_cu_seqlens,
                    cp_cu_seqlens_host,
                    cp_max_seqlen,
                    fa3_or_min_block_attention,
                )
            )
        else:
            outputs.append(
                ring_varlen_forward(
                    dist.group.WORLD,
                    q_cp,
                    k_cp,
                    v_cp,
                    False,
                    lambda q_, k_, v_, causal_: fa3_or_min_block_attention(
                        q_,
                        k_,
                        v_,
                        cp_cu_seqlens,
                        cp_cu_seqlens,
                        cp_cu_seqlens_host,
                        cp_cu_seqlens_host,
                        cp_max_seqlen,
                        cp_max_seqlen,
                        causal_,
                    ),
                )
            )

    if local_total > 0:
        q_local = q[cp_total:cp_total + local_total]
        k_local = k[cp_total:cp_total + local_total]
        v_local = v[cp_total:cp_total + local_total]
        local_out, _ = fa3_or_min_block_attention(
            q_local,
            k_local,
            v_local,
            local_cu_seqlens,
            local_cu_seqlens,
            local_cu_seqlens_host,
            local_cu_seqlens_host,
            local_max_seqlen,
            local_max_seqlen,
            is_causal,
        )
        outputs.append(local_out)

    if not outputs:
        return q.new_empty(q.shape)
    return torch.cat(outputs, dim=0).contiguous()


def mega_ring_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    max_seqlen: int,
    is_causal: bool,
    remote_k: min_fa3_op.TKParallelTensor,
    remote_v: min_fa3_op.TKParallelTensor,
    num_comp_sm: int,
    num_comm_sm: int,
    half_cu_seqlens: Optional[torch.Tensor],
    half_cu_seqlens_host: Optional[torch.Tensor],
    global_seqlens_host: Optional[torch.Tensor],
    cp_threshold: int,
    ready_once: bool,
) -> torch.Tensor:
    """Launch legacy all-CP or hybrid fused mega-ring attention."""
    return min_fa3_op.forward_varlen_mega_ring(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        is_causal,
        cu_seqlens_q_host=cu_seqlens_host,
        cu_seqlens_k_host=cu_seqlens_host,
        half_cu_seqlens=half_cu_seqlens,
        half_cu_seqlens_host=half_cu_seqlens_host,
        remote_k=remote_k,
        remote_v=remote_v,
        num_comp_sm=num_comp_sm,
        num_comm_sm=num_comm_sm,
        global_seqlens_host=global_seqlens_host,
        cp_threshold=cp_threshold,
        ready_once=ready_once,
    )


def reference_attention(
    q_i: torch.Tensor,
    k_i: torch.Tensor,
    v_i: torch.Tensor,
    is_causal: bool,
    query_pos: Optional[torch.Tensor] = None,
    key_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute one batch reference with PyTorch tensor ops."""
    if q_i.size(0) == 0:
        return q_i.new_empty(q_i.shape)

    q_i = q_i.float()
    k_i = k_i.float()
    v_i = v_i.float()
    qhead_per_kvhead = q_i.size(1) // k_i.size(1)
    if qhead_per_kvhead != 1:
        k_i = k_i.repeat_interleave(qhead_per_kvhead, dim=1)
        v_i = v_i.repeat_interleave(qhead_per_kvhead, dim=1)

    scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * (q_i.size(-1) ** -0.5)
    if is_causal:
        if query_pos is None:
            query_pos = torch.arange(q_i.size(0), device=q_i.device, dtype=torch.int64)
        if key_pos is None:
            key_pos = torch.arange(k_i.size(0), device=q_i.device, dtype=torch.int64)
        causal_mask = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v_i).to(dtype=torch.bfloat16)


def reference_layout_varlen(
    q: torch.Tensor,
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    gathered_k: torch.Tensor,
    gathered_v: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    layout: Layout,
    is_causal: bool,
    local_rank: int,
    local_world_size: int,
) -> torch.Tensor:
    """Compute the reference for one rank-local all-CP or hybrid layout."""
    outputs: list[torch.Tensor] = []
    total_tokens = int(cu_seqlens_host[-1].item())

    for batch_idx, local_len in enumerate(layout.local_lengths):
        q_start = int(cu_seqlens_host[batch_idx].item())
        q_end = int(cu_seqlens_host[batch_idx + 1].item())
        if local_len == 0:
            outputs.append(q.new_empty((0, q.size(1), q.size(2))))
            continue
        q_i = q[q_start:q_end]

        if layout.global_lengths[batch_idx] <= layout.cp_threshold:
            k_i = local_k[q_start:q_end]
            v_i = local_v[q_start:q_end]
            outputs.append(reference_attention(q_i, k_i, v_i, is_causal))
            continue

        k_blocks = []
        v_blocks = []
        key_positions = []
        half_len = local_len // 2
        for rank_idx in range(local_world_size):
            rank_offset = rank_idx * total_tokens
            k_blocks.append(gathered_k[rank_offset + q_start:rank_offset + q_end])
            v_blocks.append(gathered_v[rank_offset + q_start:rank_offset + q_end])
            if is_causal:
                front_pos = torch.arange(half_len, device=q.device, dtype=torch.int64) + rank_idx * half_len
                back_pos = (
                    torch.arange(half_len, device=q.device, dtype=torch.int64)
                    + (2 * local_world_size - 1 - rank_idx) * half_len
                )
                key_positions.append(torch.cat([front_pos, back_pos], dim=0))

        k_i = torch.cat(k_blocks, dim=0)
        v_i = torch.cat(v_blocks, dim=0)
        if is_causal:
            query_front = torch.arange(half_len, device=q.device, dtype=torch.int64) + local_rank * half_len
            query_back = (
                torch.arange(half_len, device=q.device, dtype=torch.int64)
                + (2 * local_world_size - 1 - local_rank) * half_len
            )
            query_pos = torch.cat([query_front, query_back], dim=0)
            key_pos = torch.cat(key_positions, dim=0)
        else:
            query_pos = None
            key_pos = None
        outputs.append(reference_attention(q_i, k_i, v_i, is_causal, query_pos, key_pos))

    return torch.cat(outputs, dim=0).contiguous()


def build_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    layout: Layout,
    is_causal: bool,
    local_rank: int,
    local_world_size: int,
) -> torch.Tensor:
    """Gather K/V and compute the method-specific reference."""
    gathered_k = gather_rank_blocks(k, local_rank, local_world_size)
    gathered_v = gather_rank_blocks(v, local_rank, local_world_size)
    return reference_layout_varlen(
        q,
        k,
        v,
        gathered_k,
        gathered_v,
        cu_seqlens_host,
        layout,
        is_causal,
        local_rank,
        local_world_size,
    )


def build_fa3_run(
    case: GlobalCase,
    local_rank: int,
    local_world_size: int,
    args: argparse.Namespace,
) -> MethodRun:
    """Build the Python FA3 all-CP run."""
    layout = make_all_cp_layout(case, local_world_size)
    q, k, v = make_inputs(layout.local_lengths, case.q_heads, case.kv_heads, case.head_dim, local_rank, args.seed, 1, case.is_causal)
    cu_seqlens, cu_seqlens_host = make_cu_seqlens(layout.local_lengths, q.device)
    ref = build_ref(q, k, v, cu_seqlens_host, layout, case.is_causal, local_rank, local_world_size) if args.check else None
    note = "all-CP FA3 blocks" if flash_attn_varlen_func3 is not None else "all-CP fallback: min_fa3_varlen blocks"

    def fn() -> torch.Tensor:
        return fa3_all_cp_forward(q, k, v, cu_seqlens, cu_seqlens_host, layout, case.is_causal)

    return MethodRun("fa3_all_cp", fn, ref, note)


def build_fa3_hybrid_run(
    case: GlobalCase,
    local_rank: int,
    local_world_size: int,
    args: argparse.Namespace,
) -> MethodRun:
    """Build the Python FA3 hybrid run."""
    layout = make_hybrid_layout(case, local_rank, local_world_size)
    q, k, v = make_inputs(layout.local_lengths, case.q_heads, case.kv_heads, case.head_dim, local_rank, args.seed, 4, case.is_causal)
    cu_seqlens, cu_seqlens_host = make_cu_seqlens(layout.local_lengths, q.device)
    ref = build_ref(q, k, v, cu_seqlens_host, layout, case.is_causal, local_rank, local_world_size) if args.check else None

    cp_count = sum(int(length > case.cp_threshold) for length in case.global_lengths)
    cp_lengths = tuple(layout.local_lengths[:cp_count])
    local_lengths = tuple(length for length in layout.local_lengths[cp_count:] if length > 0)
    cp_total = sum(cp_lengths)
    local_total = sum(local_lengths)

    cp_cu_seqlens = None
    cp_cu_seqlens_host = None
    cp_max_seqlen = 0
    if cp_total > 0:
        cp_cu_seqlens, cp_cu_seqlens_host = make_cu_seqlens(cp_lengths, q.device)
        cp_max_seqlen = max(cp_lengths)

    local_cu_seqlens = None
    local_cu_seqlens_host = None
    local_max_seqlen = 0
    if local_total > 0:
        local_cu_seqlens, local_cu_seqlens_host = make_cu_seqlens(local_lengths, q.device)
        local_max_seqlen = max(local_lengths)

    note = (
        "hybrid FA3 batched CP + local"
        if flash_attn_varlen_func3 is not None
        else "hybrid fallback: min_fa3_varlen batched CP + local"
    )

    def fn() -> torch.Tensor:
        return fa3_hybrid_forward(
            q,
            k,
            v,
            cp_total,
            cp_cu_seqlens,
            cp_cu_seqlens_host,
            cp_max_seqlen,
            local_total,
            local_cu_seqlens,
            local_cu_seqlens_host,
            local_max_seqlen,
            case.is_causal,
        )

    return MethodRun("fa3_hybrid", fn, ref, note)


def build_all_cp_run(
    case: GlobalCase,
    local_rank: int,
    local_world_size: int,
    args: argparse.Namespace,
) -> MethodRun:
    """Build the legacy all-CP mega-ring run."""
    layout = make_all_cp_layout(case, local_world_size)
    q, k, v = make_inputs(layout.local_lengths, case.q_heads, case.kv_heads, case.head_dim, local_rank, args.seed, 2, case.is_causal)
    cu_seqlens, cu_seqlens_host = make_cu_seqlens(layout.local_lengths, q.device)
    remote_k, remote_v = make_mega_parallel_tensors(k, v, local_rank, local_world_size)
    half_cu_seqlens = None
    half_cu_seqlens_host = None
    if case.is_causal:
        half_lengths = [length // 2 for length in layout.local_lengths]
        half_cu_seqlens, half_cu_seqlens_host = make_cu_seqlens(half_lengths, q.device)
    ref = build_ref(q, k, v, cu_seqlens_host, layout, case.is_causal, local_rank, local_world_size) if args.check else None

    def fn() -> torch.Tensor:
        return mega_ring_forward(
            q,
            remote_k.data_,
            remote_v.data_,
            cu_seqlens,
            cu_seqlens_host,
            max(layout.local_lengths),
            case.is_causal,
            remote_k,
            remote_v,
            args.num_comp_sm,
            args.num_comm_sm,
            half_cu_seqlens,
            half_cu_seqlens_host,
            None,
            case.cp_threshold,
            False,
        )

    return MethodRun("mega_ring_all_cp", fn, ref, "legacy all-CP mega-ring")


def build_hybrid_run(
    case: GlobalCase,
    local_rank: int,
    local_world_size: int,
    args: argparse.Namespace,
) -> MethodRun:
    """Build the fused hybrid mega-ring run."""
    layout = make_hybrid_layout(case, local_rank, local_world_size)
    q, k, v = make_inputs(layout.local_lengths, case.q_heads, case.kv_heads, case.head_dim, local_rank, args.seed, 3, case.is_causal)
    cu_seqlens, cu_seqlens_host = make_cu_seqlens(layout.local_lengths, q.device)
    remote_k, remote_v = make_mega_parallel_tensors(k, v, local_rank, local_world_size)
    global_seqlens_host = torch.tensor(layout.global_lengths, dtype=torch.int32)
    ref = build_ref(q, k, v, cu_seqlens_host, layout, case.is_causal, local_rank, local_world_size) if args.check else None
    cp_count = sum(int(length > case.cp_threshold) for length in case.global_lengths)

    def fn() -> torch.Tensor:
        return mega_ring_forward(
            q,
            remote_k.data_,
            remote_v.data_,
            cu_seqlens,
            cu_seqlens_host,
            max(layout.local_lengths),
            case.is_causal,
            remote_k,
            remote_v,
            args.num_comp_sm,
            args.num_comm_sm,
            None,
            None,
            global_seqlens_host,
            case.cp_threshold,
            args.mega_ring_ready_once,
        )

    path_note = "ready-once" if args.mega_ring_ready_once else "step-path"
    return MethodRun("mega_ring_hybrid", fn, ref, f"hybrid CP batches={cp_count}, {path_note}")


def build_method_runs(
    methods: list[str],
    case: GlobalCase,
    local_rank: int,
    local_world_size: int,
    args: argparse.Namespace,
) -> list[MethodRun]:
    """Create requested method runs for one global case."""
    runs: list[MethodRun] = []
    for method in methods:
        if method == "fa3_all_cp":
            runs.append(build_fa3_run(case, local_rank, local_world_size, args))
        elif method == "fa3_hybrid":
            runs.append(build_fa3_hybrid_run(case, local_rank, local_world_size, args))
        elif method == "mega_ring_all_cp":
            runs.append(build_all_cp_run(case, local_rank, local_world_size, args))
        elif method == "mega_ring_hybrid":
            runs.append(build_hybrid_run(case, local_rank, local_world_size, args))
        else:
            raise RuntimeError(f"unhandled method {method}")
    return runs


def measure_distributed_ms(fn: Callable[[], torch.Tensor], warmup_iters: int, num_iters: int) -> TimingResult:
    """Measure average CUDA-event latency for local rank time and max rank time."""
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    cuda_barrier()

    local_samples_ms: list[float] = []
    max_samples_ms: list[float] = []
    for _ in range(num_iters):
        cuda_barrier()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_tensor = torch.tensor([elapsed_ms], device="cuda", dtype=torch.float64)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        local_samples_ms.append(elapsed_ms)
        max_samples_ms.append(elapsed_tensor.item())
    cuda_barrier()

    local_avg_ms = sum(local_samples_ms) / len(local_samples_ms)
    max_avg_ms = sum(max_samples_ms) / len(max_samples_ms)
    local_avg_tensor = torch.tensor([local_avg_ms], device="cuda", dtype=torch.float64)
    rank_time_tensors = [torch.empty_like(local_avg_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(rank_time_tensors, local_avg_tensor)
    rank_times_ms = [tensor.item() for tensor in rank_time_tensors] if dist.get_rank() == 0 else None
    return TimingResult(local_avg_ms, max_avg_ms, rank_times_ms)


def aggregate_global_scores(global_lengths: tuple[int, ...], is_causal: bool) -> int:
    """Count global attention score elements for the shared workload."""
    total_scores = 0
    for global_len in global_lengths:
        if is_causal:
            total_scores += global_len * (global_len + 1) // 2
        else:
            total_scores += global_len * global_len
    return total_scores


def aggregate_tflops(score_count: int, q_heads: int, head_dim: int, time_ms: float) -> float:
    """Compute aggregate TFLOPS from global attention score count and elapsed time."""
    flops = 4 * score_count * q_heads * head_dim
    return float(flops) / (time_ms * 1e-3) / 1e12


def check_output(
    method: str,
    fn: Callable[[], torch.Tensor],
    ref: torch.Tensor,
    atol: float,
    rtol: float,
) -> str:
    """Run a method check and synchronize assertion failures across ranks."""
    local_error = None
    try:
        out = fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)
    except Exception as exc:
        local_error = f"{method}: {exc}"
    raise_if_any_rank_failed(local_error, dist.group.WORLD)
    return "ok"


def run_case(
    case: GlobalCase,
    methods: list[str],
    local_rank: int,
    local_world_size: int,
    args: argparse.Namespace,
) -> dict[str, Result]:
    """Run all requested methods for one global shape/mode case."""
    runs = build_method_runs(methods, case, local_rank, local_world_size, args)
    score_count = aggregate_global_scores(case.global_lengths, case.is_causal)

    results: dict[str, Result] = {}
    for run in runs:
        try:
            timing = measure_distributed_ms(run.timing_fn, args.warmup_iters, args.num_iters)
            time_ms = timing.max_time_ms
            agg_tflops = aggregate_tflops(score_count, case.q_heads, case.head_dim, time_ms)
            avg_gpu_tflops = agg_tflops / local_world_size
            check = "skip"
            if args.check and run.checkable and run.ref is not None:
                check = check_output(run.name, run.timing_fn, run.ref, args.atol, args.rtol)
            results[run.name] = Result(agg_tflops=agg_tflops, avg_gpu_tflops=avg_gpu_tflops, time_ms=time_ms, check=check, note=run.note, rank_times_ms=timing.rank_times_ms)
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            msg = str(exc).splitlines()[0]
            results[run.name] = Result(None, None, None, "oom", msg)
        except Exception as exc:
            results[run.name] = Result(None, None, None, "error", str(exc))
        cuda_barrier()
    return results


def make_cases(args: argparse.Namespace, global_lengths: list[int]) -> list[GlobalCase]:
    """Expand CLI arguments into concrete benchmark cases."""
    causal_values = {
        "noncausal": [False],
        "causal": [True],
        "both": [False, True],
    }[args.mode]
    return [
        GlobalCase(tuple(global_lengths), args.qhead, args.kvhead, args.headdim, is_causal, args.cp_threshold)
        for is_causal in causal_values
    ]


def validate_causal_cp_local_lengths(lengths: list[int] | tuple[int, ...], label: str) -> None:
    """Validate causal mega-ring CP half-length constraints."""
    for idx, local_len in enumerate(lengths):
        if local_len % 2 != 0:
            raise SystemExit(f"{label} batch {idx} requires even local length for causal CP, got {local_len}")
        half_len = local_len // 2
        if half_len <= 0 or half_len % 128 != 0:
            raise SystemExit(
                f"{label} batch {idx} requires local_len / 2 to be 128-aligned for causal CP, "
                f"got local_len={local_len}, half_len={half_len}"
            )


def validate_causal_fa3_all_cp_lengths(lengths: list[int] | tuple[int, ...]) -> None:
    """Validate causal Python zigzag all-CP layout constraints."""
    for idx, local_len in enumerate(lengths):
        if local_len % 2 != 0:
            raise SystemExit(f"fa3_all_cp batch {idx} requires even local length for causal zigzag, got {local_len}")


def validate_cp_batches_first(global_lengths: list[int], cp_threshold: int) -> None:
    """Require CP batches before local-only short batches.

    The hybrid mega-ring benchmark uses one rank-local cu_seqlens array per
    rank. Local-only batches may be full length on one rank and zero length on
    another, so any CP batch placed after a local-only batch could have different
    offsets across ranks. Keeping all CP batches first preserves identical CP
    offsets.
    """
    seen_local = False
    for idx, length in enumerate(global_lengths):
        is_cp = length > cp_threshold
        if not is_cp:
            seen_local = True
        elif seen_local:
            raise SystemExit(
                "For mega_ring_hybrid benchmark layout, put CP sequences before local-only sequences "
                "so CP batch offsets are identical across ranks. "
                f"batch {idx} has global length {length} > threshold {cp_threshold} after a local-only batch."
            )


def validate_hybrid_local_balance(global_lengths: list[int], cp_threshold: int, local_world_size: int) -> None:
    """Require equal total rank-local token count for hybrid TKParallelTensor buffers."""
    cp_mask = cp_mask_for_global_lengths(tuple(global_lengths), cp_threshold)
    _, local_loads = assign_local_only_batches(tuple(global_lengths), cp_mask, local_world_size)
    if len(set(local_loads)) != 1:
        raise SystemExit(
            "Hybrid local-only assignment must have equal total local-only tokens per rank for this benchmark. "
            f"local-only loads={local_loads}. Use balanced short sequences, for example one 1024 sequence per rank."
        )


def validate_args(
    args: argparse.Namespace,
    methods: list[str],
    global_lengths: list[int],
    local_world_size: int,
    sm_configs: list[SmConfig],
) -> None:
    """Validate arguments against the minimal demo's supported configuration."""
    if any(length <= 0 for length in global_lengths):
        raise SystemExit(f"all --global-seqlens entries must be positive, got {global_lengths}")
    if args.cp_threshold < 0:
        raise SystemExit(f"--cp-threshold must be non-negative, got {args.cp_threshold}")
    if args.headdim != 128:
        raise SystemExit(f"This demo requires D=128, got D={args.headdim}")
    if args.qhead % args.kvhead != 0:
        raise SystemExit(f"qhead must be divisible by kvhead, got qhead={args.qhead}, kvhead={args.kvhead}")
    if args.num_iters <= 0:
        raise SystemExit(f"--num-iters must be positive, got {args.num_iters}")
    if args.warmup_iters < 0:
        raise SystemExit(f"--warmup-iters must be non-negative, got {args.warmup_iters}")

    for sm_config in sm_configs:
        if sm_config.num_comp_sm <= 0:
            raise SystemExit(f"num_comp_sm must be positive, got {sm_config.num_comp_sm}")
        if sm_config.num_comm_sm < 0:
            raise SystemExit(f"num_comm_sm must be non-negative, got {sm_config.num_comm_sm}")

    if any(method in methods for method in ("mega_ring_all_cp", "mega_ring_hybrid")):
        if args.kvhead * args.headdim != 1024:
            raise SystemExit(
                "mega-ring communication path requires kvhead * headdim == 1024, "
                f"got kvhead={args.kvhead}, headdim={args.headdim}"
            )
        if local_world_size > 1 and any(sm_config.num_comm_sm <= 0 for sm_config in sm_configs):
            raise SystemExit("multi-rank mega-ring paths require num_comm_sm > 0")

    if any(method in methods for method in ("fa3_all_cp", "mega_ring_all_cp")):
        for idx, global_len in enumerate(global_lengths):
            if global_len % local_world_size != 0:
                raise SystemExit(
                    "all-CP benchmark requires every global length to be divisible by world_size. "
                    f"batch={idx}, global_len={global_len}, world_size={local_world_size}"
                )

    if any(method in methods for method in ("fa3_hybrid", "mega_ring_hybrid")):
        validate_cp_batches_first(global_lengths, args.cp_threshold)
        if "mega_ring_hybrid" in methods or args.check:
            validate_hybrid_local_balance(global_lengths, args.cp_threshold, local_world_size)
        for idx, global_len in enumerate(global_lengths):
            if global_len > args.cp_threshold and global_len % local_world_size != 0:
                raise SystemExit(
                    "hybrid CP batch requires global length to be divisible by world_size. "
                    f"batch={idx}, global_len={global_len}, world_size={local_world_size}"
                )

    if args.mode in ("causal", "both"):
        if "fa3_all_cp" in methods:
            validate_causal_fa3_all_cp_lengths(
                [global_len // local_world_size for global_len in global_lengths]
            )
        if "fa3_hybrid" in methods:
            validate_causal_fa3_all_cp_lengths(
                [
                    global_len // local_world_size
                    for global_len in global_lengths
                    if global_len > args.cp_threshold
                ]
            )
        if "mega_ring_all_cp" in methods:
            validate_causal_cp_local_lengths(
                [global_len // local_world_size for global_len in global_lengths],
                "all-CP",
            )
        if "mega_ring_hybrid" in methods:
            validate_causal_cp_local_lengths(
                [
                    global_len // local_world_size
                    for global_len in global_lengths
                    if global_len > args.cp_threshold
                ],
                "hybrid CP",
            )


def print_results(case: GlobalCase, results: dict[str, Result], methods: list[str]) -> None:
    """Print one result table on rank 0."""
    mode = "causal" if case.is_causal else "noncausal"
    cp_count = sum(int(length > case.cp_threshold) for length in case.global_lengths)
    print(
        f"\nB={len(case.global_lengths)}, global_seqlens={list(case.global_lengths)}, "
        f"CP batches={cp_count}, threshold={case.cp_threshold}, QH={case.q_heads}, "
        f"KVH={case.kv_heads}, D={case.head_dim}, mode={mode}"
    )
    rows: list[tuple[str, str, str, str, str, str]] = []
    for method in methods:
        result = results.get(method)
        if result is None:
            continue
        if result.time_ms is None:
            time_s = "N/A"
        elif result.rank_times_ms is None:
            time_s = f"max_across_ranks={result.time_ms:.3f}"
        else:
            rank_times_s = ", ".join(f"t{rank}={time_ms:.3f}" for rank, time_ms in enumerate(result.rank_times_ms))
            time_s = f"{rank_times_s} | max_across_ranks={result.time_ms:.3f}"
        agg_tflops_s = "N/A" if result.agg_tflops is None else f"{result.agg_tflops:.1f}"
        avg_gpu_tflops_s = "N/A" if result.avg_gpu_tflops is None else f"{result.avg_gpu_tflops:.1f}"
        rows.append((method, time_s, agg_tflops_s, avg_gpu_tflops_s, result.check, result.note))

    time_width = max((64, *(len(row[1]) for row in rows)))
    print(f"{'Method':<20} {'Time ms':<{time_width}} {'Agg TFLOPS':>12} {'Avg/GPU':>10} {'Check':>10}  Note")
    for method, time_s, agg_tflops_s, avg_gpu_tflops_s, check, note in rows:
        print(f"{method:<20} {time_s:<{time_width}} {agg_tflops_s:>12} {avg_gpu_tflops_s:>10} {check:>10}  {note}")


def main() -> None:
    """Program entry point."""
    global flash_attn_varlen_func3

    args = parse_args()
    methods = parse_methods(args.methods)
    sm_configs = (
        parse_sm_config_spec(args.sm_configs)
        if args.sm_configs is not None
        else [SmConfig(args.num_comp_sm, args.num_comm_sm)]
    )
    global_lengths = parse_lengths(args.global_seqlens, "--global-seqlens")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    local_rank, local_world_size = init_distributed()
    try:
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) != (9, 0):
            raise SystemExit(f"This demo requires SM90 Hopper, got {(major, minor)}")
        if not available_on_all_ranks(flash_attn_varlen_func3 is not None):
            flash_attn_varlen_func3 = None
        validate_args(args, methods, global_lengths, local_world_size, sm_configs)
        cases = make_cases(args, global_lengths)

        if local_rank == 0:
            sm_configs_s = ",".join(
                f"{sm_config.num_comp_sm}:{sm_config.num_comm_sm}" for sm_config in sm_configs
            )
            print(
                f"Config: world_size={local_world_size}, methods={methods}, "
                f"global_seqlens={args.global_seqlens}, qhead={args.qhead}, kvhead={args.kvhead}, "
                f"D={args.headdim}, mode={args.mode}, threshold={args.cp_threshold}, "
                f"sm_configs={sm_configs_s}, warmup={args.warmup_iters}, "
                f"iters={args.num_iters}, check={args.check}"
            )
            if args.check:
                print("Checks compare each method against its own rank-local layout reference.")
            print("Agg TFLOPS uses the same global workload FLOP count for all methods.")

        for sm_config in sm_configs:
            args.num_comp_sm = sm_config.num_comp_sm
            args.num_comm_sm = sm_config.num_comm_sm
            if local_rank == 0:
                print(
                    f"\nSM config: num_comp_sm={args.num_comp_sm}, num_comm_sm={args.num_comm_sm}",
                    flush=True,
                )
            for case in cases:
                if local_rank == 0:
                    print(f"\nRunning global hybrid benchmark causal={case.is_causal}", flush=True)
                results = run_case(case, methods, local_rank, local_world_size, args)
                if local_rank == 0:
                    print_results(case, results, methods)
                cuda_barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
