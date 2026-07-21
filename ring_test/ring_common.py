"""Shared helpers for the multi-rank varlen ring-attention test.

This module intentionally keeps the Python-side ring implementation separate
from the benchmark entry point. The functions here are small building blocks:
ring communication, online output/LSE merging, backend adapters, and reference
computation.
"""

from __future__ import annotations

import inspect
from functools import cache
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


# A block attention function consumes local Q plus the current rank-local K/V
# block and returns both block output and softmax LSE for online ring reduction.
BlockAttention = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], tuple[torch.Tensor, torch.Tensor]]
ZigzagBlockAttention = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        bool,
    ],
    tuple[torch.Tensor, torch.Tensor],
]


@cache
def _get_default_args(func):
    """Return a cached argument/default map for FlashAttention wrapper variants."""
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args


def get_default_args(func):
    """Handle both plain Python functions and CustomOpDef-style wrapped functions."""
    if inspect.isfunction(func):
        return _get_default_args(func)
    return _get_default_args(func._init_fn)


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Merge a new attention block into a running output using the stable online
    # softmax update. LSE has layout [total_q, heads, 1] in this helper.
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge one block attention result into the running ring output.

    FlashAttention-style kernels return block-local output and logsumexp (LSE).
    Ring attention combines blocks by keeping the running output in fp32 and
    merging each new block with the online softmax identity.
    """
    if out is None:
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        return out, lse
    return _update_out_and_lse(out, lse, block_out, block_lse)


def get_half_index(cu_seqlens: torch.Tensor, *, front: bool):
    """Return the flattened token index for each sequence's front/back half."""
    total_tokens = int(cu_seqlens[-1].item())
    if cu_seqlens.numel() == 2:
        midpoint = total_tokens // 2
        return slice(None, midpoint) if front else slice(midpoint, None)

    index = torch.zeros((total_tokens,), dtype=torch.bool, device=cu_seqlens.device)
    for batch_idx in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[batch_idx].item())
        end = int(cu_seqlens[batch_idx + 1].item())
        midpoint = start + (end - start) // 2
        if front:
            index[start:midpoint] = True
        else:
            index[midpoint:end] = True
    return index


def normalize_lse(block_lse: torch.Tensor, total_q: int, num_heads: int) -> torch.Tensor:
    """Normalize backend-specific LSE layouts to [heads, total_q].

    PyTorch, FA2, and FA3 wrappers may expose LSE in slightly different shapes,
    especially for batched varlen calls. The merge helper expects a single
    canonical layout.
    """
    lse = block_lse
    if lse.dim() == 3 and lse.size(-1) == 1:
        lse = lse.squeeze(-1)
    if lse.dim() == 2:
        if lse.shape == (num_heads, total_q):
            return lse.contiguous()
        if lse.shape == (total_q, num_heads):
            return lse.transpose(0, 1).contiguous()
    if lse.dim() == 3:
        batch_size = lse.size(0)
        if lse.shape[1] == num_heads and batch_size * lse.shape[2] == total_q:
            return lse.permute(1, 0, 2).reshape(num_heads, total_q).contiguous()
        if lse.shape[2] == num_heads and batch_size * lse.shape[1] == total_q:
            return lse.permute(2, 0, 1).reshape(num_heads, total_q).contiguous()
    raise RuntimeError(f"unsupported LSE shape {tuple(block_lse.shape)} for total_q={total_q}, heads={num_heads}")


def take_output_and_lse(result, total_q: int, num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract `(out, lse)` from a backend result tuple and normalize the LSE."""
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("expected attention backend to return at least (out, softmax_lse)")
    block_out = result[0]
    block_lse = normalize_lse(result[1], total_q, num_heads)
    return block_out, block_lse


class RingComm:
    """Small P2P ring wrapper for passing K/V blocks between ranks.

    Each rank sends to `(rank + 1) % world_size` and receives from
    `(rank - 1) % world_size`. After each step, the received K/V block becomes
    the next block consumed by local attention.
    """

    def __init__(self, process_group: Optional[dist.ProcessGroup]):
        self._process_group = process_group
        self._ops: list[dist.P2POp] = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Queue one async send and one async receive for a tensor."""
        if recv_tensor is None:
            recv_tensor = torch.empty_like(to_send)

        self._ops.append(dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group))
        self._ops.append(dist.P2POp(dist.irecv, recv_tensor, self.recv_rank, group=self._process_group))
        return recv_tensor

    def commit(self) -> None:
        """Submit all queued P2P ops as one batched NCCL operation."""
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self) -> None:
        """Wait for the current batched P2P operation and reset the queue."""
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Queue and submit the K and V transfers for one ring step."""
        next_k = self.send_recv(k, k_buffer)
        next_v = self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v


def ring_varlen_forward(
    process_group: Optional[dist.ProcessGroup],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    block_attention: BlockAttention,
    *,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run a full Python-side ring attention forward for one backend.

    `block_attention` is the per-step attention implementation. For noncausal
    mode every rank attends to every K/V block. For causal mode, rank `r`
    consumes only local/history blocks (`step <= r`); step 0 is the local block
    and uses the causal mask, while history blocks are noncausal.
    """
    comm = RingComm(process_group)
    out = None
    lse = None
    cur_k = k.contiguous()
    cur_v = v.contiguous()

    for step in range(comm.world_size):
        # Start moving the current K/V block before computing on it, matching
        # the usual ring attention overlap pattern.
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(cur_k, cur_v)
        else:
            next_k, next_v = None, None

        if not is_causal or step <= comm.rank:
            block_out, block_lse = block_attention(q, cur_k, cur_v, is_causal and step == 0)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            # The received block becomes the current block for the next step.
            comm.wait()
            cur_k, cur_v = next_k, next_v

    if out is None:
        raise RuntimeError("ring attention produced no output blocks")
    output = out.to(q.dtype)
    if return_lse:
        if lse is None:
            raise RuntimeError("ring attention produced no LSE")
        return output, lse.squeeze(-1).transpose(0, 1).contiguous()
    return output


def zigzag_ring_varlen_forward(
    process_group: Optional[dist.ProcessGroup],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    max_seqlen: int,
    block_attention: ZigzagBlockAttention,
    *,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run load-balanced causal zigzag ring attention for one backend.

    Each rank-local sequence is interpreted as [front half | back half]. Step 0
    computes the local full-sequence causal block. Earlier global KV ranks use
    full-Q/half-KV dense blocks, while later global KV ranks use half-Q/full-KV
    dense blocks and update only the back-half output rows.
    """
    if max_seqlen % 2 != 0:
        raise RuntimeError(f"zigzag causal ring requires an even max_seqlen, got {max_seqlen}")

    comm = RingComm(process_group)
    half_index0 = get_half_index(cu_seqlens, front=True)
    half_index1 = get_half_index(cu_seqlens, front=False)
    half_cu_seqlens = cu_seqlens // 2
    half_cu_seqlens_host = cu_seqlens_host // 2
    half_max_seqlen = max_seqlen // 2

    out = None
    lse = None
    q1 = q[half_index1].contiguous()
    cur_k = k.contiguous()
    cur_v = v.contiguous()

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(cur_k, cur_v)
        else:
            next_k, next_v = None, None

        if step == 0:
            block_out, block_lse = block_attention(
                q,
                cur_k,
                cur_v,
                cu_seqlens,
                cu_seqlens,
                cu_seqlens_host,
                cu_seqlens_host,
                max_seqlen,
                max_seqlen,
                True,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0 = cur_k[half_index0].contiguous()
            v0 = cur_v[half_index0].contiguous()
            block_out, block_lse = block_attention(
                q,
                k0,
                v0,
                cu_seqlens,
                half_cu_seqlens,
                cu_seqlens_host,
                half_cu_seqlens_host,
                max_seqlen,
                half_max_seqlen,
                False,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = block_attention(
                q1,
                cur_k,
                cur_v,
                half_cu_seqlens,
                cu_seqlens,
                half_cu_seqlens_host,
                cu_seqlens_host,
                half_max_seqlen,
                max_seqlen,
                False,
            )
            out1, lse1 = update_out_and_lse(out[half_index1], lse[half_index1], block_out, block_lse)
            out[half_index1] = out1
            lse[half_index1] = lse1

        if step + 1 != comm.world_size:
            comm.wait()
            cur_k, cur_v = next_k, next_v

    if out is None:
        raise RuntimeError("zigzag ring attention produced no output blocks")
    output = out.to(q.dtype)
    if return_lse:
        return output, lse.squeeze(-1).transpose(0, 1).contiguous()
    return output


def pytorch_varlen_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one PyTorch SDPA block and return output plus LSE.

    PyTorch's flash SDPA op expects B/H/S/D layout. The public varlen demo uses
    flattened [B * S, H, D], so this adapter reshapes around the call.
    """
    q_bshd = q.view(batch_size, seqlen_q, q.size(1), q.size(2))
    k_bshd = k.view(batch_size, seqlen_k, k.size(1), k.size(2))
    v_bshd = v.view(batch_size, seqlen_k, v.size(1), v.size(2))
    qt = q_bshd.transpose(1, 2).contiguous()
    kt = k_bshd.transpose(1, 2).contiguous()
    vt = v_bshd.transpose(1, 2).contiguous()
    if qt.size(1) != kt.size(1):
        repeat = qt.size(1) // kt.size(1)
        kt = kt.repeat_interleave(repeat, dim=1)
        vt = vt.repeat_interleave(repeat, dim=1)

    block_out, block_lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention.default(
        qt,
        kt,
        vt,
        0.0,
        is_causal,
        False,
    )
    block_out = block_out.transpose(1, 2).contiguous().view(q.size(0), q.size(1), q.size(2))
    block_lse = normalize_lse(block_lse, q.size(0), q.size(1))
    return block_out, block_lse


def flash_varlen_block_attention(
    method: str,
    flash_varlen_func,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one FA2/FA3 varlen block and return output plus normalized LSE.

    FlashAttention package versions expose different keyword names for asking
    the wrapper to return softmax LSE, so select the supported one from the
    callable signature.
    """
    kwargs = {"causal": is_causal}
    default_args = get_default_args(flash_varlen_func)
    if "return_attn_probs" in default_args:
        kwargs["return_attn_probs"] = True
    elif "return_softmax" in default_args:
        kwargs["return_softmax"] = True
    else:
        raise RuntimeError(f"{method} flash_attn_varlen_func does not expose a softmax LSE return flag")

    if method == "fa2":
        result = flash_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            **kwargs,
        )
    elif method == "fa3":
        result = flash_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            **kwargs,
        )
    else:
        raise ValueError(f"unsupported FlashAttention method '{method}'")
    return take_output_and_lse(result, q.size(0), q.size(1))


def min_fa3_varlen_block_attention(
    forward_varlen_func,
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
    """Run one min_fa3 varlen block and return output plus normalized LSE."""
    result = forward_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal,
        cu_seqlens_q_host=cu_seqlens_q_host,
        cu_seqlens_k_host=cu_seqlens_k_host,
        return_lse=True,
    )
    return take_output_and_lse(result, q.size(0), q.size(1))


def gather_rank_tensor(tensor: torch.Tensor, process_group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    """Gather same-shaped local tensors into a [world_size, ...] tensor."""
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    chunks = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(chunks, tensor, group=process_group)
    chunks[rank] = tensor
    return torch.stack(chunks, dim=0).contiguous()


def reference_ring_varlen(
    q: torch.Tensor,
    k_by_rank: torch.Tensor,
    v_by_rank: torch.Tensor,
    batch_size: int,
    local_seqlen: int,
    local_rank: int,
    is_causal: bool,
) -> torch.Tensor:
    """Compute a full-rank PyTorch reference for the local rank's Q block.

    This reference concatenates every rank's K/V block in rank order and applies
    the global ring causal mask when requested. It is deliberately simple and is
    used only for correctness checks, not benchmark timing.
    """
    world_size = k_by_rank.size(0)
    q_heads = q.size(1)
    kv_heads = k_by_rank.size(2)
    head_dim = q.size(2)
    qhead_per_kvhead = q_heads // kv_heads
    scale = head_dim ** -0.5
    outputs: list[torch.Tensor] = []

    q_b = q.view(batch_size, local_seqlen, q_heads, head_dim).float()
    k_rb = k_by_rank.view(world_size, batch_size, local_seqlen, kv_heads, head_dim)
    v_rb = v_by_rank.view(world_size, batch_size, local_seqlen, kv_heads, head_dim)
    key_pos = torch.arange(world_size * local_seqlen, device=q.device, dtype=torch.int64)
    query_pos = torch.arange(local_seqlen, device=q.device, dtype=torch.int64) + local_rank * local_seqlen

    for batch_idx in range(batch_size):
        q_i = q_b[batch_idx]
        k_i = k_rb[:, batch_idx].reshape(world_size * local_seqlen, kv_heads, head_dim).float()
        v_i = v_rb[:, batch_idx].reshape(world_size * local_seqlen, kv_heads, head_dim).float()
        if qhead_per_kvhead != 1:
            k_i = k_i.repeat_interleave(qhead_per_kvhead, dim=1)
            v_i = v_i.repeat_interleave(qhead_per_kvhead, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * scale
        if is_causal:
            # Query positions are local to this rank but offset into the global
            # sequence. Key positions span all rank-local K/V blocks.
            causal_mask = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
            scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probs, v_i).to(dtype=q.dtype))

    return torch.cat(outputs, dim=0).contiguous()



def reference_zigzag_ring_varlen(
    q: torch.Tensor,
    k_by_rank: torch.Tensor,
    v_by_rank: torch.Tensor,
    batch_size: int,
    local_seqlen: int,
    local_rank: int,
) -> torch.Tensor:
    """Compute the full-rank reference for causal zigzag [front | back] layout."""
    if local_seqlen % 2 != 0:
        raise RuntimeError(f"zigzag reference requires an even local seqlen, got {local_seqlen}")

    world_size = k_by_rank.size(0)
    q_heads = q.size(1)
    kv_heads = k_by_rank.size(2)
    head_dim = q.size(2)
    qhead_per_kvhead = q_heads // kv_heads
    half_len = local_seqlen // 2
    scale = head_dim ** -0.5
    outputs: list[torch.Tensor] = []

    q_b = q.view(batch_size, local_seqlen, q_heads, head_dim).float()
    k_rb = k_by_rank.view(world_size, batch_size, local_seqlen, kv_heads, head_dim)
    v_rb = v_by_rank.view(world_size, batch_size, local_seqlen, kv_heads, head_dim)
    half_arange = torch.arange(half_len, device=q.device, dtype=torch.int64)
    key_pos = torch.cat(
        [
            torch.cat(
                [
                    half_arange + rank_idx * half_len,
                    half_arange + (2 * world_size - 1 - rank_idx) * half_len,
                ],
                dim=0,
            )
            for rank_idx in range(world_size)
        ],
        dim=0,
    )
    query_pos = torch.cat(
        [
            half_arange + local_rank * half_len,
            half_arange + (2 * world_size - 1 - local_rank) * half_len,
        ],
        dim=0,
    )

    for batch_idx in range(batch_size):
        q_i = q_b[batch_idx]
        k_i = k_rb[:, batch_idx].reshape(world_size * local_seqlen, kv_heads, head_dim).float()
        v_i = v_rb[:, batch_idx].reshape(world_size * local_seqlen, kv_heads, head_dim).float()
        if qhead_per_kvhead != 1:
            k_i = k_i.repeat_interleave(qhead_per_kvhead, dim=1)
            v_i = v_i.repeat_interleave(qhead_per_kvhead, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * scale
        causal_mask = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probs, v_i).to(dtype=q.dtype))

    return torch.cat(outputs, dim=0).contiguous()


def raise_if_any_rank_failed(local_error: Optional[str], process_group: Optional[dist.ProcessGroup]) -> None:
    """Turn per-rank check failures into one synchronized exception path."""
    failed = torch.tensor([1 if local_error is not None else 0], device="cuda", dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM, group=process_group)
    if failed.item() == 0:
        return
    if local_error is not None:
        raise AssertionError(local_error)
    raise AssertionError("another rank failed this check")
