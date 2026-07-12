"""All-CP all-gather attention baseline for the varlen ring benchmarks.

The all-gather/FlashAttention/reduce-scatter structure is adapted from
ring-flash-attention's ``llama3_flash_attn_varlen.py``. The zigzag sequence
packing is local to this demo and matches its existing ring benchmarks.

The causal path follows the repository's zigzag local layout: every rank owns
``[front | back]`` halves from two non-adjacent positions in the global
sequence.  Standard causal FlashAttention can express those positions with two
bottom-right-aligned varlen calls after K/V have been gathered and reordered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

import min_fa3_op
from ring_common import get_default_args

try:
    from flash_attn_interface import (
        _flash_attn_varlen_backward as _fa3_varlen_backward,
        _flash_attn_varlen_forward as _fa3_varlen_forward,
    )
except ImportError:
    _fa3_varlen_forward = None
    _fa3_varlen_backward = None


EXTERNAL_FA3_AVAILABLE = _fa3_varlen_forward is not None and _fa3_varlen_backward is not None


def select_allgather_backend(process_group: Optional[dist.ProcessGroup]) -> str:
    """Select one backend consistently across every rank in the process group."""
    available = torch.tensor(
        [1 if EXTERNAL_FA3_AVAILABLE else 0],
        device=torch.device("cuda", torch.cuda.current_device()),
        dtype=torch.int32,
    )
    dist.all_reduce(available, op=dist.ReduceOp.MIN, group=process_group)
    return "external_fa3" if bool(available.item()) else "min_fa3"


def backend_note(backend: str) -> str:
    if backend == "external_fa3":
        return "all-gather; external FA3; zigzag causal"
    if backend == "min_fa3":
        return "all-gather; local min_fa3 fallback; zigzag causal"
    raise ValueError(f"unknown all-gather attention backend: {backend}")


def _external_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _fa3_varlen_forward is None:
        raise RuntimeError("external FA3 varlen forward is unavailable")
    params = get_default_args(_fa3_varlen_forward).copy()
    params.update(
        {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": max_q,
            "max_seqlen_k": max_k,
            "dropout_p": 0.0,
            "softmax_scale": q.shape[-1] ** -0.5,
            "causal": causal,
            "alibi_slopes": None,
            "return_softmax": False,
        }
    )
    result = _fa3_varlen_forward(**params)
    if len(result) == 8:
        return result[0], result[5]
    if len(result) == 4:
        return result[0], result[1]
    raise RuntimeError(f"unsupported external FA3 forward result length: {len(result)}")


def _external_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    causal: bool,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _fa3_varlen_backward is None:
        raise RuntimeError("external FA3 varlen backward is unavailable")
    params = get_default_args(_fa3_varlen_backward).copy()
    params.update(
        {
            "dout": dout,
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "softmax_lse": lse,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": max_q,
            "max_seqlen_k": max_k,
            "dropout_p": 0.0,
            "softmax_scale": q.shape[-1] ** -0.5,
            "causal": causal,
            "alibi_slopes": None,
            "deterministic": False,
        }
    )
    _fa3_varlen_backward(**params)
    return dq, dk, dv


def _local_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    cu_q_host: torch.Tensor,
    cu_k_host: torch.Tensor,
    max_q: int,
    max_k: int,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = min_fa3_op.forward_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        causal,
        cu_seqlens_q_host=cu_q_host,
        cu_seqlens_k_host=cu_k_host,
        return_lse=True,
    )
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("min_fa3 varlen forward did not return output and LSE")
    return result[0], result[1]


def _local_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    causal: bool,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return min_fa3_op.backward_varlen(
        dout,
        q,
        k,
        v,
        out,
        lse,
        cu_q,
        cu_k,
        max_q,
        max_k,
        causal,
        dq=dq,
        dk=dk,
        dv=dv,
    )


def _uniform_cu(batch_size: int, seqlen: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.arange(0, (batch_size + 1) * seqlen, seqlen, dtype=torch.int32)
    return host.to(device=device), host


@dataclass
class _ForwardState:
    out_front: Optional[torch.Tensor] = None
    out_back: Optional[torch.Tensor] = None
    lse_front: Optional[torch.Tensor] = None
    lse_back: Optional[torch.Tensor] = None


class AllGatherAttention:
    """Reusable all-gather attention runner with preallocated communication buffers."""

    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        local_seqlen: int,
        causal: bool,
        backend: str,
        *,
        enable_backward: bool = False,
    ) -> None:
        if backend not in ("external_fa3", "min_fa3"):
            raise ValueError(f"unsupported all-gather attention backend: {backend}")
        if causal and local_seqlen % 2 != 0:
            raise ValueError(f"zigzag causal all-gather requires an even local seqlen, got {local_seqlen}")
        expected_tokens = batch_size * local_seqlen
        if q.size(0) != expected_tokens or k.size(0) != expected_tokens or v.size(0) != expected_tokens:
            raise ValueError("Q/K/V token counts must equal batch_size * local_seqlen")

        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.q = q
        self.k = k
        self.v = v
        self.batch_size = batch_size
        self.local_seqlen = local_seqlen
        self.causal = causal
        self.backend = backend
        self.enable_backward = enable_backward
        self.half = local_seqlen // 2
        self.local_tokens = expected_tokens
        self.global_seqlen = self.world_size * local_seqlen
        self.state = _ForwardState()

        gathered_shape = (self.world_size * expected_tokens, k.size(1), k.size(2))
        ordered_shape = (batch_size * self.global_seqlen, k.size(1), k.size(2))
        self.gathered_k = torch.empty(gathered_shape, dtype=k.dtype, device=k.device)
        self.gathered_v = torch.empty_like(self.gathered_k)
        self.ordered_k = torch.empty(ordered_shape, dtype=k.dtype, device=k.device)
        self.ordered_v = torch.empty_like(self.ordered_k)
        self.out = torch.empty_like(q)
        if enable_backward:
            if not causal:
                self.full_dk = torch.empty_like(self.ordered_k)
                self.full_dv = torch.empty_like(self.ordered_v)
            self.rank_major_dk = torch.empty(gathered_shape, dtype=torch.float32, device=k.device)
            self.rank_major_dv = torch.empty_like(self.rank_major_dk)
            self.local_dk_fp32 = torch.empty_like(k, dtype=torch.float32)
            self.local_dv_fp32 = torch.empty_like(v, dtype=torch.float32)
            self.local_dk = torch.empty_like(k)
            self.local_dv = torch.empty_like(v)
            self.dq = torch.empty_like(q)

        self.full_cu_q, self.full_cu_q_host = _uniform_cu(batch_size, local_seqlen, q.device)
        self.global_cu_k, self.global_cu_k_host = _uniform_cu(batch_size, self.global_seqlen, q.device)

        if causal:
            front_k_len = (self.rank + 1) * self.half
            back_k_len = (2 * self.world_size - self.rank) * self.half
            q_half_shape = (batch_size * self.half, q.size(1), q.size(2))
            front_k_shape = (batch_size * front_k_len, k.size(1), k.size(2))
            back_k_shape = (batch_size * back_k_len, k.size(1), k.size(2))
            self.q_front = torch.empty(q_half_shape, dtype=q.dtype, device=q.device)
            self.q_back = torch.empty_like(self.q_front)
            self.k_front = torch.empty(front_k_shape, dtype=k.dtype, device=k.device)
            self.v_front = torch.empty_like(self.k_front)
            self.k_back = torch.empty(back_k_shape, dtype=k.dtype, device=k.device)
            self.v_back = torch.empty_like(self.k_back)
            if enable_backward:
                self.dout_front = torch.empty_like(self.q_front)
                self.dout_back = torch.empty_like(self.q_back)
                self.dq_front = torch.empty_like(self.q_front)
                self.dq_back = torch.empty_like(self.q_back)
                self.dk_front = torch.empty_like(self.k_front)
                self.dv_front = torch.empty_like(self.v_front)
                self.dk_back = torch.empty_like(self.k_back)
                self.dv_back = torch.empty_like(self.v_back)
                self.ordered_dk = torch.empty_like(self.ordered_k, dtype=torch.float32)
                self.ordered_dv = torch.empty_like(self.ordered_v, dtype=torch.float32)
            self.half_cu_q, self.half_cu_q_host = _uniform_cu(batch_size, self.half, q.device)
            self.front_cu_k, self.front_cu_k_host = _uniform_cu(batch_size, front_k_len, q.device)
            self.back_cu_k, self.back_cu_k_host = _uniform_cu(batch_size, back_k_len, q.device)

    @property
    def note(self) -> str:
        note = backend_note(self.backend)
        return note if self.causal else note.replace("; zigzag causal", "; noncausal")

    def _gather_and_order_kv(self) -> None:
        dist.all_gather_into_tensor(self.gathered_k, self.k, group=self.process_group)
        dist.all_gather_into_tensor(self.gathered_v, self.v, group=self.process_group)
        gathered_k = self.gathered_k.view(
            self.world_size, self.batch_size, self.local_seqlen, self.k.size(1), self.k.size(2)
        )
        gathered_v = self.gathered_v.view_as(gathered_k)
        ordered_k = self.ordered_k.view(
            self.batch_size, self.global_seqlen, self.k.size(1), self.k.size(2)
        )
        ordered_v = self.ordered_v.view_as(ordered_k)

        for batch_idx in range(self.batch_size):
            if not self.causal:
                for rank_idx in range(self.world_size):
                    dst = slice(rank_idx * self.local_seqlen, (rank_idx + 1) * self.local_seqlen)
                    ordered_k[batch_idx, dst].copy_(gathered_k[rank_idx, batch_idx])
                    ordered_v[batch_idx, dst].copy_(gathered_v[rank_idx, batch_idx])
                continue

            for rank_idx in range(self.world_size):
                front_dst = slice(rank_idx * self.half, (rank_idx + 1) * self.half)
                back_block = 2 * self.world_size - 1 - rank_idx
                back_dst = slice(back_block * self.half, (back_block + 1) * self.half)
                ordered_k[batch_idx, front_dst].copy_(gathered_k[rank_idx, batch_idx, : self.half])
                ordered_v[batch_idx, front_dst].copy_(gathered_v[rank_idx, batch_idx, : self.half])
                ordered_k[batch_idx, back_dst].copy_(gathered_k[rank_idx, batch_idx, self.half :])
                ordered_v[batch_idx, back_dst].copy_(gathered_v[rank_idx, batch_idx, self.half :])

    def _run_forward_block(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        cu_q_host: torch.Tensor,
        cu_k_host: torch.Tensor,
        max_q: int,
        max_k: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "external_fa3":
            return _external_forward(q, k, v, cu_q, cu_k, max_q, max_k, causal)
        return _local_forward(
            q, k, v, cu_q, cu_k, cu_q_host, cu_k_host, max_q, max_k, causal
        )

    def forward(self) -> torch.Tensor:
        self._gather_and_order_kv()
        if not self.causal:
            out, lse = self._run_forward_block(
                self.q,
                self.ordered_k,
                self.ordered_v,
                self.full_cu_q,
                self.global_cu_k,
                self.full_cu_q_host,
                self.global_cu_k_host,
                self.local_seqlen,
                self.global_seqlen,
                False,
            )
            self.out.copy_(out)
            self.state = _ForwardState(out_front=out, lse_front=lse)
            return self.out

        ordered_k = self.ordered_k.view(
            self.batch_size, self.global_seqlen, self.k.size(1), self.k.size(2)
        )
        ordered_v = self.ordered_v.view_as(ordered_k)
        q = self.q.view(self.batch_size, self.local_seqlen, self.q.size(1), self.q.size(2))
        k_front = self.k_front.view(self.batch_size, -1, self.k.size(1), self.k.size(2))
        v_front = self.v_front.view_as(k_front)
        k_back = self.k_back.view(self.batch_size, -1, self.k.size(1), self.k.size(2))
        v_back = self.v_back.view_as(k_back)
        q_front = self.q_front.view(self.batch_size, self.half, self.q.size(1), self.q.size(2))
        q_back = self.q_back.view_as(q_front)
        for batch_idx in range(self.batch_size):
            q_front[batch_idx].copy_(q[batch_idx, : self.half])
            q_back[batch_idx].copy_(q[batch_idx, self.half :])
            k_front[batch_idx].copy_(ordered_k[batch_idx, : k_front.size(1)])
            v_front[batch_idx].copy_(ordered_v[batch_idx, : v_front.size(1)])
            k_back[batch_idx].copy_(ordered_k[batch_idx, : k_back.size(1)])
            v_back[batch_idx].copy_(ordered_v[batch_idx, : v_back.size(1)])

        out_front, lse_front = self._run_forward_block(
            self.q_front,
            self.k_front,
            self.v_front,
            self.half_cu_q,
            self.front_cu_k,
            self.half_cu_q_host,
            self.front_cu_k_host,
            self.half,
            self.k_front.size(0) // self.batch_size,
            True,
        )
        out_back, lse_back = self._run_forward_block(
            self.q_back,
            self.k_back,
            self.v_back,
            self.half_cu_q,
            self.back_cu_k,
            self.half_cu_q_host,
            self.back_cu_k_host,
            self.half,
            self.k_back.size(0) // self.batch_size,
            True,
        )
        out = self.out.view(self.batch_size, self.local_seqlen, self.q.size(1), self.q.size(2))
        packed_front = out_front.view(self.batch_size, self.half, self.q.size(1), self.q.size(2))
        packed_back = out_back.view_as(packed_front)
        for batch_idx in range(self.batch_size):
            out[batch_idx, : self.half].copy_(packed_front[batch_idx])
            out[batch_idx, self.half :].copy_(packed_back[batch_idx])
        self.state = _ForwardState(out_front, out_back, lse_front, lse_back)
        return self.out

    def _run_backward_block(
        self,
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        causal: bool,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.backend == "external_fa3":
            return _external_backward(
                dout, q, k, v, out, lse, cu_q, cu_k, max_q, max_k, causal, dq, dk, dv
            )
        return _local_backward(
            dout, q, k, v, out, lse, cu_q, cu_k, max_q, max_k, causal, dq, dk, dv
        )

    def _ordered_grads_to_rank_major(self, ordered: torch.Tensor, rank_major: torch.Tensor) -> None:
        ordered_view = ordered.view(
            self.batch_size, self.global_seqlen, self.k.size(1), self.k.size(2)
        )
        rank_view = rank_major.view(
            self.world_size, self.batch_size, self.local_seqlen, self.k.size(1), self.k.size(2)
        )
        for batch_idx in range(self.batch_size):
            for rank_idx in range(self.world_size):
                if not self.causal:
                    src = slice(rank_idx * self.local_seqlen, (rank_idx + 1) * self.local_seqlen)
                    rank_view[rank_idx, batch_idx].copy_(ordered_view[batch_idx, src])
                    continue
                front_src = slice(rank_idx * self.half, (rank_idx + 1) * self.half)
                back_block = 2 * self.world_size - 1 - rank_idx
                back_src = slice(back_block * self.half, (back_block + 1) * self.half)
                rank_view[rank_idx, batch_idx, : self.half].copy_(ordered_view[batch_idx, front_src])
                rank_view[rank_idx, batch_idx, self.half :].copy_(ordered_view[batch_idx, back_src])

    def backward(self, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enable_backward:
            raise RuntimeError("all-gather attention runner was created without backward workspaces")
        if self.state.out_front is None or self.state.lse_front is None:
            raise RuntimeError("all-gather attention backward requires a prepared forward")

        if not self.causal:
            dq, dk, dv = self._run_backward_block(
                dout,
                self.q,
                self.ordered_k,
                self.ordered_v,
                self.state.out_front,
                self.state.lse_front,
                self.full_cu_q,
                self.global_cu_k,
                self.local_seqlen,
                self.global_seqlen,
                False,
                self.dq,
                self.full_dk,
                self.full_dv,
            )
            self._ordered_grads_to_rank_major(dk, self.rank_major_dk)
            self._ordered_grads_to_rank_major(dv, self.rank_major_dv)
        else:
            if self.state.out_back is None or self.state.lse_back is None:
                raise RuntimeError("zigzag all-gather backward is missing back-half forward state")
            dout_view = dout.view(
                self.batch_size, self.local_seqlen, self.q.size(1), self.q.size(2)
            )
            packed_dout_front = self.dout_front.view(
                self.batch_size, self.half, self.q.size(1), self.q.size(2)
            )
            packed_dout_back = self.dout_back.view_as(packed_dout_front)
            for batch_idx in range(self.batch_size):
                packed_dout_front[batch_idx].copy_(dout_view[batch_idx, : self.half])
                packed_dout_back[batch_idx].copy_(dout_view[batch_idx, self.half :])

            dq_front, dk_front, dv_front = self._run_backward_block(
                self.dout_front,
                self.q_front,
                self.k_front,
                self.v_front,
                self.state.out_front,
                self.state.lse_front,
                self.half_cu_q,
                self.front_cu_k,
                self.half,
                self.k_front.size(0) // self.batch_size,
                True,
                self.dq_front,
                self.dk_front,
                self.dv_front,
            )
            dq_back, dk_back, dv_back = self._run_backward_block(
                self.dout_back,
                self.q_back,
                self.k_back,
                self.v_back,
                self.state.out_back,
                self.state.lse_back,
                self.half_cu_q,
                self.back_cu_k,
                self.half,
                self.k_back.size(0) // self.batch_size,
                True,
                self.dq_back,
                self.dk_back,
                self.dv_back,
            )
            dq = self.dq
            dq_view = dq.view(self.batch_size, self.local_seqlen, self.q.size(1), self.q.size(2))
            dq_front_view = dq_front.view(self.batch_size, self.half, self.q.size(1), self.q.size(2))
            dq_back_view = dq_back.view_as(dq_front_view)
            for batch_idx in range(self.batch_size):
                dq_view[batch_idx, : self.half].copy_(dq_front_view[batch_idx])
                dq_view[batch_idx, self.half :].copy_(dq_back_view[batch_idx])

            self.ordered_dk.zero_()
            self.ordered_dv.zero_()
            ordered_dk = self.ordered_dk.view(
                self.batch_size, self.global_seqlen, self.k.size(1), self.k.size(2)
            )
            ordered_dv = self.ordered_dv.view_as(ordered_dk)
            dk_front = dk_front.view(self.batch_size, -1, self.k.size(1), self.k.size(2))
            dv_front = dv_front.view_as(dk_front)
            dk_back = dk_back.view(self.batch_size, -1, self.k.size(1), self.k.size(2))
            dv_back = dv_back.view_as(dk_back)
            for batch_idx in range(self.batch_size):
                ordered_dk[batch_idx, : dk_front.size(1)].add_(dk_front[batch_idx])
                ordered_dv[batch_idx, : dv_front.size(1)].add_(dv_front[batch_idx])
                ordered_dk[batch_idx, : dk_back.size(1)].add_(dk_back[batch_idx])
                ordered_dv[batch_idx, : dv_back.size(1)].add_(dv_back[batch_idx])
            self._ordered_grads_to_rank_major(self.ordered_dk, self.rank_major_dk)
            self._ordered_grads_to_rank_major(self.ordered_dv, self.rank_major_dv)

        dist.reduce_scatter_tensor(
            self.local_dk_fp32, self.rank_major_dk, group=self.process_group
        )
        dist.reduce_scatter_tensor(
            self.local_dv_fp32, self.rank_major_dv, group=self.process_group
        )
        self.local_dk.copy_(self.local_dk_fp32)
        self.local_dv.copy_(self.local_dv_fp32)
        return dq, self.local_dk, self.local_dv


__all__ = [
    "AllGatherAttention",
    "EXTERNAL_FA3_AVAILABLE",
    "backend_note",
    "select_allgather_backend",
]
