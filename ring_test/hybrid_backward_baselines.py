"""Varlen all-CP baselines for the hierarchical backward benchmark."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from allgather_attention import (
    _external_backward,
    _external_forward,
    _local_backward,
    _local_forward,
)
from ring_common import RingComm, get_half_index, zigzag_ring_varlen_forward


def make_cu_seqlens(
    lengths: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.zeros((len(lengths) + 1,), dtype=torch.int32)
    for idx, length in enumerate(lengths):
        host[idx + 1] = host[idx] + length
    return host.to(device=device), host


class _BlockBackend:
    def __init__(self, backend: str) -> None:
        if backend not in ("external_fa3", "min_fa3"):
            raise ValueError(f"unsupported block backend: {backend}")
        self.backend = backend

    @property
    def backend_name(self) -> str:
        return "external FA3" if self.backend == "external_fa3" else "local min_fa3 fallback"

    def forward_block(
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
            q,
            k,
            v,
            cu_q,
            cu_k,
            cu_q_host,
            cu_k_host,
            max_q,
            max_k,
            causal,
        )

    def backward_block(
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
                dq,
                dk,
                dv,
            )
        return _local_backward(
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
            dq,
            dk,
            dv,
        )


class VarlenAllGatherBackward(_BlockBackend):
    """Per-sequence zigzag all-gather with batched varlen FA backward."""

    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        local_lengths: list[int],
        backend: str,
    ) -> None:
        super().__init__(backend)
        if not local_lengths or any(length <= 0 or length % 2 for length in local_lengths):
            raise ValueError("causal all-gather requires positive even local lengths")
        if q.size(0) != sum(local_lengths) or k.size(0) != q.size(0) or v.shape != k.shape:
            raise ValueError("Q/K/V do not match the packed local lengths")

        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.q = q
        self.k = k
        self.v = v
        self.local_total = sum(local_lengths)
        self.global_total = self.world_size * self.local_total

        global_order: list[int] = []
        q_front_indices: list[int] = []
        q_back_indices: list[int] = []
        k_front_indices: list[int] = []
        k_back_indices: list[int] = []
        half_lengths: list[int] = []
        front_k_lengths: list[int] = []
        back_k_lengths: list[int] = []
        local_offset = 0
        global_offset = 0
        for local_len in local_lengths:
            half_len = local_len // 2
            front_k_len = (self.rank + 1) * half_len
            back_k_len = (2 * self.world_size - self.rank) * half_len
            half_lengths.append(half_len)
            front_k_lengths.append(front_k_len)
            back_k_lengths.append(back_k_len)
            q_front_indices.extend(range(local_offset, local_offset + half_len))
            q_back_indices.extend(range(local_offset + half_len, local_offset + local_len))
            for source_rank in range(self.world_size):
                source = source_rank * self.local_total + local_offset
                global_order.extend(range(source, source + half_len))
            for source_rank in reversed(range(self.world_size)):
                source = source_rank * self.local_total + local_offset + half_len
                global_order.extend(range(source, source + half_len))
            k_front_indices.extend(range(global_offset, global_offset + front_k_len))
            k_back_indices.extend(range(global_offset, global_offset + back_k_len))
            local_offset += local_len
            global_offset += local_len * self.world_size

        device = q.device
        self.global_order = torch.tensor(global_order, device=device, dtype=torch.int64)
        self.q_front_indices = torch.tensor(q_front_indices, device=device, dtype=torch.int64)
        self.q_back_indices = torch.tensor(q_back_indices, device=device, dtype=torch.int64)
        self.k_front_indices = torch.tensor(k_front_indices, device=device, dtype=torch.int64)
        self.k_back_indices = torch.tensor(k_back_indices, device=device, dtype=torch.int64)
        self.half_cu, self.half_cu_host = make_cu_seqlens(half_lengths, device)
        self.front_k_cu, self.front_k_cu_host = make_cu_seqlens(front_k_lengths, device)
        self.back_k_cu, self.back_k_cu_host = make_cu_seqlens(back_k_lengths, device)
        self.max_half = max(half_lengths)
        self.max_front_k = max(front_k_lengths)
        self.max_back_k = max(back_k_lengths)

        gathered_shape = (self.global_total, k.size(1), k.size(2))
        self.gathered_k = torch.empty(gathered_shape, dtype=k.dtype, device=device)
        self.gathered_v = torch.empty_like(self.gathered_k)
        self.ordered_k = torch.empty_like(self.gathered_k)
        self.ordered_v = torch.empty_like(self.gathered_v)
        self.q_front = q.index_select(0, self.q_front_indices).contiguous()
        self.q_back = q.index_select(0, self.q_back_indices).contiguous()
        self.k_front = torch.empty(
            (len(k_front_indices), k.size(1), k.size(2)), dtype=k.dtype, device=device
        )
        self.v_front = torch.empty_like(self.k_front)
        self.k_back = torch.empty(
            (len(k_back_indices), k.size(1), k.size(2)), dtype=k.dtype, device=device
        )
        self.v_back = torch.empty_like(self.k_back)
        self.out = torch.empty_like(q)
        self.out_front: torch.Tensor | None = None
        self.out_back: torch.Tensor | None = None
        self.lse_front: torch.Tensor | None = None
        self.lse_back: torch.Tensor | None = None

        self.dout_front = torch.empty_like(self.q_front)
        self.dout_back = torch.empty_like(self.q_back)
        self.dq_front = torch.empty_like(self.q_front)
        self.dq_back = torch.empty_like(self.q_back)
        self.dk_front = torch.empty_like(self.k_front)
        self.dv_front = torch.empty_like(self.v_front)
        self.dk_back = torch.empty_like(self.k_back)
        self.dv_back = torch.empty_like(self.v_back)
        self.dq = torch.empty_like(q)
        self.ordered_dk = torch.empty_like(self.ordered_k, dtype=torch.float32)
        self.ordered_dv = torch.empty_like(self.ordered_v, dtype=torch.float32)
        self.rank_major_dk = torch.empty_like(self.gathered_k, dtype=torch.float32)
        self.rank_major_dv = torch.empty_like(self.gathered_v, dtype=torch.float32)
        self.local_dk_fp32 = torch.empty_like(k, dtype=torch.float32)
        self.local_dv_fp32 = torch.empty_like(v, dtype=torch.float32)
        self.local_dk = torch.empty_like(k)
        self.local_dv = torch.empty_like(v)

    @property
    def note(self) -> str:
        return f"per-sequence all-gather; {self.backend_name}; zigzag causal"

    def forward(self) -> torch.Tensor:
        dist.all_gather_into_tensor(self.gathered_k, self.k, group=self.process_group)
        dist.all_gather_into_tensor(self.gathered_v, self.v, group=self.process_group)
        torch.index_select(self.gathered_k, 0, self.global_order, out=self.ordered_k)
        torch.index_select(self.gathered_v, 0, self.global_order, out=self.ordered_v)
        torch.index_select(self.ordered_k, 0, self.k_front_indices, out=self.k_front)
        torch.index_select(self.ordered_v, 0, self.k_front_indices, out=self.v_front)
        torch.index_select(self.ordered_k, 0, self.k_back_indices, out=self.k_back)
        torch.index_select(self.ordered_v, 0, self.k_back_indices, out=self.v_back)

        self.out_front, self.lse_front = self.forward_block(
            self.q_front,
            self.k_front,
            self.v_front,
            self.half_cu,
            self.front_k_cu,
            self.half_cu_host,
            self.front_k_cu_host,
            self.max_half,
            self.max_front_k,
            True,
        )
        self.out_back, self.lse_back = self.forward_block(
            self.q_back,
            self.k_back,
            self.v_back,
            self.half_cu,
            self.back_k_cu,
            self.half_cu_host,
            self.back_k_cu_host,
            self.max_half,
            self.max_back_k,
            True,
        )
        self.out.index_copy_(0, self.q_front_indices, self.out_front)
        self.out.index_copy_(0, self.q_back_indices, self.out_back)
        return self.out

    def backward(self, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.out_front is None or self.out_back is None:
            raise RuntimeError("all-gather backward requires a prepared forward")
        if self.lse_front is None or self.lse_back is None:
            raise RuntimeError("all-gather backward is missing forward LSE")
        torch.index_select(dout, 0, self.q_front_indices, out=self.dout_front)
        torch.index_select(dout, 0, self.q_back_indices, out=self.dout_back)
        dq_front, dk_front, dv_front = self.backward_block(
            self.dout_front,
            self.q_front,
            self.k_front,
            self.v_front,
            self.out_front,
            self.lse_front,
            self.half_cu,
            self.front_k_cu,
            self.max_half,
            self.max_front_k,
            True,
            self.dq_front,
            self.dk_front,
            self.dv_front,
        )
        dq_back, dk_back, dv_back = self.backward_block(
            self.dout_back,
            self.q_back,
            self.k_back,
            self.v_back,
            self.out_back,
            self.lse_back,
            self.half_cu,
            self.back_k_cu,
            self.max_half,
            self.max_back_k,
            True,
            self.dq_back,
            self.dk_back,
            self.dv_back,
        )
        self.dq.index_copy_(0, self.q_front_indices, dq_front)
        self.dq.index_copy_(0, self.q_back_indices, dq_back)
        self.ordered_dk.zero_()
        self.ordered_dv.zero_()
        self.ordered_dk.index_add_(0, self.k_front_indices, dk_front.float())
        self.ordered_dv.index_add_(0, self.k_front_indices, dv_front.float())
        self.ordered_dk.index_add_(0, self.k_back_indices, dk_back.float())
        self.ordered_dv.index_add_(0, self.k_back_indices, dv_back.float())
        self.rank_major_dk.index_copy_(0, self.global_order, self.ordered_dk)
        self.rank_major_dv.index_copy_(0, self.global_order, self.ordered_dv)
        dist.reduce_scatter_tensor(
            self.local_dk_fp32, self.rank_major_dk, group=self.process_group
        )
        dist.reduce_scatter_tensor(
            self.local_dv_fp32, self.rank_major_dv, group=self.process_group
        )
        self.local_dk.copy_(self.local_dk_fp32)
        self.local_dv.copy_(self.local_dv_fp32)
        return self.dq, self.local_dk, self.local_dv


class VarlenFa3RingBackward(_BlockBackend):
    """Complete varlen zigzag NCCL ring using FA3 block backward."""

    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dout: torch.Tensor,
        local_lengths: list[int],
        backend: str,
    ) -> None:
        super().__init__(backend)
        if not local_lengths or any(length <= 0 or length % 2 for length in local_lengths):
            raise ValueError("causal ring requires positive even local lengths")
        self.process_group = process_group
        self.q = q
        self.k = k
        self.v = v
        self.dout = dout
        self.max_local = max(local_lengths)
        self.cu, self.cu_host = make_cu_seqlens(local_lengths, q.device)
        self.half_cu = self.cu // 2
        self.half_max = self.max_local // 2
        self.front_index = get_half_index(self.cu, front=True)
        self.back_index = get_half_index(self.cu, front=False)
        self.q_back = q[self.back_index].contiguous()
        self.dout_back = dout[self.back_index].contiguous()
        self.k_ring = [torch.empty_like(k), torch.empty_like(k)]
        self.v_ring = [torch.empty_like(v), torch.empty_like(v)]
        self.dk_ring = [torch.empty_like(k, dtype=torch.float32) for _ in range(2)]
        self.dv_ring = [torch.empty_like(v, dtype=torch.float32) for _ in range(2)]
        self.dq_scratch = torch.empty_like(q)
        self.dk_scratch = torch.empty_like(k)
        self.dv_scratch = torch.empty_like(v)
        self.dq_accum = torch.empty_like(q, dtype=torch.float32)
        self.out: torch.Tensor | None = None
        self.lse: torch.Tensor | None = None
        self.out_back: torch.Tensor | None = None
        self.lse_back: torch.Tensor | None = None

    @property
    def note(self) -> str:
        return f"NCCL zigzag ring; {self.backend_name} block backward"

    def forward(self) -> torch.Tensor:
        result = zigzag_ring_varlen_forward(
            self.process_group,
            self.q,
            self.k,
            self.v,
            self.cu,
            self.cu_host,
            self.max_local,
            self.forward_block,
            return_lse=True,
        )
        if not isinstance(result, tuple):
            raise RuntimeError("zigzag forward did not return output and LSE")
        self.out, self.lse = result
        self.out_back = self.out[self.back_index].contiguous()
        self.lse_back = self.lse[:, self.back_index].contiguous()
        return self.out

    def _block_backward(
        self,
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        *,
        q_is_half: bool,
        k_is_half: bool,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.backward_block(
            dout,
            q,
            k,
            v,
            out,
            lse,
            self.half_cu if q_is_half else self.cu,
            self.half_cu if k_is_half else self.cu,
            self.half_max if q_is_half else self.max_local,
            self.half_max if k_is_half else self.max_local,
            causal,
            self.dq_scratch[: q.size(0)],
            self.dk_scratch[: k.size(0)],
            self.dv_scratch[: v.size(0)],
        )

    def backward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.out is None or self.lse is None:
            raise RuntimeError("FA3 ring backward requires a prepared forward")
        if self.out_back is None or self.lse_back is None:
            raise RuntimeError("FA3 ring backward is missing back-half state")
        kv_comm = RingComm(self.process_group)
        dkv_comm = RingComm(self.process_group)
        cur_k, cur_v = self.k, self.v
        cur_dk, cur_dv = self.dk_ring[0], self.dv_ring[0]
        next_dk = next_dv = None

        for step in range(kv_comm.world_size):
            if step + 1 < kv_comm.world_size:
                buffer_idx = step % 2
                next_k, next_v = kv_comm.send_recv_kv(
                    cur_k,
                    cur_v,
                    self.k_ring[buffer_idx],
                    self.v_ring[buffer_idx],
                )
            else:
                next_k = next_v = None

            if step == 0:
                dq_block, dk_block, dv_block = self._block_backward(
                    self.dout,
                    self.q,
                    cur_k,
                    cur_v,
                    self.out,
                    self.lse,
                    q_is_half=False,
                    k_is_half=False,
                    causal=True,
                )
                self.dq_accum.copy_(dq_block)
                cur_dk.copy_(dk_block)
                cur_dv.copy_(dv_block)
            else:
                if step <= kv_comm.rank:
                    k_front = cur_k[self.front_index].contiguous()
                    v_front = cur_v[self.front_index].contiguous()
                    dq_block, dk_block, dv_block = self._block_backward(
                        self.dout,
                        self.q,
                        k_front,
                        v_front,
                        self.out,
                        self.lse,
                        q_is_half=False,
                        k_is_half=True,
                        causal=False,
                    )
                    self.dq_accum += dq_block
                else:
                    dq_block, dk_block, dv_block = self._block_backward(
                        self.dout_back,
                        self.q_back,
                        cur_k,
                        cur_v,
                        self.out_back,
                        self.lse_back,
                        q_is_half=True,
                        k_is_half=False,
                        causal=False,
                    )
                    self.dq_accum[self.back_index] += dq_block

                dkv_comm.wait()
                if next_dk is None or next_dv is None:
                    raise RuntimeError("dKV ring did not produce a receive buffer")
                cur_dk, cur_dv = next_dk, next_dv
                if step <= kv_comm.rank:
                    cur_dk[self.front_index] += dk_block
                    cur_dv[self.front_index] += dv_block
                else:
                    cur_dk += dk_block
                    cur_dv += dv_block

            if step + 1 < kv_comm.world_size:
                kv_comm.wait()
                if next_k is None or next_v is None:
                    raise RuntimeError("KV ring did not produce a receive buffer")
                cur_k, cur_v = next_k, next_v

            recv_idx = (step + 1) % 2
            next_dk, next_dv = dkv_comm.send_recv_kv(
                cur_dk,
                cur_dv,
                self.dk_ring[recv_idx],
                self.dv_ring[recv_idx],
            )

        dkv_comm.wait()
        if next_dk is None or next_dv is None:
            raise RuntimeError("dKV ring did not return owner gradients")
        return (
            self.dq_accum.to(self.q.dtype),
            next_dk.to(self.k.dtype),
            next_dv.to(self.v.dtype),
        )


__all__ = ["VarlenAllGatherBackward", "VarlenFa3RingBackward"]
