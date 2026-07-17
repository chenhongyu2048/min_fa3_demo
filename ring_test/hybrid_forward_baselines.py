"""Varlen all-CP baselines for the hierarchical forward benchmark."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from allgather_attention import _external_forward, _local_forward
from ring_common import ring_varlen_forward, zigzag_ring_varlen_forward
from zepplin import ZepplinPlan, zepplin_note


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
        return (
            "external FA3"
            if self.backend == "external_fa3"
            else "in-repo min_fa3 fallback"
        )

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


class VarlenAllGatherForward(_BlockBackend):
    """All-gather K/V and run one batched varlen attention per visible Q half."""

    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        local_lengths: list[int],
        is_causal: bool,
        backend: str,
    ) -> None:
        super().__init__(backend)
        self.process_group = process_group
        self.q = q
        self.k = k
        self.v = v
        self.local_lengths = local_lengths
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.is_causal = is_causal
        self.local_total = sum(local_lengths)
        self.global_lengths = [length * self.world_size for length in local_lengths]
        self.gathered_k = torch.empty(
            (self.world_size * self.local_total, k.size(1), k.size(2)),
            dtype=k.dtype,
            device=k.device,
        )
        self.gathered_v = torch.empty_like(self.gathered_k)
        self.ordered_k = torch.empty_like(self.gathered_k)
        self.ordered_v = torch.empty_like(self.gathered_v)

        global_order: list[int] = []
        q_front_indices: list[int] = []
        q_back_indices: list[int] = []
        k_front_indices: list[int] = []
        k_back_indices: list[int] = []
        local_offset = 0
        global_offset = 0
        half_lengths: list[int] = []
        front_k_lengths: list[int] = []
        back_k_lengths: list[int] = []
        for local_len in local_lengths:
            if is_causal:
                half_len = local_len // 2
                half_lengths.append(half_len)
                front_k_len = (self.rank + 1) * half_len
                back_k_len = (2 * self.world_size - self.rank) * half_len
                front_k_lengths.append(front_k_len)
                back_k_lengths.append(back_k_len)
                q_front_indices.extend(range(local_offset, local_offset + half_len))
                q_back_indices.extend(
                    range(local_offset + half_len, local_offset + local_len)
                )
                for source_rank in range(self.world_size):
                    source = source_rank * self.local_total + local_offset
                    global_order.extend(range(source, source + half_len))
                for source_rank in reversed(range(self.world_size)):
                    source = source_rank * self.local_total + local_offset + half_len
                    global_order.extend(range(source, source + half_len))
                k_front_indices.extend(range(global_offset, global_offset + front_k_len))
                k_back_indices.extend(range(global_offset, global_offset + back_k_len))
            else:
                for source_rank in range(self.world_size):
                    source = source_rank * self.local_total + local_offset
                    global_order.extend(range(source, source + local_len))
            local_offset += local_len
            global_offset += local_len * self.world_size

        self.global_order = torch.tensor(
            global_order, device=q.device, dtype=torch.int64
        )
        self.local_cu, self.local_cu_host = make_cu_seqlens(local_lengths, q.device)
        self.global_cu, self.global_cu_host = make_cu_seqlens(
            self.global_lengths, q.device
        )
        self.max_local = max(local_lengths)
        self.max_global = max(self.global_lengths)

        if is_causal:
            self.q_front_indices = torch.tensor(
                q_front_indices, device=q.device, dtype=torch.int64
            )
            self.q_back_indices = torch.tensor(
                q_back_indices, device=q.device, dtype=torch.int64
            )
            self.k_front_indices = torch.tensor(
                k_front_indices, device=q.device, dtype=torch.int64
            )
            self.k_back_indices = torch.tensor(
                k_back_indices, device=q.device, dtype=torch.int64
            )
            self.q_front = q.index_select(0, self.q_front_indices).contiguous()
            self.q_back = q.index_select(0, self.q_back_indices).contiguous()
            self.k_front = torch.empty(
                (len(k_front_indices), k.size(1), k.size(2)),
                dtype=k.dtype,
                device=k.device,
            )
            self.v_front = torch.empty_like(self.k_front)
            self.k_back = torch.empty(
                (len(k_back_indices), k.size(1), k.size(2)),
                dtype=k.dtype,
                device=k.device,
            )
            self.v_back = torch.empty_like(self.k_back)
            self.half_cu, self.half_cu_host = make_cu_seqlens(
                half_lengths, q.device
            )
            self.front_k_cu, self.front_k_cu_host = make_cu_seqlens(
                front_k_lengths, q.device
            )
            self.back_k_cu, self.back_k_cu_host = make_cu_seqlens(
                back_k_lengths, q.device
            )
            self.out = torch.empty_like(q)

    @property
    def note(self) -> str:
        return f"per-sequence all-gather; {self.backend_name}"

    def forward(self) -> torch.Tensor:
        dist.all_gather_into_tensor(
            self.gathered_k, self.k, group=self.process_group
        )
        dist.all_gather_into_tensor(
            self.gathered_v, self.v, group=self.process_group
        )
        torch.index_select(self.gathered_k, 0, self.global_order, out=self.ordered_k)
        torch.index_select(self.gathered_v, 0, self.global_order, out=self.ordered_v)
        if not self.is_causal:
            out, _ = self.forward_block(
                self.q,
                self.ordered_k,
                self.ordered_v,
                self.local_cu,
                self.global_cu,
                self.local_cu_host,
                self.global_cu_host,
                self.max_local,
                self.max_global,
                False,
            )
            return out

        torch.index_select(
            self.ordered_k, 0, self.k_front_indices, out=self.k_front
        )
        torch.index_select(
            self.ordered_v, 0, self.k_front_indices, out=self.v_front
        )
        torch.index_select(self.ordered_k, 0, self.k_back_indices, out=self.k_back)
        torch.index_select(self.ordered_v, 0, self.k_back_indices, out=self.v_back)
        out_front, _ = self.forward_block(
            self.q_front,
            self.k_front,
            self.v_front,
            self.half_cu,
            self.front_k_cu,
            self.half_cu_host,
            self.front_k_cu_host,
            max(self.local_lengths) // 2,
            max(
                length * (self.rank + 1) // 2 for length in self.local_lengths
            ),
            True,
        )
        out_back, _ = self.forward_block(
            self.q_back,
            self.k_back,
            self.v_back,
            self.half_cu,
            self.back_k_cu,
            self.half_cu_host,
            self.back_k_cu_host,
            max(self.local_lengths) // 2,
            max(
                length * (2 * self.world_size - self.rank) // 2
                for length in self.local_lengths
            ),
            True,
        )
        self.out.index_copy_(0, self.q_front_indices, out_front)
        self.out.index_copy_(0, self.q_back_indices, out_back)
        return self.out


def fa3_ring_forward(
    process_group: Optional[dist.ProcessGroup],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    local_lengths: list[int],
    is_causal: bool,
    backend: str,
) -> torch.Tensor:
    block_backend = _BlockBackend(backend)
    max_local_len = max(local_lengths)
    if is_causal:
        return zigzag_ring_varlen_forward(
            process_group,
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens_host,
            max_local_len,
            block_backend.forward_block,
        )
    return ring_varlen_forward(
        process_group,
        q,
        k,
        v,
        False,
        lambda q_, k_, v_, causal_: block_backend.forward_block(
            q_,
            k_,
            v_,
            cu_seqlens,
            cu_seqlens,
            cu_seqlens_host,
            cu_seqlens_host,
            max_local_len,
            max_local_len,
            causal_,
        ),
    )


class ZepplinForward(_BlockBackend):
    """Synchronize, run the all-rank Gworld ring, then rank-local G1 attention."""

    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        plan: ZepplinPlan,
        backend: str,
    ) -> None:
        super().__init__(backend)
        self.process_group = process_group
        self.q = q
        self.k = k
        self.v = v
        self.plan = plan
        self.rank = dist.get_rank(process_group)
        if dist.get_world_size(process_group) != plan.world_size:
            raise ValueError("zepplin plan world size does not match process group")

        self.short_lengths = plan.short_lengths_for_rank(self.rank)
        self.long_lengths = plan.long_local_lengths()
        self.short_total = sum(self.short_lengths)
        expected_total = self.short_total + sum(self.long_lengths)
        if q.size(0) != expected_total or k.size(0) != expected_total:
            raise ValueError("Q/K do not match the zepplin rank-local packed layout")
        if v.shape != k.shape:
            raise ValueError("zepplin K/V shapes must match")

        self.out = torch.empty_like(q)
        if self.short_lengths:
            self.short_cu, self.short_cu_host = make_cu_seqlens(
                self.short_lengths, q.device
            )
        if self.long_lengths:
            self.long_cu, self.long_cu_host = make_cu_seqlens(
                self.long_lengths, q.device
            )

    @property
    def note(self) -> str:
        return zepplin_note(self.plan, self.backend_name)

    def forward(self) -> torch.Tensor:
        dist.barrier(group=self.process_group)

        if self.long_lengths:
            long_out = fa3_ring_forward(
                self.process_group,
                self.q[self.short_total :],
                self.k[self.short_total :],
                self.v[self.short_total :],
                self.long_cu,
                self.long_cu_host,
                self.long_lengths,
                self.plan.is_causal,
                self.backend,
            )
            self.out[self.short_total :].copy_(long_out)

        if self.short_lengths:
            short_out, _ = self.forward_block(
                self.q[: self.short_total],
                self.k[: self.short_total],
                self.v[: self.short_total],
                self.short_cu,
                self.short_cu,
                self.short_cu_host,
                self.short_cu_host,
                max(self.short_lengths),
                max(self.short_lengths),
                self.plan.is_causal,
            )
            self.out[: self.short_total].copy_(short_out)
        return self.out


__all__ = ["VarlenAllGatherForward", "ZepplinForward", "fa3_ring_forward"]
