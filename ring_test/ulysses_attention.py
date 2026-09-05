"""TND Ulysses and USP baselines for the local CP benchmarks.

The head/sequence all-to-all layout follows the copied Transformer Engine and
MagiAttention Ulysses baselines under ``third_party/``.  Attention and ring
execution reuse the existing local benchmark adapters so external FA3 remains
preferred with the in-repository minimal FA3 as its fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist

try:
    from .utils import make_cu_seqlens, sequence_shards_to_global_order
except ImportError:  # Direct ``python ring_test/...`` entry points.
    from utils import make_cu_seqlens, sequence_shards_to_global_order


def _make_block_backend(backend: str) -> Any:
    try:
        from .hybrid_backward_baselines import _BlockBackend
    except ImportError:
        from hybrid_backward_baselines import _BlockBackend
    return _BlockBackend(backend)


@dataclass(frozen=True)
class HeadShardPlan:
    original_heads: int
    effective_heads: int
    degree: int
    heads_per_rank: int
    replica_factor: int
    owner_head_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class USPTopology:
    world_size: int
    ulysses_degree: int
    ring_degree: int
    a2a_groups: tuple[tuple[int, ...], ...]
    ring_groups: tuple[tuple[int, ...], ...]


def make_head_shard_plan(
    heads: int,
    degree: int,
    *,
    allow_replication: bool,
) -> HeadShardPlan:
    """Resolve contiguous head owners, optionally replicating small KVH."""
    if heads <= 0 or degree <= 0:
        raise ValueError("heads and all-to-all degree must be positive")
    if heads >= degree:
        if heads % degree:
            raise ValueError(f"heads={heads} must be divisible by degree={degree}")
        effective_heads = heads
        replica_factor = 1
    else:
        if not allow_replication or degree % heads:
            raise ValueError(
                f"heads={heads} cannot be sharded over degree={degree}"
            )
        effective_heads = degree
        replica_factor = degree // heads

    heads_per_rank = effective_heads // degree
    expanded = tuple(
        head
        for head in range(heads)
        for _ in range(replica_factor)
    )
    owner_head_indices = tuple(
        expanded[rank * heads_per_rank : (rank + 1) * heads_per_rank]
        for rank in range(degree)
    )
    return HeadShardPlan(
        original_heads=heads,
        effective_heads=effective_heads,
        degree=degree,
        heads_per_rank=heads_per_rank,
        replica_factor=replica_factor,
        owner_head_indices=owner_head_indices,
    )


def make_usp_topology(world_size: int, kv_heads: int) -> USPTopology:
    """Build the row-major Ulysses x ring decomposition used by USP."""
    if world_size <= 0 or kv_heads <= 0:
        raise ValueError("world_size and kv_heads must be positive")
    ulysses_degree = min(world_size, kv_heads)
    if world_size % ulysses_degree or kv_heads % ulysses_degree:
        raise ValueError(
            f"USP requires U={ulysses_degree} to divide CP={world_size} "
            f"and KVH={kv_heads}"
        )
    ring_degree = world_size // ulysses_degree
    a2a_groups = tuple(
        tuple(range(row * ulysses_degree, (row + 1) * ulysses_degree))
        for row in range(ring_degree)
    )
    ring_groups = tuple(
        tuple(range(column, world_size, ulysses_degree))
        for column in range(ulysses_degree)
    )
    return USPTopology(
        world_size,
        ulysses_degree,
        ring_degree,
        a2a_groups,
        ring_groups,
    )


def sequence_a2a_order(
    local_lengths: Sequence[int], degree: int, *, zigzag: bool
) -> tuple[int, ...]:
    """Map source-rank-major local TND rows to packed per-sequence rows."""
    lengths = [int(length) for length in local_lengths]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("local_lengths must contain positive values")
    group_lengths = [length * degree for length in lengths]
    return tuple(
        sequence_shards_to_global_order(group_lengths, degree, zigzag)
    )


_USP_GROUP_CACHE: dict[
    tuple[int, tuple[int, ...], int],
    tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup]],
] = {}


def _create_usp_process_groups(
    process_group: Optional[dist.ProcessGroup], topology: USPTopology
) -> tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup]]:
    parent = dist.group.WORLD if process_group is None else process_group
    parent_ranks = tuple(dist.get_process_group_ranks(parent))
    if parent_ranks != tuple(range(dist.get_world_size())):
        raise ValueError("USP currently requires the full WORLD process group")
    key = (id(parent), parent_ranks, topology.ulysses_degree)
    cached = _USP_GROUP_CACHE.get(key)
    if cached is not None:
        return cached

    if topology.ulysses_degree == topology.world_size:
        result = (parent, None)
        _USP_GROUP_CACHE[key] = result
        return result
    if topology.ulysses_degree == 1:
        result = (None, parent)
        _USP_GROUP_CACHE[key] = result
        return result

    global_rank = dist.get_rank()
    a2a_group: Optional[dist.ProcessGroup] = None
    ring_group: Optional[dist.ProcessGroup] = None
    for members in topology.a2a_groups:
        group = dist.new_group(ranks=list(members))
        if global_rank in members:
            a2a_group = group
    for members in topology.ring_groups:
        group = dist.new_group(ranks=list(members))
        if global_rank in members:
            ring_group = group
    if a2a_group is None or ring_group is None:
        raise RuntimeError("current rank was not assigned to both USP groups")
    result = (a2a_group, ring_group)
    _USP_GROUP_CACHE[key] = result
    return result


class _SeqHeadAllToAll:
    """Reusable equal-split TND sequence-to-head all-to-all transform."""

    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup],
        degree: int,
        local_lengths: Sequence[int],
        plan: HeadShardPlan,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        zigzag: bool,
        enable_backward: bool,
    ) -> None:
        self.process_group = process_group
        self.degree = degree
        self.local_lengths = tuple(int(length) for length in local_lengths)
        self.local_tokens = sum(self.local_lengths)
        self.plan = plan
        self.head_dim = head_dim
        self.owner_indices = tuple(
            torch.tensor(indices, dtype=torch.int64, device=device)
            for indices in plan.owner_head_indices
        )
        order = torch.tensor(
            sequence_a2a_order(self.local_lengths, degree, zigzag=zigzag),
            dtype=torch.int64,
            device=device,
        )
        self.order = order
        self.inverse_order = torch.argsort(order)
        shape = (
            degree,
            self.local_tokens,
            plan.heads_per_rank,
            head_dim,
        )
        self.send = torch.empty(shape, dtype=dtype, device=device)
        self.recv = torch.empty_like(self.send)
        self.attention = torch.empty(
            (degree * self.local_tokens, plan.heads_per_rank, head_dim),
            dtype=dtype,
            device=device,
        )
        self.local = torch.empty(
            (self.local_tokens, plan.effective_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.fp32_send = self.fp32_recv = self.fp32_local = None
        if enable_backward:
            self.fp32_send = torch.empty(shape, dtype=torch.float32, device=device)
            self.fp32_recv = torch.empty_like(self.fp32_send)
            self.fp32_local = torch.empty(
                (self.local_tokens, plan.effective_heads, head_dim),
                dtype=torch.float32,
                device=device,
            )

    def before_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape != (
            self.local_tokens,
            self.plan.original_heads,
            self.head_dim,
        ):
            raise ValueError(
                "all-to-all input shape does not match its head-shard plan: "
                f"got {tuple(tensor.shape)}"
            )
        for rank, indices in enumerate(self.owner_indices):
            torch.index_select(tensor, 1, indices, out=self.send[rank])
        if self.degree == 1:
            self.recv.copy_(self.send)
        else:
            dist.all_to_all_single(
                self.recv.view(-1),
                self.send.view(-1),
                group=self.process_group,
            )
        torch.index_select(
            self.recv.view(-1, self.plan.heads_per_rank, self.head_dim),
            0,
            self.order,
            out=self.attention,
        )
        return self.attention

    def after_attention(
        self, tensor: torch.Tensor, *, fp32: bool = False
    ) -> torch.Tensor:
        expected = (
            self.degree * self.local_tokens,
            self.plan.heads_per_rank,
            self.head_dim,
        )
        if tensor.shape != expected:
            raise ValueError(
                f"inverse all-to-all input must have shape {expected}, "
                f"got {tuple(tensor.shape)}"
            )
        if fp32:
            if self.fp32_send is None or self.fp32_recv is None or self.fp32_local is None:
                raise RuntimeError("FP32 inverse buffers were not allocated")
            send, recv, local = self.fp32_send, self.fp32_recv, self.fp32_local
            if tensor.dtype != torch.float32:
                raise ValueError("FP32 inverse all-to-all requires an FP32 input")
        else:
            send, recv, local = self.send, self.recv, self.local
            if tensor.dtype != send.dtype:
                raise ValueError("inverse all-to-all dtype does not match its channel")

        torch.index_select(
            tensor,
            0,
            self.inverse_order,
            out=send.view(-1, self.plan.heads_per_rank, self.head_dim),
        )
        if self.degree == 1:
            recv.copy_(send)
        else:
            dist.all_to_all_single(
                recv.view(-1), send.view(-1), group=self.process_group
            )
        for rank in range(self.degree):
            begin = rank * self.plan.heads_per_rank
            end = begin + self.plan.heads_per_rank
            local[:, begin:end].copy_(recv[rank])
        return local


class UlyssesAttention:
    """Full-CP sequence-to-head all-to-all attention baseline."""

    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_lengths: Sequence[int],
        is_causal: bool,
        backend: str,
        *,
        enable_backward: bool = False,
        _a2a_group: Optional[dist.ProcessGroup] = None,
        _a2a_degree: Optional[int] = None,
        _ring_group: Optional[dist.ProcessGroup] = None,
        _ring_degree: int = 1,
        _zigzag: bool = False,
        _replicate_kv: bool = True,
    ) -> None:
        self.process_group = dist.group.WORLD if process_group is None else process_group
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)
        self.a2a_group = self.process_group if _a2a_group is None else _a2a_group
        self.a2a_degree = self.world_size if _a2a_degree is None else _a2a_degree
        self.ring_group = _ring_group
        self.ring_degree = _ring_degree
        self.zigzag = _zigzag
        self.global_lengths = tuple(int(length) for length in global_lengths)
        self.is_causal = bool(is_causal)
        self.enable_backward = enable_backward
        self.backend = _make_block_backend(backend)
        self._validate_inputs(q, k, v)
        self.local_lengths = tuple(
            length // self.world_size for length in self.global_lengths
        )
        self.local_tokens = sum(self.local_lengths)
        if q.size(0) != self.local_tokens:
            raise ValueError(
                f"Q/K/V contain {q.size(0)} rows, expected {self.local_tokens}"
            )
        self.q_plan = make_head_shard_plan(
            q.size(1), self.a2a_degree, allow_replication=False
        )
        self.kv_plan = make_head_shard_plan(
            k.size(1), self.a2a_degree, allow_replication=_replicate_kv
        )
        self.q_channel = self._make_channel(q, self.q_plan)
        self.k_channel = self._make_channel(
            k, self.kv_plan, fp32_inverse=enable_backward
        )
        self.v_channel = self._make_channel(
            v, self.kv_plan, fp32_inverse=enable_backward
        )
        self.do_channel = (
            self._make_channel(q, self.q_plan) if enable_backward else None
        )
        self.layer_lengths = tuple(
            length // self.ring_degree for length in self.global_lengths
        )
        self.cu, self.cu_host = make_cu_seqlens(list(self.layer_lengths), q.device)
        self.max_seqlen = max(self.layer_lengths)
        self.q, self.k, self.v = q, k, v
        self.out = torch.empty_like(q)
        self.dq = torch.empty_like(q) if enable_backward else None
        self.dk = torch.empty_like(k) if enable_backward else None
        self.dv = torch.empty_like(v) if enable_backward else None
        self.dk_fp32 = (
            torch.empty_like(k, dtype=torch.float32) if enable_backward else None
        )
        self.dv_fp32 = (
            torch.empty_like(v, dtype=torch.float32) if enable_backward else None
        )
        self.layer_dq = (
            torch.empty_like(self.q_channel.attention) if enable_backward else None
        )
        self.layer_dk = (
            torch.empty_like(self.k_channel.attention) if enable_backward else None
        )
        self.layer_dv = (
            torch.empty_like(self.v_channel.attention) if enable_backward else None
        )
        self.layer_dk_fp32 = (
            torch.empty_like(self.k_channel.attention, dtype=torch.float32)
            if enable_backward
            else None
        )
        self.layer_dv_fp32 = (
            torch.empty_like(self.v_channel.attention, dtype=torch.float32)
            if enable_backward
            else None
        )
        self.saved_out: Optional[torch.Tensor] = None
        self.saved_lse: Optional[torch.Tensor] = None

    def _validate_inputs(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        if not self.global_lengths or any(length <= 0 for length in self.global_lengths):
            raise ValueError("global_lengths must contain positive values")
        if any(length % self.world_size for length in self.global_lengths):
            raise ValueError("every global sequence length must be divisible by CP size")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("Q/K/V must have TND shape [tokens, heads, head_dim]")
        if k.shape != v.shape or q.size(0) != k.size(0):
            raise ValueError("Q/K/V token counts and K/V shapes must match")
        if q.size(2) != 128 or k.size(2) != 128:
            raise ValueError("Ulysses/USP baselines require head_dim=128")
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("Ulysses/USP baselines require BF16 Q/K/V")
        if q.device != k.device or q.device != v.device or not q.is_cuda:
            raise ValueError("Q/K/V must be CUDA tensors on the same device")
        if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
            raise ValueError("Q/K/V must be contiguous")
        if q.size(1) % k.size(1):
            raise ValueError("QH must be divisible by KVH")
        if self.enable_backward and not self.is_causal:
            raise ValueError("the benchmark USP/Ulysses backward is causal-only")
        if self.zigzag and any(
            length % (2 * self.world_size) for length in self.global_lengths
        ):
            raise ValueError(
                "causal USP with a ring requires every global length to be "
                "divisible by 2 * CP"
            )

    def _make_channel(
        self,
        tensor: torch.Tensor,
        plan: HeadShardPlan,
        *,
        fp32_inverse: bool = False,
    ) -> _SeqHeadAllToAll:
        return _SeqHeadAllToAll(
            self.a2a_group,
            self.a2a_degree,
            self.local_lengths,
            plan,
            tensor.size(2),
            tensor.dtype,
            tensor.device,
            zigzag=self.zigzag,
            enable_backward=fp32_inverse,
        )

    @property
    def note(self) -> str:
        replica = (
            f"KV replica x{self.kv_plan.replica_factor}"
            if self.kv_plan.replica_factor > 1
            else "no KV replication"
        )
        mode = "causal contiguous" if self.is_causal else "noncausal contiguous"
        return (
            f"full-CP QKVO all-to-all; {replica}; {self.backend.backend_name}; "
            f"{mode}; backward redoes QKV all-to-all"
        )

    def bind_inputs(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        for tensor, previous, name in ((q, self.q, "Q"), (k, self.k, "K"), (v, self.v, "V")):
            if tensor.shape != previous.shape or tensor.dtype != previous.dtype:
                raise ValueError(f"rebound {name} shape/dtype must match construction")
            if tensor.device != previous.device or not tensor.is_contiguous():
                raise ValueError(f"rebound {name} device/layout must match construction")
        self.q, self.k, self.v = q, k, v

    def _attention_forward(
        self, q_layer: torch.Tensor, k_layer: torch.Tensor, v_layer: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend.forward_block(
            q_layer,
            k_layer,
            v_layer,
            self.cu,
            self.cu,
            self.cu_host,
            self.cu_host,
            self.max_seqlen,
            self.max_seqlen,
            self.is_causal,
        )

    def forward(self) -> torch.Tensor:
        q_layer = self.q_channel.before_attention(self.q)
        k_layer = self.k_channel.before_attention(self.k)
        v_layer = self.v_channel.before_attention(self.v)
        attention_out, attention_lse = self._attention_forward(
            q_layer, k_layer, v_layer
        )
        if self.enable_backward:
            self.saved_out, self.saved_lse = attention_out, attention_lse
        self.out.copy_(self.q_channel.after_attention(attention_out))
        return self.out

    def backward(
        self, dout: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enable_backward:
            raise RuntimeError("runner was created without backward buffers")
        if any(
            tensor is None
            for tensor in (
                self.do_channel,
                self.dq,
                self.dk,
                self.dv,
                self.dk_fp32,
                self.dv_fp32,
                self.layer_dq,
                self.layer_dk,
                self.layer_dv,
                self.layer_dk_fp32,
                self.layer_dv_fp32,
            )
        ):
            raise RuntimeError("backward buffers were not initialized")
        if self.saved_out is None or self.saved_lse is None:
            raise RuntimeError("backward requires a prepared forward")
        if dout.shape != self.out.shape or dout.dtype != self.out.dtype:
            raise ValueError("dout must match the local output")
        q_layer = self.q_channel.before_attention(self.q)
        k_layer = self.k_channel.before_attention(self.k)
        v_layer = self.v_channel.before_attention(self.v)
        dout_layer = self.do_channel.before_attention(dout.contiguous())
        dq_layer, dk_layer, dv_layer = self.backend.backward_block(
            dout_layer,
            q_layer,
            k_layer,
            v_layer,
            self.saved_out,
            self.saved_lse,
            self.cu,
            self.cu,
            self.max_seqlen,
            self.max_seqlen,
            self.is_causal,
            self.layer_dq,
            self.layer_dk,
            self.layer_dv,
        )
        self.dq.copy_(self.q_channel.after_attention(dq_layer))
        self.layer_dk_fp32.copy_(dk_layer)
        self.layer_dv_fp32.copy_(dv_layer)
        dk_effective = self.k_channel.after_attention(
            self.layer_dk_fp32, fp32=True
        )
        dv_effective = self.v_channel.after_attention(
            self.layer_dv_fp32, fp32=True
        )
        if self.kv_plan.replica_factor > 1:
            shape = (
                self.local_tokens,
                self.kv_plan.original_heads,
                self.kv_plan.replica_factor,
                self.k.size(2),
            )
            torch.sum(dk_effective.view(shape), dim=2, out=self.dk_fp32)
            torch.sum(dv_effective.view(shape), dim=2, out=self.dv_fp32)
        else:
            self.dk_fp32.copy_(dk_effective)
            self.dv_fp32.copy_(dv_effective)
        self.dk.copy_(self.dk_fp32)
        self.dv.copy_(self.dv_fp32)
        return self.dq, self.dk, self.dv


class USPAttention(UlyssesAttention):
    """Ulysses head sharding followed by a ring over sequence super-shards."""

    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_lengths: Sequence[int],
        is_causal: bool,
        backend: str,
        *,
        enable_backward: bool = False,
    ) -> None:
        parent = dist.group.WORLD if process_group is None else process_group
        world_size = dist.get_world_size(parent)
        topology = make_usp_topology(world_size, k.size(1))
        a2a_group, ring_group = _create_usp_process_groups(
            parent, topology
        )
        self.topology = topology
        super().__init__(
            parent,
            q,
            k,
            v,
            global_lengths,
            is_causal,
            backend,
            enable_backward=enable_backward,
            _a2a_group=a2a_group,
            _a2a_degree=topology.ulysses_degree,
            _ring_group=ring_group,
            _ring_degree=topology.ring_degree,
            _zigzag=is_causal and topology.ring_degree > 1,
            _replicate_kv=False,
        )
        self.ring_backward: Optional[Any] = None
        if enable_backward and topology.ring_degree > 1:
            try:
                from .hybrid_backward_baselines import VarlenFa3RingBackward
            except ImportError:
                from hybrid_backward_baselines import VarlenFa3RingBackward
            self.ring_backward = VarlenFa3RingBackward(
                ring_group,
                self.q_channel.attention,
                self.k_channel.attention,
                self.v_channel.attention,
                self.do_channel.attention,
                list(self.layer_lengths),
                backend,
            )

    @property
    def note(self) -> str:
        layout = (
            "zigzag causal"
            if self.is_causal and self.topology.ring_degree > 1
            else ("causal contiguous" if self.is_causal else "noncausal contiguous")
        )
        return (
            f"USP U={self.topology.ulysses_degree}, R={self.topology.ring_degree}; "
            f"QKVO all-to-all + FA3 ring; {self.backend.backend_name}; {layout}; "
            "backward redoes QKV all-to-all and ring K/V"
        )

    def _attention_forward(
        self, q_layer: torch.Tensor, k_layer: torch.Tensor, v_layer: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.topology.ring_degree == 1:
            return super()._attention_forward(q_layer, k_layer, v_layer)
        if self.enable_backward:
            if self.ring_backward is None:
                raise RuntimeError("USP ring backward runner was not initialized")
            self.ring_backward.bind_inputs(q_layer, k_layer, v_layer)
            out = self.ring_backward.forward()
            if self.ring_backward.lse is None:
                raise RuntimeError("USP ring forward did not retain LSE")
            return out, self.ring_backward.lse
        try:
            from .hybrid_forward_baselines import fa3_ring_forward
        except ImportError:
            from hybrid_forward_baselines import fa3_ring_forward
        result = fa3_ring_forward(
            self.ring_group,
            q_layer,
            k_layer,
            v_layer,
            self.cu,
            self.cu_host,
            list(self.layer_lengths),
            self.is_causal,
            self.backend.backend,
            return_lse=True,
        )
        if not isinstance(result, tuple):
            raise RuntimeError("USP ring forward did not return output and LSE")
        return result

    def backward(
        self, dout: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.topology.ring_degree == 1:
            return super().backward(dout)
        if not self.enable_backward or self.ring_backward is None:
            raise RuntimeError("runner was created without ring backward buffers")
        if self.saved_out is None or self.saved_lse is None:
            raise RuntimeError("backward requires a prepared forward")
        q_layer = self.q_channel.before_attention(self.q)
        k_layer = self.k_channel.before_attention(self.k)
        v_layer = self.v_channel.before_attention(self.v)
        dout_layer = self.do_channel.before_attention(dout.contiguous())
        self.ring_backward.bind_inputs(q_layer, k_layer, v_layer, dout_layer)
        dq_layer, dk_layer_fp32, dv_layer_fp32 = self.ring_backward.backward(
            return_fp32=True
        )
        self.dq.copy_(self.q_channel.after_attention(dq_layer))
        self.dk_fp32.copy_(
            self.k_channel.after_attention(dk_layer_fp32, fp32=True)
        )
        self.dv_fp32.copy_(
            self.v_channel.after_attention(dv_layer_fp32, fp32=True)
        )
        self.dk.copy_(self.dk_fp32)
        self.dv.copy_(self.dv_fp32)
        return self.dq, self.dk, self.dv


__all__ = [
    "HeadShardPlan",
    "USPAttention",
    "USPTopology",
    "UlyssesAttention",
    "make_head_shard_plan",
    "make_usp_topology",
    "sequence_a2a_order",
]
