"""Single-layer Megatron adapters for the existing CP attention benchmarks.

The Transformer layer, norms, projections, BDA, and MLP remain Megatron-Core
modules.  Only ``SelfAttention.core_attention`` is replaced by the dispatcher
below.  Existing CP runners retain ownership of their communication plans and
preallocated scratch buffers.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from itertools import accumulate
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch
import torch.distributed as dist


THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
for _path in (THIS_DIR, DEMO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


METHOD_ORDER = (
    "allgather_attention",
    "llama3_allgather_attention",
    "fa3_ring",
    "megatron_hybrid_cp",
    "magi_attention",
    "zeppelin",
    "mega_ring_all_cp",
    "mega_ring_hybrid",
)
MEGA_RING_METHODS = frozenset(("mega_ring_all_cp", "mega_ring_hybrid"))
ALL_CP_METHODS = frozenset(
    (
        "allgather_attention",
        "llama3_allgather_attention",
        "fa3_ring",
        "mega_ring_all_cp",
    )
)


@dataclass(frozen=True)
class SmConfig:
    num_comp_sm: int
    num_comm_sm: int

    def __post_init__(self) -> None:
        if self.num_comp_sm <= 0 or self.num_comm_sm <= 0:
            raise ValueError("MegaRing SM counts must both be positive")

    @property
    def label(self) -> str:
        return f"{self.num_comp_sm}:{self.num_comm_sm}"


@dataclass(frozen=True)
class PhysicalLayout:
    method: str
    original_lengths: tuple[int, ...]
    execution_lengths: tuple[int, ...]
    rank_lengths: tuple[tuple[int, ...], ...]
    note: str

    @property
    def original_tokens(self) -> int:
        return sum(self.original_lengths)

    @property
    def execution_tokens(self) -> int:
        return sum(self.execution_lengths)

    @property
    def padding_tokens(self) -> int:
        return self.execution_tokens - self.original_tokens

    @property
    def rank_token_loads(self) -> tuple[int, ...]:
        return tuple(sum(lengths) for lengths in self.rank_lengths)

    def local_lengths(self, rank: int) -> tuple[int, ...]:
        if not self.rank_lengths:
            raise RuntimeError(f"{self.method} local layout is resolved at runtime")
        return self.rank_lengths[rank]


def parse_methods(spec: str) -> list[str]:
    methods: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        expanded = METHOD_ORDER if token == "all" else (token,)
        for method in expanded:
            if method not in METHOD_ORDER:
                raise ValueError(
                    f"unknown method {method!r}; expected one of {METHOD_ORDER} or all"
                )
            if method not in methods:
                methods.append(method)
    if not methods:
        raise ValueError("at least one CP method is required")
    return methods


def parse_sm_configs(spec: str | None, device_sm_count: int) -> list[SmConfig]:
    if spec is None or not spec.strip():
        if device_sm_count <= 8:
            raise ValueError(
                f"automatic MegaRing split requires more than 8 SMs, got {device_sm_count}"
            )
        return [SmConfig(device_sm_count - 8, 8)]
    configs: list[SmConfig] = []
    for token in spec.split(","):
        fields = token.strip().split(":")
        if len(fields) != 2:
            raise ValueError(f"invalid SM config {token!r}; expected COMP:COMM")
        config = SmConfig(int(fields[0]), int(fields[1]))
        if config.num_comp_sm + config.num_comm_sm > device_sm_count:
            raise ValueError(
                f"SM config {config.label} exceeds device SM count {device_sm_count}"
            )
        configs.append(config)
    if not configs:
        raise ValueError("--sm-configs must provide at least one COMP:COMM pair")
    return configs


def _all_cp_rank_lengths(
    execution_lengths: Sequence[int], world_size: int
) -> tuple[tuple[int, ...], ...]:
    if any(length % world_size for length in execution_lengths):
        raise ValueError("all-CP execution lengths must be divisible by CP size")
    local = tuple(length // world_size for length in execution_lengths)
    return tuple(local for _ in range(world_size))


def validate_hybrid_metadata(
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    world_size: int,
) -> None:
    if not global_lengths or not (
        len(global_lengths) == len(ring_sizes) == len(ring_starts)
    ):
        raise ValueError("hybrid lengths, ring sizes, and ring starts must align")
    previous_size = world_size
    for sample, (length, ring_size, ring_start) in enumerate(
        zip(global_lengths, ring_sizes, ring_starts)
    ):
        if ring_size not in (1, 2, 4, 8) or ring_size > world_size:
            raise ValueError(f"sample {sample} has unsupported ring size {ring_size}")
        if ring_size > previous_size:
            raise ValueError("hybrid samples must remain in non-increasing ring-size order")
        if ring_start < 0 or ring_start % ring_size:
            raise ValueError(f"sample {sample} has unaligned ring start {ring_start}")
        if ring_start + ring_size > world_size:
            raise ValueError(f"sample {sample} ring exceeds the CP group")
        if length <= 0 or length % ring_size:
            raise ValueError(f"sample {sample} length is not divisible by its ring")
        previous_size = ring_size


def build_physical_layout(
    method: str,
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    world_size: int,
    *,
    megatron_max_seqlen_per_rank: int = 8192,
    zeppelin_threshold: int = 4096,
) -> PhysicalLayout:
    """Build CPU-only physical work metadata for strict preflight and reporting."""
    from baseline.megatron_hybrid_cp import build_hybrid_cp_plan_for_fa3_ring
    from ring_test.utils import align_mega_ring_all_cp_lengths, local_lengths_for_rank
    from ring_test.zeppelin import make_zeppelin_plan

    if method not in METHOD_ORDER:
        raise ValueError(f"unknown method {method!r}")
    original = tuple(int(length) for length in global_lengths)
    if not original or any(length <= 0 for length in original):
        raise ValueError("global lengths must contain positive values")
    if world_size not in (4, 8):
        raise ValueError(
            f"this benchmark supports CP=4 smoke tests or formal CP=8 runs, got {world_size}"
        )
    validate_hybrid_metadata(original, ring_sizes, ring_starts, world_size)

    if method in ("allgather_attention", "llama3_allgather_attention", "fa3_ring"):
        rank_lengths = _all_cp_rank_lengths(original, world_size)
        if any(length % 2 for length in rank_lengths[0]):
            raise ValueError(f"{method} causal local sequence lengths must be even")
        if method == "llama3_allgather_attention" and sum(original) % (2 * world_size):
            raise ValueError("Llama3 packed zigzag requires total tokens divisible by 16")
        note = (
            "whole-packed two-block zigzag"
            if method == "llama3_allgather_attention"
            else "per-sequence zigzag all-CP"
        )
        return PhysicalLayout(method, original, original, rank_lengths, note)

    if method == "mega_ring_all_cp":
        execution = tuple(align_mega_ring_all_cp_lengths(list(original)))
        return PhysicalLayout(
            method,
            original,
            execution,
            _all_cp_rank_lengths(execution, world_size),
            "per-sequence 2048-token alignment; fused all-CP MegaRing",
        )

    if method == "mega_ring_hybrid":
        rank_lengths = tuple(
            tuple(
                local_lengths_for_rank(
                    list(original), list(ring_sizes), list(ring_starts), rank
                )
            )
            for rank in range(world_size)
        )
        return PhysicalLayout(
            method,
            original,
            original,
            rank_lengths,
            "BR-PBS G8/G4/G2/G1 placement; fused hybrid MegaRing",
        )

    if method == "megatron_hybrid_cp":
        plan = build_hybrid_cp_plan_for_fa3_ring(
            original,
            world_size,
            True,
            megatron_max_seqlen_per_rank,
        )
        rank_lengths = tuple(
            tuple(plan.local_lengths_for_rank(rank)) for rank in range(world_size)
        )
        return PhysicalLayout(
            method,
            original,
            tuple(plan.global_lengths),
            rank_lengths,
            f"Megatron scheduler; {plan.num_execution_groups} execution groups",
        )

    if method == "zeppelin":
        plan = make_zeppelin_plan(
            list(original), world_size, True, zeppelin_threshold
        )
        rank_lengths = tuple(
            tuple(plan.packed_lengths_for_rank(rank)) for rank in range(world_size)
        )
        return PhysicalLayout(
            method,
            original,
            tuple(plan.execution_lengths),
            rank_lengths,
            (
                f"Zeppelin L={plan.threshold}, final_s0={plan.effective_threshold}, "
                f"iterations={plan.iterations}"
            ),
        )

    # Magi's dispatch owns padding and rank-local placement.  Runtime setup
    # replaces this placeholder with exact metadata before any timed layer call.
    return PhysicalLayout(
        method,
        original,
        original,
        (),
        "Magi dispatch/padding resolved before layer timing",
    )


class ExplicitAttentionAdapter(Protocol):
    note: str

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor: ...

    def backward(
        self, dout: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class _ExplicitCPAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        adapter: ExplicitAttentionAdapter,
    ) -> torch.Tensor:
        ctx.adapter = adapter
        return adapter.forward(q, k, v)

    @staticmethod
    def backward(
        ctx: Any, dout: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        dq, dk, dv = ctx.adapter.backward(dout.contiguous())
        return dq, dk, dv, None


def explicit_cp_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    adapter: ExplicitAttentionAdapter,
) -> torch.Tensor:
    """Public seam used by the CPU fake-adapter test."""
    return _ExplicitCPAttention.apply(q, k, v, adapter)


class CoreAttentionDispatcher(torch.nn.Module):
    """Megatron ``CoreAttentionBuilder`` implementation with a mutable adapter."""

    def __init__(self, **_: Any) -> None:
        super().__init__()
        self.adapter: Any | None = None
        self.native_autograd = False

    def set_adapter(self, adapter: Any, *, native_autograd: bool = False) -> None:
        self.adapter = adapter
        self.native_autograd = native_autograd

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        attn_mask_type: Any,
        attention_bias: torch.Tensor | None,
        packed_seq_params: Any,
    ) -> torch.Tensor:
        del attention_mask, attn_mask_type, attention_bias, packed_seq_params
        if self.adapter is None:
            raise RuntimeError("no CP attention adapter is bound to the Transformer layer")
        q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
        if self.native_autograd:
            return self.adapter.forward(q, k, v)
        return explicit_cp_attention(q, k, v, self.adapter)


class _RunnerAdapter:
    def __init__(self, runner: Any, backward_takes_dout: bool) -> None:
        self.runner = runner
        self.backward_takes_dout = backward_takes_dout
        self.q: torch.Tensor | None = None
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.note = runner.note

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        self.q, self.k, self.v = q, k, v
        self.runner.bind_inputs(q, k, v)
        if hasattr(self.runner, "forward_all"):
            return self.runner.forward_all()
        return self.runner.forward()

    def backward(
        self, dout: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.q is None or self.k is None or self.v is None:
            raise RuntimeError("CP backward requires a prepared forward")
        if self.backward_takes_dout:
            return self.runner.backward(dout)
        self.runner.bind_inputs(self.q, self.k, self.v, dout)
        if hasattr(self.runner, "backward_all"):
            return self.runner.backward_all()
        return self.runner.backward()


class _MagiAdapter:
    def __init__(self, projected: Any) -> None:
        self.projected = projected
        self.note = projected.note

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return self.projected.forward(q, k, v)


class _MegaRingAdapter:
    """Timed data population plus fused MegaRing forward/backward."""

    def __init__(
        self,
        *,
        method: str,
        dummy_q: torch.Tensor,
        local_lengths: Sequence[int],
        rank_capacity: int,
        global_lengths: Sequence[int],
        ring_sizes: Sequence[int],
        ring_starts: Sequence[int],
        rank: int,
        world_size: int,
        sm_config: SmConfig,
    ) -> None:
        import min_fa3_op

        if method not in MEGA_RING_METHODS:
            raise ValueError(f"invalid MegaRing method {method}")
        self.op = min_fa3_op
        self.method = method
        self.rank = rank
        self.world_size = world_size
        self.rank_capacity = rank_capacity
        self.local_tokens = sum(local_lengths)
        self.sm_config = sm_config
        self.cu_host = torch.tensor(
            [0, *accumulate(local_lengths)], dtype=torch.int32
        )
        self.cu = self.cu_host.to(device=dummy_q.device)
        self.max_local_len = max(
            max(global_lengths[index] // ring_sizes[index], 1)
            for index in range(len(global_lengths))
        )
        self.global_host = torch.tensor(global_lengths, dtype=torch.int32)
        self.ring_sizes_host = torch.tensor(ring_sizes, dtype=torch.int32)
        self.ring_starts_host = torch.tensor(ring_starts, dtype=torch.int32)

        kv_heads = 8
        head_dim = 128
        arena_shape = [world_size * rank_capacity, kv_heads, head_dim]
        self.remote_k = min_fa3_op.TKParallelTensor(
            arena_shape, torch.bfloat16, rank, world_size, False
        )
        self.remote_v = min_fa3_op.TKParallelTensor(
            arena_shape, torch.bfloat16, rank, world_size, False
        )
        padded_capacity = (
            (rank_capacity + len(global_lengths) * 128 + 127) // 128
        ) * 128
        accum_numel = kv_heads * padded_capacity * head_dim
        self.remote_dk = min_fa3_op.TKParallelTensor(
            [accum_numel], torch.float32, rank, world_size, False
        )
        self.remote_dv = min_fa3_op.TKParallelTensor(
            [accum_numel], torch.float32, rank, world_size, False
        )
        self.completion = min_fa3_op.TKParallelTensor(
            [1], torch.int32, rank, world_size, False
        )
        self.workspace = min_fa3_op._create_backward_varlen_mega_ring_workspace(
            dummy_q,
            self.cu_host,
            self.cu_host,
            world_size,
            kv_heads,
            rank_capacity,
        )
        self.out = torch.empty_like(dummy_q)
        self.lse = torch.empty(
            (dummy_q.size(1), dummy_q.size(0)),
            dtype=torch.float32,
            device=dummy_q.device,
        )
        self.q: torch.Tensor | None = None
        self.note = (
            f"fused {method}; SM={sm_config.label}; rank_capacity={rank_capacity}; "
            "K/V arena population+barrier and backward accumulator reset+barrier timed"
        )

    def _populate_kv(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.size(0) != self.local_tokens or v.size(0) != self.local_tokens:
            raise ValueError("MegaRing projected K/V do not match the physical layout")
        owner_begin = self.rank * self.rank_capacity
        owner_end = owner_begin + self.rank_capacity
        self.remote_k.data_[owner_begin:owner_end].fill_(-123.0)
        self.remote_v.data_[owner_begin:owner_end].fill_(-123.0)
        self.remote_k.data_[owner_begin : owner_begin + k.size(0)].copy_(k)
        self.remote_v.data_[owner_begin : owner_begin + v.size(0)].copy_(v)
        torch.cuda.synchronize()
        dist.barrier()

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        self.q = q
        self._populate_kv(k, v)
        result = self.op.forward_varlen_mega_ring(
            q,
            self.remote_k.data_,
            self.remote_v.data_,
            self.cu,
            self.cu,
            self.max_local_len,
            self.max_local_len,
            True,
            cu_seqlens_q_host=self.cu_host,
            cu_seqlens_k_host=self.cu_host,
            remote_k=self.remote_k,
            remote_v=self.remote_v,
            num_comp_sm=self.sm_config.num_comp_sm,
            num_comm_sm=self.sm_config.num_comm_sm,
            global_seqlens_host=self.global_host,
            ring_sizes_host=self.ring_sizes_host,
            ring_starts_host=self.ring_starts_host,
            out=self.out,
            lse=self.lse,
            return_lse=True,
        )
        if not isinstance(result, tuple):
            raise RuntimeError("MegaRing forward did not return output and LSE")
        self.out, self.lse = result
        return self.out

    def backward(
        self, dout: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.q is None:
            raise RuntimeError("MegaRing backward requires a prepared forward")
        self.remote_dk.data_.zero_()
        self.remote_dv.data_.zero_()
        self.completion.data_.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        return self.op.backward_varlen_mega_ring(
            dout,
            self.q,
            self.remote_k.data_,
            self.remote_v.data_,
            self.out,
            self.lse,
            self.cu,
            self.cu,
            self.max_local_len,
            self.max_local_len,
            cu_seqlens_q_host=self.cu_host,
            cu_seqlens_k_host=self.cu_host,
            remote_k=self.remote_k,
            remote_v=self.remote_v,
            remote_dk_accum=self.remote_dk,
            remote_dv_accum=self.remote_dv,
            remote_dkv_completion=self.completion,
            num_comp_sm=self.sm_config.num_comp_sm,
            num_comm_sm=self.sm_config.num_comm_sm,
            global_seqlens_host=self.global_host,
            ring_sizes_host=self.ring_sizes_host,
            ring_starts_host=self.ring_starts_host,
            workspace=self.workspace,
        )


@dataclass
class PreparedMethod:
    method: str
    adapter: Any
    native_autograd: bool
    layout: PhysicalLayout
    local_packed_lengths: tuple[int, ...]
    metadata: dict[str, Any]


def _empty_qkv(
    local_tokens: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.empty((local_tokens, 32, 128), dtype=torch.bfloat16, device=device)
    k = torch.empty((local_tokens, 8, 128), dtype=torch.bfloat16, device=device)
    v = torch.empty_like(k)
    dout = torch.empty_like(q)
    return q, k, v, dout


def prepare_method(
    *,
    method: str,
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    rank: int,
    world_size: int,
    device: torch.device,
    block_backend: str,
    hybrid_groups: Any,
    sm_config: SmConfig | None,
    allgather_heads_k_stride: int,
    megatron_max_seqlen_per_rank: int,
    zeppelin_threshold: int,
    magi_overlap_degree: int,
    seed: int,
) -> PreparedMethod:
    """Allocate a method plan and all reusable workspaces outside timing."""
    from baseline.magi_attention import MagiAttentionConfig, MagiProjectedAttention
    from baseline.megatron_hybrid_cp import (
        MegatronHybridCPAttention,
        build_hybrid_cp_plan_for_fa3_ring,
    )
    from ring_test.allgather_attention import Llama3AllGatherAttention
    from ring_test.hybrid_backward_baselines import (
        VarlenAllGatherBackward,
        VarlenFa3RingBackward,
        ZeppelinBackward,
    )
    from ring_test.utils import local_lengths_for_rank
    from ring_test.zeppelin import make_zeppelin_plan

    layout = build_physical_layout(
        method,
        global_lengths,
        ring_sizes,
        ring_starts,
        world_size,
        megatron_max_seqlen_per_rank=megatron_max_seqlen_per_rank,
        zeppelin_threshold=zeppelin_threshold,
    )
    metadata: dict[str, Any] = {}

    if method == "magi_attention":
        projected = MagiProjectedAttention(
            dist.group.WORLD,
            global_lengths,
            32,
            8,
            128,
            True,
            device,
            config=MagiAttentionConfig(
                overlap_degree=magi_overlap_degree,
                seed=seed,
            ),
        )
        loads = [None for _ in range(world_size)]
        dist.all_gather_object(loads, projected.local_tokens)
        rank_lengths = tuple((int(tokens),) for tokens in loads)
        physical_tokens = sum(int(tokens) for tokens in loads)
        layout = replace(
            layout,
            execution_lengths=(physical_tokens,),
            rank_lengths=rank_lengths,
            note=projected.note,
        )
        metadata.update(
            chunk_size=projected.metadata.chunk_size,
            overlap_degree=projected.metadata.overlap_degree,
            magi_key_padded_tokens=projected.metadata.padded_tokens,
        )
        return PreparedMethod(
            method,
            _MagiAdapter(projected),
            True,
            layout,
            (projected.local_tokens,),
            metadata,
        )

    local_lengths = layout.local_lengths(rank)
    positive_local_lengths = tuple(length for length in local_lengths if length > 0)
    local_tokens = sum(local_lengths)
    q, k, v, dout = _empty_qkv(local_tokens, device)

    if method == "allgather_attention":
        runner = VarlenAllGatherBackward(
            dist.group.WORLD,
            q,
            k,
            v,
            list(local_lengths),
            block_backend,
            heads_k_stride=allgather_heads_k_stride,
        )
        adapter: Any = _RunnerAdapter(runner, True)
    elif method == "llama3_allgather_attention":
        runner = Llama3AllGatherAttention(
            dist.group.WORLD,
            q,
            k,
            v,
            list(global_lengths),
            True,
            block_backend,
            heads_k_stride=allgather_heads_k_stride,
            enable_backward=True,
        )
        adapter = _RunnerAdapter(runner, True)
        positive_local_lengths = (local_tokens,)
    elif method == "fa3_ring":
        runner = VarlenFa3RingBackward(
            dist.group.WORLD,
            q,
            k,
            v,
            dout,
            list(local_lengths),
            block_backend,
        )
        adapter = _RunnerAdapter(runner, False)
    elif method == "megatron_hybrid_cp":
        plan = build_hybrid_cp_plan_for_fa3_ring(
            global_lengths,
            world_size,
            True,
            megatron_max_seqlen_per_rank,
        )
        runner = MegatronHybridCPAttention(
            plan,
            hybrid_groups,
            q,
            k,
            v,
            True,
            block_backend,
            dout=dout,
        )
        adapter = _RunnerAdapter(runner, False)
        metadata.update(
            execution_groups=plan.num_execution_groups,
            cp_sizes=[assignment.cp_size for assignment in plan.assignments],
        )
    elif method == "zeppelin":
        plan = make_zeppelin_plan(
            list(global_lengths), world_size, True, zeppelin_threshold
        )
        runner = ZeppelinBackward(
            dist.group.WORLD,
            q,
            k,
            v,
            dout,
            plan,
            block_backend,
        )
        adapter = _RunnerAdapter(runner, False)
        metadata.update(
            threshold=plan.threshold,
            effective_threshold=plan.effective_threshold,
            group_sizes=[assignment.group_size for assignment in plan.assignments],
        )
    elif method in MEGA_RING_METHODS:
        if sm_config is None:
            raise ValueError("MegaRing preparation requires an SM configuration")
        if method == "mega_ring_all_cp":
            mega_ring_sizes = [world_size] * len(layout.execution_lengths)
            mega_ring_starts = [0] * len(layout.execution_lengths)
            rank_capacity = local_tokens
        else:
            mega_ring_sizes = list(ring_sizes)
            mega_ring_starts = list(ring_starts)
            rank_capacity = (
                (max(layout.rank_token_loads) + 127) // 128
            ) * 128
        adapter = _MegaRingAdapter(
            method=method,
            dummy_q=q,
            local_lengths=local_lengths,
            rank_capacity=rank_capacity,
            global_lengths=layout.execution_lengths,
            ring_sizes=mega_ring_sizes,
            ring_starts=mega_ring_starts,
            rank=rank,
            world_size=world_size,
            sm_config=sm_config,
        )
        metadata.update(
            num_comp_sm=sm_config.num_comp_sm,
            num_comm_sm=sm_config.num_comm_sm,
            rank_capacity=rank_capacity,
        )
    else:
        raise AssertionError(f"unhandled method {method}")

    return PreparedMethod(
        method,
        adapter,
        False,
        layout,
        positive_local_lengths or (local_tokens,),
        metadata,
    )


def make_packed_seq_params(
    local_lengths: Sequence[int], device: torch.device
) -> Any:
    """Construct Megatron packed metadata after the physical layout is fixed."""
    from megatron.core.packed_seq_params import PackedSeqParams

    lengths = tuple(int(length) for length in local_lengths if length > 0)
    if not lengths:
        raise ValueError("a rank must execute at least one physical token")
    cu_host = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32)
    cu = cu_host.to(device=device)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
    )


def build_megatron_layer(
    megatron_path: Path, device: torch.device, context_parallel_size: int
) -> Any:
    """Build the locked Llama-3-8B-style single TE Transformer layer."""
    if str(megatron_path) not in sys.path:
        sys.path.insert(0, str(megatron_path))

    import torch.nn.functional as functional
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_layer import TransformerLayer

    config = TransformerConfig(
        num_layers=1,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        kv_channels=128,
        ffn_hidden_size=14336,
        gated_linear_unit=True,
        activation_func=functional.silu,
        normalization="RMSNorm",
        layernorm_epsilon=1.0e-5,
        add_bias_linear=False,
        add_qkv_bias=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=context_parallel_size,
        sequence_parallel=False,
    )
    submodules = get_gpt_layer_with_transformer_engine_submodules()
    submodules.self_attention.submodules.core_attention = CoreAttentionDispatcher
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    layer = TransformerLayer(
        config,
        submodules,
        layer_number=1,
        pg_collection=pg_collection,
    )
    layer.to(device=device)
    layer.train()
    if not isinstance(layer.self_attention.core_attention, CoreAttentionDispatcher):
        raise RuntimeError("Megatron layer did not install the CP attention dispatcher")
    return layer


__all__ = [
    "ALL_CP_METHODS",
    "CoreAttentionDispatcher",
    "MEGA_RING_METHODS",
    "METHOD_ORDER",
    "PhysicalLayout",
    "PreparedMethod",
    "SmConfig",
    "build_megatron_layer",
    "build_physical_layout",
    "explicit_cp_attention",
    "make_packed_seq_params",
    "parse_methods",
    "parse_sm_configs",
    "prepare_method",
    "validate_hybrid_metadata",
]
