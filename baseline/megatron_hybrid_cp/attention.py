"""Phase-separated FA3 execution for a Megatron hybrid-CP plan."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

# The ring helpers predate the ``ring_test`` package imports and are also used
# as direct script modules.  Make that existing local module layout available
# without importing Megatron-LM or adding third_party to sys.path.
_DEMO_DIR = Path(__file__).resolve().parents[2]
_RING_TEST_DIR = _DEMO_DIR / "ring_test"
if str(_RING_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_RING_TEST_DIR))

from hybrid_backward_baselines import (  # noqa: E402
    VarlenFa3RingBackward,
    _BlockBackend,
)
from ring_common import (  # noqa: E402
    ring_varlen_forward,
    zigzag_ring_varlen_forward,
)

from .plan import (
    HybridCPPlan,
    HybridCPProcessGroups,
    _needs_inter_group_barrier,
)


@dataclass(frozen=True)
class PackedHybridCPInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    dout: Optional[torch.Tensor]


def make_packed_hybrid_cp_inputs(
    plan: HybridCPPlan,
    rank: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    *,
    is_causal: bool,
    seed: int = 0,
    with_dout: bool = False,
) -> PackedHybridCPInputs:
    """Allocate deterministic rank-local tensors in plan assignment order."""
    local_tokens = sum(plan.local_lengths_for_rank(rank))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + rank * 1009 + int(is_causal) * 1_000_003)
    q = torch.randn(
        (local_tokens, q_heads, head_dim),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).to(torch.bfloat16)
    k = torch.randn(
        (local_tokens, kv_heads, head_dim),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).to(torch.bfloat16)
    v = (
        torch.randn(
            (local_tokens, kv_heads, head_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(0.5)
        .add_(rank * 0.125)
        .to(torch.bfloat16)
    )
    dout = None
    if with_dout:
        dout = torch.randn(
            q.shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        ).to(torch.bfloat16)
    return PackedHybridCPInputs(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        None if dout is None else dout.contiguous(),
    )


class _LocalSampleRunner:
    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dout: Optional[torch.Tensor],
        is_causal: bool,
        backend: str,
    ) -> None:
        self.q = q
        self.k = k
        self.v = v
        self.dout = dout
        self.is_causal = is_causal
        self.backend = _BlockBackend(backend)
        self.length = q.size(0)
        self.cu_host = torch.tensor([0, self.length], dtype=torch.int32)
        self.cu = self.cu_host.to(device=q.device)
        self.out: Optional[torch.Tensor] = None
        self.lse: Optional[torch.Tensor] = None
        self.dq = torch.empty_like(q)
        self.dk = torch.empty_like(k)
        self.dv = torch.empty_like(v)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.out, self.lse = self.backend.forward_block(
            self.q,
            self.k,
            self.v,
            self.cu,
            self.cu,
            self.cu_host,
            self.cu_host,
            self.length,
            self.length,
            self.is_causal,
        )
        return self.out, self.lse

    def backward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dout is None:
            raise RuntimeError("backward was not enabled for this sample")
        if self.out is None or self.lse is None:
            raise RuntimeError("local backward requires a prepared forward")
        return self.backend.backward_block(
            self.dout,
            self.q,
            self.k,
            self.v,
            self.out,
            self.lse,
            self.cu,
            self.cu,
            self.length,
            self.length,
            self.is_causal,
            self.dq,
            self.dk,
            self.dv,
        )


class _RingForwardSampleRunner:
    def __init__(
        self,
        process_group: dist.ProcessGroup,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        backend: str,
    ) -> None:
        self.process_group = process_group
        self.q = q
        self.k = k
        self.v = v
        self.is_causal = is_causal
        self.backend = _BlockBackend(backend)
        self.length = q.size(0)
        self.cu_host = torch.tensor([0, self.length], dtype=torch.int32)
        self.cu = self.cu_host.to(device=q.device)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_causal:
            result = zigzag_ring_varlen_forward(
                self.process_group,
                self.q,
                self.k,
                self.v,
                self.cu,
                self.cu_host,
                self.length,
                self.backend.forward_block,
                return_lse=True,
            )
        else:
            result = ring_varlen_forward(
                self.process_group,
                self.q,
                self.k,
                self.v,
                False,
                lambda q, k, v, causal: self.backend.forward_block(
                    q,
                    k,
                    v,
                    self.cu,
                    self.cu,
                    self.cu_host,
                    self.cu_host,
                    self.length,
                    self.length,
                    causal,
                ),
                return_lse=True,
            )
        if not isinstance(result, tuple):
            raise RuntimeError("ring forward did not return output and LSE")
        return result


class _RingBackwardSampleRunner:
    def __init__(
        self,
        process_group: dist.ProcessGroup,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dout: torch.Tensor,
        backend: str,
    ) -> None:
        self.runner = VarlenFa3RingBackward(
            process_group,
            q,
            k,
            v,
            dout,
            [q.size(0)],
            backend,
        )

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.runner.forward()
        if self.runner.lse is None:
            raise RuntimeError("ring forward did not save LSE")
        return out, self.runner.lse

    def backward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.runner.backward()


class MegatronHybridCPAttention:
    """Execute every sample in a standalone Megatron hybrid-CP plan."""

    def __init__(
        self,
        plan: HybridCPPlan,
        process_groups: HybridCPProcessGroups,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        backend: str,
        *,
        dout: Optional[torch.Tensor] = None,
    ) -> None:
        if process_groups.world_size != plan.world_size:
            raise ValueError("plan and process-group world sizes do not match")
        self.plan = plan
        self.process_groups = process_groups
        self.rank = dist.get_rank(process_groups.world_group)
        self.q = q
        self.k = k
        self.v = v
        self.dout = dout
        self.is_causal = is_causal
        self.backend = backend
        self._validate_inputs()

        self.sample_ids = plan.sample_ids_for_rank(self.rank)
        self.sample_slices: dict[int, slice] = {}
        offset = 0
        for sample_id in self.sample_ids:
            assignment = plan.assignment(sample_id)
            local_length = assignment.global_length // assignment.cp_size
            self.sample_slices[sample_id] = slice(offset, offset + local_length)
            offset += local_length

        self.out = torch.empty_like(q)
        self.lse = torch.empty(
            (q.size(1), q.size(0)), device=q.device, dtype=torch.float32
        )
        self.dq = torch.empty_like(q) if dout is not None else None
        self.dk = torch.empty_like(k) if dout is not None else None
        self.dv = torch.empty_like(v) if dout is not None else None
        self._runners: dict[int, object] = {}
        for sample_id in self.sample_ids:
            assignment = plan.assignment(sample_id)
            token_slice = self.sample_slices[sample_id]
            sample_q = q[token_slice]
            sample_k = k[token_slice]
            sample_v = v[token_slice]
            sample_dout = None if dout is None else dout[token_slice]
            if assignment.cp_size == 1:
                runner: object = _LocalSampleRunner(
                    sample_q,
                    sample_k,
                    sample_v,
                    sample_dout,
                    is_causal,
                    backend,
                )
            else:
                process_group = process_groups.group_for(
                    assignment.cp_size, assignment.rank_start
                )
                if process_group is None:
                    raise AssertionError("distributed sample has no process group")
                if sample_dout is None:
                    runner = _RingForwardSampleRunner(
                        process_group,
                        sample_q,
                        sample_k,
                        sample_v,
                        is_causal,
                        backend,
                    )
                else:
                    if not is_causal:
                        raise ValueError("hybrid-CP backward supports causal mode only")
                    runner = _RingBackwardSampleRunner(
                        process_group,
                        sample_q,
                        sample_k,
                        sample_v,
                        sample_dout,
                        backend,
                    )
            self._runners[sample_id] = runner
        self._forward_ready = False

    def _validate_inputs(self) -> None:
        tensors = [self.q, self.k, self.v]
        if self.dout is not None:
            tensors.append(self.dout)
        if any(tensor.device.type != "cuda" for tensor in tensors):
            raise ValueError("hybrid-CP attention requires CUDA tensors")
        if any(tensor.device != self.q.device for tensor in tensors[1:]):
            raise ValueError("Q/K/V/dO must be on the same CUDA device")
        if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
            raise ValueError("hybrid-CP attention requires BF16 tensors")
        if any(tensor.dim() != 3 for tensor in tensors):
            raise ValueError("Q/K/V/dO must use [tokens, heads, dim] layout")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("Q/K/V/dO must be contiguous")
        if self.q.size(2) != 128 or self.k.size(2) != 128 or self.v.size(2) != 128:
            raise ValueError("hybrid-CP attention requires head dim 128")
        if self.q.size(1) % self.k.size(1):
            raise ValueError("Q head count must be divisible by KV head count")
        if self.k.shape != self.v.shape:
            raise ValueError("K and V shapes must match")
        if self.q.size(0) != self.k.size(0):
            raise ValueError("Q/K/V token counts must match")
        if self.dout is not None and self.dout.shape != self.q.shape:
            raise ValueError("dO shape must match Q")
        expected_tokens = sum(self.plan.local_lengths_for_rank(self.rank))
        if self.q.size(0) != expected_tokens:
            raise ValueError(
                f"rank {self.rank} input has {self.q.size(0)} tokens, "
                f"but plan requires {expected_tokens}"
            )
        for assignment in self.plan.assignments:
            if assignment.global_length % assignment.cp_size:
                raise ValueError(
                    f"sample {assignment.sample_id} length "
                    f"{assignment.global_length} is not divisible by "
                    f"CP{assignment.cp_size}"
                )
            local_length = assignment.global_length // assignment.cp_size
            if self.is_causal and assignment.cp_size > 1 and (
                local_length % 2 or (local_length // 2) % 128
            ):
                raise ValueError(
                    f"sample {assignment.sample_id} causal local half length "
                    f"{local_length // 2} is not 128-aligned"
                )

    @property
    def note(self) -> str:
        backend_name = (
            "external FA3"
            if self.backend == "external_fa3"
            else "in-repo min_fa3 fallback"
        )
        return (
            f"Megatron scheduler; {self.plan.num_execution_groups} execution groups; "
            f"P2P ring; {backend_name}; forward-order phase replay"
        )

    def _barrier_after_group(self, group_index: int) -> None:
        if _needs_inter_group_barrier(
            group_index, self.plan.num_execution_groups
        ):
            dist.barrier(group=self.process_groups.world_group)

    def forward_all(self) -> torch.Tensor:
        for group_index, group in enumerate(self.plan.execution_groups):
            for sample_id in group.sample_ids_by_rank[self.rank]:
                runner = self._runners[sample_id]
                out, lse = runner.forward()  # type: ignore[attr-defined]
                token_slice = self.sample_slices[sample_id]
                self.out[token_slice].copy_(out)
                self.lse[:, token_slice].copy_(lse)
            self._barrier_after_group(group_index)
        self._forward_ready = True
        return self.out

    def backward_all(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dout is None or self.dq is None or self.dk is None or self.dv is None:
            raise RuntimeError("backward was not enabled for this runner")
        if not self.is_causal:
            raise RuntimeError("hybrid-CP backward supports causal mode only")
        if not self._forward_ready:
            raise RuntimeError("backward_all requires a complete forward_all phase")
        for group_index, group in enumerate(self.plan.execution_groups):
            for sample_id in group.sample_ids_by_rank[self.rank]:
                runner = self._runners[sample_id]
                dq, dk, dv = runner.backward()  # type: ignore[attr-defined]
                token_slice = self.sample_slices[sample_id]
                self.dq[token_slice].copy_(dq)
                self.dk[token_slice].copy_(dk)
                self.dv[token_slice].copy_(dv)
            self._barrier_after_group(group_index)
        return self.dq, self.dk, self.dv


__all__ = [
    "MegatronHybridCPAttention",
    "PackedHybridCPInputs",
    "make_packed_hybrid_cp_inputs",
]
