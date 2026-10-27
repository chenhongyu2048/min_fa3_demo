"""Communication-free controls for the Megatron hybrid-CP experiment."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .attention import MegatronHybridCPAttention
from hybrid_backward_baselines import VarlenFa3RingBackward, _BlockBackend
from ring_common import zigzag_ring_varlen_forward


class _ComputeOnlyRing(VarlenFa3RingBackward):
    """Replay ring compute with resident KV and rank-local dKV contributions."""

    def __init__(self, source: VarlenFa3RingBackward, kv: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        super().__init__(
            source.process_group, source.q, source.k, source.v, source.dout,
            [source.q.size(0)], source.backend,
        )
        self.cp_rank = dist.get_rank(self.process_group)
        self.cp_size = len(kv)
        self.kv = kv
        self.step_kv = [kv[(self.cp_rank - step) % self.cp_size] for step in range(self.cp_size)]
        shape = (self.cp_size, *self.k.shape)
        self.dk_parts = torch.zeros(shape, device=self.k.device, dtype=torch.float32)
        self.dv_parts = torch.zeros_like(self.dk_parts)

    def prepare_backward(self) -> None:
        # Incoming remote contributions are zero for this rank's partial dKV.
        self.dk_parts.zero_()
        self.dv_parts.zero_()

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.index_select(self.q, 0, self.back_index, out=self.q_back)
        self.out, self.lse = zigzag_ring_varlen_forward(
            self.process_group, self.q, self.k, self.v,
            self.cu, self.cu_host, self.max_local, self.forward_block,
            return_lse=True, execution_mode="comp_only", gathered_kv=self.step_kv,
        )
        self.out_back = self.out[self.back_index].contiguous()
        self.lse_back = self.lse[:, self.back_index].contiguous()
        return self.out, self.lse

    def backward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.index_select(self.q, 0, self.back_index, out=self.q_back)
        torch.index_select(self.dout, 0, self.back_index, out=self.dout_back)
        for step, (k, v) in enumerate(self.step_kv):
            owner = (self.cp_rank - step) % self.cp_size
            if step == 0:
                dq, dk, dv = self._block_backward(
                    self.dout, self.q, k, v, self.out, self.lse,
                    q_is_half=False, k_is_half=False, causal=True,
                )
                self.dq_accum.copy_(dq)
                self.dk_parts[owner].copy_(dk)
                self.dv_parts[owner].copy_(dv)
            elif step <= self.cp_rank:
                dq, dk, dv = self._block_backward(
                    self.dout, self.q,
                    k[self.front_index].contiguous(), v[self.front_index].contiguous(),
                    self.out, self.lse,
                    q_is_half=False, k_is_half=True, causal=False,
                )
                self.dq_accum += dq
                self.dk_parts[owner, self.front_index] += dk
                self.dv_parts[owner, self.front_index] += dv
            else:
                dq, dk, dv = self._block_backward(
                    self.dout_back, self.q_back, k, v, self.out_back, self.lse_back,
                    q_is_half=True, k_is_half=False, causal=False,
                )
                self.dq_accum[self.back_index] += dq
                self.dk_parts[owner] += dk
                self.dv_parts[owner] += dv
        return (
            self.dq_accum.to(self.q.dtype),
            self.dk_parts[self.cp_rank].to(self.k.dtype),
            self.dv_parts[self.cp_rank].to(self.v.dtype),
        )

    def reduce_dkv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble owner gradients for validation outside the timed interval."""
        dist.all_reduce(self.dk_parts, group=self.process_group)
        dist.all_reduce(self.dv_parts, group=self.process_group)
        return self.dk_parts[self.cp_rank], self.dv_parts[self.cp_rank]


class _OneStepSample(_BlockBackend):
    """One varlen FA3 call for both zigzag Q halves and their causal prefixes."""

    def __init__(self, source: VarlenFa3RingBackward, kv: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        super().__init__(source.backend)
        self.process_group = source.process_group
        self.cp_rank = dist.get_rank(self.process_group)
        self.cp_size = len(kv)
        self.q, self.dout = source.q, source.dout
        self.half = self.q.size(0) // 2
        self.front_length = (self.cp_rank + 1) * self.half
        self.back_length = (2 * self.cp_size - self.cp_rank) * self.half

        def pack(index: int) -> torch.Tensor:
            logical = torch.cat(
                [part[index][:self.half] for part in kv]
                + [part[index][self.half:] for part in reversed(kv)], dim=0,
            )
            return torch.cat((logical[:self.front_length], logical[:self.back_length]), dim=0)

        self.k, self.v = pack(0), pack(1)
        self.cu_q_host = torch.tensor([0, self.half, 2 * self.half], dtype=torch.int32)
        self.cu_k_host = torch.tensor(
            [0, self.front_length, self.front_length + self.back_length], dtype=torch.int32,
        )
        self.cu_q = self.cu_q_host.to(self.q.device)
        self.cu_k = self.cu_k_host.to(self.q.device)
        self.dq = torch.empty_like(self.q)
        self.dk = torch.empty_like(self.k)
        self.dv = torch.empty_like(self.v)
        shape = (self.cp_size * self.q.size(0), *source.k.shape[1:])
        self.dk_global = torch.zeros(shape, device=self.q.device, dtype=torch.float32)
        self.dv_global = torch.zeros_like(self.dk_global)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.out, self.lse = self.forward_block(
            self.q, self.k, self.v, self.cu_q, self.cu_k,
            self.cu_q_host, self.cu_k_host, self.half, self.back_length, True,
        )
        return self.out, self.lse

    def prepare_backward(self) -> None:
        self.dk_global.zero_()
        self.dv_global.zero_()

    def _owner_part(self, tensor: torch.Tensor) -> torch.Tensor:
        front = self.cp_rank * self.half
        back = (2 * self.cp_size - 1 - self.cp_rank) * self.half
        return torch.cat((tensor[front:front + self.half], tensor[back:back + self.half]), dim=0)

    def backward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dq, dk, dv = self.backward_block(
            self.dout, self.q, self.k, self.v, self.out, self.lse,
            self.cu_q, self.cu_k, self.half, self.back_length, True,
            self.dq, self.dk, self.dv,
        )
        # The two causal prefixes share keys; combine their dKV locally.
        for packed, logical in ((dk, self.dk_global), (dv, self.dv_global)):
            logical[:self.back_length].copy_(packed[self.front_length:])
            logical[:self.front_length] += packed[:self.front_length]
        return dq, self._owner_part(self.dk_global).to(dk.dtype), self._owner_part(self.dv_global).to(dv.dtype)

    def reduce_dkv(self) -> tuple[torch.Tensor, torch.Tensor]:
        dist.all_reduce(self.dk_global, group=self.process_group)
        dist.all_reduce(self.dv_global, group=self.process_group)
        return self._owner_part(self.dk_global), self._owner_part(self.dv_global)


class MegatronHybridCPComputeAttention(MegatronHybridCPAttention):
    """Keep the Megatron schedule while removing KV and dKV communication.

    Backward retains this rank's contributions for every KV owner. Call
    ``reduce_dkv_for_check`` only outside timing to assemble complete gradients.
    """

    def __init__(self, *args, one_step: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.dout is None or not self.is_causal:
            raise ValueError("motivation controls require causal forward and backward")
        for group_index, group in enumerate(self.plan.execution_groups):
            for sample_id in group.sample_ids_by_rank[self.rank]:
                assignment = self.plan.assignment(sample_id)
                if assignment.cp_size == 1:
                    continue
                source = self._runners[sample_id].runner
                parts_k = [torch.empty_like(source.k) for _ in assignment.ranks]
                parts_v = [torch.empty_like(source.v) for _ in assignment.ranks]
                dist.all_gather(parts_k, source.k, group=source.process_group)
                dist.all_gather(parts_v, source.v, group=source.process_group)
                runner_type = _OneStepSample if one_step else _ComputeOnlyRing
                self._runners[sample_id] = runner_type(source, list(zip(parts_k, parts_v)))
            self._barrier_after_group(group_index)

    def prepare_backward(self) -> None:
        """Reset resident partial-gradient buffers before starting the timer."""
        for runner in self._runners.values():
            if isinstance(runner, (_ComputeOnlyRing, _OneStepSample)):
                runner.prepare_backward()

    def reduce_dkv_for_check(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for group_index, group in enumerate(self.plan.execution_groups):
            for sample_id in group.sample_ids_by_rank[self.rank]:
                runner = self._runners[sample_id]
                if self.plan.assignment(sample_id).cp_size == 1:
                    continue
                dk, dv = runner.reduce_dkv()
                token_slice = self.sample_slices[sample_id]
                self.dk[token_slice].copy_(dk)
                self.dv[token_slice].copy_(dv)
            self._barrier_after_group(group_index)
        return self.dq, self.dk, self.dv
