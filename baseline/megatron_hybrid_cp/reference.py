"""Dense correctness references for the standalone hybrid-CP layout."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .attention import MegatronHybridCPAttention


def _gather_packed(tensor: torch.Tensor, world_group: dist.ProcessGroup) -> torch.Tensor:
    local_size = torch.tensor([tensor.size(0)], device=tensor.device, dtype=torch.int64)
    dist.all_reduce(local_size, op=dist.ReduceOp.MAX, group=world_group)
    capacity = int(local_size.item())
    padded = torch.zeros(
        (capacity, tensor.size(1), tensor.size(2)),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    padded[: tensor.size(0)].copy_(tensor)
    parts = [torch.empty_like(padded) for _ in range(dist.get_world_size(world_group))]
    dist.all_gather(parts, padded, group=world_group)
    return torch.stack(parts)


def _rank_slices(runner: MegatronHybridCPAttention) -> list[dict[int, slice]]:
    result: list[dict[int, slice]] = []
    for rank in range(runner.plan.world_size):
        offset = 0
        slices: dict[int, slice] = {}
        for sample_id in runner.plan.sample_ids_for_rank(rank):
            assignment = runner.plan.assignment(sample_id)
            local_length = assignment.global_length // assignment.cp_size
            slices[sample_id] = slice(offset, offset + local_length)
            offset += local_length
        result.append(slices)
    return result


def _logical_tensor(
    gathered: torch.Tensor,
    sample_id: int,
    runner: MegatronHybridCPAttention,
    slices: list[dict[int, slice]],
) -> torch.Tensor:
    assignment = runner.plan.assignment(sample_id)
    parts = [
        gathered[rank, slices[rank][sample_id]] for rank in assignment.ranks
    ]
    if not runner.is_causal or assignment.cp_size == 1:
        return torch.cat(parts, dim=0)
    half = parts[0].size(0) // 2
    return torch.cat(
        [part[:half] for part in parts]
        + [part[half:] for part in reversed(parts)],
        dim=0,
    )


def _local_from_logical(
    logical: torch.Tensor,
    sample_id: int,
    runner: MegatronHybridCPAttention,
) -> torch.Tensor:
    assignment = runner.plan.assignment(sample_id)
    if not runner.is_causal or assignment.cp_size == 1:
        local_length = assignment.global_length // assignment.cp_size
        subgroup_rank = runner.rank - assignment.rank_start
        begin = subgroup_rank * local_length
        return logical[begin : begin + local_length]
    half = assignment.global_length // assignment.cp_size // 2
    subgroup_rank = runner.rank - assignment.rank_start
    front_begin = subgroup_rank * half
    back_begin = (2 * assignment.cp_size - 1 - subgroup_rank) * half
    return torch.cat(
        (
            logical[front_begin : front_begin + half],
            logical[back_begin : back_begin + half],
        ),
        dim=0,
    )


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    repeat = q.size(1) // k.size(1)
    expanded_k = k.repeat_interleave(repeat, dim=1) if repeat != 1 else k
    expanded_v = v.repeat_interleave(repeat, dim=1) if repeat != 1 else v
    scores = torch.einsum("qhd,khd->hqk", q, expanded_k) * (q.size(-1) ** -0.5)
    if is_causal:
        mask = torch.ones(
            (q.size(0), k.size(0)), device=q.device, dtype=torch.bool
        ).tril()
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), expanded_v)
    return out, lse


def forward_reference(
    runner: MegatronHybridCPAttention,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_group = runner.process_groups.world_group
    gathered_q = _gather_packed(runner.q, world_group)
    gathered_k = _gather_packed(runner.k, world_group)
    gathered_v = _gather_packed(runner.v, world_group)
    slices = _rank_slices(runner)
    expected_out = torch.empty_like(runner.q)
    expected_lse = torch.empty_like(runner.lse)
    for sample_id in runner.sample_ids:
        q = _logical_tensor(gathered_q, sample_id, runner, slices).float()
        k = _logical_tensor(gathered_k, sample_id, runner, slices).float()
        v = _logical_tensor(gathered_v, sample_id, runner, slices).float()
        out, lse = _attention(q, k, v, runner.is_causal)
        token_slice = runner.sample_slices[sample_id]
        expected_out[token_slice].copy_(
            _local_from_logical(out, sample_id, runner).to(torch.bfloat16)
        )
        local_lse = _local_from_logical(
            lse.transpose(0, 1), sample_id, runner
        ).transpose(0, 1)
        expected_lse[:, token_slice].copy_(local_lse)
    return expected_out, expected_lse


def backward_reference(
    runner: MegatronHybridCPAttention,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if runner.dout is None:
        raise ValueError("backward reference requires dO")
    world_group = runner.process_groups.world_group
    gathered_q = _gather_packed(runner.q, world_group)
    gathered_k = _gather_packed(runner.k, world_group)
    gathered_v = _gather_packed(runner.v, world_group)
    gathered_dout = _gather_packed(runner.dout, world_group)
    slices = _rank_slices(runner)
    expected_dq = torch.empty_like(runner.q)
    expected_dk = torch.empty_like(runner.k)
    expected_dv = torch.empty_like(runner.v)
    with torch.enable_grad():
        for sample_id in runner.sample_ids:
            q = _logical_tensor(gathered_q, sample_id, runner, slices).float()
            k = _logical_tensor(gathered_k, sample_id, runner, slices).float()
            v = _logical_tensor(gathered_v, sample_id, runner, slices).float()
            dout = _logical_tensor(
                gathered_dout, sample_id, runner, slices
            ).float()
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
            out, _ = _attention(q, k, v, True)
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
            token_slice = runner.sample_slices[sample_id]
            expected_dq[token_slice].copy_(
                _local_from_logical(dq, sample_id, runner).to(torch.bfloat16)
            )
            expected_dk[token_slice].copy_(
                _local_from_logical(dk, sample_id, runner).to(torch.bfloat16)
            )
            expected_dv[token_slice].copy_(
                _local_from_logical(dv, sample_id, runner).to(torch.bfloat16)
            )
    return expected_dq, expected_dk, expected_dv


__all__ = ["backward_reference", "forward_reference"]
