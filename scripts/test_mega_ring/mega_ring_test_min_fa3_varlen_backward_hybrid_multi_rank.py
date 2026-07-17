"""Distributed correctness test for hierarchical causal mega-ring backward."""

import argparse

import torch
import torch.distributed as dist

import min_fa3_op
from mega_ring_test_min_fa3_varlen_hybrid_multi_rank import (
    SENTINEL,
    assert_all_ranks,
    close_error,
    expected_loaded_mask,
    hierarchical_reference,
    init_distributed,
    local_lengths_for_rank,
    make_cu_seqlens,
    make_local_qkv,
    parse_int_list,
)


def make_padded_accum_reference(
    gradient: torch.Tensor,
    cu_host: torch.Tensor,
    kv_heads: int,
    padded_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = torch.zeros(
        (kv_heads, padded_capacity, 128), device=gradient.device, dtype=torch.float32
    )
    active = torch.zeros((padded_capacity,), device=gradient.device, dtype=torch.bool)
    for batch_idx in range(cu_host.numel() - 1):
        src_begin = int(cu_host[batch_idx])
        src_end = int(cu_host[batch_idx + 1])
        dst_begin = src_begin + batch_idx * 128
        dst_end = dst_begin + src_end - src_begin
        if dst_end > dst_begin:
            reference[:, dst_begin:dst_end].copy_(
                gradient[src_begin:src_end].float().permute(1, 0, 2)
            )
            active[dst_begin:dst_end] = True
    return reference, active


def expected_owner_completion(
    local_lengths: list[int], ring_sizes: list[int]
) -> int:
    return sum(
        ring_size
        for ring_size in (8, 4, 2, 1)
        if any(
            length > 0 and batch_ring_size == ring_size
            for length, batch_ring_size in zip(local_lengths, ring_sizes)
        )
    )


def run_case(args: argparse.Namespace, rank: int, world_size: int) -> None:
    global_lengths = parse_int_list(args.global_seqlens, "--global-seqlens")
    ring_sizes = parse_int_list(args.ring_sizes, "--ring-sizes")
    ring_starts = parse_int_list(args.ring_starts, "--ring-starts")
    if not (len(global_lengths) == len(ring_sizes) == len(ring_starts)):
        raise SystemExit("global lengths, ring sizes, and ring starts must match")

    all_rank_lengths = [
        local_lengths_for_rank(global_lengths, ring_sizes, ring_starts, source_rank)
        for source_rank in range(world_size)
    ]
    local_lengths = all_rank_lengths[rank]
    local_total = sum(local_lengths)
    device = torch.device("cuda", rank)
    cu, cu_host = make_cu_seqlens(local_lengths, device)
    q, local_k, local_v = make_local_qkv(
        local_total, args.qhead, args.kvhead, args.headdim, rank, True, device
    )

    capacity_tensor = torch.tensor([local_total], device=device, dtype=torch.int32)
    dist.all_reduce(capacity_tensor, op=dist.ReduceOp.MAX)
    rank_capacity = ((int(capacity_tensor.item()) + 127) // 128) * 128
    padded_k = torch.full(
        (rank_capacity, args.kvhead, args.headdim),
        SENTINEL,
        device=device,
        dtype=torch.bfloat16,
    )
    padded_v = torch.full_like(padded_k, SENTINEL)
    padded_k[:local_total].copy_(local_k)
    padded_v[:local_total].copy_(local_v)
    gathered_k_parts = [torch.empty_like(padded_k) for _ in range(world_size)]
    gathered_v_parts = [torch.empty_like(padded_v) for _ in range(world_size)]
    dist.all_gather(gathered_k_parts, padded_k)
    dist.all_gather(gathered_v_parts, padded_v)
    gathered_k = torch.stack(gathered_k_parts)
    gathered_v = torch.stack(gathered_v_parts)

    arena_shape = [world_size * rank_capacity, args.kvhead, args.headdim]
    remote_k = min_fa3_op.TKParallelTensor(
        arena_shape, torch.bfloat16, rank, world_size, False
    )
    remote_v = min_fa3_op.TKParallelTensor(
        arena_shape, torch.bfloat16, rank, world_size, False
    )
    k, v = remote_k.data_, remote_v.data_
    k.fill_(SENTINEL)
    v.fill_(SENTINEL)
    owner_begin = rank * rank_capacity
    k[owner_begin : owner_begin + local_total].copy_(local_k)
    v[owner_begin : owner_begin + local_total].copy_(local_v)

    global_host = torch.tensor(global_lengths, dtype=torch.int32)
    ring_sizes_host = torch.tensor(ring_sizes, dtype=torch.int32)
    ring_starts_host = torch.tensor(ring_starts, dtype=torch.int32)
    max_local_len = max(max(lengths) for lengths in all_rank_lengths)
    torch.cuda.synchronize()
    dist.barrier()

    out, lse = min_fa3_op.forward_varlen_mega_ring(
        q,
        k,
        v,
        cu,
        cu,
        max_local_len,
        max_local_len,
        True,
        cu_seqlens_q_host=cu_host,
        cu_seqlens_k_host=cu_host,
        remote_k=remote_k,
        remote_v=remote_v,
        num_comp_sm=args.num_comp_sm,
        num_comm_sm=args.num_comm_sm,
        global_seqlens_host=global_host,
        ring_sizes_host=ring_sizes_host,
        ring_starts_host=ring_starts_host,
        return_lse=True,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(20260715 + rank)
    dout = (
        torch.randn(q.shape, device=device, dtype=torch.float32, generator=generator) * 0.25
    ).to(torch.bfloat16).contiguous()

    q_ref = q.detach().clone().requires_grad_(True)
    gathered_k_ref = gathered_k.detach().clone().requires_grad_(True)
    gathered_v_ref = gathered_v.detach().clone().requires_grad_(True)
    out_ref, lse_ref = hierarchical_reference(
        q_ref,
        gathered_k_ref,
        gathered_v_ref,
        all_rank_lengths,
        cu_host,
        global_lengths,
        ring_sizes,
        ring_starts,
        rank,
        True,
    )
    if local_total > 0:
        dq_ref, dk_ref_all, dv_ref_all = torch.autograd.grad(
            out_ref, (q_ref, gathered_k_ref, gathered_v_ref), dout
        )
    else:
        dq_ref = torch.zeros_like(q_ref)
        dk_ref_all = torch.zeros_like(gathered_k_ref)
        dv_ref_all = torch.zeros_like(gathered_v_ref)
    dist.all_reduce(dk_ref_all)
    dist.all_reduce(dv_ref_all)
    dk_ref = dk_ref_all[rank, :local_total]
    dv_ref = dv_ref_all[rank, :local_total]

    padded_rank_capacity = rank_capacity + len(global_lengths) * 128
    accum_numel = args.kvhead * padded_rank_capacity * 128
    remote_dk = min_fa3_op.TKParallelTensor(
        [accum_numel], torch.float32, rank, world_size, False
    )
    remote_dv = min_fa3_op.TKParallelTensor(
        [accum_numel], torch.float32, rank, world_size, False
    )
    remote_completion = min_fa3_op.TKParallelTensor(
        [1], torch.int32, rank, world_size, False
    )

    dq = dk = dv = None
    for _ in range(args.repeat):
        remote_dk.data_.zero_()
        remote_dv.data_.zero_()
        remote_completion.data_.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        dq, dk, dv = min_fa3_op.backward_varlen_mega_ring(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu,
            cu,
            max_local_len,
            max_local_len,
            cu_seqlens_q_host=cu_host,
            cu_seqlens_k_host=cu_host,
            remote_k=remote_k,
            remote_v=remote_v,
            remote_dk_accum=remote_dk,
            remote_dv_accum=remote_dv,
            remote_dkv_completion=remote_completion,
            num_comp_sm=args.num_comp_sm,
            num_comm_sm=args.num_comm_sm,
            global_seqlens_host=global_host,
            ring_sizes_host=ring_sizes_host,
            ring_starts_host=ring_starts_host,
        )

    _, active = make_padded_accum_reference(
        dk, cu_host, args.kvhead, padded_rank_capacity
    )
    dk_accum = remote_dk.data_.view(args.kvhead, padded_rank_capacity, 128)
    dv_accum = remote_dv.data_.view_as(dk_accum)
    padding = ~active
    errors = [
        close_error("forward output", out.float(), out_ref.float(), atol=0.2, rtol=0.2),
        close_error("forward LSE", lse, lse_ref, atol=0.2, rtol=0.2),
        close_error("dQ", dq.float(), dq_ref.float(), atol=1.0, rtol=0.2),
        close_error("dK", dk.float(), dk_ref.float(), atol=0.5, rtol=0.2),
        close_error("dV", dv.float(), dv_ref.float(), atol=0.5, rtol=0.2),
        close_error("dK accumulator padding", dk_accum[:, padding], torch.zeros_like(dk_accum[:, padding]), atol=0.0, rtol=0.0),
        close_error("dV accumulator padding", dv_accum[:, padding], torch.zeros_like(dv_accum[:, padding]), atol=0.0, rtol=0.0),
    ]
    if active.any():
        if not torch.isfinite(dk_accum[:, active]).all() or not torch.isfinite(dv_accum[:, active]).all():
            errors.append("FP32 owner accumulator active rows contain non-finite values")
        if torch.count_nonzero(dk_accum[:, active]) == 0 or torch.count_nonzero(dv_accum[:, active]) == 0:
            errors.append("FP32 owner accumulator active rows were not reduced")
    loaded = expected_loaded_mask(
        all_rank_lengths,
        rank_capacity,
        global_lengths,
        ring_sizes,
        ring_starts,
        rank,
        True,
        device,
    )
    expected_arena_k = gathered_k.reshape_as(k)
    expected_arena_v = gathered_v.reshape_as(v)
    errors.extend(
        (
            close_error("backward-loaded K", k[loaded], expected_arena_k[loaded], atol=0.0, rtol=0.0),
            close_error("backward-loaded V", v[loaded], expected_arena_v[loaded], atol=0.0, rtol=0.0),
        )
    )
    completion = int(remote_completion.data_.item())
    completion_expected = expected_owner_completion(local_lengths, ring_sizes)
    if completion != completion_expected:
        errors.append(f"completion={completion}, expected={completion_expected}")

    error_details = [error for error in errors if error is not None]
    local_error = None
    if error_details:
        local_error = (
            f"rank={rank}, local_lengths={local_lengths}, completion={completion}\n"
            + "\n\n".join(error_details)
        )
    assert_all_ranks(local_error)
    if rank == 0:
        print(
            "hierarchical mega-ring backward: ok "
            f"(global={global_lengths}, rings={ring_sizes}, starts={ring_starts}, "
            f"QH={args.qhead}, KVH={args.kvhead}, repeat={args.repeat})",
            flush=True,
        )
    dist.barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hierarchical hybrid mega-ring backward checks")
    parser.add_argument("--global-seqlens", default="2048,1024,512,256")
    parser.add_argument("--ring-sizes", default="8,4,2,1")
    parser.add_argument("--ring-starts", default="0,4,2,7")
    parser.add_argument("--qhead", type=int, default=16)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--num-comp-sm", type=int, default=100)
    parser.add_argument("--num-comm-sm", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.headdim != 128 or args.kvhead * args.headdim != 1024:
        raise SystemExit("This path requires D=128 and KVH * D == 1024")
    if args.qhead % args.kvhead:
        raise SystemExit("qhead must be divisible by kvhead")
    rank, world_size = init_distributed()
    if world_size != 8:
        raise SystemExit("The default hybrid coverage requires 8 local ranks")
    try:
        run_case(args, rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
