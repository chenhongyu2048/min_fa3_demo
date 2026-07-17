"""Distributed correctness test for causal mega-ring varlen backward."""

import argparse
import os

import torch
import torch.distributed as dist

import min_fa3_op
from mega_ring_test_min_fa3_varlen_ring_multi_rank import (
    assert_close_named,
    gather_rank_blocks,
    make_cu_seqlens,
    expected_loaded_row_mask,
    parse_seqlen_spec,
    raise_if_any_rank_failed,
    reference_mega_ring_varlen,
)


def init_distributed() -> tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    if dist.get_world_size() != local_world_size:
        raise RuntimeError("mega-ring backward test is single-node only")
    return local_rank, local_world_size


def make_centered_rank_local_qkv(
    total_q: int,
    total_k: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Keep logits away from saturated softmax regions so the dense FP32
    # autograd reference is a useful numerical sanity check for BF16 FA3.
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260711 + rank)
    q = (
        torch.randn(
            total_q, q_heads, head_dim,
            device="cuda", dtype=torch.float32, generator=generator,
        )
        * 0.25
    ).to(torch.bfloat16).contiguous()
    k = (
        torch.randn(
            total_k, kv_heads, head_dim,
            device="cuda", dtype=torch.float32, generator=generator,
        )
        * 0.25
    ).to(torch.bfloat16).contiguous()
    v = (
        torch.randn(
            total_k, kv_heads, head_dim,
            device="cuda", dtype=torch.float32, generator=generator,
        )
        * 0.25
    ).to(torch.bfloat16).contiguous()
    return q, k, v


def dense_error_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    actual_f = actual.float()
    reference_f = reference.float()
    diff = (actual_f - reference_f).abs()
    rmse = diff.square().mean().sqrt()
    reference_rms = reference_f.square().mean().sqrt()
    tolerance_scale = reference_rms.clamp_min(atol)
    tolerance = atol + rtol * reference_f.abs()
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(rmse),
        "reference_rms": float(reference_rms),
        # Raw relative RMSE is diagnostic only: it is ill-conditioned when the
        # reference gradient is near zero.
        "relative_rmse": float(rmse / reference_rms.clamp_min(1e-8)),
        "tolerance_normalized_rmse": float(rmse / tolerance_scale),
        "outside_tolerance": float((diff > tolerance).float().mean()),
    }


def assert_dense_sanity(
    name: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    metrics = dense_error_metrics(actual, reference, atol=atol, rtol=rtol)
    # Dense FP32 autograd and BF16/online-softmax FA3 do not use identical
    # forward probabilities. Causal dV amplifies small probability differences
    # for early keys, so judge the dense comparison by both global error and a
    # tightly bounded tail instead of requiring every element to pass.
    limits = {
        "max_abs": 2.0,
        "tolerance_normalized_rmse": 0.1,
        "outside_tolerance": 1.0e-3,
    }
    failed = {
        key: (metrics[key], limit)
        for key, limit in limits.items()
        if metrics[key] > limit
    }
    if failed:
        raise AssertionError(
            f"{name} dense sanity failed: metrics={metrics}, limits={limits}, failed={failed}"
        )
    return metrics


def pack_half(
    tensor: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    *,
    front: bool,
) -> torch.Tensor:
    pieces = []
    for batch_idx in range(cu_seqlens_host.numel() - 1):
        start = int(cu_seqlens_host[batch_idx])
        end = int(cu_seqlens_host[batch_idx + 1])
        middle = start + (end - start) // 2
        pieces.append(tensor[start:middle] if front else tensor[middle:end])
    return torch.cat(pieces, dim=0).contiguous()


def pack_half_lse(
    lse: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    *,
    front: bool,
) -> torch.Tensor:
    pieces = []
    for batch_idx in range(cu_seqlens_host.numel() - 1):
        start = int(cu_seqlens_host[batch_idx])
        end = int(cu_seqlens_host[batch_idx + 1])
        middle = start + (end - start) // 2
        pieces.append(lse[:, start:middle] if front else lse[:, middle:end])
    return torch.cat(pieces, dim=1).contiguous()


def add_packed_half(
    destination: torch.Tensor,
    packed: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    *,
    front: bool,
) -> None:
    packed_offset = 0
    for batch_idx in range(cu_seqlens_host.numel() - 1):
        start = int(cu_seqlens_host[batch_idx])
        end = int(cu_seqlens_host[batch_idx + 1])
        middle = start + (end - start) // 2
        dst_start, dst_end = (start, middle) if front else (middle, end)
        half_len = dst_end - dst_start
        destination[dst_start:dst_end] += packed[packed_offset : packed_offset + half_len].float()
        packed_offset += half_len


def stepwise_min_fa3_grads(
    dout: torch.Tensor,
    q: torch.Tensor,
    all_k: torch.Tensor,
    all_v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_full: torch.Tensor,
    cu_full_host: torch.Tensor,
    cu_half: torch.Tensor,
    max_seqlen: int,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_tokens = q.size(0)
    dq_sum = torch.zeros_like(q, dtype=torch.float32)
    dk_by_owner = torch.zeros(
        world_size, total_tokens, all_k.size(1), all_k.size(2),
        device=q.device, dtype=torch.float32,
    )
    dv_by_owner = torch.zeros_like(dk_by_owner)
    q_back = pack_half(q, cu_full_host, front=False)
    dout_back = pack_half(dout, cu_full_host, front=False)
    out_back = pack_half(out, cu_full_host, front=False)
    lse_back = pack_half_lse(lse, cu_full_host, front=False)

    for step in range(world_size):
        owner = (rank - step) % world_size
        owner_start = owner * total_tokens
        owner_k = all_k[owner_start : owner_start + total_tokens]
        owner_v = all_v[owner_start : owner_start + total_tokens]
        if step == 0:
            dq_step, dk_step, dv_step = min_fa3_op.backward_varlen(
                dout, q, owner_k, owner_v, out, lse,
                cu_full, cu_full, max_seqlen, max_seqlen, True,
            )
            dq_sum += dq_step.float()
            dk_by_owner[owner] += dk_step.float()
            dv_by_owner[owner] += dv_step.float()
        elif step <= rank:
            k_front = pack_half(owner_k, cu_full_host, front=True)
            v_front = pack_half(owner_v, cu_full_host, front=True)
            dq_step, dk_step, dv_step = min_fa3_op.backward_varlen(
                dout, q, k_front, v_front, out, lse,
                cu_full, cu_half, max_seqlen, max_seqlen // 2, False,
            )
            dq_sum += dq_step.float()
            add_packed_half(dk_by_owner[owner], dk_step, cu_full_host, front=True)
            add_packed_half(dv_by_owner[owner], dv_step, cu_full_host, front=True)
        else:
            dq_step, dk_step, dv_step = min_fa3_op.backward_varlen(
                dout_back, q_back, owner_k, owner_v, out_back, lse_back,
                cu_half, cu_full, max_seqlen // 2, max_seqlen, False,
            )
            add_packed_half(dq_sum, dq_step, cu_full_host, front=False)
            dk_by_owner[owner] += dk_step.float()
            dv_by_owner[owner] += dv_step.float()

    # Every query rank contributes to every KV owner. This all-reduce matches
    # the source-rank remote bulk-reduce performed by the fused kernel.
    dist.all_reduce(dk_by_owner)
    dist.all_reduce(dv_by_owner)
    return dq_sum, dk_by_owner[rank], dv_by_owner[rank]


def run_case(
    args: argparse.Namespace,
    seqlen: int,
    rank: int,
    world_size: int,
) -> None:
    if seqlen % 256 != 0:
        raise ValueError("causal zigzag requires seqlen / 2 to be 128-aligned")
    total_tokens = args.b * seqlen
    device = torch.device("cuda")
    cu_q, cu_q_host = make_cu_seqlens(args.b, seqlen, device)
    cu_k, cu_k_host = cu_q, cu_q_host
    half_cu, _ = make_cu_seqlens(args.b, seqlen // 2, device)
    global_seqlens_host = torch.full(
        (args.b,), seqlen * world_size, dtype=torch.int32
    )
    ring_sizes_host = torch.full((args.b,), world_size, dtype=torch.int32)
    ring_starts_host = torch.zeros((args.b,), dtype=torch.int32)
    q, local_k, local_v = make_centered_rank_local_qkv(
        total_tokens, total_tokens, args.qhead, args.kvhead, args.headdim, rank
    )
    expected_k = gather_rank_blocks(local_k, rank, world_size)
    expected_v = gather_rank_blocks(local_v, rank, world_size)

    remote_k = min_fa3_op.TKParallelTensor(
        list(expected_k.shape), torch.bfloat16, rank, world_size, False
    )
    remote_v = min_fa3_op.TKParallelTensor(
        list(expected_v.shape), torch.bfloat16, rank, world_size, False
    )
    k = remote_k.data_
    v = remote_v.data_
    k.zero_()
    v.zero_()
    start = rank * total_tokens
    k[start : start + total_tokens].copy_(local_k)
    v[start : start + total_tokens].copy_(local_v)
    torch.cuda.synchronize()
    dist.barrier()

    if world_size == 1:
        out, lse = min_fa3_op.forward_varlen(
            q, k, v, cu_q, cu_k, seqlen, seqlen, True,
            cu_seqlens_q_host=cu_q_host,
            cu_seqlens_k_host=cu_k_host,
            return_lse=True,
        )
    else:
        out, lse = min_fa3_op.forward_varlen_mega_ring(
            q,
            k,
            v,
            cu_q,
            cu_k,
            seqlen,
            seqlen,
            True,
            cu_seqlens_q_host=cu_q_host,
            cu_seqlens_k_host=cu_k_host,
            remote_k=remote_k,
            remote_v=remote_v,
            num_comp_sm=args.num_comp_sm,
            num_comm_sm=args.num_comm_sm,
            global_seqlens_host=global_seqlens_host,
            ring_sizes_host=ring_sizes_host,
            ring_starts_host=ring_starts_host,
            return_lse=True,
        )
    dout_generator = torch.Generator(device="cuda")
    dout_generator.manual_seed(20261711 + rank)
    dout = (
        torch.randn(
            q.shape, device=q.device, dtype=torch.float32,
            generator=dout_generator,
        )
        * 0.25
    ).to(torch.bfloat16).contiguous()

    total_k_padded = ((total_tokens + args.b * 128 + 127) // 128) * 128
    accum_numel = args.kvhead * total_k_padded * 128
    remote_dk = min_fa3_op.TKParallelTensor(
        [accum_numel], torch.float32, rank, world_size, False
    )
    remote_dv = min_fa3_op.TKParallelTensor(
        [accum_numel], torch.float32, rank, world_size, False
    )
    remote_completion = min_fa3_op.TKParallelTensor(
        [1], torch.int32, rank, world_size, False
    )
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
        cu_q,
        cu_k,
        seqlen,
        seqlen,
        cu_seqlens_q_host=cu_q_host,
        cu_seqlens_k_host=cu_k_host,
        remote_k=remote_k,
        remote_v=remote_v,
        remote_dk_accum=remote_dk,
        remote_dv_accum=remote_dv,
        remote_dkv_completion=remote_completion,
        num_comp_sm=args.num_comp_sm,
        num_comm_sm=args.num_comm_sm,
        global_seqlens_host=global_seqlens_host,
        ring_sizes_host=ring_sizes_host,
        ring_starts_host=ring_starts_host,
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = expected_k.detach().clone().requires_grad_(True)
    v_ref = expected_v.detach().clone().requires_grad_(True)
    out_ref = reference_mega_ring_varlen(
        q_ref, k_ref, v_ref, cu_q_host, cu_k_host, True, rank, world_size
    )
    dq_ref, dk_all_ref, dv_all_ref = torch.autograd.grad(
        out_ref, (q_ref, k_ref, v_ref), dout
    )
    dq_stepwise, dk_stepwise, dv_stepwise = stepwise_min_fa3_grads(
        dout, q, expected_k, expected_v, out, lse,
        cu_q, cu_q_host, half_cu, seqlen, rank, world_size,
    )
    # Preserve the owner dimension across the collective. Slicing rank-local
    # owners before all-reduce would add owner 0 from rank 0 to owner 1 from
    # rank 1 at the same tensor positions.
    dk_ref_by_owner = dk_all_ref.reshape(
        world_size, total_tokens, args.kvhead, args.headdim
    ).contiguous()
    dv_ref_by_owner = dv_all_ref.reshape_as(dk_ref_by_owner).contiguous()
    dist.all_reduce(dk_ref_by_owner)
    dist.all_reduce(dv_ref_by_owner)
    dk_ref = dk_ref_by_owner[rank]
    dv_ref = dv_ref_by_owner[rank]
    loaded_rows = expected_loaded_row_mask(
        args.b, seqlen, True, rank, world_size, k.device
    )
    expected_loaded_k = torch.zeros_like(expected_k)
    expected_loaded_v = torch.zeros_like(expected_v)
    expected_loaded_k[loaded_rows] = expected_k[loaded_rows]
    expected_loaded_v[loaded_rows] = expected_v[loaded_rows]

    local_error = None
    try:
        assert_close_named("forward output", out.float(), out_ref.float(), atol=0.2, rtol=0.2)
        assert_close_named("backward-loaded K", k.float(), expected_loaded_k.float(), atol=0.0, rtol=0.0)
        assert_close_named("backward-loaded V", v.float(), expected_loaded_v.float(), atol=0.0, rtol=0.0)
        assert_close_named("dQ vs stepwise min_fa3", dq.float(), dq_stepwise, atol=0.3, rtol=0.3)
        assert_close_named("dK vs stepwise min_fa3", dk.float(), dk_stepwise, atol=0.3, rtol=0.3)
        assert_close_named("dV vs stepwise min_fa3", dv.float(), dv_stepwise, atol=0.3, rtol=0.3)
        dense_metrics = {
            "dQ": assert_dense_sanity("dQ", dq, dq_ref, atol=args.atol, rtol=args.rtol),
            "dK": assert_dense_sanity("dK", dk, dk_ref, atol=args.atol, rtol=args.rtol),
            "dV": assert_dense_sanity("dV", dv, dv_ref, atol=args.atol, rtol=args.rtol),
        }
        if remote_completion.data_.item() != world_size:
            raise AssertionError(
                f"owner completion={remote_completion.data_.item()}, expected {world_size}"
            )
    except AssertionError as exc:
        local_error = f"rank={rank}, world_size={world_size}\n{exc}"
    raise_if_any_rank_failed(local_error, rank)
    dense_metrics_by_rank: list[dict[str, dict[str, float]] | None] = [None] * world_size
    dist.all_gather_object(dense_metrics_by_rank, dense_metrics)
    if rank == 0:
        print(
            f"mega-ring varlen backward: ok (world_size={world_size}, B={args.b}, "
            f"S={seqlen}, QH={args.qhead}, KVH={args.kvhead}, "
            f"dense_metrics_by_rank={dense_metrics_by_rank})",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=2)
    parser.add_argument(
        "--seqlen",
        "--seqlens",
        dest="seqlen",
        type=str,
        default="256",
        help="comma-separated sequence lengths",
    )
    parser.add_argument("--qhead", type=int, default=16)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--num-comp-sm", type=int, default=64)
    parser.add_argument("--num-comm-sm", type=int, default=8)
    parser.add_argument("--atol", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    seqlen_cases = parse_seqlen_spec(parsed.seqlen)
    invalid_seqlens = [seqlen for seqlen in seqlen_cases if seqlen % 256 != 0]
    if invalid_seqlens:
        raise SystemExit(
            "causal zigzag requires every seqlen / 2 to be 128-aligned; "
            f"invalid lengths: {invalid_seqlens}"
        )
    local_rank, local_world_size = init_distributed()
    try:
        for seqlen in seqlen_cases:
            run_case(parsed, seqlen, local_rank, local_world_size)
    finally:
        dist.destroy_process_group()
