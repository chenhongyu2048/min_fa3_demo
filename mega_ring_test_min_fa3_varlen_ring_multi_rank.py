import argparse
import os

import torch
import torch.distributed as dist

import min_fa3_op


def parse_seqlen_spec(spec: str) -> list[int]:
    cases: list[int] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        cases.append(int(token))
    if not cases:
        raise SystemExit("--seqlen must provide at least one case")
    return cases


def init_distributed() -> tuple[int, int]:
    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit("Run this test with torchrun so LOCAL_RANK and LOCAL_WORLD_SIZE are set")

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    if dist.get_world_size() != local_world_size:
        raise SystemExit(
            "ThunderKittens mega-ring attention demo is single-node only: "
            f"world_size={dist.get_world_size()}, local_world_size={local_world_size}"
        )

    return local_rank, local_world_size


def make_cu_seqlens(batch_size: int, seqlen: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.arange(0, (batch_size + 1) * seqlen, seqlen, dtype=torch.int32)
    return host.to(device=device), host


def raise_if_any_rank_failed(local_error: str | None, local_rank: int) -> None:
    failed = torch.tensor([1 if local_error is not None else 0], device="cuda", dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM)
    if failed.item() == 0:
        return
    if local_error is not None:
        raise AssertionError(local_error)
    raise AssertionError(f"another rank failed this case; rank {local_rank} had no local assertion failure")


def reference_mega_ring_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    is_causal: bool,
    local_rank: int,
    local_world_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    batch_size = cu_seqlens_q.numel() - 1
    local_total_k = cu_seqlens_k[-1].item()
    scale = q.size(-1) ** -0.5
    qhead_per_kvhead = q.size(1) // k.size(1)

    for batch_idx in range(batch_size):
        q_start = cu_seqlens_q[batch_idx].item()
        q_end = cu_seqlens_q[batch_idx + 1].item()
        k_start = cu_seqlens_k[batch_idx].item()
        k_end = cu_seqlens_k[batch_idx + 1].item()

        q_i = q[q_start:q_end].float()
        k_blocks = []
        v_blocks = []
        key_positions = []
        for rank_idx in range(local_world_size):
            block_start = rank_idx * local_total_k + k_start
            block_end = rank_idx * local_total_k + k_end
            k_blocks.append(k[block_start:block_end])
            v_blocks.append(v[block_start:block_end])
            key_positions.append(torch.arange(k_start, k_end, device=q.device, dtype=torch.int64) + rank_idx * local_total_k)

        k_i = torch.cat(k_blocks, dim=0).float()
        v_i = torch.cat(v_blocks, dim=0).float()
        key_pos = torch.cat(key_positions, dim=0)
        if qhead_per_kvhead != 1:
            k_i = k_i.repeat_interleave(qhead_per_kvhead, dim=1)
            v_i = v_i.repeat_interleave(qhead_per_kvhead, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * scale
        if is_causal:
            query_pos = torch.arange(q_start, q_end, device=q.device, dtype=torch.int64) + local_rank * local_total_k
            causal_mask = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
            scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probs, v_i).to(dtype=q.dtype))

    return torch.cat(outputs, dim=0)


def make_rank_local_qkv(
    total_q: int,
    total_k: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    local_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(total_q, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    base_k = torch.arange(total_k * kv_heads * head_dim, device="cuda", dtype=torch.float32)
    base_v = torch.arange(total_k * kv_heads * head_dim, device="cuda", dtype=torch.float32)
    k = ((base_k % 16).reshape(total_k, kv_heads, head_dim) + local_rank * 16.0).to(torch.bfloat16).contiguous()
    v = ((base_v % 16).reshape(total_k, kv_heads, head_dim) + local_rank * 16.0 + 7.0).to(torch.bfloat16).contiguous()
    return q, k, v


def gather_rank_blocks(local_tensor: torch.Tensor, local_rank: int, local_world_size: int) -> torch.Tensor:
    blocks = [torch.empty_like(local_tensor) for _ in range(local_world_size)]
    dist.all_gather(blocks, local_tensor)
    blocks[local_rank] = local_tensor
    return torch.cat(blocks, dim=0).contiguous()


def run_case(
    batch_size: int,
    seqlen: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
    num_comp_sm: int,
    num_comm_sm: int,
    local_rank: int,
    local_world_size: int,
) -> None:
    total_tokens = batch_size * seqlen
    cu_seqlens_q, cu_seqlens_q_host = make_cu_seqlens(batch_size, seqlen, torch.device("cuda"))
    cu_seqlens_k, cu_seqlens_k_host = make_cu_seqlens(batch_size, seqlen, torch.device("cuda"))

    q, local_k, local_v = make_rank_local_qkv(
        total_tokens,
        total_tokens,
        q_heads,
        kv_heads,
        head_dim,
        local_rank,
    )
    expected_k = gather_rank_blocks(local_k, local_rank, local_world_size)
    expected_v = gather_rank_blocks(local_v, local_rank, local_world_size)
    local_block_start = local_rank * total_tokens
    local_block_end = local_block_start + total_tokens
    remote_k = min_fa3_op.TKParallelTensor(
        list(expected_k.shape),
        torch.bfloat16,
        local_rank,
        local_world_size,
        False,
    )
    remote_v = min_fa3_op.TKParallelTensor(
        list(expected_v.shape),
        torch.bfloat16,
        local_rank,
        local_world_size,
        False,
    )
    # Use the VMM-backed TK storage as the ordinary K/V tensors too.
    k = remote_k.data_
    v = remote_v.data_
    k.zero_()
    v.zero_()
    k[local_block_start:local_block_end].copy_(local_k)
    v[local_block_start:local_block_end].copy_(local_v)

    torch.cuda.synchronize()
    dist.barrier()
    out = min_fa3_op.forward_varlen_mega_ring(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlen,
        seqlen,
        is_causal,
        cu_seqlens_q_host=cu_seqlens_q_host,
        cu_seqlens_k_host=cu_seqlens_k_host,
        remote_k=remote_k,
        remote_v=remote_v,
        num_comp_sm=num_comp_sm,
        num_comm_sm=num_comm_sm,
    )
    torch.cuda.synchronize()
    dist.barrier()

    ref = reference_mega_ring_varlen(
        q,
        expected_k,
        expected_v,
        cu_seqlens_q_host,
        cu_seqlens_k_host,
        is_causal,
        local_rank,
        local_world_size,
    )
    local_error: str | None = None
    try:
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-1, rtol=2e-1)
        if num_comm_sm > 0:
            torch.testing.assert_close(k.float(), expected_k.float(), atol=0.0, rtol=0.0)
            torch.testing.assert_close(v.float(), expected_v.float(), atol=0.0, rtol=0.0)
    except AssertionError as exc:
        local_error = str(exc)

    raise_if_any_rank_failed(local_error, local_rank)

    if local_rank == 0:
        print(
            f"distributed mega ring varlen case causal={is_causal}: ok "
            f"(world_size={local_world_size}, B={batch_size}, S={seqlen}, "
            f"QH={q_heads}, KVH={kv_heads}, D={head_dim}, "
            f"num_comp_sm={num_comp_sm}, num_comm_sm={num_comm_sm})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the distributed minimal Hopper FA3 varlen mega-ring demo test.")
    parser.add_argument("--b", type=int, default=2, help="Batch size B.")
    parser.add_argument(
        "--seqlen",
        "--seqlens",
        dest="seqlen",
        type=str,
        default="128",
        help="Comma-separated sequence lengths S. Q and each rank-local K/V block all use the same S.",
    )
    parser.add_argument("--qhead", type=int, default=16, help="Number of query/output heads")
    parser.add_argument("--kvhead", type=int, default=8, help="Number of key/value heads")
    parser.add_argument("--headdim", type=int, default=128, help="Head dimension D")
    parser.add_argument("--num-comp-sm", type=int, default=1, help="Number of compute CTAs")
    parser.add_argument("--num-comm-sm", type=int, default=1, help="Number of communication CTAs")
    parser.add_argument(
        "--mode",
        choices=("noncausal", "causal", "both"),
        default="both",
        help="Which attention mode to test.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seqlen_cases = parse_seqlen_spec(args.seqlen)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (9, 0):
        raise SystemExit(f"This demo requires SM90 Hopper, got {(major, minor)}")
    if args.headdim != 128:
        raise SystemExit(f"This demo requires D=128, got D={args.headdim}")
    if args.qhead % args.kvhead != 0:
        raise SystemExit(
            f"This demo requires qhead % kvhead == 0 for GQA/MQA, got qhead={args.qhead}, kvhead={args.kvhead}"
        )
    if args.kvhead * args.headdim != 1024:
        raise SystemExit(
            "Mega ring communication path requires kvhead * headdim == 1024, "
            f"got kvhead={args.kvhead}, headdim={args.headdim}"
        )

    local_rank, local_world_size = init_distributed()
    cases = [(args.num_comp_sm, 0), (args.num_comp_sm, args.num_comm_sm)]

    try:
        for seqlen in seqlen_cases:
            seen: set[tuple[int, int]] = set()
            for num_comp_sm, num_comm_sm in cases:
                key = (num_comp_sm, num_comm_sm)
                if key in seen:
                    continue
                seen.add(key)
                if local_world_size > 1 and num_comm_sm == 0:
                    continue
                if args.mode in ("noncausal", "both"):
                    run_case(
                        args.b,
                        seqlen,
                        args.qhead,
                        args.kvhead,
                        args.headdim,
                        False,
                        num_comp_sm,
                        num_comm_sm,
                        local_rank,
                        local_world_size,
                    )
                if args.mode in ("causal", "both"):
                    run_case(
                        args.b,
                        seqlen,
                        args.qhead,
                        args.kvhead,
                        args.headdim,
                        True,
                        num_comp_sm,
                        num_comm_sm,
                        local_rank,
                        local_world_size,
                    )
                dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
