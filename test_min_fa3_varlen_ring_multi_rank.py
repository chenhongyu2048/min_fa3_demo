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
            "ThunderKittens ring-attention demo is single-node only: "
            f"world_size={dist.get_world_size()}, local_world_size={local_world_size}"
        )

    return local_rank, local_world_size


def make_cu_seqlens(batch_size: int, seqlen: int, device: torch.device) -> torch.Tensor:
    return torch.arange(0, (batch_size + 1) * seqlen, seqlen, device=device, dtype=torch.int32)


def reference_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    q_bhsd = q.permute(0, 2, 1, 3).float()
    k_bhsd = k.permute(0, 2, 1, 3).float()
    v_bhsd = v.permute(0, 2, 1, 3).float()
    enable_gqa = q_bhsd.size(1) != k_bhsd.size(1)
    if enable_gqa:
        out = torch.nn.functional.scaled_dot_product_attention(
            q_bhsd, k_bhsd, v_bhsd, is_causal=is_causal, enable_gqa=True
        )
    else:
        out = torch.nn.functional.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, is_causal=is_causal)
    return out.permute(0, 2, 1, 3).to(dtype=q.dtype)


def reference_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    batch_size = cu_seqlens_q.numel() - 1
    for batch_idx in range(batch_size):
        q_start = cu_seqlens_q[batch_idx].item()
        q_end = cu_seqlens_q[batch_idx + 1].item()
        k_start = cu_seqlens_k[batch_idx].item()
        k_end = cu_seqlens_k[batch_idx + 1].item()
        q_i = q[q_start:q_end].unsqueeze(0)
        k_i = k[k_start:k_end].unsqueeze(0)
        v_i = v[k_start:k_end].unsqueeze(0)
        outputs.append(reference_flash(q_i, k_i, v_i, is_causal).squeeze(0))
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
    k = (base_k.reshape(total_k, kv_heads, head_dim) + local_rank * 1000.0).to(torch.bfloat16).contiguous()
    v = (base_v.reshape(total_k, kv_heads, head_dim) + local_rank * 2000.0 + 7.0).to(torch.bfloat16).contiguous()
    return q, k, v


def run_case(
    batch_size: int,
    seqlen: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
    src_rank: int,
    num_comp_sm: int,
    num_comm_sm: int,
    ring_step: int,
    local_rank: int,
    local_world_size: int,
) -> None:
    total_tokens = batch_size * seqlen
    cu_seqlens_q = make_cu_seqlens(batch_size, seqlen, torch.device("cuda"))
    cu_seqlens_k = make_cu_seqlens(batch_size, seqlen, torch.device("cuda"))

    q, k, v = make_rank_local_qkv(total_tokens, total_tokens, q_heads, kv_heads, head_dim, local_rank)
    remote_k = min_fa3_op.create_parallel_tensor(k, local_rank=local_rank, local_world_size=local_world_size)
    remote_v = min_fa3_op.create_parallel_tensor(v, local_rank=local_rank, local_world_size=local_world_size)
    prefetch_k = torch.empty_like(k)
    prefetch_v = torch.empty_like(v)

    expected_k = k.clone()
    expected_v = v.clone()
    dist.broadcast(expected_k, src=src_rank)
    dist.broadcast(expected_v, src=src_rank)

    dist.barrier()
    out = min_fa3_op.forward_varlen_ring(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlen,
        seqlen,
        is_causal,
        remote_k=remote_k,
        remote_v=remote_v,
        src_rank=src_rank,
        num_comp_sm=num_comp_sm,
        num_comm_sm=num_comm_sm,
        ring_step=ring_step,
        prefetch_k=prefetch_k,
        prefetch_v=prefetch_v,
    )
    torch.cuda.synchronize()
    dist.barrier()

    local_ref = reference_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, is_causal)
    torch.testing.assert_close(out.float(), local_ref.float(), atol=2e-1, rtol=2e-1)

    if num_comm_sm > 0:
        torch.testing.assert_close(prefetch_k.float(), expected_k.float(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(prefetch_v.float(), expected_v.float(), atol=0.0, rtol=0.0)

    if local_rank == 0:
        print(
            f"distributed ring varlen case causal={is_causal}: ok "
            f"(world_size={local_world_size}, src_rank={src_rank}, B={batch_size}, S={seqlen}, "
            f"QH={q_heads}, KVH={kv_heads}, D={head_dim}, num_comp_sm={num_comp_sm}, num_comm_sm={num_comm_sm})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the distributed minimal Hopper FA3 varlen ring-attention demo test.")
    parser.add_argument("--b", type=int, default=2, help="Batch size B.")
    parser.add_argument(
        "--seqlen",
        "--seqlens",
        dest="seqlen",
        type=str,
        default="128",
        help="Comma-separated sequence lengths S. Q, K, and V all use the same S.",
    )
    parser.add_argument("--qhead", type=int, default=16, help="Number of query/output heads")
    parser.add_argument("--kvhead", type=int, default=8, help="Number of key/value heads")
    parser.add_argument("--headdim", type=int, default=128, help="Head dimension D")
    parser.add_argument("--src-rank", type=int, default=0, help="Source rank to read K/V from.")
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

    local_rank, local_world_size = init_distributed()
    if args.src_rank < 0 or args.src_rank >= local_world_size:
        raise SystemExit(f"--src-rank must be in [0, {local_world_size}), got src_rank={args.src_rank}")

    cases = [(args.num_comp_sm, 0), (args.num_comp_sm, args.num_comm_sm)]

    try:
        for seqlen in seqlen_cases:
            seen: set[tuple[int, int]] = set()
            for num_comp_sm, num_comm_sm in cases:
                key = (num_comp_sm, num_comm_sm)
                if key in seen:
                    continue
                seen.add(key)
                if args.mode in ("noncausal", "both"):
                    run_case(
                        args.b,
                        seqlen,
                        args.qhead,
                        args.kvhead,
                        args.headdim,
                        False,
                        args.src_rank,
                        num_comp_sm,
                        num_comm_sm,
                        0,
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
                        args.src_rank,
                        num_comp_sm,
                        num_comm_sm,
                        0,
                        local_rank,
                        local_world_size,
                    )
                dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
