import argparse

import torch

import min_fa3_op


def format_oom(exc: torch.OutOfMemoryError) -> str:
    return str(exc).splitlines()[0]


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
    if outputs:
        return torch.cat(outputs, dim=0)
    return torch.empty_like(q)


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
    device = torch.device("cuda")
    total_tokens = batch_size * seqlen
    q = torch.randn(total_tokens, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(total_tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(total_tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    cu_seqlens_q = make_cu_seqlens(batch_size, seqlen, device)
    cu_seqlens_k = make_cu_seqlens(batch_size, seqlen, device)
    remote_k = min_fa3_op.create_parallel_tensor(k, local_rank=local_rank, local_world_size=local_world_size)
    remote_v = min_fa3_op.create_parallel_tensor(v, local_rank=local_rank, local_world_size=local_world_size)
    prefetch_k = torch.empty_like(k)
    prefetch_v = torch.empty_like(v)

    out_ring = min_fa3_op.forward_varlen_ring(
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
    ref = reference_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, is_causal)
    base = min_fa3_op.forward_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlen,
        seqlen,
        is_causal,
        manual_block_count=num_comp_sm,
    )

    # Ring mode now always runs the epilogue merge against neutral-initialized out/lse,
    # so a single local block should still match both the direct FA3 varlen path and
    # the PyTorch reference.
    torch.testing.assert_close(out_ring.float(), ref.float(), atol=2e-1, rtol=2e-1)
    torch.testing.assert_close(out_ring.float(), base.float(), atol=2e-1, rtol=2e-1)
    if num_comm_sm > 0:
        torch.cuda.synchronize()
        torch.testing.assert_close(prefetch_k.float(), k.float(), atol=0.0, rtol=0.0)
        torch.testing.assert_close(prefetch_v.float(), v.float(), atol=0.0, rtol=0.0)
    print(
        f"ring varlen case causal={is_causal}: ok "
        f"(B={batch_size}, S={seqlen}, QH={q_heads}, KVH={kv_heads}, D={head_dim}, "
        f"num_comp_sm={num_comp_sm}, num_comm_sm={num_comm_sm})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal Hopper FA3 varlen ring-attention demo test.")
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

    # cases = [(args.num_comp_sm, 0), (args.num_comp_sm, args.num_comm_sm)]
    cases = [(args.num_comp_sm, args.num_comm_sm), ]

    for seqlen in seqlen_cases:
        seen: set[tuple[int, int]] = set()
        for num_comp_sm, num_comm_sm in cases:
            key = (num_comp_sm, num_comm_sm)
            if key in seen:
                continue
            seen.add(key)
            if args.mode in ("noncausal", "both"):
                try:
                    run_case(
                        args.b,
                        seqlen,
                        args.qhead,
                        args.kvhead,
                        args.headdim,
                        False,
                        0,
                        num_comp_sm,
                        num_comm_sm,
                        0,
                        torch.cuda.current_device(),
                        1,
                    )
                except torch.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    print(
                        f"ring varlen case causal=False: skipped due to OOM "
                        f"(B={args.b}, S={seqlen}, num_comp_sm={num_comp_sm}, num_comm_sm={num_comm_sm}) "
                        f"[{format_oom(exc)}]"
                    )
            if args.mode in ("causal", "both"):
                try:
                    run_case(
                        args.b,
                        seqlen,
                        args.qhead,
                        args.kvhead,
                        args.headdim,
                        True,
                        0,
                        num_comp_sm,
                        num_comm_sm,
                        0,
                        torch.cuda.current_device(),
                        1,
                    )
                except torch.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    print(
                        f"ring varlen case causal=True: skipped due to OOM "
                        f"(B={args.b}, S={seqlen}, num_comp_sm={num_comp_sm}, num_comm_sm={num_comm_sm}) "
                        f"[{format_oom(exc)}]"
                    )
