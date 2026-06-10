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
    return torch.cat(outputs, dim=0)


def run_case(
    batch_size: int,
    seqlen: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
    manual_block_count: int | None,
) -> None:
    device = torch.device("cuda")
    total_tokens = batch_size * seqlen
    q = torch.randn(total_tokens, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(total_tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(total_tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    cu_seqlens_q = make_cu_seqlens(batch_size, seqlen, device)
    cu_seqlens_k = make_cu_seqlens(batch_size, seqlen, device)

    out = min_fa3_op.forward_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlen,
        seqlen,
        is_causal,
        manual_block_count=manual_block_count,
    )
    ref = reference_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, is_causal)

    assert out.shape == q.shape, (out.shape, q.shape)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-1, rtol=2e-1)
    print(
        f"varlen case causal={is_causal}: ok "
        f"(B={batch_size}, S={seqlen}, QH={q_heads}, KVH={kv_heads}, D={head_dim})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal Hopper FA3 varlen demo test.")
    parser.add_argument("--b", type=int, default=1, help="Batch size B.")
    parser.add_argument(
        "--seqlen",
        "--seqlens",
        dest="seqlen",
        type=str,
        default="128",
        help="Comma-separated sequence lengths S. Q, K, and V all use the same S.",
    )
    parser.add_argument("--qhead", "--qo-head", dest="qhead", type=int, default=32, help="Number of query/output heads")
    parser.add_argument("--kvhead", "--kv-head", dest="kvhead", type=int, default=32, help="Number of key/value heads")
    parser.add_argument("--headdim", "--d", dest="headdim", type=int, default=128, help="Head dimension D")
    parser.add_argument(
        "--mode",
        choices=("noncausal", "causal", "both"),
        default="both",
        help="Which attention mode to test.",
    )
    parser.add_argument(
        "--manual-block-count",
        type=int,
        default=None,
        help="Optional grid.x thread-block count override. Defaults to the automatic get_grid_shape(...) result.",
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

    for seqlen in seqlen_cases:
        if args.mode in ("noncausal", "both"):
            try:
                run_case(args.b, seqlen, args.qhead, args.kvhead, args.headdim, False, args.manual_block_count)
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                print(
                    f"varlen case causal=False: skipped due to OOM "
                    f"(B={args.b}, S={seqlen}, QH={args.qhead}, KVH={args.kvhead}, D={args.headdim}) "
                    f"[{format_oom(exc)}]"
                )
        if args.mode in ("causal", "both"):
            try:
                run_case(args.b, seqlen, args.qhead, args.kvhead, args.headdim, True, args.manual_block_count)
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                print(
                    f"varlen case causal=True: skipped due to OOM "
                    f"(B={args.b}, S={seqlen}, QH={args.qhead}, KVH={args.kvhead}, D={args.headdim}) "
                    f"[{format_oom(exc)}]"
                )
