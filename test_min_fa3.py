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
        if "x" in token:
            raise SystemExit("--seqlen only accepts one length per case; rectangular SqxSk input is no longer supported")
        cases.append(int(token))
    if not cases:
        raise SystemExit("--seqlen must provide at least one case")
    return cases


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
    q = torch.randn(batch_size, seqlen, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, seqlen, kv_heads, head_dim, device=device, dtype=torch.bfloat16)

    out = min_fa3_op.forward(q, k, v, is_causal, manual_block_count=manual_block_count)
    ref = reference_flash(q, k, v, is_causal)

    assert out.shape == q.shape, (out.shape, q.shape)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-1, rtol=2e-1)
    print(
        f"case causal={is_causal}: ok "
        f"(B={batch_size}, S={seqlen}, QH={q_heads}, KVH={kv_heads}, D={head_dim})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal Hopper FA3 demo test.")
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
                    f"case causal=False: skipped due to OOM "
                    f"(B={args.b}, S={seqlen}, QH={args.qhead}, KVH={args.kvhead}, D={args.headdim}) "
                    f"[{format_oom(exc)}]"
                )
        if args.mode in ("causal", "both"):
            try:
                run_case(args.b, seqlen, args.qhead, args.kvhead, args.headdim, True, args.manual_block_count)
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                print(
                    f"case causal=True: skipped due to OOM "
                    f"(B={args.b}, S={seqlen}, QH={args.qhead}, KVH={args.kvhead}, D={args.headdim}) "
                    f"[{format_oom(exc)}]"
                )
