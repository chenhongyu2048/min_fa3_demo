import argparse

import torch

import min_fa3_op
from mega_ring_test_min_fa3_varlen_ring_local import reference_varlen


def make_cu_seqlens(lengths: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.zeros((len(lengths) + 1,), dtype=torch.int32)
    for idx, length in enumerate(lengths):
        host[idx + 1] = host[idx] + int(length)
    return host.to(device=device), host


def parse_lengths(spec: str) -> list[int]:
    lengths = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not lengths:
        raise SystemExit("--seqlens must provide at least one length")
    return lengths


def run_case(
    lengths: list[int],
    global_lengths: list[int],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
    num_comp_sm: int,
    num_comm_sm: int,
    cp_threshold: int,
) -> None:
    device = torch.device("cuda")
    total_tokens = sum(lengths)
    max_seqlen = max(lengths)

    q = torch.randn(total_tokens, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    cu_seqlens, cu_seqlens_host = make_cu_seqlens(lengths, device)
    global_seqlens_host = torch.tensor(global_lengths, dtype=torch.int32)

    remote_k = min_fa3_op.TKParallelTensor(
        [total_tokens, kv_heads, head_dim],
        torch.bfloat16,
        torch.cuda.current_device(),
        1,
        False,
    )
    remote_v = min_fa3_op.TKParallelTensor(
        [total_tokens, kv_heads, head_dim],
        torch.bfloat16,
        torch.cuda.current_device(),
        1,
        False,
    )
    k = remote_k.data_
    v = remote_v.data_
    k.normal_()
    v.normal_()

    out_hybrid = min_fa3_op.forward_varlen_mega_ring(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        is_causal,
        cu_seqlens_q_host=cu_seqlens_host,
        cu_seqlens_k_host=cu_seqlens_host,
        remote_k=remote_k,
        remote_v=remote_v,
        num_comp_sm=num_comp_sm,
        num_comm_sm=num_comm_sm,
        global_seqlens_host=global_seqlens_host,
        cp_threshold=cp_threshold,
    )
    out_base = min_fa3_op.forward_varlen(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        is_causal,
        cu_seqlens_q_host=cu_seqlens_host,
        cu_seqlens_k_host=cu_seqlens_host,
        manual_block_count=num_comp_sm,
    )
    ref = reference_varlen(q, k, v, cu_seqlens_host, cu_seqlens_host, is_causal)

    torch.testing.assert_close(out_hybrid.float(), out_base.float(), atol=2e-1, rtol=2e-1)
    torch.testing.assert_close(out_hybrid.float(), ref.float(), atol=2e-1, rtol=2e-1)
    cp_count = sum(int(length > cp_threshold) for length in global_lengths)
    print(
        f"mega ring hybrid local case causal={is_causal}: ok "
        f"(lengths={lengths}, cp_count={cp_count}, threshold={cp_threshold})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local hybrid mega-ring varlen correctness checks.")
    parser.add_argument("--seqlens", type=str, default="1152,4096,1408")
    parser.add_argument("--all-local-seqlens", type=str, default="1152,1408")
    parser.add_argument("--qhead", type=int, default=16)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--num-comp-sm", type=int, default=1)
    parser.add_argument("--num-comm-sm", type=int, default=0)
    parser.add_argument("--cp-threshold", type=int, default=2048)
    parser.add_argument("--mode", choices=("noncausal", "causal", "both"), default="both")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (9, 0):
        raise SystemExit(f"This demo requires SM90 Hopper, got {(major, minor)}")
    if args.headdim != 128:
        raise SystemExit(f"This demo requires D=128, got D={args.headdim}")
    if args.qhead % args.kvhead != 0:
        raise SystemExit(f"qhead must be divisible by kvhead, got {args.qhead} and {args.kvhead}")
    if args.kvhead * args.headdim != 1024:
        raise SystemExit("Mega ring communication path requires kvhead * headdim == 1024")

    mixed_lengths = parse_lengths(args.seqlens)
    all_local_lengths = parse_lengths(args.all_local_seqlens)
    cases = [
        (mixed_lengths, mixed_lengths),
        (all_local_lengths, all_local_lengths),
    ]
    for lengths, global_lengths in cases:
        if args.mode in ("noncausal", "both"):
            run_case(
                lengths,
                global_lengths,
                args.qhead,
                args.kvhead,
                args.headdim,
                False,
                args.num_comp_sm,
                args.num_comm_sm,
                args.cp_threshold,
            )
        if args.mode in ("causal", "both"):
            run_case(
                lengths,
                global_lengths,
                args.qhead,
                args.kvhead,
                args.headdim,
                True,
                args.num_comp_sm,
                args.num_comm_sm,
                args.cp_threshold,
            )
