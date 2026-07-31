import argparse
import math

import torch

import min_fa3_op


def parse_int_list(spec: str, name: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise SystemExit(f"{name} must provide at least one integer")
    return values


def reference_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    sq = q.size(1)
    qhead_per_khead = q.size(2) // k_cache.size(2)
    for batch_idx, cache_len in enumerate(cache_seqlens):
        q_i = q[batch_idx].float()
        k_i = k_cache[batch_idx, :cache_len].float().repeat_interleave(qhead_per_khead, dim=1)
        v_i = v_cache[batch_idx, :cache_len].float().repeat_interleave(qhead_per_khead, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q_i, k_i) / math.sqrt(q.size(3))
        query_idx = torch.arange(sq, device=q.device)[:, None]
        key_idx = torch.arange(cache_len, device=q.device)[None, :]
        scores.masked_fill_(key_idx > cache_len - sq + query_idx, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probabilities, v_i))
        lses.append(torch.logsumexp(scores, dim=-1))
    return torch.stack(outputs), torch.stack(lses)


def make_ragged_lengths(batch_size: int, sq: int, capacity: int) -> list[int]:
    if batch_size == 1:
        return [capacity]
    candidates = [max(sq, 129), max(sq, 1024), capacity]
    return [min(capacity, candidates[idx % len(candidates)]) for idx in range(batch_size)]


def run_case(
    batch_size: int,
    sq: int,
    capacity: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    num_splits: int,
) -> None:
    if sq > capacity:
        return
    lengths = make_ragged_lengths(batch_size, sq, capacity)
    q = torch.randn(batch_size, sq, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(batch_size, capacity, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    cache_seqlens = torch.tensor(lengths, device="cuda", dtype=torch.int32)

    out, lse = min_fa3_op.forward_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens,
        num_splits=num_splits,
        return_lse=True,
    )
    out_ref, lse_ref = reference_kvcache(q, k_cache, v_cache, lengths)
    torch.testing.assert_close(out.float(), out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, lse_ref, atol=3e-3, rtol=3e-3)

    # Forced Split and forced NoSplit must implement the same public operation.
    if num_splits > 1:
        out_nosplit = min_fa3_op.forward_kvcache(
            q, k_cache, v_cache, cache_seqlens, num_splits=1
        )
        torch.testing.assert_close(out.float(), out_nosplit.float(), atol=3e-2, rtol=3e-2)

    print(
        "kvcache case: ok "
        f"(B={batch_size}, Sq={sq}, Sk={capacity}, QH={q_heads}, KVH={kv_heads}, "
        f"D={head_dim}, lengths={lengths}, num_splits={num_splits})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test minimal Hopper FA3 KV-cache decode/chunk prefill.")
    parser.add_argument("--b", type=int, default=3, help="Batch size B")
    parser.add_argument(
        "--seqlen",
        type=str,
        default="129,1024,3131",
        help="Comma-separated dense KV-cache capacities",
    )
    parser.add_argument("--sq", type=str, default="1,8,32,120,128", help="Comma-separated query chunk lengths")
    parser.add_argument("--qhead", type=int, default=8, help="Number of query/output heads")
    parser.add_argument("--kvhead", type=int, default=2, help="Number of key/value heads")
    parser.add_argument("--headdim", type=int, default=128, help="Head dimension D")
    parser.add_argument(
        "--mode",
        choices=("auto", "nosplit", "split", "all"),
        default="all",
        help="Split dispatch modes to test",
    )
    parser.add_argument(
        "--num-splits",
        type=str,
        default="2,3,8,128",
        help="Comma-separated forced Split values used by split/all mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit(f"This demo requires SM90 Hopper, got {torch.cuda.get_device_capability()}")
    if args.b <= 0:
        raise SystemExit("--b must be positive")
    if args.headdim != 128:
        raise SystemExit(f"This demo requires D=128, got {args.headdim}")
    if args.qhead <= 0 or args.kvhead <= 0 or args.qhead % args.kvhead:
        raise SystemExit("--qhead and --kvhead must be positive and qhead must be divisible by kvhead")

    split_values: list[int] = []
    if args.mode in ("auto", "all"):
        split_values.append(0)
    if args.mode in ("nosplit", "all"):
        split_values.append(1)
    if args.mode in ("split", "all"):
        split_values.extend(parse_int_list(args.num_splits, "--num-splits"))
    if any(value < 0 or value > 128 for value in split_values):
        raise SystemExit("split values must be in [0, 128]")

    torch.manual_seed(0)
    for capacity in parse_int_list(args.seqlen, "--seqlen"):
        for sq in parse_int_list(args.sq, "--sq"):
            for num_splits in split_values:
                run_case(args.b, sq, capacity, args.qhead, args.kvhead, args.headdim, num_splits)
