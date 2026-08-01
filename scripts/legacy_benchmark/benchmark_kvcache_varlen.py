import argparse
import statistics
from collections.abc import Callable

import torch

import min_fa3_op


def parse_lengths(spec: str, batch_size: int, name: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if len(values) == 1:
        values *= batch_size
    if len(values) != batch_size:
        raise SystemExit(f"{name} must contain one value or exactly B={batch_size} values")
    if any(value <= 0 for value in values):
        raise SystemExit(f"{name} values must be positive")
    return values


def make_cu_seqlens(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    host = torch.tensor(offsets, dtype=torch.int32)
    return host.cuda(), host


def attention_flops(
    q_lengths: list[int],
    k_lengths: list[int],
    q_heads: int,
    head_dim: int,
    is_causal: bool,
) -> int:
    valid_pairs = 0
    for q_len, k_len in zip(q_lengths, k_lengths):
        valid_pairs += q_len * k_len
        if is_causal:
            valid_pairs -= q_len * (q_len - 1) // 2
    return 2 * q_heads * valid_pairs * (head_dim + head_dim)


def logical_attention_io_bytes(
    q_lengths: list[int],
    k_lengths: list[int],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> int:
    total_q = sum(q_lengths)
    total_k = sum(k_lengths)
    q_bytes = total_q * q_heads * head_dim * 2
    kv_bytes = total_k * kv_heads * head_dim * 2 * 2
    o_bytes = total_q * q_heads * head_dim * 2
    lse_bytes = total_q * q_heads * 4
    return q_bytes + kv_bytes + o_bytes + lse_bytes


def median_time_ms(fn: Callable[[], object], warmup_iters: int, num_iters: int) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))


def profile_kernel_times(fn: Callable[[], object]) -> dict[str, float]:
    totals = {"prepare": 0.0, "attention": 0.0, "combine": 0.0}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        fn()
        torch.cuda.synchronize()
    for event in profile.key_averages():
        time_ms = event.device_time_total / 1000.0
        if time_ms <= 0:
            continue
        if "prepare_varlen_num_blocks" in event.key:
            totals["prepare"] += time_ms
        elif "FlashAttnFwdCombine" in event.key:
            totals["combine"] += time_ms
        elif "FlashAttnFwdSm90" in event.key:
            totals["attention"] += time_ms
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark packed-varlen Hopper FA3 KV-cache forward.")
    parser.add_argument("--b", type=int, default=3, help="Batch size B")
    parser.add_argument(
        "--sq",
        type=str,
        default="1,8,32",
        help="One broadcast query length or exactly B comma-separated lengths",
    )
    parser.add_argument(
        "--seqlen",
        type=str,
        default="129,1024,3131",
        help="One broadcast KV length or exactly B comma-separated lengths",
    )
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--mode", choices=("auto", "nosplit", "split", "all"), default="all")
    parser.add_argument("--num-splits", type=int, default=8)
    parser.add_argument(
        "--mask",
        choices=("default", "causal", "noncausal"),
        default="default",
        help="Packed-batch mask selection",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--profile-kernels", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("This benchmark requires an SM90 Hopper GPU")
    if args.b <= 0:
        raise SystemExit("--b must be positive")
    if args.headdim != 128:
        raise SystemExit("This benchmark requires --headdim 128")
    if args.qhead <= 0 or args.kvhead <= 0 or args.qhead % args.kvhead:
        raise SystemExit("qhead and kvhead must be positive and qhead must be divisible by kvhead")
    if args.num_splits < 2 or args.num_splits > 128:
        raise SystemExit("--num-splits must be in [2, 128]")
    if args.warmup_iters < 0 or args.num_iters <= 0:
        raise SystemExit("--warmup-iters must be nonnegative and --num-iters must be positive")

    q_lengths = parse_lengths(args.sq, args.b, "--sq")
    k_lengths = parse_lengths(args.seqlen, args.b, "--seqlen")
    max_seqlen_q = max(q_lengths)
    max_seqlen_k = max(k_lengths)
    is_causal_arg = {"default": None, "causal": True, "noncausal": False}[args.mask]
    effective_is_causal = max_seqlen_q != 1 if is_causal_arg is None else is_causal_arg
    if effective_is_causal and any(q_len > k_len for q_len, k_len in zip(q_lengths, k_lengths)):
        raise SystemExit("causal attention requires every query length to be <= its KV length")

    split_modes = {
        "auto": [("auto", 0)],
        "nosplit": [("nosplit", 1)],
        "split": [(f"split{args.num_splits}", args.num_splits)],
        "all": [("auto", 0), ("nosplit", 1), (f"split{args.num_splits}", args.num_splits)],
    }[args.mode]

    torch.manual_seed(0)
    cu_q, cu_q_host = make_cu_seqlens(q_lengths)
    cu_k, cu_k_host = make_cu_seqlens(k_lengths)
    q = torch.randn(sum(q_lengths), args.qhead, args.headdim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(k_lengths), args.kvhead, args.headdim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    dense_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    q_start = 0
    k_start = 0
    for q_len, k_len in zip(q_lengths, k_lengths):
        q_i = q[q_start : q_start + q_len].unsqueeze(0)
        k_i = k[k_start : k_start + k_len].unsqueeze(0)
        v_i = v[k_start : k_start + k_len].unsqueeze(0)
        cache_seqlens = torch.tensor([k_len], device="cuda", dtype=torch.int32)
        dense_inputs.append((q_i, k_i, v_i, cache_seqlens))
        q_start += q_len
        k_start += k_len

    model_flops = attention_flops(
        q_lengths, k_lengths, args.qhead, args.headdim, effective_is_causal
    )
    logical_io_bytes = logical_attention_io_bytes(
        q_lengths, k_lengths, args.qhead, args.kvhead, args.headdim
    )
    print(
        f"B={args.b} q_lengths={q_lengths} k_lengths={k_lengths} "
        f"QH={args.qhead} KVH={args.kvhead} D={args.headdim} "
        f"mask={args.mask} (effective_causal={effective_is_causal})"
    )
    print(
        "mode       fused_ms    loop_ms  speedup  fused_TF/s  loop_TF/s  "
        "logical_IO_MB  fused_GB/s  loop_GB/s  prepare_ms  attention_ms  combine_ms"
    )

    for mode_name, num_splits in split_modes:
        def fused_fn() -> object:
            return min_fa3_op.forward_kvcache_varlen(
                q,
                k,
                v,
                cu_q,
                cu_k,
                max_seqlen_q,
                max_seqlen_k,
                cu_seqlens_q_host=cu_q_host,
                cu_seqlens_k_host=cu_k_host,
                num_splits=num_splits,
                is_causal=is_causal_arg,
            )

        def dense_loop_fn() -> list[torch.Tensor]:
            return [
                min_fa3_op.forward_kvcache(
                    q_i,
                    k_i,
                    v_i,
                    cache_seqlens,
                    num_splits=num_splits,
                    is_causal=effective_is_causal,
                )
                for q_i, k_i, v_i, cache_seqlens in dense_inputs
            ]

        fused_ms = median_time_ms(fused_fn, args.warmup_iters, args.num_iters)
        loop_ms = median_time_ms(dense_loop_fn, args.warmup_iters, args.num_iters)
        kernel_times = profile_kernel_times(fused_fn) if args.profile_kernels else {}
        print(
            f"{mode_name:<10} "
            f"{fused_ms:>9.4f} "
            f"{loop_ms:>10.4f} "
            f"{loop_ms / fused_ms:>8.3f} "
            f"{model_flops / fused_ms / 1e9:>11.2f} "
            f"{model_flops / loop_ms / 1e9:>10.2f} "
            f"{logical_io_bytes / 1e6:>14.3f} "
            f"{logical_io_bytes / fused_ms / 1e6:>11.1f} "
            f"{logical_io_bytes / loop_ms / 1e6:>10.1f} "
            f"{kernel_times.get('prepare', float('nan')):>11.4f} "
            f"{kernel_times.get('attention', float('nan')):>13.4f} "
            f"{kernel_times.get('combine', float('nan')):>11.4f}"
        )
