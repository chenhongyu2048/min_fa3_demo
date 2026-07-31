import argparse
import statistics

import torch

import min_fa3_op


def parse_int_list(spec: str, name: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise SystemExit(f"{name} must provide at least one integer")
    return values


def causal_attention_flops(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    q_heads: int,
    head_dim: int,
) -> int:
    # Bottom-right causal QK pairs. QK and PV each contribute 2 FLOPs per
    # contracted element; this demo fixes Dv == D.
    valid_qk_pairs = seqlen_q * seqlen_k - seqlen_q * (seqlen_q - 1) // 2
    return 2 * batch_size * q_heads * valid_qk_pairs * (head_dim + head_dim)


def logical_attention_io_bytes(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> int:
    # Minimum logical tensor traffic. This intentionally excludes scheduler
    # metadata and Split's internal FP32 partial O/LSE traffic.
    q_bytes = batch_size * seqlen_q * q_heads * head_dim * 2
    kv_bytes = batch_size * seqlen_k * kv_heads * head_dim * 2 * 2
    o_bytes = batch_size * seqlen_q * q_heads * head_dim * 2
    lse_bytes = batch_size * seqlen_q * q_heads * 4
    return q_bytes + kv_bytes + o_bytes + lse_bytes


def median_time_ms(fn, warmup_iters: int, num_iters: int) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for idx in range(num_iters):
        starts[idx].record()
        fn()
        ends[idx].record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))


def profile_kernel_times(fn) -> dict[str, float]:
    totals = {"prepare": 0.0, "attention": 0.0, "combine": 0.0, "other": 0.0}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        fn()
        torch.cuda.synchronize()
    for event in profile.key_averages():
        time_ms = event.device_time_total / 1000.0
        if time_ms <= 0:
            continue
        name = event.key
        if "prepare_varlen_num_blocks" in name:
            totals["prepare"] += time_ms
        elif "FlashAttnFwdCombine" in name:
            totals["combine"] += time_ms
        elif "FlashAttnFwdSm90" in name:
            totals["attention"] += time_ms
        else:
            totals["other"] += time_ms
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark minimal Hopper FA3 KV-cache forward.")
    parser.add_argument("--b", type=int, default=4, help="Batch size B")
    parser.add_argument("--seqlen", type=str, default="1024,4096,16384", help="KV-cache capacities")
    parser.add_argument("--sq", type=str, default="1,32,128", help="Query chunk lengths")
    parser.add_argument("--qhead", type=int, default=32, help="Number of query/output heads")
    parser.add_argument("--kvhead", type=int, default=8, help="Number of key/value heads")
    parser.add_argument("--headdim", type=int, default=128, help="Head dimension D")
    parser.add_argument(
        "--mode",
        choices=("auto", "nosplit", "split", "all"),
        default="all",
        help="Split dispatch modes to benchmark",
    )
    parser.add_argument("--num-splits", type=int, default=8, help="Forced split count for split/all mode")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--profile-kernels", action="store_true", help="Report one-call CUDA kernel breakdown")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("This benchmark requires an SM90 Hopper GPU")
    if args.headdim != 128:
        raise SystemExit("This benchmark requires --headdim 128")
    if args.qhead <= 0 or args.kvhead <= 0 or args.qhead % args.kvhead:
        raise SystemExit("qhead and kvhead must be positive and qhead must be divisible by kvhead")
    if args.num_splits < 2 or args.num_splits > 128:
        raise SystemExit("--num-splits must be in [2, 128]")

    split_modes = {
        "auto": [("auto", 0)],
        "nosplit": [("nosplit", 1)],
        "split": [(f"split{args.num_splits}", args.num_splits)],
        "all": [("auto", 0), ("nosplit", 1), (f"split{args.num_splits}", args.num_splits)],
    }[args.mode]

    torch.manual_seed(0)
    print(
        "mode       B   Sq  Sk_capacity  QH  KVH    end_to_end_ms  causal_GFLOP  logical_IO_MB  "
        "TFLOP/s  effective_GB/s   prepare_ms   attention_ms   combine_ms"
    )
    for capacity in parse_int_list(args.seqlen, "--seqlen"):
        for sq in parse_int_list(args.sq, "--sq"):
            if sq > capacity:
                continue
            model_flops = causal_attention_flops(
                args.b, sq, capacity, args.qhead, args.headdim
            )
            logical_io_bytes = logical_attention_io_bytes(
                args.b, sq, capacity, args.qhead, args.kvhead, args.headdim
            )
            q = torch.randn(args.b, sq, args.qhead, args.headdim, device="cuda", dtype=torch.bfloat16)
            k_cache = torch.randn(args.b, capacity, args.kvhead, args.headdim, device="cuda", dtype=torch.bfloat16)
            v_cache = torch.randn_like(k_cache)
            cache_seqlens = torch.full((args.b,), capacity, device="cuda", dtype=torch.int32)
            for name, num_splits in split_modes:
                fn = lambda: min_fa3_op.forward_kvcache(
                    q, k_cache, v_cache, cache_seqlens, num_splits=num_splits
                )
                end_to_end_ms = median_time_ms(fn, args.warmup_iters, args.num_iters)
                kernel_times = profile_kernel_times(fn) if args.profile_kernels else {}
                causal_gflops = model_flops / 1e9
                logical_io_mb = logical_io_bytes / 1e6
                tflops = model_flops / end_to_end_ms / 1e9
                effective_gbps = logical_io_bytes / end_to_end_ms / 1e6
                print(
                    f"{name:<10} {args.b:>2} {sq:>4} {capacity:>12} {args.qhead:>3} {args.kvhead:>4} "
                    f"{end_to_end_ms:>16.4f} "
                    f"{causal_gflops:>13.3f} "
                    f"{logical_io_mb:>14.3f} "
                    f"{tflops:>8.2f} "
                    f"{effective_gbps:>15.1f} "
                    f"{kernel_times.get('prepare', float('nan')):>12.4f} "
                    f"{kernel_times.get('attention', float('nan')):>14.4f} "
                    f"{kernel_times.get('combine', float('nan')):>12.4f}"
                )
