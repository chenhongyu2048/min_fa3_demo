"""
Benchmark: min_fa3_demo vs PyTorch vs FA2 vs FA3

Examples:
    python benchmark.py
    python benchmark.py --b 4 --seqlen 512,1024,2048,4096,8192 --qo-head 32 --kv-head 32 --d 128 --mode both
    python benchmark.py --seqlen 512,1024,2048 --mode noncausal
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

THIS_DIR = Path(__file__).resolve().parent
HOPPER_DIR = THIS_DIR.parent
REPO_ROOT = HOPPER_DIR.parent

for path in (THIS_DIR, HOPPER_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import min_fa3_op

print("Loaded min_fa3_op (minimal Hopper demo)")

try:
    from flash_attn import flash_attn_func as flash_attn_func2
    print("Loaded flash_attn (FA2)")
except ImportError:
    flash_attn_func2 = None
    print("FA2 not available")

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func3
    print("Loaded flash_attn_interface (FA3)")
except ImportError:
    flash_attn_func3 = None
    print("FA3 not available")


@dataclass(frozen=True)
class Case:
    batch_size: int
    seqlen: int
    qo_heads: int
    kv_heads: int
    head_dim: int
    is_causal: bool


@dataclass(frozen=True)
class Result:
    time_ms: float
    tflops: float


Method = Callable[..., float | None]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark min_fa3_demo against PyTorch / FA2 / FA3")
    parser.add_argument("--b", type=int, default=4, help="Batch size B")
    parser.add_argument(
        "--seqlen",
        "--seqlens",
        dest="seqlen",
        type=str,
        default="512,1024,2048,4096",
        help="Comma-separated sequence lengths S. Q, K, and V all use the same S.",
    )
    parser.add_argument("--qhead", "--qo-head", dest="qhead", type=int, default=32, help="Number of query/output heads")
    parser.add_argument("--kvhead", "--kv-head", dest="kvhead", type=int, default=32, help="Number of key/value heads")
    parser.add_argument("--headdim", "--d", dest="headdim", type=int, default=128, help="Head dimension D")
    parser.add_argument(
        "--mode",
        choices=("noncausal", "causal", "both"),
        default="both",
        help="Benchmark noncausal, causal, or both modes",
    )
    parser.add_argument(
        "--manual-block-count",
        type=int,
        default=None,
        help="Optional grid.x thread-block count override for min_fa3_demo. Defaults to the automatic get_grid_shape(...) result.",
    )
    parser.add_argument("--num-iters", type=int, default=50, help="Timing iterations")
    parser.add_argument("--warmup-iters", type=int, default=10, help="Warmup iterations")
    return parser.parse_args()


def get_flops(case: Case) -> int:
    return 4 * case.batch_size * case.seqlen * case.seqlen * case.qo_heads * case.head_dim // (2 if case.is_causal else 1)


def parse_cases(args: argparse.Namespace) -> list[Case]:
    lengths = parse_seqlen_spec(args.seqlen)

    causal_values = {
        "noncausal": [False],
        "causal": [True],
        "both": [False, True],
    }[args.mode]

    return [
        Case(args.b, seqlen, args.qhead, args.kvhead, args.headdim, is_causal)
        for is_causal in causal_values
        for seqlen in lengths
    ]


def make_inputs(case: Case, dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(case.batch_size, case.seqlen, case.qo_heads, case.head_dim, dtype=dtype, device="cuda")
    k = torch.randn(case.batch_size, case.seqlen, case.kv_heads, case.head_dim, dtype=dtype, device="cuda")
    v = torch.randn(case.batch_size, case.seqlen, case.kv_heads, case.head_dim, dtype=dtype, device="cuda")
    return q, k, v


def median_time_ms(fn: Callable[[], None], warmup_iters: int, num_iters: int) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for i in range(num_iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(start_events, end_events))


def bench_pytorch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool, warmup_iters: int, num_iters: int) -> float | None:
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    enable_gqa = qt.size(1) != kt.size(1)

    def run() -> None:
        if enable_gqa:
            torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=is_causal, enable_gqa=True)
        else:
            torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=is_causal)

    return median_time_ms(run, warmup_iters, num_iters)


def bench_fa2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool, warmup_iters: int, num_iters: int) -> float | None:
    if flash_attn_func2 is None:
        return None
    return median_time_ms(lambda: flash_attn_func2(q, k, v, causal=is_causal), warmup_iters, num_iters)


def bench_fa3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool, warmup_iters: int, num_iters: int) -> float | None:
    if flash_attn_func3 is None:
        return None
    return median_time_ms(lambda: flash_attn_func3(q, k, v, causal=is_causal), warmup_iters, num_iters)


def bench_min_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    warmup_iters: int,
    num_iters: int,
    manual_block_count: int | None,
) -> float | None:
    if k.size(2) != v.size(2):
        return None
    if q.size(2) % k.size(2) != 0:
        return None
    if q.size(3) != 128 or k.size(3) != 128 or v.size(3) != 128:
        return None
    return median_time_ms(
        lambda: min_fa3_op.forward(q, k, v, is_causal, manual_block_count=manual_block_count),
        warmup_iters,
        num_iters,
    )


def bench_case(case: Case, warmup_iters: int, num_iters: int, manual_block_count: int | None) -> dict[str, Result | None]:
    methods: list[tuple[str, Method]] = [
        ("PyTorch_SDPA", bench_pytorch),
        ("FA2", bench_fa2),
        ("FA3", bench_fa3),
        ("min_fa3_demo", bench_min_fa3),
    ]
    results: dict[str, Result | None] = {}
    try:
        q, k, v = make_inputs(case)
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        print(f"[case OOM while allocating inputs: {format_case_key(case)}] [{format_oom(exc)}]")
        for name, _ in methods:
            results[name] = None
        return results

    for name, fn in methods:
        try:
            if name == "min_fa3_demo":
                time_ms = bench_min_fa3(q, k, v, case.is_causal, warmup_iters, num_iters, manual_block_count)
            else:
                time_ms = fn(q, k, v, case.is_causal, warmup_iters, num_iters)
            if time_ms is None:
                results[name] = None
                continue
            tflops = get_flops(case) / (time_ms * 1e-3) / 1e12
            results[name] = Result(time_ms=time_ms, tflops=tflops)
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            print(f"[{name} OOM: {format_case_key(case)}] [{format_oom(exc)}]")
            results[name] = None
        except Exception as exc:
            print(f"[{name} error: {exc}]")
            results[name] = None
    torch.cuda.empty_cache()
    return results


def format_case_key(case: Case) -> str:
    return (
        f"B={case.batch_size},S={case.seqlen},"
        f"QH={case.qo_heads},KVH={case.kv_heads},D={case.head_dim},"
        f"causal={case.is_causal}"
    )


def print_table(results_by_case: dict[Case, dict[str, Result | None]]) -> None:
    methods = ["PyTorch_SDPA", "FA2", "FA3", "min_fa3_demo"]
    print("\n" + "=" * 100)
    print("Benchmark Results")
    print("=" * 100)
    for case, results in results_by_case.items():
        print(format_case_key(case))
        print(f"{'Method':<16} {'Time (ms)':>12} {'TFLOPS':>12}")
        for method in methods:
            result = results.get(method)
            if result is None:
                print(f"{method:<16} {'N/A':>12} {'N/A':>12}")
            else:
                print(f"{method:<16} {result.time_ms:>12.3f} {result.tflops:>12.1f}")
        print("-" * 100)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.headdim != 128:
        print("Warning: min_fa3_demo only supports D=128. It will be skipped for other D values.")
    if args.qhead % args.kvhead != 0:
        raise SystemExit(
            f"This benchmark requires qhead % kvhead == 0 for GQA/MQA, got qhead={args.qhead}, kvhead={args.kvhead}"
        )

    cases = parse_cases(args)
    print(
        f"Config: B={args.b}, qhead={args.qhead}, kvhead={args.kvhead}, D={args.headdim}, "
        f"mode={args.mode}, num_iters={args.num_iters}, warmup_iters={args.warmup_iters}, "
        f"manual_block_count={args.manual_block_count}"
    )
    print(f"Seqlen: {args.seqlen}")

    results_by_case: dict[Case, dict[str, Result | None]] = {}
    for case in cases:
        print(f"Running {format_case_key(case)}", flush=True)
        results_by_case[case] = bench_case(case, args.warmup_iters, args.num_iters, args.manual_block_count)

    print_table(results_by_case)


if __name__ == "__main__":
    main()
