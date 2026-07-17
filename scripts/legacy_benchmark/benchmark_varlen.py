"""
Benchmark: min_fa3 varlen demo vs PyTorch vs FA2 vs FA3

Examples:
    python benchmark_varlen.py
    python benchmark_varlen.py --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 --headdim 128 --mode both
    python benchmark_varlen.py --seqlen 512,1024,2048 --mode noncausal
"""

from __future__ import annotations

import argparse
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
    from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func2

    print("Loaded flash_attn varlen (FA2)")
except ImportError:
    flash_attn_varlen_func2 = None
    print("FA2 varlen not available")

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func3

    print("Loaded flash_attn_interface varlen (FA3)")
except ImportError:
    flash_attn_varlen_func3 = None
    print("FA3 varlen not available")


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
    parser = argparse.ArgumentParser(description="Benchmark min_fa3 varlen demo against PyTorch / FA2 / FA3")
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
        help="Optional grid.x thread-block count override for min_fa3_varlen. Defaults to the automatic get_grid_shape(...) result.",
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


def make_cu_seqlens(batch_size: int, seqlen: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.arange(0, (batch_size + 1) * seqlen, seqlen, dtype=torch.int32)
    return host.to(device=device), host


def make_inputs(case: Case, dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, ...]:
    total_tokens = case.batch_size * case.seqlen
    q = torch.randn(total_tokens, case.qo_heads, case.head_dim, dtype=dtype, device="cuda")
    k = torch.randn(total_tokens, case.kv_heads, case.head_dim, dtype=dtype, device="cuda")
    v = torch.randn(total_tokens, case.kv_heads, case.head_dim, dtype=dtype, device="cuda")
    cu_seqlens_q, cu_seqlens_q_host = make_cu_seqlens(case.batch_size, case.seqlen, q.device)
    cu_seqlens_k, cu_seqlens_k_host = make_cu_seqlens(case.batch_size, case.seqlen, q.device)
    return q, k, v, cu_seqlens_q, cu_seqlens_k, cu_seqlens_q_host, cu_seqlens_k_host


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


def take_output(result):
    return result[0] if isinstance(result, tuple) else result


def bench_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_q_host: torch.Tensor,
    cu_seqlens_k_host: torch.Tensor,
    batch_size: int,
    is_causal: bool,
    warmup_iters: int,
    num_iters: int,
) -> float | None:
    seqlen_q = int(cu_seqlens_q_host[1] - cu_seqlens_q_host[0])
    seqlen_k = int(cu_seqlens_k_host[1] - cu_seqlens_k_host[0])
    q_bshd = q.view(batch_size, seqlen_q, q.size(1), q.size(2))
    k_bshd = k.view(batch_size, seqlen_k, k.size(1), k.size(2))
    v_bshd = v.view(batch_size, seqlen_k, v.size(1), v.size(2))
    qt = q_bshd.transpose(1, 2)
    kt = k_bshd.transpose(1, 2)
    vt = v_bshd.transpose(1, 2)
    enable_gqa = qt.size(1) != kt.size(1)

    def run() -> None:
        if enable_gqa:
            torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=is_causal, enable_gqa=True)
        else:
            torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=is_causal)

    return median_time_ms(run, warmup_iters, num_iters)


def bench_fa2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_q_host: torch.Tensor,
    cu_seqlens_k_host: torch.Tensor,
    batch_size: int,
    is_causal: bool,
    warmup_iters: int,
    num_iters: int,
) -> float | None:
    if flash_attn_varlen_func2 is None:
        return None
    max_seqlen_q = int(cu_seqlens_q_host[1] - cu_seqlens_q_host[0])
    max_seqlen_k = int(cu_seqlens_k_host[1] - cu_seqlens_k_host[0])
    return median_time_ms(
        lambda: take_output(
            flash_attn_varlen_func2(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=is_causal)
        ),
        warmup_iters,
        num_iters,
    )


def bench_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_q_host: torch.Tensor,
    cu_seqlens_k_host: torch.Tensor,
    batch_size: int,
    is_causal: bool,
    warmup_iters: int,
    num_iters: int,
) -> float | None:
    if flash_attn_varlen_func3 is None:
        return None
    max_seqlen_q = int(cu_seqlens_q_host[1] - cu_seqlens_q_host[0])
    max_seqlen_k = int(cu_seqlens_k_host[1] - cu_seqlens_k_host[0])
    return median_time_ms(
        lambda: take_output(
            flash_attn_varlen_func3(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=is_causal)
        ),
        warmup_iters,
        num_iters,
    )


def bench_min_fa3_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_q_host: torch.Tensor,
    cu_seqlens_k_host: torch.Tensor,
    batch_size: int,
    is_causal: bool,
    warmup_iters: int,
    num_iters: int,
    manual_block_count: int | None,
) -> float | None:
    if k.size(1) != v.size(1):
        return None
    if q.size(1) % k.size(1) != 0:
        return None
    if q.size(2) != 128 or k.size(2) != 128 or v.size(2) != 128:
        return None
    max_seqlen_q = int(cu_seqlens_q_host[1] - cu_seqlens_q_host[0])
    max_seqlen_k = int(cu_seqlens_k_host[1] - cu_seqlens_k_host[0])
    return median_time_ms(
        lambda: min_fa3_op.forward_varlen(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            is_causal,
            cu_seqlens_q_host=cu_seqlens_q_host,
            cu_seqlens_k_host=cu_seqlens_k_host,
            manual_block_count=manual_block_count,
        ),
        warmup_iters,
        num_iters,
    )


def bench_case(case: Case, warmup_iters: int, num_iters: int, manual_block_count: int | None) -> dict[str, Result | None]:
    methods: list[tuple[str, Method]] = [
        ("PyTorch_SDPA", bench_pytorch),
        ("FA2_varlen", bench_fa2),
        ("FA3_varlen", bench_fa3),
        ("min_fa3_varlen", bench_min_fa3_varlen),
    ]
    results: dict[str, Result | None] = {}
    try:
        q, k, v, cu_seqlens_q, cu_seqlens_k, cu_seqlens_q_host, cu_seqlens_k_host = make_inputs(case)
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        print(f"[case OOM while allocating inputs: {format_case_key(case)}] [{format_oom(exc)}]")
        for name, _ in methods:
            results[name] = None
        return results

    for name, fn in methods:
        try:
            if name == "min_fa3_varlen":
                time_ms = bench_min_fa3_varlen(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    cu_seqlens_q_host,
                    cu_seqlens_k_host,
                    case.batch_size,
                    case.is_causal,
                    warmup_iters,
                    num_iters,
                    manual_block_count,
                )
            else:
                time_ms = fn(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    cu_seqlens_q_host,
                    cu_seqlens_k_host,
                    case.batch_size,
                    case.is_causal,
                    warmup_iters,
                    num_iters,
                )
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
    methods = ["PyTorch_SDPA", "FA2_varlen", "FA3_varlen", "min_fa3_varlen"]
    print("\n" + "=" * 100)
    print("Varlen Benchmark Results")
    print("=" * 100)
    for case, results in results_by_case.items():
        print(format_case_key(case))
        print(f"{'Method':<18} {'Time (ms)':>12} {'TFLOPS':>12}")
        for method in methods:
            result = results.get(method)
            if result is None:
                print(f"{method:<18} {'N/A':>12} {'N/A':>12}")
            else:
                print(f"{method:<18} {result.time_ms:>12.3f} {result.tflops:>12.1f}")
        print("-" * 100)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.headdim != 128:
        print("Warning: min_fa3_varlen only supports D=128. It will be skipped for other D values.")
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
