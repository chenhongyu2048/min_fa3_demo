"""Benchmark the minimal varlen backward against the installed FA3 backward."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Callable

import torch

import min_fa3_op

try:
    from flash_attn_interface import _flash_attn_backward, _flash_attn_forward
except ImportError:
    _flash_attn_backward = None
    _flash_attn_forward = None


@dataclass(frozen=True)
class Case:
    batch_size: int
    seqlen: int
    q_heads: int
    kv_heads: int
    head_dim: int
    is_causal: bool


@dataclass(frozen=True)
class Result:
    time_ms: float
    tflops: float


def parse_lengths(spec: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise SystemExit("--seqlen must provide at least one case")
    return values


def median_time_ms(fn: Callable[[], object], warmup_iters: int, num_iters: int) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for index in range(num_iters):
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))


def backward_flops(case: Case) -> float:
    flops = 10 * case.batch_size * case.seqlen * case.seqlen * case.q_heads * case.head_dim
    return flops / (2 if case.is_causal else 1)


def make_inputs(case: Case) -> tuple[torch.Tensor, ...]:
    total = case.batch_size * case.seqlen
    q = torch.randn(total, case.q_heads, case.head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, case.kv_heads, case.head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    dout = torch.randn_like(q)
    host = torch.arange(0, total + 1, case.seqlen, dtype=torch.int32)
    return q, k, v, dout, host.cuda(), host


def bench_min_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    cu: torch.Tensor,
    cu_host: torch.Tensor,
    case: Case,
    deterministic: bool,
    warmup_iters: int,
    num_iters: int,
) -> float:
    out, lse = min_fa3_op.forward_varlen(
        q, k, v, cu, cu, case.seqlen, case.seqlen, case.is_causal,
        cu_seqlens_q_host=cu_host, cu_seqlens_k_host=cu_host, return_lse=True
    )
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    return median_time_ms(
        lambda: min_fa3_op.backward_varlen(
            dout, q, k, v, out, lse, cu, cu, case.seqlen, case.seqlen, case.is_causal,
            deterministic=deterministic, dq=dq, dk=dk, dv=dv
        ),
        warmup_iters,
        num_iters,
    )


def bench_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    cu: torch.Tensor,
    cu_host: torch.Tensor,
    case: Case,
    deterministic: bool,
    warmup_iters: int,
    num_iters: int,
) -> float | None:
    if _flash_attn_forward is None or _flash_attn_backward is None:
        return None
    out, lse, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=case.seqlen,
        max_seqlen_k=case.seqlen,
        causal=case.is_causal,
    )
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    return median_time_ms(
        lambda: _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=case.seqlen,
            max_seqlen_k=case.seqlen,
            dq=dq,
            dk=dk,
            dv=dv,
            softmax_scale=q.size(-1) ** -0.5,
            is_causal=case.is_causal,
            deterministic=deterministic,
        ),
        warmup_iters,
        num_iters,
    )


def run_case(case: Case, args: argparse.Namespace) -> dict[str, Result | None]:
    q, k, v, dout, cu, cu_host = make_inputs(case)
    results: dict[str, Result | None] = {}
    methods = (("FA3_varlen", bench_fa3), ("min_fa3_varlen", bench_min_fa3))
    for name, method in methods:
        try:
            elapsed = method(
                q, k, v, dout, cu, cu_host, case, args.deterministic,
                args.warmup_iters, args.num_iters
            )
            results[name] = None if elapsed is None else Result(
                elapsed, backward_flops(case) / (elapsed * 1e-3) / 1e12
            )
        except torch.OutOfMemoryError as exc:
            print(f"[{name} OOM: {str(exc).splitlines()[0]}]")
            torch.cuda.empty_cache()
            results[name] = None
        except Exception as exc:
            print(f"[{name} error: {exc}]")
            results[name] = None
    return results


def print_results(case: Case, results: dict[str, Result | None]) -> None:
    fa3 = results.get("FA3_varlen")
    print(
        f"B={case.batch_size},S={case.seqlen},QH={case.q_heads},KVH={case.kv_heads},"
        f"D={case.head_dim},causal={case.is_causal}"
    )
    print(f"{'Method':<18} {'Time (ms)':>12} {'Bwd TFLOPS':>14} {'vs FA3':>10}")
    for name in ("FA3_varlen", "min_fa3_varlen"):
        result = results.get(name)
        if result is None:
            print(f"{name:<18} {'N/A':>12} {'N/A':>14} {'N/A':>10}")
            continue
        relative = "1.000x" if name == "FA3_varlen" else (
            f"{fa3.time_ms / result.time_ms:.3f}x" if fa3 is not None else "N/A"
        )
        print(f"{name:<18} {result.time_ms:>12.3f} {result.tflops:>14.1f} {relative:>10}")
    print("-" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark minimal FA3 varlen backward against FA3")
    parser.add_argument("--b", type=int, default=4)
    parser.add_argument("--seqlen", type=str, default="512,1024,2048,4096")
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=32)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--mode", choices=("noncausal", "causal", "both"), default="both")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit(f"This benchmark requires SM90, got {torch.cuda.get_device_capability()}")
    if args.headdim != 128 or args.qhead % args.kvhead != 0:
        raise SystemExit("This benchmark requires D=128 and qhead divisible by kvhead")
    causal_values = {"noncausal": (False,), "causal": (True,), "both": (False, True)}[args.mode]
    for causal in causal_values:
        for seqlen in parse_lengths(args.seqlen):
            case = Case(args.b, seqlen, args.qhead, args.kvhead, args.headdim, causal)
            print_results(case, run_case(case, args))
            torch.cuda.empty_cache()
