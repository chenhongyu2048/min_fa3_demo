"""
Benchmark: min_fa3 varlen mega-ring local demo vs PyTorch vs FA2 vs FA3 vs base min_fa3 varlen

This benchmark follows mega_ring_test_min_fa3_varlen_ring_local.py: K/V are
VMM-backed TKParallelTensor storage, and the ordinary k/v tensors passed to the
FA3 path are direct references to remote_k.data_ / remote_v.data_.

Examples:
    python benchmark_varlen_mega_ring_local.py
    python benchmark_varlen_mega_ring_local.py --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 --headdim 128 --mode both
    python benchmark_varlen_mega_ring_local.py --b 4 --seqlen 1024 --qhead 32 --kvhead 8 --num-comp-sm 128 --num-comm-sm 4 --mode causal
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
            raise SystemExit("--seqlen only accepts one length per case; rectangular SqxSk input is not supported")
        cases.append(int(token))
    if not cases:
        raise SystemExit("--seqlen must provide at least one case")
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark min_fa3 varlen mega-ring local demo against PyTorch / FA2 / FA3 / base min_fa3 varlen"
    )
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
    parser.add_argument("--kvhead", "--kv-head", dest="kvhead", type=int, default=8, help="Number of key/value heads")
    parser.add_argument("--headdim", "--d", dest="headdim", type=int, default=128, help="Head dimension D")
    parser.add_argument(
        "--mode",
        choices=("noncausal", "causal", "both"),
        default="both",
        help="Benchmark noncausal, causal, or both modes",
    )
    parser.add_argument("--num-comp-sm", type=int, default=128, help="Number of compute CTAs for min_fa3_varlen_mega_ring")
    parser.add_argument(
        "--num-comm-sm",
        type=int,
        default=0,
        help="Number of communication CTAs for min_fa3_varlen_mega_ring",
    )
    parser.add_argument("--num-iters", type=int, default=50, help="Timing iterations")
    parser.add_argument("--warmup-iters", type=int, default=50, help="Warmup iterations")
    return parser.parse_args()


def get_flops(case: Case) -> int:
    return 4 * case.batch_size * case.seqlen * case.seqlen * case.qo_heads * case.head_dim // (
        2 if case.is_causal else 1
    )


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


def make_cu_seqlens(batch_size: int, seqlen: int, device: torch.device) -> torch.Tensor:
    return torch.arange(0, (batch_size + 1) * seqlen, seqlen, device=device, dtype=torch.int32)


def make_inputs(
    case: Case,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    min_fa3_op.TKParallelTensor,
    min_fa3_op.TKParallelTensor,
]:
    total_tokens = case.batch_size * case.seqlen
    local_rank = torch.cuda.current_device()
    local_world_size = 1
    q = torch.randn(total_tokens, case.qo_heads, case.head_dim, dtype=dtype, device="cuda")
    remote_k = min_fa3_op.TKParallelTensor(
        [total_tokens, case.kv_heads, case.head_dim],
        dtype,
        local_rank,
        local_world_size,
        False,
    )
    remote_v = min_fa3_op.TKParallelTensor(
        [total_tokens, case.kv_heads, case.head_dim],
        dtype,
        local_rank,
        local_world_size,
        False,
    )
    k = remote_k.data_
    v = remote_v.data_
    k.normal_()
    v.normal_()
    cu_seqlens_q = make_cu_seqlens(case.batch_size, case.seqlen, q.device)
    cu_seqlens_k = make_cu_seqlens(case.batch_size, case.seqlen, q.device)
    return q, k, v, cu_seqlens_q, cu_seqlens_k, remote_k, remote_v


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
    batch_size: int,
    is_causal: bool,
    warmup_iters: int,
    num_iters: int,
) -> float | None:
    seqlen_q = (cu_seqlens_q[1] - cu_seqlens_q[0]).item()
    seqlen_k = (cu_seqlens_k[1] - cu_seqlens_k[0]).item()
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
    batch_size: int,
    is_causal: bool,
    warmup_iters: int,
    num_iters: int,
) -> float | None:
    if flash_attn_varlen_func2 is None:
        return None
    max_seqlen_q = (cu_seqlens_q[1] - cu_seqlens_q[0]).item()
    max_seqlen_k = (cu_seqlens_k[1] - cu_seqlens_k[0]).item()
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
    batch_size: int,
    is_causal: bool,
    warmup_iters: int,
    num_iters: int,
) -> float | None:
    if flash_attn_varlen_func3 is None:
        return None
    max_seqlen_q = (cu_seqlens_q[1] - cu_seqlens_q[0]).item()
    max_seqlen_k = (cu_seqlens_k[1] - cu_seqlens_k[0]).item()
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
    batch_size: int,
    is_causal: bool,
    warmup_iters: int,
    num_iters: int,
    manual_block_count: int,
) -> float | None:
    if k.size(1) != v.size(1):
        return None
    if q.size(1) % k.size(1) != 0:
        return None
    if q.size(2) != 128 or k.size(2) != 128 or v.size(2) != 128:
        return None
    max_seqlen_q = (cu_seqlens_q[1] - cu_seqlens_q[0]).item()
    max_seqlen_k = (cu_seqlens_k[1] - cu_seqlens_k[0]).item()
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
            manual_block_count=manual_block_count,
        ),
        warmup_iters,
        num_iters,
    )


def bench_min_fa3_varlen_mega_ring(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    batch_size: int,
    is_causal: bool,
    warmup_iters: int,
    num_iters: int,
    num_comp_sm: int,
    num_comm_sm: int,
    remote_k: min_fa3_op.TKParallelTensor,
    remote_v: min_fa3_op.TKParallelTensor,
) -> float | None:
    if k.size(1) != v.size(1):
        return None
    if q.size(1) % k.size(1) != 0:
        return None
    if q.size(2) != 128 or k.size(2) != 128 or v.size(2) != 128:
        return None
    if k.size(1) * k.size(2) != 1024:
        return None
    max_seqlen_q = (cu_seqlens_q[1] - cu_seqlens_q[0]).item()
    max_seqlen_k = (cu_seqlens_k[1] - cu_seqlens_k[0]).item()
    return median_time_ms(
        lambda: min_fa3_op.forward_varlen_mega_ring(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            is_causal,
            remote_k=remote_k,
            remote_v=remote_v,
            num_comp_sm=num_comp_sm,
            num_comm_sm=num_comm_sm,
        ),
        warmup_iters,
        num_iters,
    )


def bench_case(
    case: Case,
    warmup_iters: int,
    num_iters: int,
    num_comp_sm: int,
    num_comm_sm: int,
) -> dict[str, Result | None]:
    methods: list[tuple[str, Method]] = [
        ("PyTorch_SDPA", bench_pytorch),
        ("FA2_varlen", bench_fa2),
        ("FA3_varlen", bench_fa3),
        ("min_fa3_varlen", bench_min_fa3_varlen),
        ("min_fa3_varlen_mega_ring", bench_min_fa3_varlen_mega_ring),
    ]
    results: dict[str, Result | None] = {}
    try:
        q, k, v, cu_seqlens_q, cu_seqlens_k, remote_k, remote_v = make_inputs(case)
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
                    case.batch_size,
                    case.is_causal,
                    warmup_iters,
                    num_iters,
                    num_comp_sm,
                )
            elif name == "min_fa3_varlen_mega_ring":
                time_ms = bench_min_fa3_varlen_mega_ring(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    case.batch_size,
                    case.is_causal,
                    warmup_iters,
                    num_iters,
                    num_comp_sm,
                    num_comm_sm,
                    remote_k,
                    remote_v,
                )
            else:
                time_ms = fn(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
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
    methods = ["PyTorch_SDPA", "FA2_varlen", "FA3_varlen", "min_fa3_varlen", "min_fa3_varlen_mega_ring"]
    print("\n" + "=" * 100)
    print("Varlen Mega-Ring Local Benchmark Results")
    print("=" * 100)
    for case, results in results_by_case.items():
        print(format_case_key(case))
        print(f"{'Method':<28} {'Time (ms)':>12} {'TFLOPS':>12}")
        for method in methods:
            result = results.get(method)
            if result is None:
                print(f"{method:<28} {'N/A':>12} {'N/A':>12}")
            else:
                print(f"{method:<28} {result.time_ms:>12.3f} {result.tflops:>12.1f}")
        print("-" * 100)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (9, 0):
        raise SystemExit(f"This demo requires SM90 Hopper, got {(major, minor)}")
    if args.headdim != 128:
        raise SystemExit(f"This demo requires D=128, got D={args.headdim}")
    if args.qhead % args.kvhead != 0:
        raise SystemExit(
            f"This benchmark requires qhead % kvhead == 0 for GQA/MQA, got qhead={args.qhead}, kvhead={args.kvhead}"
        )
    if args.kvhead * args.headdim != 1024:
        raise SystemExit(
            "Mega ring communication path requires kvhead * headdim == 1024, "
            f"got kvhead={args.kvhead}, headdim={args.headdim}"
        )
    if args.num_comp_sm <= 0:
        raise SystemExit(f"--num-comp-sm must be positive, got num_comp_sm={args.num_comp_sm}")
    if args.num_comm_sm < 0:
        raise SystemExit(f"--num-comm-sm must be non-negative, got num_comm_sm={args.num_comm_sm}")
    if args.num_iters <= 0:
        raise SystemExit(f"--num-iters must be positive, got num_iters={args.num_iters}")
    if args.warmup_iters < 0:
        raise SystemExit(f"--warmup-iters must be non-negative, got warmup_iters={args.warmup_iters}")

    cases = parse_cases(args)
    print(
        f"Config: B={args.b}, qhead={args.qhead}, kvhead={args.kvhead}, D={args.headdim}, "
        f"mode={args.mode}, num_iters={args.num_iters}, warmup_iters={args.warmup_iters}, "
        f"num_comp_sm={args.num_comp_sm}, num_comm_sm={args.num_comm_sm}"
    )
    print(f"Seqlen: {args.seqlen}")

    results_by_case: dict[Case, dict[str, Result | None]] = {}
    for case in cases:
        print(f"Running {format_case_key(case)}", flush=True)
        results_by_case[case] = bench_case(
            case,
            args.warmup_iters,
            args.num_iters,
            args.num_comp_sm,
            args.num_comm_sm,
        )

    print_table(results_by_case)


if __name__ == "__main__":
    main()
