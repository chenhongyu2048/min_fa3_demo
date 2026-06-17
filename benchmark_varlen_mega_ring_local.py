"""
Benchmark: min_fa3 varlen mega-ring local demo vs PyTorch vs FA2 vs FA3 vs base min_fa3 varlen vs min_fa3 varlen ring
We add reduction to PyTorch / FA2 / FA3 to monitor the time spent in the
attention kernel and the time spent in the output/LSE merging logic.

This benchmark follows mega_ring_test_min_fa3_varlen_ring_local.py: K/V are
VMM-backed TKParallelTensor storage, and the ordinary k/v tensors passed to the
FA3 path are direct references to remote_k.data_ / remote_v.data_.

Examples:
    python benchmark_varlen_mega_ring_local.py
    python benchmark_varlen_mega_ring_local.py --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 --headdim 128 --mode both
    python benchmark_varlen_mega_ring_local.py --b 4 --seqlen 1024 --qhead 32 --kvhead 8 --num-comp-sm 128 --num-comm-sm 4 --mode causal
    nsys profile -t cuda,nvtx,osrt -o my_report --stats=true python benchmark_varlen_mega_ring_local.py --profile --b 16 --seqlen 1024 --qhead 32 --kvhead 8 --headdim 128 --mode noncausal
"""

from __future__ import annotations

import argparse
import statistics
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

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
    attn_ms: float | None = None
    reduction_ms: float | None = None


@dataclass(frozen=True)
class TimingBreakdown:
    total_ms: float
    attn_ms: float | None = None
    reduction_ms: float | None = None


Method = Callable[..., TimingBreakdown | None]


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


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
        description=(
            "Benchmark min_fa3 varlen mega-ring local demo against PyTorch / FA2 / FA3 / "
            "base min_fa3 varlen / min_fa3 varlen ring"
        )
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
    parser.add_argument(
        "--num-comp-sm",
        type=int,
        default=128,
        help="Number of compute CTAs for min_fa3_varlen_ring and min_fa3_varlen_mega_ring",
    )
    parser.add_argument(
        "--num-comm-sm",
        type=int,
        default=0,
        help="Number of communication CTAs for min_fa3_varlen_ring and min_fa3_varlen_mega_ring",
    )
    parser.add_argument("--num-iters", type=int, default=50, help="Timing iterations")
    parser.add_argument("--warmup-iters", type=int, default=50, help="Warmup iterations")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Emit NVTX ranges around measured timing loops for nsys profiling",
    )
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


def make_cu_seqlens(batch_size: int, seqlen: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.arange(0, (batch_size + 1) * seqlen, seqlen, dtype=torch.int32)
    return host.to(device=device), host


def make_inputs(
    case: Case,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
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
    cu_seqlens_q, cu_seqlens_q_host = make_cu_seqlens(case.batch_size, case.seqlen, q.device)
    cu_seqlens_k, cu_seqlens_k_host = make_cu_seqlens(case.batch_size, case.seqlen, q.device)
    return q, k, v, cu_seqlens_q, cu_seqlens_k, cu_seqlens_q_host, cu_seqlens_k_host, remote_k, remote_v


@contextmanager
def nvtx_range(label: str | None):
    if label is None:
        yield
        return
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def median_time_ms(
    fn: Callable[[], None],
    warmup_iters: int,
    num_iters: int,
    profile_label: str | None = None,
) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    with nvtx_range(profile_label):
        for i in range(num_iters):
            start_events[i].record()
            fn()
            end_events[i].record()
        torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(start_events, end_events))


def median_split_time_ms(
    attn_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    reduction_fn: Callable[[torch.Tensor, torch.Tensor], None],
    warmup_iters: int,
    num_iters: int,
    profile_label: str | None = None,
) -> TimingBreakdown:
    for _ in range(warmup_iters):
        block_out, block_lse = attn_fn()
        reduction_fn(block_out, block_lse)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    attn_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    with nvtx_range(profile_label):
        for i in range(num_iters):
            start_events[i].record()
            block_out, block_lse = attn_fn()
            attn_end_events[i].record()
            reduction_fn(block_out, block_lse)
            end_events[i].record()
        torch.cuda.synchronize()
    total_ms = statistics.median(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    attn_ms = statistics.median(s.elapsed_time(e) for s, e in zip(start_events, attn_end_events))
    reduction_ms = statistics.median(s.elapsed_time(e) for s, e in zip(attn_end_events, end_events))
    return TimingBreakdown(total_ms=total_ms, attn_ms=attn_ms, reduction_ms=reduction_ms)


def make_batch_reduction_buffers(
    batch_size: int,
    seqlen: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float32, device=device)
    lse = torch.randn(batch_size, seqlen, num_heads, 1, dtype=torch.float32, device=device)
    return out, lse


def make_varlen_reduction_buffers(
    total_q: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.randn(total_q, num_heads, head_dim, dtype=torch.float32, device=device)
    lse = torch.randn(total_q, num_heads, 1, dtype=torch.float32, device=device)
    return out, lse


def take_output_and_lse(result) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("expected attention backend to return at least (out, softmax_lse)")
    return result[0], result[1]


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
    profile_label: str | None = None,
) -> TimingBreakdown | None:
    seqlen_q = int(cu_seqlens_q_host[1] - cu_seqlens_q_host[0])
    seqlen_k = int(cu_seqlens_k_host[1] - cu_seqlens_k_host[0])
    q_bshd = q.view(batch_size, seqlen_q, q.size(1), q.size(2))
    k_bshd = k.view(batch_size, seqlen_k, k.size(1), k.size(2))
    v_bshd = v.view(batch_size, seqlen_k, v.size(1), v.size(2))
    qt = q_bshd.transpose(1, 2).contiguous()
    kt = k_bshd.transpose(1, 2).contiguous()
    vt = v_bshd.transpose(1, 2).contiguous()
    enable_gqa = qt.size(1) != kt.size(1)
    old_out, old_lse = make_batch_reduction_buffers(batch_size, seqlen_q, q.size(1), q.size(2), q.device)

    def attn_fn() -> tuple[torch.Tensor, torch.Tensor]:
        block_k = kt.repeat_interleave(qt.size(1) // kt.size(1), dim=1) if enable_gqa else kt
        block_v = vt.repeat_interleave(qt.size(1) // vt.size(1), dim=1) if enable_gqa else vt
        block_out, block_lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention.default(
            qt,
            block_k,
            block_v,
            0.0,
            is_causal,
            False,
        )
        return block_out.transpose(1, 2), block_lse

    def reduction_fn(block_out: torch.Tensor, block_lse: torch.Tensor) -> None:
        merged_out, merged_lse = update_out_and_lse(old_out, old_lse, block_out, block_lse)
        del merged_out, merged_lse

    return median_split_time_ms(attn_fn, reduction_fn, warmup_iters, num_iters, profile_label)


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
    profile_label: str | None = None,
) -> TimingBreakdown | None:
    if flash_attn_varlen_func2 is None:
        return None
    max_seqlen_q = int(cu_seqlens_q_host[1] - cu_seqlens_q_host[0])
    max_seqlen_k = int(cu_seqlens_k_host[1] - cu_seqlens_k_host[0])
    old_out, old_lse = make_varlen_reduction_buffers(q.size(0), q.size(1), q.size(2), q.device)

    def attn_fn() -> tuple[torch.Tensor, torch.Tensor]:
        return take_output_and_lse(
            flash_attn_varlen_func2(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=is_causal,
                return_attn_probs=True,
            )
        )

    def reduction_fn(block_out: torch.Tensor, block_lse: torch.Tensor) -> None:
        merged_out, merged_lse = update_out_and_lse(old_out, old_lse, block_out, block_lse)
        del merged_out, merged_lse

    return median_split_time_ms(attn_fn, reduction_fn, warmup_iters, num_iters, profile_label)


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
    profile_label: str | None = None,
) -> TimingBreakdown | None:
    if flash_attn_varlen_func3 is None:
        return None
    max_seqlen_q = int(cu_seqlens_q_host[1] - cu_seqlens_q_host[0])
    max_seqlen_k = int(cu_seqlens_k_host[1] - cu_seqlens_k_host[0])
    old_out, old_lse = make_varlen_reduction_buffers(q.size(0), q.size(1), q.size(2), q.device)

    def attn_fn() -> tuple[torch.Tensor, torch.Tensor]:
        return take_output_and_lse(
            flash_attn_varlen_func3(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=is_causal,
                return_attn_probs=True,
            )
        )

    def reduction_fn(block_out: torch.Tensor, block_lse: torch.Tensor) -> None:
        merged_out, merged_lse = update_out_and_lse(old_out, old_lse, block_out, block_lse)
        del merged_out, merged_lse

    return median_split_time_ms(attn_fn, reduction_fn, warmup_iters, num_iters, profile_label)


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
    manual_block_count: int,
    profile_label: str | None = None,
) -> TimingBreakdown | None:
    if k.size(1) != v.size(1):
        return None
    if q.size(1) % k.size(1) != 0:
        return None
    if q.size(2) != 128 or k.size(2) != 128 or v.size(2) != 128:
        return None
    max_seqlen_q = int(cu_seqlens_q_host[1] - cu_seqlens_q_host[0])
    max_seqlen_k = int(cu_seqlens_k_host[1] - cu_seqlens_k_host[0])
    return TimingBreakdown(
        total_ms=median_time_ms(
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
            profile_label,
        )
    )


def bench_min_fa3_varlen_ring(
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
    num_comp_sm: int,
    num_comm_sm: int,
    remote_k: min_fa3_op.TKParallelTensor,
    remote_v: min_fa3_op.TKParallelTensor,
    prefetch_k: torch.Tensor,
    prefetch_v: torch.Tensor,
    profile_label: str | None = None,
) -> TimingBreakdown | None:
    if k.size(1) != v.size(1):
        return None
    if q.size(1) % k.size(1) != 0:
        return None
    if q.size(2) != 128 or k.size(2) != 128 or v.size(2) != 128:
        return None
    max_seqlen_q = int(cu_seqlens_q_host[1] - cu_seqlens_q_host[0])
    max_seqlen_k = int(cu_seqlens_k_host[1] - cu_seqlens_k_host[0])
    return TimingBreakdown(
        total_ms=median_time_ms(
            lambda: min_fa3_op.forward_varlen_ring(
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
                remote_k=remote_k,
                remote_v=remote_v,
                src_rank=0,
                num_comp_sm=num_comp_sm,
                num_comm_sm=num_comm_sm,
                ring_step=0,
                prefetch_k=prefetch_k,
                prefetch_v=prefetch_v,
            ),
            warmup_iters,
            num_iters,
            profile_label,
        )
    )


def bench_min_fa3_varlen_mega_ring(
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
    num_comp_sm: int,
    num_comm_sm: int,
    remote_k: min_fa3_op.TKParallelTensor,
    remote_v: min_fa3_op.TKParallelTensor,
    profile_label: str | None = None,
) -> TimingBreakdown | None:
    if k.size(1) != v.size(1):
        return None
    if q.size(1) % k.size(1) != 0:
        return None
    if q.size(2) != 128 or k.size(2) != 128 or v.size(2) != 128:
        return None
    if k.size(1) * k.size(2) != 1024:
        return None
    max_seqlen_q = int(cu_seqlens_q_host[1] - cu_seqlens_q_host[0])
    max_seqlen_k = int(cu_seqlens_k_host[1] - cu_seqlens_k_host[0])
    return TimingBreakdown(
        total_ms=median_time_ms(
            lambda: min_fa3_op.forward_varlen_mega_ring(
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
                remote_k=remote_k,
                remote_v=remote_v,
                num_comp_sm=num_comp_sm,
                num_comm_sm=num_comm_sm,
            ),
            warmup_iters,
            num_iters,
            profile_label,
        )
    )


def bench_case(
    case: Case,
    warmup_iters: int,
    num_iters: int,
    num_comp_sm: int,
    num_comm_sm: int,
    profile: bool = False,
) -> dict[str, Result | None]:
    methods: list[tuple[str, Method]] = [
        ("PyTorch_SDPA", bench_pytorch),
        ("FA2_varlen", bench_fa2),
        ("FA3_varlen", bench_fa3),
        ("min_fa3_varlen", bench_min_fa3_varlen),
        ("min_fa3_varlen_ring", bench_min_fa3_varlen_ring),
        ("min_fa3_varlen_mega_ring", bench_min_fa3_varlen_mega_ring),
    ]
    results: dict[str, Result | None] = {}
    try:
        (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_seqlens_q_host,
            cu_seqlens_k_host,
            remote_k,
            remote_v,
        ) = make_inputs(case)
        prefetch_k = torch.empty_like(k)
        prefetch_v = torch.empty_like(v)
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        print(f"[case OOM while allocating inputs: {format_case_key(case)}] [{format_oom(exc)}]")
        for name, _ in methods:
            results[name] = None
        return results

    for name, fn in methods:
        profile_label = format_profile_label(name, case, num_comp_sm, num_comm_sm) if profile else None
        try:
            if name == "min_fa3_varlen":
                timing = bench_min_fa3_varlen(
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
                    num_comp_sm,
                    profile_label,
                )
            elif name == "min_fa3_varlen_ring":
                timing = bench_min_fa3_varlen_ring(
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
                    num_comp_sm,
                    num_comm_sm,
                    remote_k,
                    remote_v,
                    prefetch_k,
                    prefetch_v,
                    profile_label,
                )
            elif name == "min_fa3_varlen_mega_ring":
                timing = bench_min_fa3_varlen_mega_ring(
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
                    num_comp_sm,
                    num_comm_sm,
                    remote_k,
                    remote_v,
                    profile_label,
                )
            else:
                timing = fn(
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
                    profile_label,
                )
            if timing is None:
                results[name] = None
                continue
            tflops = get_flops(case) / (timing.total_ms * 1e-3) / 1e12
            results[name] = Result(
                time_ms=timing.total_ms,
                tflops=tflops,
                attn_ms=timing.attn_ms,
                reduction_ms=timing.reduction_ms,
            )
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


def format_profile_label(method: str, case: Case, num_comp_sm: int, num_comm_sm: int) -> str:
    label = f"{method} {format_case_key(case)}"
    if method == "min_fa3_varlen":
        return f"{label},manual_block_count={num_comp_sm}"
    if method in ("min_fa3_varlen_ring", "min_fa3_varlen_mega_ring"):
        return f"{label},num_comp_sm={num_comp_sm},num_comm_sm={num_comm_sm}"
    return label


def print_table(results_by_case: dict[Case, dict[str, Result | None]]) -> None:
    methods = [
        "PyTorch_SDPA",
        "FA2_varlen",
        "FA3_varlen",
        "min_fa3_varlen",
        "min_fa3_varlen_ring",
        "min_fa3_varlen_mega_ring",
    ]
    print("\n" + "=" * 110)
    print("Varlen Mega-Ring Local Benchmark Results")
    print("=" * 110)
    for case, results in results_by_case.items():
        print(format_case_key(case))
        print(f"{'Method':<28} {'Attn (ms)':>12} {'Reduce (ms)':>12} {'Time (ms)':>12} {'TFLOPS':>12}")
        for method in methods:
            result = results.get(method)
            if result is None:
                print(f"{method:<28} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
            else:
                attn_ms = f"{result.attn_ms:>12.3f}" if result.attn_ms is not None else f"{'N/A':>12}"
                reduction_ms = (
                    f"{result.reduction_ms:>12.3f}" if result.reduction_ms is not None else f"{'N/A':>12}"
                )
                print(f"{method:<28} {attn_ms} {reduction_ms} {result.time_ms:>12.3f} {result.tflops:>12.1f}")
        print("-" * 110)


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
        f"num_comp_sm={args.num_comp_sm}, num_comm_sm={args.num_comm_sm}, profile={args.profile}"
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
            args.profile,
        )

    print_table(results_by_case)


if __name__ == "__main__":
    main()
