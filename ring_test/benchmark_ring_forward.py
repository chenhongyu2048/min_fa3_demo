"""Torchrun entry point for multi-rank forward attention benchmarks.

The script compares an all-gather baseline and Python-side ring attention using
PyTorch/FA2/FA3 block kernels with the local min_fa3 varlen, min_fa3
single-step ring, and fused min_fa3 mega-ring paths. Timing is end-to-end per
method call and reports the maximum elapsed time across ranks.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist

THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import min_fa3_op

from allgather_attention import AllGatherAttention, select_allgather_backend
from ring_common import (
    flash_varlen_block_attention,
    gather_rank_tensor,
    min_fa3_varlen_block_attention,
    pytorch_varlen_block_attention,
    raise_if_any_rank_failed,
    reference_ring_varlen,
    reference_zigzag_ring_varlen,
    ring_varlen_forward,
    zigzag_ring_varlen_forward,
)

try:
    from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func2
except ImportError:
    print("FA2 flash_attn_varlen_func is not available, skipping FA2 benchmarks", flush=True)
    flash_attn_varlen_func2 = None

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func3
except ImportError:
    print("FA3 flash_attn_varlen_func is not available, skipping FA3 benchmarks", flush=True)
    flash_attn_varlen_func3 = None


METHOD_ORDER = [
    "allgather_attention",
    # "pytorch",
    # "fa2",
    "fa3",
    "min_varlen",
    "min_varlen_ring",
    "min_varlen_mega_ring",
]


@dataclass(frozen=True)
class Case:
    """One benchmark problem shape and attention mode."""

    batch_size: int
    seqlen: int
    q_heads: int
    kv_heads: int
    head_dim: int
    is_causal: bool


@dataclass(frozen=True)
class SmConfig:
    """One compute/communication SM allocation."""

    num_comp_sm: int
    num_comm_sm: int


@dataclass
class MethodRun:
    """Callable bundle for a benchmark method.

    `timing_fn` is what is measured and, for complete methods, what is checked.
    `checkable=False` marks timing-only paths.
    """

    name: str
    timing_fn: Callable[[], torch.Tensor]
    note: str = ""
    checkable: bool = True


@dataclass
class Result:
    """Printable benchmark result for one method."""

    time_ms: Optional[float]
    tflops: Optional[float]
    check: str
    note: str = ""
    rank_times_ms: Optional[list[float]] = None
    avg_gpu_tflops: Optional[float] = None


@dataclass
class TimingResult:
    """CUDA-event timing summary for one measured method."""

    local_time_ms: float
    max_time_ms: float
    rank_times_ms: Optional[list[float]]


def parse_seqlen_spec(spec: str) -> list[int]:
    """Parse the unified comma-separated local sequence length CLI format."""
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


def parse_methods(spec: str) -> list[str]:
    """Parse the method list while preserving user order and removing duplicates."""
    methods: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "all":
            methods.extend(METHOD_ORDER)
        elif token in METHOD_ORDER:
            methods.append(token)
        else:
            raise SystemExit(f"unknown method '{token}', expected one of {METHOD_ORDER} or all")

    deduped: list[str] = []
    for method in methods:
        if method not in deduped:
            deduped.append(method)
    return deduped


def parse_sm_config_spec(spec: str) -> list[SmConfig]:
    """Parse comma-separated num_comp_sm:num_comm_sm pairs."""
    configs: list[SmConfig] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise SystemExit(f"invalid --sm-configs token '{token}', expected num_comp_sm:num_comm_sm")
        try:
            configs.append(SmConfig(int(parts[0]), int(parts[1])))
        except ValueError as exc:
            raise SystemExit(
                f"invalid --sm-configs token '{token}', expected integer num_comp_sm:num_comm_sm"
            ) from exc
    if not configs:
        raise SystemExit("--sm-configs must provide at least one num_comp_sm:num_comm_sm pair")
    return configs


def parse_args() -> argparse.Namespace:
    """Define the ring_test command-line interface."""
    parser = argparse.ArgumentParser(
        description="Distributed forward-only ring-attention test/benchmark for varlen backends."
    )
    parser.add_argument("--b", type=int, default=1, help="Batch size B per rank.")
    parser.add_argument(
        "--seqlen",
        "--seqlens",
        dest="seqlen",
        type=str,
        default="256",
        help="Comma-separated local sequence lengths S per rank.",
    )
    parser.add_argument("--qhead", type=int, default=8, help="Number of query/output heads.")
    parser.add_argument("--kvhead", type=int, default=8, help="Number of key/value heads.")
    parser.add_argument("--headdim", type=int, default=128, help="Head dimension D.")
    parser.add_argument(
        "--mode",
        choices=("noncausal", "causal", "both"),
        default="both",
        help="Run noncausal, causal, or both.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help=f"Comma-separated methods from {METHOD_ORDER}, or all.",
    )
    parser.add_argument("--num-comp-sm", type=int, default=1, help="Compute CTAs for min_fa3 ring kernels.")
    parser.add_argument("--num-comm-sm", type=int, default=1, help="Communication CTAs for min_fa3 ring kernels.")
    parser.add_argument(
        "--sm-configs",
        type=str,
        default=None,
        help=(
            "Comma-separated num_comp_sm:num_comm_sm pairs to run in one invocation, "
            "for example 128:4,124:8,116:16. Overrides --num-comp-sm/--num-comm-sm."
        ),
    )
    parser.add_argument("--warmup-iters", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--num-iters", type=int, default=20, help="Measured iterations.")
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed.")
    parser.add_argument("--check", dest="check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float, default=2e-1, help="Correctness absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=2e-1, help="Correctness relative tolerance.")
    return parser.parse_args()


def init_distributed() -> tuple[int, int]:
    """Initialize single-node NCCL distributed execution.

    TKParallelTensor remote-load paths are local IPC paths, so this script
    intentionally rejects multi-node world sizes where WORLD_SIZE differs from
    LOCAL_WORLD_SIZE.
    """
    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit("Run with torchrun so LOCAL_RANK and LOCAL_WORLD_SIZE are set")

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        dist.init_process_group(backend="nccl", device_id=device)
    except TypeError:
        dist.init_process_group(backend="nccl")

    if dist.get_world_size() != local_world_size:
        raise SystemExit(
            "This ring_test path is single-node only because TKParallelTensor uses local IPC: "
            f"world_size={dist.get_world_size()}, local_world_size={local_world_size}"
        )
    return local_rank, local_world_size


def available_on_all_ranks(local_available: bool) -> bool:
    """Return True only if the optional backend is importable on every rank."""
    available = torch.tensor([1 if local_available else 0], device="cuda", dtype=torch.int32)
    dist.all_reduce(available, op=dist.ReduceOp.MIN)
    return bool(available.item())


def cuda_barrier() -> None:
    """Barrier with an explicit CUDA device to avoid NCCL device-selection warnings."""
    try:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    except TypeError:
        dist.barrier()


def make_cu_seqlens(batch_size: int, seqlen: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create matching device and host cumulative sequence length tensors."""
    host = torch.arange(0, (batch_size + 1) * seqlen, seqlen, dtype=torch.int32)
    return host.to(device=device), host


def make_inputs(case: Case, local_rank: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate deterministic rank-local Q/K/V inputs for one case."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + local_rank * 1009 + (1 if case.is_causal else 0))
    total_tokens = case.batch_size * case.seqlen
    q = torch.randn(
        total_tokens,
        case.q_heads,
        case.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn(
        total_tokens,
        case.kv_heads,
        case.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    v = torch.randn(
        total_tokens,
        case.kv_heads,
        case.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    return q.contiguous(), k.contiguous(), v.contiguous()


def make_ring_parallel_tensors(
    k: torch.Tensor,
    v: torch.Tensor,
    local_rank: int,
    local_world_size: int,
) -> tuple[min_fa3_op.TKParallelTensor, min_fa3_op.TKParallelTensor]:
    """Create same-shape TKParallelTensor wrappers for the single-step ring path."""
    remote_k = min_fa3_op.TKParallelTensor(list(k.shape), torch.bfloat16, local_rank, local_world_size, False)
    remote_v = min_fa3_op.TKParallelTensor(list(v.shape), torch.bfloat16, local_rank, local_world_size, False)
    remote_k.data_.copy_(k)
    remote_v.data_.copy_(v)
    return remote_k, remote_v


def make_mega_parallel_tensors(
    k: torch.Tensor,
    v: torch.Tensor,
    local_rank: int,
    local_world_size: int,
) -> tuple[min_fa3_op.TKParallelTensor, min_fa3_op.TKParallelTensor]:
    """Create full [world_size * local_tokens, KVH, D] TK buffers for mega-ring.

    Each rank initializes only its own block. The mega-ring kernel's communication
    CTAs remotely load the other rank blocks into the same full local buffer.
    """
    full_shape = [local_world_size * k.size(0), k.size(1), k.size(2)]
    remote_k = min_fa3_op.TKParallelTensor(full_shape, torch.bfloat16, local_rank, local_world_size, False)
    remote_v = min_fa3_op.TKParallelTensor(full_shape, torch.bfloat16, local_rank, local_world_size, False)
    remote_k.data_.zero_()
    remote_v.data_.zero_()
    start = local_rank * k.size(0)
    end = start + k.size(0)
    remote_k.data_[start:end].copy_(k)
    remote_v.data_[start:end].copy_(v)
    return remote_k, remote_v


def min_varlen_local_steps_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_q_host: torch.Tensor,
    cu_seqlens_k_host: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool,
    rank: int,
    world_size: int,
    num_comp_sm: int,
) -> torch.Tensor:
    """Launch the base min_fa3 varlen kernel once per visible ring step.

    This path intentionally performs no inter-rank communication and no Python
    output/LSE reduction. It exists to compare the cost of repeatedly launching
    the local base varlen kernel under the same step count.
    """
    out = None
    for step in range(world_size):
        if not is_causal or step <= rank:
            out = min_fa3_op.forward_varlen(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                is_causal and step == 0,
                cu_seqlens_q_host=cu_seqlens_q_host,
                cu_seqlens_k_host=cu_seqlens_k_host,
                manual_block_count=num_comp_sm,
            )
    if out is None:
        raise RuntimeError("min_varlen local step loop produced no output")
    return out


def min_varlen_ring_steps_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_q_host: torch.Tensor,
    cu_seqlens_k_host: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool,
    rank: int,
    world_size: int,
    remote_k: min_fa3_op.TKParallelTensor,
    remote_v: min_fa3_op.TKParallelTensor,
    num_comp_sm: int,
    num_comm_sm: int,
) -> torch.Tensor:
    """Launch the min_fa3 single-step ring kernel across ring steps.

    The same output and LSE buffers are passed through every visible ring step,
    so the kernel's device-side online reduction produces the full-ring result.
    """
    if world_size > 1 and num_comm_sm <= 0:
        raise RuntimeError("min_varlen_ring multi-rank path requires num_comm_sm > 0")

    prefetch_k = [torch.empty_like(k), torch.empty_like(k)]
    prefetch_v = [torch.empty_like(v), torch.empty_like(v)]
    cur_k = k
    cur_v = v
    out = torch.zeros_like(q)
    lse = torch.full((q.size(1), q.size(0)), float("-inf"), device=q.device, dtype=torch.float32)
    produced_output = False

    for step in range(world_size):
        if not is_causal or step <= rank:
            # At step s, rank r consumes the K/V block from rank (r - s).
            # The communication CTAs prefetch rank (r - s - 1) for the next step.
            # Obviously the prefetching doesn't depend on the prefetch buffer of the `next_src_rank`.
            next_src_rank = (rank - step - 1 + world_size) % world_size
            buffer_idx = step % 2
            launch_comm_sm = num_comm_sm if step + 1 < world_size else 0
            out, lse = min_fa3_op.forward_varlen_ring(
                q,
                cur_k,
                cur_v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                is_causal and step == 0,
                cu_seqlens_q_host=cu_seqlens_q_host,
                cu_seqlens_k_host=cu_seqlens_k_host,
                remote_k=remote_k,
                remote_v=remote_v,
                src_rank=next_src_rank,
                num_comp_sm=num_comp_sm,
                num_comm_sm=launch_comm_sm,
                ring_step=step,
                prefetch_k=prefetch_k[buffer_idx],
                prefetch_v=prefetch_v[buffer_idx],
                out=out,
                lse=lse,
                return_lse=True,
            )
            produced_output = True
            if step + 1 < world_size:
                # The prefetch buffer written by this kernel launch is consumed
                # as the local K/V input for the next visible step.
                cur_k = prefetch_k[buffer_idx]
                cur_v = prefetch_v[buffer_idx]

    if not produced_output:
        raise RuntimeError("min_varlen_ring step loop produced no output")
    return out


def build_method_runs(
    methods: list[str],
    case: Case,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_q_host: torch.Tensor,
    cu_seqlens_k_host: torch.Tensor,
    local_rank: int,
    local_world_size: int,
    args: argparse.Namespace,
) -> list[MethodRun]:
    """Create per-method timing/check callables for one input allocation.

    Capturing `remote_k`/`remote_v` as default arguments is intentional. Python
    closures capture variables by reference, and later methods allocate
    different TKParallelTensor shapes.
    """
    runs: list[MethodRun] = []
    max_seqlen_q = case.seqlen
    max_seqlen_k = case.seqlen
    half_cu_seqlens = None
    half_cu_seqlens_host = None
    if case.is_causal and case.seqlen % 256 == 0:
        half_cu_seqlens, half_cu_seqlens_host = make_cu_seqlens(case.batch_size, case.seqlen // 2, q.device)

    for method in methods:
        if method == "allgather_attention":
            runner = AllGatherAttention(
                dist.group.WORLD,
                q,
                k,
                v,
                case.batch_size,
                case.seqlen,
                case.is_causal,
                args.allgather_backend,
            )
            runs.append(MethodRun(method, runner.forward, runner.note))
        elif method == "pytorch":
            def fn(method=method):
                # Full Python-side ring: P2P K/V exchange, one PyTorch block
                # attention call per visible step, and online LSE merge.
                if case.is_causal:
                    return zigzag_ring_varlen_forward(
                        dist.group.WORLD,
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_q_host,
                        max_seqlen_q,
                        lambda q_, k_, v_, _cu_q, _cu_k, _cu_q_host, _cu_k_host, max_q_, max_k_, causal_: pytorch_varlen_block_attention(
                            q_, k_, v_, case.batch_size, max_q_, max_k_, causal_
                        ),
                    )
                return ring_varlen_forward(
                    dist.group.WORLD,
                    q,
                    k,
                    v,
                    False,
                    lambda q_, k_, v_, causal_: pytorch_varlen_block_attention(
                        q_, k_, v_, case.batch_size, max_seqlen_q, max_seqlen_k, causal_
                    ),
                )

            runs.append(MethodRun(method, fn, "zigzag causal" if case.is_causal else ""))
        elif method == "fa2":
            if flash_attn_varlen_func2 is None:
                runs.append(MethodRun(method, lambda: q, "not available", checkable=False))
                continue

            def fn(method=method):
                if case.is_causal:
                    return zigzag_ring_varlen_forward(
                        dist.group.WORLD,
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_q_host,
                        max_seqlen_q,
                        lambda q_, k_, v_, cu_q_, cu_k_, _cu_q_host, _cu_k_host, max_q_, max_k_, causal_: flash_varlen_block_attention(
                            method,
                            flash_attn_varlen_func2,
                            q_,
                            k_,
                            v_,
                            cu_q_,
                            cu_k_,
                            max_q_,
                            max_k_,
                            causal_,
                        ),
                    )
                return ring_varlen_forward(
                    dist.group.WORLD,
                    q,
                    k,
                    v,
                    False,
                    lambda q_, k_, v_, causal_: flash_varlen_block_attention(
                        method,
                        flash_attn_varlen_func2,
                        q_,
                        k_,
                        v_,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        causal_,
                    ),
                )

            runs.append(MethodRun(method, fn, "zigzag causal" if case.is_causal else ""))
        elif method == "fa3":
            if flash_attn_varlen_func3 is None:
                def fn(method=method):
                    if case.is_causal:
                        return zigzag_ring_varlen_forward(
                            dist.group.WORLD,
                            q,
                            k,
                            v,
                            cu_seqlens_q,
                            cu_seqlens_q_host,
                            max_seqlen_q,
                            lambda q_, k_, v_, cu_q_, cu_k_, cu_q_host_, cu_k_host_, max_q_, max_k_, causal_: min_fa3_varlen_block_attention(
                                min_fa3_op.forward_varlen,
                                q_,
                                k_,
                                v_,
                                cu_q_,
                                cu_k_,
                                cu_q_host_,
                                cu_k_host_,
                                max_q_,
                                max_k_,
                                causal_,
                            ),
                        )
                    return ring_varlen_forward(
                        dist.group.WORLD,
                        q,
                        k,
                        v,
                        False,
                        lambda q_, k_, v_, causal_: min_fa3_varlen_block_attention(
                            min_fa3_op.forward_varlen,
                            q_,
                            k_,
                            v_,
                            cu_seqlens_q,
                            cu_seqlens_k,
                            cu_seqlens_q_host,
                            cu_seqlens_k_host,
                            max_seqlen_q,
                            max_seqlen_k,
                            causal_,
                        ),
                    )

                runs.append(MethodRun(method, fn, "fallback: min_fa3_varlen block"))
                continue

            def fn(method=method):
                if case.is_causal:
                    return zigzag_ring_varlen_forward(
                        dist.group.WORLD,
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_q_host,
                        max_seqlen_q,
                        lambda q_, k_, v_, cu_q_, cu_k_, _cu_q_host, _cu_k_host, max_q_, max_k_, causal_: flash_varlen_block_attention(
                            method,
                            flash_attn_varlen_func3,
                            q_,
                            k_,
                            v_,
                            cu_q_,
                            cu_k_,
                            max_q_,
                            max_k_,
                            causal_,
                        ),
                    )
                return ring_varlen_forward(
                    dist.group.WORLD,
                    q,
                    k,
                    v,
                    False,
                    lambda q_, k_, v_, causal_: flash_varlen_block_attention(
                        method,
                        flash_attn_varlen_func3,
                        q_,
                        k_,
                        v_,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        causal_,
                    ),
                )

            runs.append(MethodRun(method, fn, "zigzag causal" if case.is_causal else ""))
        elif method == "min_varlen":
            def fn(method=method):
                # Timing-only local step loop. This is not a complete ring
                # implementation because it never exchanges K/V across ranks.
                return min_varlen_local_steps_forward(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    cu_seqlens_q_host,
                    cu_seqlens_k_host,
                    max_seqlen_q,
                    max_seqlen_k,
                    case.is_causal,
                    local_rank,
                    local_world_size,
                    args.num_comp_sm,
                )

            runs.append(MethodRun(method, fn, "timing-only local step loop", checkable=False))
        elif method == "min_varlen_ring":
            remote_k, remote_v = make_ring_parallel_tensors(k, v, local_rank, local_world_size)

            def fn(method=method, remote_k=remote_k, remote_v=remote_v):
                # Single-step ring path: each launch consumes one ring block and
                # updates the running O/LSE buffers in the kernel epilogue.
                return min_varlen_ring_steps_forward(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    cu_seqlens_q_host,
                    cu_seqlens_k_host,
                    max_seqlen_q,
                    max_seqlen_k,
                    case.is_causal,
                    local_rank,
                    local_world_size,
                    remote_k,
                    remote_v,
                    args.num_comp_sm,
                    args.num_comm_sm,
                )

            runs.append(
                MethodRun(
                    method,
                    fn,
                    "ordinary causal ring timing-only" if case.is_causal else "",
                    checkable=not case.is_causal,
                )
            )
        elif method == "min_varlen_mega_ring":
            if case.is_causal and half_cu_seqlens is None:
                runs.append(MethodRun(method, lambda: q, "not available", checkable=False))
                continue
            remote_k, remote_v = make_mega_parallel_tensors(k, v, local_rank, local_world_size)

            def fn(method=method, remote_k=remote_k, remote_v=remote_v):
                # Fused mega-ring path: one kernel launch covers all ring steps,
                # remote K/V loads, and output/LSE reduction.
                return min_fa3_op.forward_varlen_mega_ring(
                    q,
                    remote_k.data_,
                    remote_v.data_,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    case.is_causal,
                    cu_seqlens_q_host=cu_seqlens_q_host,
                    cu_seqlens_k_host=cu_seqlens_k_host,
                    half_cu_seqlens=half_cu_seqlens,
                    half_cu_seqlens_host=half_cu_seqlens_host,
                    remote_k=remote_k,
                    remote_v=remote_v,
                    num_comp_sm=args.num_comp_sm,
                    num_comm_sm=args.num_comm_sm,
                )

            runs.append(
                MethodRun(
                    method,
                    fn,
                    "zigzag causal" if case.is_causal else "",
                )
            )
        else:
            raise RuntimeError(f"unhandled method {method}")

    return runs


def measure_distributed_ms(fn: Callable[[], torch.Tensor], warmup_iters: int, num_iters: int) -> TimingResult:
    """Measure average CUDA-event latency for local rank time and max rank time."""
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    cuda_barrier()

    local_samples_ms: list[float] = []
    max_samples_ms: list[float] = []
    for _ in range(num_iters):
        cuda_barrier()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_tensor = torch.tensor([elapsed_ms], device="cuda", dtype=torch.float64)
        # Report the completion time of the slowest rank, which is the relevant
        # end-to-end latency for a distributed forward call.
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        local_samples_ms.append(elapsed_ms)
        max_samples_ms.append(elapsed_tensor.item())
    cuda_barrier()

    local_avg_ms = sum(local_samples_ms) / len(local_samples_ms)
    max_avg_ms = sum(max_samples_ms) / len(max_samples_ms)
    local_avg_tensor = torch.tensor([local_avg_ms], device="cuda", dtype=torch.float64)
    rank_time_tensors = [torch.empty_like(local_avg_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(rank_time_tensors, local_avg_tensor)
    rank_times_ms = [tensor.item() for tensor in rank_time_tensors] if dist.get_rank() == 0 else None
    return TimingResult(local_avg_ms, max_avg_ms, rank_times_ms)


def aggregate_tflops(case: Case, local_rank: int, local_world_size: int, time_ms: float) -> float:
    """Compute aggregate TFLOPS for the visible attention work across ranks."""
    if case.is_causal:
        visible_scores = case.seqlen * (local_rank * case.seqlen) + case.seqlen * (case.seqlen + 1) // 2
    else:
        visible_scores = case.seqlen * (local_world_size * case.seqlen)
    local_flops = 4 * case.batch_size * visible_scores * case.q_heads * case.head_dim
    flops_tensor = torch.tensor([float(local_flops)], device="cuda", dtype=torch.float64)
    dist.all_reduce(flops_tensor, op=dist.ReduceOp.SUM)
    return flops_tensor.item() / (time_ms * 1e-3) / 1e12


def check_output(
    method: str,
    fn: Callable[[], torch.Tensor],
    ref: torch.Tensor,
    atol: float,
    rtol: float,
) -> str:
    """Run a method check and synchronize assertion failures across ranks."""
    local_error = None
    try:
        out = fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)
    except Exception as exc:
        local_error = f"{method}: {exc}"
    raise_if_any_rank_failed(local_error, dist.group.WORLD)
    return "ok"


def run_case(
    case: Case,
    methods: list[str],
    local_rank: int,
    local_world_size: int,
    args: argparse.Namespace,
) -> dict[str, Result]:
    """Run all requested methods for one shape/mode case."""
    q, k, v = make_inputs(case, local_rank, args.seed)
    cu_seqlens_q, cu_seqlens_q_host = make_cu_seqlens(case.batch_size, case.seqlen, q.device)
    cu_seqlens_k, cu_seqlens_k_host = make_cu_seqlens(case.batch_size, case.seqlen, q.device)
    runs = build_method_runs(
        methods,
        case,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_q_host,
        cu_seqlens_k_host,
        local_rank,
        local_world_size,
        args,
    )

    ref = None
    if args.check:
        # The reference gathers all rank-local K/V blocks. Noncausal uses the
        # ordinary full-rank sequence; causal uses the default zigzag layout.
        k_by_rank = gather_rank_tensor(k, dist.group.WORLD)
        v_by_rank = gather_rank_tensor(v, dist.group.WORLD)
        if case.is_causal:
            ref = reference_zigzag_ring_varlen(
                q,
                
                k_by_rank,
                v_by_rank,
                case.batch_size,
                case.seqlen,
                local_rank,
            )
        else:
            ref = reference_ring_varlen(
                q,
                k_by_rank,
                v_by_rank,
                case.batch_size,
                case.seqlen,
                local_rank,
                False,
            )

    results: dict[str, Result] = {}
    for run in runs:
        if run.note == "not available":
            results[run.name] = Result(None, None, "skip", run.note)
            continue
        try:
            timing = measure_distributed_ms(run.timing_fn, args.warmup_iters, args.num_iters)
            time_ms = timing.max_time_ms
            tflops = aggregate_tflops(case, local_rank, local_world_size, time_ms)
            avg_gpu_tflops = tflops / local_world_size
            check = "skip"
            if args.check and run.checkable and ref is not None:
                check = check_output(run.name, run.timing_fn, ref, args.atol, args.rtol)
            elif args.check and not run.checkable:
                check = "timing-only"
            results[run.name] = Result(time_ms, tflops, check, run.note, timing.rank_times_ms, avg_gpu_tflops)
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            msg = str(exc).splitlines()[0]
            results[run.name] = Result(None, None, "oom", msg)
        except Exception as exc:
            results[run.name] = Result(None, None, "error", str(exc))
        cuda_barrier()
    return results


def make_cases(args: argparse.Namespace) -> list[Case]:
    """Expand CLI arguments into concrete benchmark cases."""
    causal_values = {
        "noncausal": [False],
        "causal": [True],
        "both": [False, True],
    }[args.mode]
    return [
        Case(args.b, seqlen, args.qhead, args.kvhead, args.headdim, is_causal)
        for is_causal in causal_values
        for seqlen in parse_seqlen_spec(args.seqlen)
    ]


def validate_args(
    args: argparse.Namespace,
    methods: list[str],
    local_world_size: int,
    sm_configs: list[SmConfig],
) -> None:
    """Validate arguments against the minimal demo's supported configuration."""
    if args.headdim != 128:
        raise SystemExit(f"This demo requires D=128, got D={args.headdim}")
    if args.qhead % args.kvhead != 0:
        raise SystemExit(f"qhead must be divisible by kvhead, got qhead={args.qhead}, kvhead={args.kvhead}")
    for sm_config in sm_configs:
        if sm_config.num_comp_sm <= 0:
            raise SystemExit(f"num_comp_sm must be positive, got {sm_config.num_comp_sm}")
        if sm_config.num_comm_sm < 0:
            raise SystemExit(f"num_comm_sm must be non-negative, got {sm_config.num_comm_sm}")
    if args.num_iters <= 0:
        raise SystemExit(f"--num-iters must be positive, got {args.num_iters}")
    if args.warmup_iters < 0:
        raise SystemExit(f"--warmup-iters must be non-negative, got {args.warmup_iters}")
    if any(method in methods for method in ("min_varlen_ring", "min_varlen_mega_ring")):
        if args.kvhead * args.headdim != 1024:
            raise SystemExit(
                "min_fa3 ring communication path requires kvhead * headdim == 1024, "
                f"got kvhead={args.kvhead}, headdim={args.headdim}"
            )
        if local_world_size > 1 and any(sm_config.num_comm_sm <= 0 for sm_config in sm_configs):
            raise SystemExit("multi-rank min_fa3 ring paths require num_comm_sm > 0")


def print_results(case: Case, results: dict[str, Result], methods: list[str]) -> None:
    """Print one result table on rank 0."""
    mode = "causal" if case.is_causal else "noncausal"
    print(
        f"\nB={case.batch_size}, local_S={case.seqlen}, QH={case.q_heads}, "
        f"KVH={case.kv_heads}, D={case.head_dim}, mode={mode}"
    )
    rows: list[tuple[str, str, str, str, str, str]] = []
    for method in methods:
        result = results.get(method)
        if result is None:
            continue
        if result.time_ms is None:
            time_s = "N/A"
        elif result.rank_times_ms is None:
            time_s = f"max_across_ranks={result.time_ms:.3f}"
        else:
            rank_times_s = ", ".join(f"t{rank}={time_ms:.3f}" for rank, time_ms in enumerate(result.rank_times_ms))
            time_s = f"{rank_times_s} | max_across_ranks={result.time_ms:.3f}"
        tflops_s = "N/A" if result.tflops is None else f"{result.tflops:.1f}"
        avg_gpu_tflops_s = "N/A" if result.avg_gpu_tflops is None else f"{result.avg_gpu_tflops:.1f}"
        rows.append((method, time_s, tflops_s, avg_gpu_tflops_s, result.check, result.note))

    time_width = max((64, *(len(row[1]) for row in rows)))
    print(
        f"{'Method':<24} {'Time ms':<{time_width}} {'Agg TFLOPS':>12} "
        f"{'Avg/GPU':>10} {'Check':>14}  Note"
    )
    for method, time_s, tflops_s, avg_gpu_tflops_s, check, note in rows:
        print(
            f"{method:<24} {time_s:<{time_width}} {tflops_s:>12} "
            f"{avg_gpu_tflops_s:>10} {check:>14}  {note}"
        )


def main() -> None:
    """Program entry point."""
    global flash_attn_varlen_func2, flash_attn_varlen_func3

    args = parse_args()
    methods = parse_methods(args.methods)
    sm_configs = (
        parse_sm_config_spec(args.sm_configs)
        if args.sm_configs is not None
        else [SmConfig(args.num_comp_sm, args.num_comm_sm)]
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    local_rank, local_world_size = init_distributed()
    try:
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) != (9, 0):
            raise SystemExit(f"This demo requires SM90 Hopper, got {(major, minor)}")
        if not available_on_all_ranks(flash_attn_varlen_func2 is not None):
            # A distributed benchmark cannot safely run an optional backend if
            # only some ranks imported it successfully.
            flash_attn_varlen_func2 = None
        if not available_on_all_ranks(flash_attn_varlen_func3 is not None):
            flash_attn_varlen_func3 = None
        args.allgather_backend = select_allgather_backend(dist.group.WORLD)
        validate_args(args, methods, local_world_size, sm_configs)
        cases = make_cases(args)

        if local_rank == 0:
            sm_configs_s = ",".join(
                f"{sm_config.num_comp_sm}:{sm_config.num_comm_sm}" for sm_config in sm_configs
            )
            print(
                f"Config: world_size={local_world_size}, methods={methods}, B={args.b}, "
                f"seqlen={args.seqlen}, qhead={args.qhead}, kvhead={args.kvhead}, "
                f"D={args.headdim}, mode={args.mode}, sm_configs={sm_configs_s}, "
                f"warmup={args.warmup_iters}, iters={args.num_iters}, "
                f"check={args.check}"
            )
            if args.check:
                print("Checks compare each rank output against a full-rank PyTorch reference.")
                print("Causal checks use the zigzag [front | back] reference by default.")
            print("Agg TFLOPS sums visible attention work across ranks; Avg/GPU is that value divided by world_size.")

        for sm_config in sm_configs:
            args.num_comp_sm = sm_config.num_comp_sm
            args.num_comm_sm = sm_config.num_comm_sm
            if local_rank == 0:
                print(
                    f"\nSM config: num_comp_sm={args.num_comp_sm}, num_comm_sm={args.num_comm_sm}",
                    flush=True,
                )
            for case in cases:
                if local_rank == 0:
                    print(f"\nRunning local_S={case.seqlen}, causal={case.is_causal}", flush=True)
                results = run_case(case, methods, local_rank, local_world_size, args)
                if local_rank == 0:
                    print_results(case, results, methods)
                cuda_barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
