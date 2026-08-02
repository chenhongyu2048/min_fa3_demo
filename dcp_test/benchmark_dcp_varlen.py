"""Packed-varlen DCP orchestration benchmark using one local min FA3 kernel."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

import min_fa3_op
from min_fa3_dcp import (
    DCPAttentionCUDAGraph,
    DCPAttentionRunner,
    SGLangDCPAttentionRunner,
    VLLMA2ADCPAttentionRunner,
    VLLMDCPAttentionRunner,
    _dcp_a2a_payload_bytes,
    make_topology,
)


VLLM_COMMIT = "a89015c6df8eeb37a843b717c97a5be1355de83d"
SGLANG_COMMIT = "8d6549bc4039d33635844495d86684677a4f0df8"

METHOD_OURS_OVERLAP = "ours_overlap_varlen"
METHOD_OURS_NO_OVERLAP = "ours_no_overlap_varlen"
METHOD_VLLM = "vllm_ag_rs_min_fa3_varlen"
METHOD_VLLM_A2A = "vllm_a2a_min_fa3_varlen"
METHOD_SGLANG = "sglang_mha_ag_ar_min_fa3_varlen"
METHOD_FULL = "full_kv_min_fa3_varlen"

STAGES = (
    "attention_end_to_end_ms",
    "q_allgather_and_reorder_ms",
    "local_chunk_attention_ms",
    "overlapped_ag_chunk_window_ms",
    "overlap_hidden_time_ms",
    "local_history_attention_ms",
    "lse_allgather_correct_ms",
    "output_collective_ms",
    "output_reduce_scatter_ms",
    "a2a_pack_ms",
    "a2a_all_to_all_ms",
    "a2a_unpack_combine_ms",
    "state_merge_ms",
)
CAPTURE_EAGER_WARMUP = 3


@dataclass
class Inputs:
    q_local: torch.Tensor
    k_history_local: torch.Tensor
    v_history_local: torch.Tensor
    k_chunk: torch.Tensor | None
    v_chunk: torch.Tensor | None
    k_reference: torch.Tensor
    v_reference: torch.Tensor
    q_lengths: list[int]
    history_lengths: list[int]
    local_history_lengths: list[int]
    reference_lengths: list[int]
    cu_q: torch.Tensor
    cu_q_host: torch.Tensor
    cu_history_local: torch.Tensor
    cu_history_local_host: torch.Tensor
    cu_reference: torch.Tensor
    cu_reference_host: torch.Tensor


def parse_lengths(spec: str, batch_size: int, name: str) -> list[int]:
    try:
        values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    except ValueError as error:
        raise SystemExit(f"{name} must be a comma-separated integer list") from error
    if len(values) == 1:
        values *= batch_size
    if len(values) != batch_size:
        raise SystemExit(f"{name} must contain one value or exactly B={batch_size} values")
    if any(value <= 0 for value in values):
        raise SystemExit(f"{name} values must be positive")
    return values


def parse_implementations(spec: str) -> tuple[str, ...]:
    values = tuple(token.strip().lower() for token in spec.split(",") if token.strip())
    allowed = {"ours", "vllm", "sglang", "full"}
    if not values or any(value not in allowed for value in values):
        raise SystemExit(
            "--implementations must contain ours,vllm,sglang,full entries"
        )
    return tuple(dict.fromkeys(values))


def expanded_method_labels(implementations: tuple[str, ...]) -> tuple[str, ...]:
    methods: list[str] = []
    if "ours" in implementations:
        methods.extend((METHOD_OURS_NO_OVERLAP, METHOD_OURS_OVERLAP))
    if "vllm" in implementations:
        methods.extend((METHOD_VLLM, METHOD_VLLM_A2A))
    if "sglang" in implementations:
        methods.append(METHOD_SGLANG)
    if "full" in implementations:
        methods.append(METHOD_FULL)
    return tuple(methods)


def make_cu(lengths: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    host = torch.tensor(offsets, dtype=torch.int32)
    return host.to(device), host


def make_group(dcp_size: int, device: torch.device) -> tuple[dist.ProcessGroup, tuple[int, ...]]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_group = None
    local_ranks: tuple[int, ...] | None = None
    for start in range(0, world_size, dcp_size):
        ranks = tuple(range(start, start + dcp_size))
        group = dist.new_group(list(ranks), backend="nccl", device_id=device)
        if rank in ranks:
            local_group = group
            local_ranks = ranks
    assert local_group is not None and local_ranks is not None
    return local_group, local_ranks


def randn_bf16(
    shape: tuple[int, ...], seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)


def shard_interleaved(
    tensor: torch.Tensor,
    lengths: list[int],
    dcp_rank: int,
    dcp_size: int,
) -> tuple[torch.Tensor, list[int]]:
    pieces = []
    local_lengths = []
    start = 0
    for length in lengths:
        piece = tensor[start : start + length][dcp_rank::dcp_size]
        pieces.append(piece)
        local_lengths.append(piece.shape[0])
        start += length
    return torch.cat(pieces).contiguous(), local_lengths


def append_chunk(
    history: torch.Tensor,
    chunk: torch.Tensor,
    history_lengths: list[int],
    q_lengths: list[int],
) -> torch.Tensor:
    pieces = []
    history_start = 0
    q_start = 0
    for history_length, q_length in zip(history_lengths, q_lengths):
        pieces.append(history[history_start : history_start + history_length])
        pieces.append(chunk[q_start : q_start + q_length])
        history_start += history_length
        q_start += q_length
    return torch.cat(pieces).contiguous()


def build_inputs(args: argparse.Namespace, topology, device: torch.device) -> Inputs:
    q_lengths = [1] * args.b if args.workload == "decode" else parse_lengths(
        args.sq, args.b, "--sq"
    )
    history_lengths = parse_lengths(args.seqlen, args.b, "--seqlen")
    rank = dist.get_rank()
    dcp_rank = topology.dcp_rank(rank)
    kv_head = topology.kv_head_for_rank(rank)
    seed = (
        91009
        + topology.q_heads * 1009
        + topology.kv_heads * 503
        + topology.dcp_size * 211
        + sum(q_lengths) * 53
        + sum(history_lengths) * 7
    )
    q_local = randn_bf16(
        (sum(q_lengths), topology.q_heads_local, args.headdim),
        seed + rank * 100_003,
        device,
    )
    k_history = randn_bf16(
        (sum(history_lengths), 1, args.headdim),
        seed + kv_head * 1_000_003,
        device,
    )
    v_history = randn_bf16(
        (sum(history_lengths), 1, args.headdim),
        seed + kv_head * 1_000_003 + 1,
        device,
    )
    k_history_local, local_lengths = shard_interleaved(
        k_history, history_lengths, dcp_rank, args.dcp_size
    )
    v_history_local, v_local_lengths = shard_interleaved(
        v_history, history_lengths, dcp_rank, args.dcp_size
    )
    assert local_lengths == v_local_lengths
    if any(length <= 0 for length in local_lengths):
        raise SystemExit("every rank-local sequence must be nonempty")

    k_chunk = None
    v_chunk = None
    if args.workload == "chunk":
        k_chunk = randn_bf16(
            (sum(q_lengths), 1, args.headdim),
            seed + kv_head * 1_000_003 + 2,
            device,
        )
        v_chunk = randn_bf16(
            (sum(q_lengths), 1, args.headdim),
            seed + kv_head * 1_000_003 + 3,
            device,
        )
        k_reference = append_chunk(k_history, k_chunk, history_lengths, q_lengths)
        v_reference = append_chunk(v_history, v_chunk, history_lengths, q_lengths)
        reference_lengths = [
            history + query for history, query in zip(history_lengths, q_lengths)
        ]
    else:
        k_reference = k_history
        v_reference = v_history
        reference_lengths = list(history_lengths)

    cu_q, cu_q_host = make_cu(q_lengths, device)
    cu_local, cu_local_host = make_cu(local_lengths, device)
    cu_reference, cu_reference_host = make_cu(reference_lengths, device)
    return Inputs(
        q_local,
        k_history_local,
        v_history_local,
        k_chunk,
        v_chunk,
        k_reference,
        v_reference,
        q_lengths,
        history_lengths,
        local_lengths,
        reference_lengths,
        cu_q,
        cu_q_host,
        cu_local,
        cu_local_host,
        cu_reference,
        cu_reference_host,
    )


def full_forward(inputs: Inputs, args: argparse.Namespace):
    return min_fa3_op.forward_kvcache_varlen(
        inputs.q_local,
        inputs.k_reference,
        inputs.v_reference,
        inputs.cu_q,
        inputs.cu_reference,
        max(inputs.q_lengths),
        max(inputs.reference_lengths),
        cu_seqlens_q_host=inputs.cu_q_host,
        cu_seqlens_k_host=inputs.cu_reference_host,
        num_splits=args.num_splits,
        return_lse=False,
        is_causal=args.workload == "chunk",
    )


def runner_call(
    runner: DCPAttentionRunner,
    inputs: Inputs,
    args: argparse.Namespace,
    overlap: bool,
    timed: bool,
):
    common = dict(
        cu_seqlens_q_host=inputs.cu_q_host,
        num_splits=args.num_splits,
        return_lse=False,
        _record_timing=timed,
    )
    if args.workload == "decode":
        return runner.forward_decode_varlen(
            inputs.q_local,
            inputs.k_history_local,
            inputs.v_history_local,
            inputs.cu_q,
            inputs.cu_history_local,
            max(inputs.q_lengths),
            max(inputs.local_history_lengths),
            cu_seqlens_k_local_host=inputs.cu_history_local_host,
            overlap_q_allgather=overlap,
            **common,
        )
    assert inputs.k_chunk is not None and inputs.v_chunk is not None
    return runner.forward_chunk_prefill_varlen(
        inputs.q_local,
        inputs.k_history_local,
        inputs.v_history_local,
        inputs.k_chunk,
        inputs.v_chunk,
        inputs.cu_q,
        inputs.cu_history_local,
        max(inputs.q_lengths),
        max(inputs.local_history_lengths),
        cu_seqlens_history_local_host=inputs.cu_history_local_host,
        overlap_q_allgather=overlap,
        **common,
    )


def capture_runner(
    runner: DCPAttentionRunner,
    inputs: Inputs,
    args: argparse.Namespace,
    overlap: bool,
) -> DCPAttentionCUDAGraph:
    common = dict(
        cu_seqlens_q_host=inputs.cu_q_host,
        num_splits=args.num_splits,
        return_lse=False,
        overlap_q_allgather=overlap,
        record_timing=True,
        capture_warmup=CAPTURE_EAGER_WARMUP,
    )
    if args.workload == "decode":
        return runner.capture_decode_varlen(
            inputs.q_local,
            inputs.k_history_local,
            inputs.v_history_local,
            inputs.cu_q,
            inputs.cu_history_local,
            max(inputs.q_lengths),
            max(inputs.local_history_lengths),
            cu_seqlens_k_local_host=inputs.cu_history_local_host,
            **common,
        )
    assert inputs.k_chunk is not None and inputs.v_chunk is not None
    return runner.capture_chunk_prefill_varlen(
        inputs.q_local,
        inputs.k_history_local,
        inputs.v_history_local,
        inputs.k_chunk,
        inputs.v_chunk,
        inputs.cu_q,
        inputs.cu_history_local,
        max(inputs.q_lengths),
        max(inputs.local_history_lengths),
        cu_seqlens_history_local_host=inputs.cu_history_local_host,
        **common,
    )


def global_max_values(values: dict[str, float], device: torch.device) -> dict[str, float]:
    tensor = torch.tensor(
        [values.get(stage, 0.0) for stage in STAGES],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return {stage: float(value) for stage, value in zip(STAGES, tensor.tolist())}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        stage: {
            "p50": percentile([sample[stage] for sample in samples], 0.50),
            "p90": percentile([sample[stage] for sample in samples], 0.90),
        }
        for stage in STAGES
    }


def measure_runner(
    runner: DCPAttentionRunner,
    call: Callable[[bool], torch.Tensor],
    inputs: Inputs,
    args: argparse.Namespace,
    device: torch.device,
    *,
    overlap: bool,
    expected: torch.Tensor,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    method_name = (
        runner.varlen_overlap_method_name if overlap else runner.varlen_method_name
    )
    check_output(f"{method_name}_eager", call(False), expected)
    captured = capture_runner(runner, inputs, args, overlap) if args.cuda_graph else None
    try:
        check_output(
            method_name,
            captured.replay() if captured is not None else call(False),
            expected,
        )
        for _ in range(args.warmup):
            captured.replay() if captured is not None else call(False)
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(args.iters):
            if captured is not None:
                captured.replay()
            else:
                call(True)
            local_timing = runner.last_timing_ms()
            local_timing["overlap_hidden_time_ms"] = (
                max(
                    0.0,
                    local_timing["q_allgather_and_reorder_ms"]
                    + local_timing["local_chunk_attention_ms"]
                    - local_timing["overlapped_ag_chunk_window_ms"],
                )
                if overlap and args.workload == "chunk"
                else 0.0
            )
            samples.append(global_max_values(local_timing, device))
        execution = {
            "execution_mode": "cuda_graph" if args.cuda_graph else "eager",
            "capture_eager_warmup": (
                CAPTURE_EAGER_WARMUP if args.cuda_graph else 0
            ),
            "post_capture_warmup": args.warmup,
            "stream_policy": (
                "compute_plus_communication" if overlap else "single_stream"
            ),
            "overlap_q_allgather": overlap,
            "graph_static_signature": (
                captured.signature if captured is not None else None
            ),
        }
        return summarize_samples(samples), execution
    finally:
        if captured is not None:
            captured.close()


def measure_full(
    call: Callable[[], torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    inputs: Inputs,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    graph: torch.cuda.CUDAGraph | None = None
    capture_stream: torch.cuda.Stream | None = None
    if args.cuda_graph:
        capture_stream = torch.cuda.Stream(device=device)
        caller_stream = torch.cuda.current_stream(device)
        capture_stream.wait_stream(caller_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(CAPTURE_EAGER_WARMUP):
                call()
        caller_stream.wait_stream(capture_stream)
        torch.cuda.synchronize(device)
        dist.barrier()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(
                graph, stream=capture_stream, capture_error_mode="global"
            ):
                static_output = call()
        except Exception:
            torch.cuda.synchronize(device)
            graph.reset()
            raise
    else:
        static_output = call()

    def replay() -> torch.Tensor:
        if graph is None or capture_stream is None:
            return call()
        current_stream = torch.cuda.current_stream(device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            graph.replay()
        current_stream.wait_stream(capture_stream)
        return static_output

    try:
        for _ in range(args.warmup):
            replay()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(args.iters):
            start.record()
            replay()
            end.record()
            end.synchronize()
            total = torch.tensor(start.elapsed_time(end), device=device, dtype=torch.float64)
            dist.all_reduce(total, op=dist.ReduceOp.MAX)
            values = {stage: 0.0 for stage in STAGES}
            values["attention_end_to_end_ms"] = float(total.item())
            samples.append(values)
        signature = {
            "operation": "full_kv_varlen",
            "bindings": {
                name: DCPAttentionRunner._tensor_signature(tensor)
                for name, tensor in {
                    "q_local": inputs.q_local,
                    "k_reference": inputs.k_reference,
                    "v_reference": inputs.v_reference,
                    "cu_q": inputs.cu_q,
                    "cu_reference": inputs.cu_reference,
                    "cu_q_host": inputs.cu_q_host,
                    "cu_reference_host": inputs.cu_reference_host,
                }.items()
            },
            "scalars": {
                "max_seqlen_q": max(inputs.q_lengths),
                "max_seqlen_k": max(inputs.reference_lengths),
                "num_splits": args.num_splits,
                "return_lse": False,
                "is_causal": args.workload == "chunk",
            },
        }
        execution = {
            "execution_mode": "cuda_graph" if args.cuda_graph else "eager",
            "capture_eager_warmup": (
                CAPTURE_EAGER_WARMUP if args.cuda_graph else 0
            ),
            "post_capture_warmup": args.warmup,
            "stream_policy": "single_stream",
            "overlap_q_allgather": False,
            "graph_static_signature": signature if args.cuda_graph else None,
        }
        return summarize_samples(samples), execution
    finally:
        if graph is not None:
            torch.cuda.synchronize(device)
            graph.reset()


def check_output(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    close = torch.tensor(
        int(torch.isclose(actual.float(), expected.float(), atol=3e-2, rtol=3e-2).all()),
        device=actual.device,
        dtype=torch.int32,
    )
    dist.all_reduce(close, op=dist.ReduceOp.MIN)
    if not close.item():
        raise RuntimeError(f"{name} failed the pre-benchmark correctness check")


def all_rank_local_lengths(inputs: Inputs, device: torch.device) -> list[list[int]]:
    local = torch.tensor(inputs.local_history_lengths, device=device, dtype=torch.int64)
    gathered = torch.empty(
        dist.get_world_size() * len(inputs.local_history_lengths),
        device=device,
        dtype=torch.int64,
    )
    dist.all_gather_into_tensor(gathered, local)
    return gathered.cpu().view(dist.get_world_size(), -1).tolist()


def effective_pairs(inputs: Inputs, workload: str) -> int:
    if workload == "decode":
        return sum(inputs.history_lengths)
    return sum(
        query * history + query * (query + 1) // 2
        for query, history in zip(inputs.q_lengths, inputs.history_lengths)
    )


def communication_report(
    method: str,
    total_q: int,
    h_local: int,
    head_dim: int,
    dcp_size: int,
) -> dict[str, object]:
    if method == METHOD_FULL:
        return {
            "output_collective": "none",
            "q_allgather_receive_bytes_per_rank": 0,
            "lse_allgather_receive_bytes_per_rank": 0,
            "output_collective_buffer_bytes_per_rank": 0,
            "output_collective_remote_bytes_per_rank": 0,
            "output_collective_payload_bytes_per_rank": 0,
            "collective_payload_bytes_per_rank_total": 0,
        }
    h_group = h_local * dcp_size
    q_receive = (dcp_size - 1) * total_q * h_local * head_dim * 2
    lse_receive = (dcp_size - 1) * total_q * h_group * 4
    if method == METHOD_VLLM_A2A:
        output_kind = "bf16_packed_all_to_all"
        output_buffer, output_remote = _dcp_a2a_payload_bytes(
            total_q, h_local, head_dim, dcp_size
        )
        lse_receive = 0
    elif method == METHOD_SGLANG:
        output_kind = "fp32_all_reduce"
        output_buffer = total_q * h_group * head_dim * 4
        output_remote = 2.0 * (dcp_size - 1) / dcp_size * output_buffer
    else:
        output_kind = "bf16_reduce_scatter"
        output_buffer = total_q * h_group * head_dim * 2
        output_remote = (dcp_size - 1) / dcp_size * output_buffer
    return {
        "output_collective": output_kind,
        "q_allgather_receive_bytes_per_rank": float(q_receive),
        "lse_allgather_receive_bytes_per_rank": float(lse_receive),
        "output_collective_buffer_bytes_per_rank": float(output_buffer),
        "output_collective_remote_bytes_per_rank": float(output_remote),
        "output_collective_payload_bytes_per_rank": float(output_remote),
        "collective_payload_bytes_per_rank_total": float(
            q_receive + lse_receive + output_remote
        ),
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment(device: torch.device) -> dict[str, object]:
    props = torch.cuda.get_device_properties(device)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": props.name,
        "world_size": dist.get_world_size(),
        "repository_commit": git_commit(),
        "vllm_source_commit": VLLM_COMMIT,
        "sglang_source_commit": SGLANG_COMMIT,
    }


def default_output_path(workload: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return Path("benchmarks/results") / f"dcp_varlen_{workload}_{timestamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare packed-varlen DCP orchestration with one local min FA3 kernel; "
            "this is not native serving-runtime backend performance."
        )
    )
    parser.add_argument("--b", type=int, default=3)
    parser.add_argument("--sq", type=str, default="1,8,32")
    parser.add_argument("--seqlen", type=str, default="129,1024,3131")
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=1)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--dcp-size", type=int, default=8)
    parser.add_argument("--workload", choices=("decode", "chunk"), default="chunk")
    parser.add_argument(
        "--implementations", type=str, default="ours,vllm,sglang,full"
    )
    parser.add_argument("--num-splits", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture each method and full-KV reference before timed replay",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise SystemExit("This benchmark requires SM90 Hopper")
        if args.tp_size != world_size:
            raise SystemExit(f"--tp-size must equal torchrun world size {world_size}")
        if args.headdim != 128 or args.b <= 0:
            raise SystemExit("--headdim must be 128 and --b must be positive")
        if args.dcp_size <= 0 or world_size % args.dcp_size:
            raise SystemExit("--dcp-size must be positive and divide world size")
        if not 0 <= args.num_splits <= 128:
            raise SystemExit("--num-splits must be in [0, 128]")
        if args.warmup < 0 or args.iters <= 0:
            raise SystemExit("--warmup must be nonnegative and --iters must be positive")
        if args.workload == "decode" and any(
            length != 1 for length in parse_lengths(args.sq, args.b, "--sq")
        ):
            raise SystemExit("decode requires every --sq value to be 1")

        implementations = parse_implementations(args.implementations)
        topology = make_topology(
            args.qhead, args.kvhead, args.tp_size, args.dcp_size
        )
        process_group, group_ranks = make_group(args.dcp_size, device)
        if group_ranks != topology.dcp_group_ranks(rank):
            raise RuntimeError("DCP process group crosses a KV replica boundary")
        inputs = build_inputs(args, topology, device)
        local_lengths_by_rank = all_rank_local_lengths(inputs, device)

        runners: dict[str, DCPAttentionRunner] = {}
        if "ours" in implementations:
            runners[METHOD_OURS_NO_OVERLAP] = DCPAttentionRunner(process_group)
            runners[METHOD_OURS_OVERLAP] = DCPAttentionRunner(process_group)
        if "vllm" in implementations:
            runners[METHOD_VLLM] = VLLMDCPAttentionRunner(process_group)
            runners[METHOD_VLLM_A2A] = VLLMA2ADCPAttentionRunner(process_group)
        if "sglang" in implementations:
            runners[METHOD_SGLANG] = SGLangDCPAttentionRunner(process_group)

        reference_output = full_forward(inputs, args)
        reports: dict[str, dict[str, object]] = {}
        for method, runner in runners.items():
            overlap = method == METHOD_OURS_OVERLAP
            call = lambda timed, runner=runner, overlap=overlap: runner_call(
                runner, inputs, args, overlap, timed
            )
            stages, execution = measure_runner(
                runner,
                call,
                inputs,
                args,
                device,
                overlap=overlap,
                expected=reference_output,
            )
            reports[method] = {
                "stages_ms": stages,
                "output_collective_kind": runner.output_collective_kind,
                "workspace_policy": runner.workspace_policy,
                "communication": communication_report(
                    method,
                    sum(inputs.q_lengths),
                    topology.q_heads_local,
                    args.headdim,
                    args.dcp_size,
                ),
                "execution": execution,
            }
            if method == METHOD_OURS_OVERLAP and args.workload == "chunk":
                reports[method]["overlap"] = {
                    "hidden_time_ms": stages["overlap_hidden_time_ms"],
                    "ag_chunk_window_ms": stages[
                        "overlapped_ag_chunk_window_ms"
                    ],
                }
        if "full" in implementations:
            stages, execution = measure_full(
                lambda: full_forward(inputs, args), args, device, inputs
            )
            reports[METHOD_FULL] = {
                "stages_ms": stages,
                "output_collective_kind": "none",
                "workspace_policy": "operator_managed",
                "communication": communication_report(
                    METHOD_FULL,
                    sum(inputs.q_lengths),
                    topology.q_heads_local,
                    args.headdim,
                    args.dcp_size,
                ),
                "execution": execution,
            }

        pairs = effective_pairs(inputs, args.workload)
        global_flops = 4 * args.headdim * args.qhead * pairs
        baseline = reports.get(METHOD_FULL)
        baseline_p50 = (
            baseline["stages_ms"]["attention_end_to_end_ms"]["p50"]
            if baseline is not None
            else None
        )
        for report in reports.values():
            p50 = report["stages_ms"]["attention_end_to_end_ms"]["p50"]
            report["speedup_vs_full_kv"] = baseline_p50 / p50 if baseline_p50 else None
            report["effective_tflops"] = global_flops / (p50 * 1.0e9)
        ours = reports.get(METHOD_OURS_NO_OVERLAP)
        if ours is not None:
            ours_p50 = ours["stages_ms"]["attention_end_to_end_ms"]["p50"]
            for report in reports.values():
                method_p50 = report["stages_ms"]["attention_end_to_end_ms"]["p50"]
                report["ours_no_overlap_speedup_vs_method"] = (
                    method_p50 / max(ours_p50, 1.0e-12)
                )

        bf16_bytes = 2
        full_kv_bytes = 2 * sum(inputs.reference_lengths) * args.headdim * bf16_bytes
        replicated_chunk_bytes = (
            2 * sum(inputs.q_lengths) * args.headdim * bf16_bytes
            if args.workload == "chunk"
            else 0
        )
        local_kv_bytes_by_rank = [
            2 * sum(lengths) * args.headdim * bf16_bytes + replicated_chunk_bytes
            for lengths in local_lengths_by_rank
        ]
        h_local = topology.q_heads_local
        h_group = h_local * args.dcp_size
        a2a_buffer_bytes, a2a_remote_bytes = _dcp_a2a_payload_bytes(
            sum(inputs.q_lengths), h_local, args.headdim, args.dcp_size
        )
        collectives = {
            "q_allgather_input_bytes_per_rank": sum(inputs.q_lengths)
            * h_local
            * args.headdim
            * bf16_bytes,
            "lse_allgather_input_bytes_per_rank": sum(inputs.q_lengths)
            * h_group
            * 4,
            "bf16_output_bytes_per_rank": sum(inputs.q_lengths)
            * h_local
            * args.headdim
            * bf16_bytes,
            "a2a_buffer_bytes_per_rank": a2a_buffer_bytes,
            "a2a_remote_bytes_per_rank": a2a_remote_bytes,
        }
        result = {
            "schema_version": 3,
            "comparison_scope": (
                "Same min_fa3_op.forward_kvcache_varlen kernel; orchestration "
                "baseline only, not native vLLM/SGLang backend performance."
            ),
            "environment": environment(device),
            "execution": {
                "execution_mode": "cuda_graph" if args.cuda_graph else "eager",
                "capture_eager_warmup": (
                    CAPTURE_EAGER_WARMUP if args.cuda_graph else 0
                ),
                "warmup_semantics": (
                    "post_capture_replay" if args.cuda_graph else "eager_call"
                ),
            },
            "parameters": {
                **vars(args),
                "output_json": str(args.output_json) if args.output_json else None,
                "implementations": list(implementations),
                "method_labels": list(expanded_method_labels(implementations)),
            },
            "topology": topology.to_dict(),
            "lengths": {
                "q_global": inputs.q_lengths,
                "history_or_cache_global": inputs.history_lengths,
                "history_or_cache_local_by_tp_rank": local_lengths_by_rank,
                "reference_k_global": inputs.reference_lengths,
            },
            "packed_tokens": {
                "total_q": sum(inputs.q_lengths),
                "total_reference_k": sum(inputs.reference_lengths),
                "total_local_history_by_tp_rank": [
                    sum(lengths) for lengths in local_lengths_by_rank
                ],
            },
            "effective_causal_or_decode_pairs": pairs,
            "global_effective_flops": global_flops,
            "kv_memory": {
                "full_kv_bytes_per_rank": full_kv_bytes,
                "local_dcp_bytes_by_tp_rank": local_kv_bytes_by_rank,
                "reduction_ratio_by_tp_rank": [
                    full_kv_bytes / local_bytes
                    for local_bytes in local_kv_bytes_by_rank
                ],
            },
            "collective_payload": collectives,
            "methods": reports,
            "pinned_sources": {
                "vllm": VLLM_COMMIT,
                "sglang": SGLANG_COMMIT,
            },
        }
        if rank == 0:
            output_path = args.output_json or default_output_path(args.workload)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            latency = ", ".join(
                f"{name}={report['stages_ms']['attention_end_to_end_ms']['p50']:.4f}ms"
                for name, report in reports.items()
            )
            print(f"packed DCP {args.workload}: {latency}", flush=True)
            print(f"wrote {output_path}", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
