"""Packed-varlen DCP orchestration benchmark using one local min FA3 kernel."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.distributed as dist

from dcp_test.benchmark_output import (
    DCPBenchmarkRow,
    effective_kv_bandwidth_gbps_per_gpu,
    print_benchmark_results,
    print_timing_breakdowns,
)
from dcp_test.baselines import (
    _dcp_a2a_payload_bytes,
    full_kv_reference_varlen,
)
from dcp_test.utils import (
    CAPTURE_EAGER_WARMUP,
    DCPGroup,
    all_rank_quantiles,
    append_packed_chunk as append_chunk,
    capture_cuda_graph_callable,
    initialize_distributed_sm90,
    make_cu_seqlens as make_cu,
    make_dcp_group,
    make_runner_set,
    measure_timed_runner,
    parse_lengths,
    quantiles,
    randn_bf16,
    require_world_size,
    shard_packed_interleaved as shard_interleaved,
)
from min_fa3_dcp import (
    DCPAttentionCUDAGraph,
    DCPAttentionRunner,
    DCPMegaAttentionRunner,
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
METHOD_MEGA = "dcp_mega_varlen"

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


def parse_implementations(spec: str) -> tuple[str, ...]:
    values = tuple(token.strip().lower() for token in spec.split(",") if token.strip())
    allowed = {"ours", "vllm", "vllm_a2a", "sglang", "full", "mega"}
    if not values or any(value not in allowed for value in values):
        raise SystemExit(
            "--implementations must contain "
            "ours,vllm,vllm_a2a,sglang,full,mega entries"
        )
    return tuple(dict.fromkeys(values))


def expanded_method_labels(implementations: tuple[str, ...]) -> tuple[str, ...]:
    methods: list[str] = []
    if "ours" in implementations:
        methods.extend((METHOD_OURS_NO_OVERLAP, METHOD_OURS_OVERLAP))
    if "vllm" in implementations:
        methods.extend((METHOD_VLLM, METHOD_VLLM_A2A))
    elif "vllm_a2a" in implementations:
        methods.append(METHOD_VLLM_A2A)
    if "sglang" in implementations:
        methods.append(METHOD_SGLANG)
    if "mega" in implementations:
        methods.append(METHOD_MEGA)
    if "full" in implementations:
        methods.append(METHOD_FULL)
    return tuple(methods)


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
    return full_kv_reference_varlen(
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
):
    common = dict(
        cu_seqlens_q_host=inputs.cu_q_host,
        num_splits=args.num_splits,
        return_lse=False,
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


def summarize_samples(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        stage: quantiles([sample[stage] for sample in samples])
        for stage in STAGES
    }


def summarize_series(values: list[float]) -> dict[str, float]:
    summary = quantiles(values)
    summary.update(min=min(values), max=max(values))
    return summary


def measure_runner(
    runner: DCPAttentionRunner,
    call: Callable[[], torch.Tensor],
    inputs: Inputs,
    args: argparse.Namespace,
    device: torch.device,
    *,
    overlap: bool,
    expected: torch.Tensor | None,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    method_name = (
        runner.varlen_overlap_method_name if overlap else runner.varlen_method_name
    )
    if expected is not None:
        check_output(f"{method_name}_eager", call(), expected)
    captured = (
        capture_runner(runner, inputs, args, overlap) if args.cuda_graph else None
    )
    try:
        if expected is not None:
            check_output(
                method_name,
                captured.replay() if captured is not None else call(),
                expected,
            )
    except Exception:
        if captured is not None:
            captured.close()
        raise

    def add_overlap_hidden_time(local_timing: dict[str, float]) -> None:
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

    phase_samples, execution = measure_timed_runner(
        runner,
        call,
        lambda: capture_runner(runner, inputs, args, overlap),
        warmup=args.warmup,
        iterations=args.iters,
        device=device,
        cuda_graph=args.cuda_graph,
        overlap_q_allgather=overlap,
        phase_timing=args.baseline_phase_timing,
        phase_names=STAGES,
        transform_timing=(
            add_overlap_hidden_time if args.baseline_phase_timing else None
        ),
        captured_graph=captured,
    )
    samples = [
        {stage: phase_samples[stage][index] for stage in STAGES}
        for index in range(args.iters)
    ]
    return summarize_samples(samples), execution


def measure_full(
    call: Callable[[], torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    inputs: Inputs,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    close_graph: Callable[[], None] = lambda: None
    if args.cuda_graph:
        replay, close_graph, _ = capture_cuda_graph_callable(call, device)
    else:
        replay = call

    try:
        for _ in range(args.warmup):
            replay()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples = []
        local_latency_samples: list[float] = []
        for _ in range(args.iters):
            start.record()
            replay()
            end.record()
            end.synchronize()
            local_elapsed_ms = start.elapsed_time(end)
            local_latency_samples.append(local_elapsed_ms)
            total = torch.tensor(local_elapsed_ms, device=device, dtype=torch.float64)
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
            "rank_latency_ms": all_rank_quantiles(
                local_latency_samples, device
            ),
        }
        return summarize_samples(samples), execution
    finally:
        close_graph()


def measure_mega(
    runner: DCPMegaAttentionRunner,
    call: Callable[[], torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    expected: torch.Tensor | None,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    def replay_after_distributed_barrier(*, return_timing_ms: bool = False):
        dist.barrier(group=runner.process_group, device_ids=[device.index])
        return runner.replay_last_forward(
            return_timing_ms=return_timing_ms,
            run_pre_barrier=False,
        )

    initial_output = call()
    if expected is not None:
        check_output(f"{METHOD_MEGA}_eager", initial_output, expected)
        runner.prepare_last_forward_replay()
        check_output(
            f"{METHOD_MEGA}_prepared_replay",
            replay_after_distributed_barrier(),
            expected,
        )
    captured = (
        runner.capture_last_forward(capture_warmup=CAPTURE_EAGER_WARMUP)
        if args.cuda_graph else None
    )
    try:
        if captured is not None and expected is not None:
            check_output(f"{METHOD_MEGA}_cuda_graph", captured.replay(), expected)
        for _ in range(args.warmup):
            if captured is not None:
                captured.replay()
            else:
                runner.prepare_last_forward_replay()
                replay_after_distributed_barrier()
        torch.cuda.synchronize(device)

        samples = []
        local_latency_samples: list[float] = []
        phase_samples = None
        if args.mega_phase_timestamps:
            phase_samples = torch.empty(
                (args.iters, len(runner.PHASE_TIMESTAMP_NAMES)),
                device=device,
                dtype=torch.int64,
            )
        graph_start = torch.cuda.Event(enable_timing=True)
        graph_end = torch.cuda.Event(enable_timing=True)
        for sample_idx in range(args.iters):
            if captured is not None:
                graph_start.record()
                captured.replay()
                graph_end.record()
                graph_end.synchronize()
                elapsed_ms = graph_start.elapsed_time(graph_end)
            else:
                runner.prepare_last_forward_replay()
                _, elapsed_ms = replay_after_distributed_barrier(
                    return_timing_ms=True
                )
            local_latency_samples.append(elapsed_ms)
            total = torch.tensor(elapsed_ms, device=device, dtype=torch.float64)
            dist.all_reduce(total, op=dist.ReduceOp.MAX)
            values = {stage: 0.0 for stage in STAGES}
            values["attention_end_to_end_ms"] = float(total.item())
            samples.append(values)
            if phase_samples is not None:
                runner.copy_last_phase_timestamps(phase_samples[sample_idx])

        phase_profile = None
        if phase_samples is not None:
            torch.cuda.synchronize(device)
            raw = phase_samples.cpu()
            if bool((raw == 0).any()):
                raise RuntimeError("mega phase timestamp buffer contains an unwritten slot")
            if not torch.equal(raw[:, 3], raw[:, 4]):
                raise RuntimeError(
                    "fused history_combine_done and publish_done timestamps differ"
                )
            relative_ns = raw - raw[:, :1]
            relative_ns_device = relative_ns.to(device)
            dist.all_reduce(relative_ns_device, op=dist.ReduceOp.MAX)
            relative_us = relative_ns_device.cpu().to(torch.float64) / 1000.0
            milestones = {
                name: summarize_series(relative_us[:, index].tolist())
                for index, name in enumerate(runner.PHASE_TIMESTAMP_NAMES)
            }
            tail_pairs = {
                "q_done_to_publish_done": (1, 4),
                "publish_done_to_receive_done": (4, 5),
                "attention_done_to_history_combine_done": (2, 3),
                "history_combine_done_to_final_combine_done": (3, 6),
            }
            local_tail_ns = torch.stack(
                [raw[:, end] - raw[:, begin] for begin, end in tail_pairs.values()],
                dim=1,
            ).to(device)
            dist.all_reduce(local_tail_ns, op=dist.ReduceOp.MAX)
            tail_us = local_tail_ns.cpu().to(torch.float64) / 1000.0
            phase_profile = {
                "clock": (
                    "SM90 %globaltimer; microseconds relative to each rank's "
                    "kernel_start"
                ),
                "aggregation": (
                    "per-iteration MAX across DCP ranks, then percentile across iterations"
                ),
                "publish_done_semantics": (
                    "all fused remote ready releases issued; recorded from the same "
                    "%globaltimer read as history_combine_done"
                ),
                "milestones_us": milestones,
                "post_global_completion_tails_us": {
                    name: summarize_series(tail_us[:, index].tolist())
                    for index, name in enumerate(tail_pairs)
                },
            }

        graph_mode = captured is not None
        dispatch = runner.last_dispatch
        execution = {
            "execution_mode": "cuda_graph" if graph_mode else "eager",
            "capture_eager_warmup": CAPTURE_EAGER_WARMUP if graph_mode else 0,
            "post_capture_warmup": args.warmup,
            "stream_policy": "single_stream_persistent_mega",
            "overlap_q_allgather": True,
            "graph_static_signature": captured.signature if graph_mode else None,
            "num_comm_sm": runner.num_comm_sm,
            "timing_boundary": (
                "cuda_graph_replay" if graph_mode else "mega_kernel_only"
            ),
            "timing_source": (
                "external_python_cuda_events"
                if graph_mode else "internal_cpp_cuda_events"
            ),
            "pre_barrier": (
                "captured_ipc_phase_barrier"
                if graph_mode else "torch_distributed_outside_timing"
            ),
            "pre_barrier_timed": graph_mode,
            "metadata_policy": (
                "generated_and_uploaded_once_before_capture"
                if graph_mode else "generated_and_uploaded_once_before_timing"
            ),
            "prepared_replay_correctness_checked": expected is not None,
            "graph_replay_correctness_checked": graph_mode and expected is not None,
            "workspace_reset_timed": graph_mode,
            "post_barrier_timed": graph_mode,
            "graph_phase_policy": (
                "device_monotonic_int32_plus_2_per_replay" if graph_mode else None
            ),
            "dispatch": asdict(dispatch) if dispatch is not None else None,
            "queue_counts": runner.last_queue_counts,
            "phase_profile": phase_profile,
            "rank_latency_ms": all_rank_quantiles(local_latency_samples, device),
        }
        return summarize_samples(samples), execution
    finally:
        if captured is not None:
            captured.close()


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
            "tile_ready_remote_bytes_per_rank": 0,
            "collective_payload_bytes_per_rank_total": 0,
        }
    h_group = h_local * dcp_size
    q_receive = (dcp_size - 1) * total_q * h_local * head_dim * 2
    lse_receive = (dcp_size - 1) * total_q * h_group * 4
    tile_ready_remote = 0
    if method == METHOD_MEGA:
        output_kind = "bf16_ipc_a2a"
        output_buffer = total_q * h_local * head_dim * 2
        output_remote = (dcp_size - 1) * output_buffer
        lse_receive = (
            (dcp_size - 1) * total_q * h_local * 4
        )
        tile_ready_remote = (
            (dcp_size - 1) * ((total_q + 15) // 16) * 4
        )
    elif method == METHOD_VLLM_A2A:
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
        "tile_ready_remote_bytes_per_rank": float(tile_ready_remote),
        "collective_payload_bytes_per_rank_total": float(
            q_receive + lse_receive + output_remote + tile_ready_remote
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


def print_results(
    args: argparse.Namespace,
    topology,
    inputs: Inputs,
    reports: dict[str, dict[str, object]],
) -> None:
    mode = "causal" if args.workload == "chunk" else "decode"
    title = (
        f"B={args.b}, q_tokens={sum(inputs.q_lengths)}, "
        f"history_tokens={sum(inputs.history_lengths)}, "
        f"QH={topology.q_heads}, KVH={topology.kv_heads}, D={args.headdim}, "
        f"TP={topology.tp_size}, DCP={topology.dcp_size}, mode={mode}"
    )
    rows = []
    for method, report in reports.items():
        latency = report["stages_ms"]["attention_end_to_end_ms"]
        execution = report["execution"]
        if method == METHOD_FULL:
            check = "reference"
        else:
            check = "ok" if args.check else "skip"
        timing_boundary = execution.get("timing_boundary")
        boundary_note = f"; timing={timing_boundary}" if timing_boundary else ""
        rows.append(
            DCPBenchmarkRow(
                method=method,
                p50_ms=latency["p50"],
                p90_ms=latency["p90"],
                aggregate_tflops=report["effective_tflops"],
                avg_gpu_tflops=(
                    report["effective_tflops"] / topology.tp_size
                ),
                kv_bandwidth_gbps_per_gpu=report["logical_kv_read"][
                    "effective_bandwidth_gbps_per_gpu"
                ],
                check=check,
                note=(
                    f"{execution['execution_mode']}; "
                    f"output={report['output_collective_kind']}; "
                    f"{report['workspace_policy']}{boundary_note}"
                ),
                rank_p50_ms=execution["rank_latency_ms"]["p50"],
            )
        )
    print_benchmark_results(title, rows)
    print_timing_breakdowns(
        "DCP CUDA-event phase breakdown",
        {
            method: report["stages_ms"]
            for method, report in reports.items()
            if report["execution"].get("phase_profile") is None
        },
        unit="ms",
        aggregation=(
            "per-iteration maximum across benchmark ranks, then p50/p90 across samples"
        ),
    )

    mega_timestamps = {}
    for method, report in reports.items():
        phase_profile = report["execution"].get("phase_profile")
        if phase_profile is None:
            continue
        mega_timestamps[method] = {
            **phase_profile["milestones_us"],
            **{
                f"post_completion/{name}": values
                for name, values in phase_profile[
                    "post_global_completion_tails_us"
                ].items()
            },
        }
    print_timing_breakdowns(
        "Mega %globaltimer phase timestamps",
        mega_timestamps,
        unit="us",
        aggregation=(
            "per-iteration maximum across benchmark ranks, then p50/p90 across samples"
        ),
        include_zero=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--mega-num-comm-sm", type=int, default=8)
    parser.add_argument("--mega-block-n", type=int, choices=(128, 176), default=128)
    parser.add_argument(
        "--mega-phase-timestamps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record low-overhead mega-kernel phase completion timestamps",
    )
    parser.add_argument(
        "--baseline-phase-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Record non-Mega CUDA-event phase breakdowns; disabling keeps only "
            "the attention start/end events used for end-to-end latency"
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture each method and full-KV reference before timed replay",
    )
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare each method against full-KV before measurement",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    device: torch.device | None = None,
    dcp_group: DCPGroup | None = None,
    manage_process_group: bool = True,
) -> dict[str, object]:
    args = parse_args(argv)
    if device is None:
        device = initialize_distributed_sm90("benchmark")
    elif not dist.is_initialized():
        raise RuntimeError("an externally supplied device requires an initialized process group")
    mega_runner: DCPMegaAttentionRunner | None = None
    case_completed = False
    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        require_world_size(args.tp_size)
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
        if "mega" in implementations:
            if args.workload != "chunk":
                raise SystemExit("the DCP mega experimental path supports chunk only")
            if args.dcp_size not in (2, 4, 8):
                raise SystemExit("the DCP mega experimental path requires DCP size 2, 4, or 8")
        topology = make_topology(
            args.qhead, args.kvhead, args.tp_size, args.dcp_size
        )
        if dcp_group is None:
            dcp_group = make_dcp_group(args.dcp_size, device)
        elif dcp_group.size != args.dcp_size:
            raise RuntimeError(
                f"supplied DCP group size {dcp_group.size} does not match "
                f"--dcp-size={args.dcp_size}"
            )
        process_group = dcp_group.process_group
        if dcp_group.ranks != topology.dcp_group_ranks(rank):
            raise RuntimeError("DCP process group crosses a KV replica boundary")
        inputs = build_inputs(args, topology, device)
        local_lengths_by_rank = all_rank_local_lengths(inputs, device)

        if rank == 0:
            execution_mode = "cuda_graph" if args.cuda_graph else "eager"
            print(
                f"Config: world_size={world_size}, "
                f"methods={list(expanded_method_labels(implementations))}, "
                f"QH={args.qhead}, KVH={args.kvhead}, D={args.headdim}, "
                f"TP={args.tp_size}, DCP={args.dcp_size}, "
                f"workload={args.workload}, execution={execution_mode}, "
                f"baseline_phase_timing={args.baseline_phase_timing}, "
                f"warmup={args.warmup}, iters={args.iters}, check={args.check}"
            )
            print(
                f"Workload: B={args.b}, q_tokens={sum(inputs.q_lengths)}, "
                f"history_tokens={sum(inputs.history_lengths)}, "
                f"q_seqlens={inputs.q_lengths}, "
                f"history_seqlens={inputs.history_lengths}"
            )
            print(
                "Agg TFLOPS uses useful attention work across all TP ranks and "
                "p50(max_across_ranks); Avg/GPU divides it by world_size."
            )
            print(
                "KV GB/s/GPU is average logical BF16 K+V bytes read per TP rank "
                "divided by p50(max_across_ranks); it is not hardware-counter HBM traffic."
            )
            print(
                f"\nRunning B={args.b}, q_tokens={sum(inputs.q_lengths)}, "
                f"history_tokens={sum(inputs.history_lengths)}, "
                f"causal={args.workload == 'chunk'}",
                flush=True,
            )

        runners = make_runner_set(
            process_group,
            implementations,
            timed=True,
            varlen=True,
            phase_timing=args.baseline_phase_timing,
        )
        mega_q = None
        if "mega" in implementations:
            mega_runner = DCPMegaAttentionRunner(
                process_group,
                dist.group.WORLD,
                max_total_q=sum(inputs.q_lengths),
                max_batch=len(inputs.q_lengths),
                Hq_local=topology.q_heads_local,
                max_num_splits=128,
                num_comm_sm=args.mega_num_comm_sm,
                block_n_override=args.mega_block_n,
                record_phase_timestamps=args.mega_phase_timestamps,
            )
            mega_q = mega_runner.q_local(sum(inputs.q_lengths))
            mega_q.copy_(inputs.q_local)

        reference_output = full_forward(inputs, args) if args.check else None
        reports: dict[str, dict[str, object]] = {}
        for method, runner in runners.items():
            overlap = method == METHOD_OURS_OVERLAP
            call = lambda runner=runner, overlap=overlap: runner_call(
                runner, inputs, args, overlap
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
        if mega_runner is not None:
            assert mega_q is not None
            assert inputs.k_chunk is not None and inputs.v_chunk is not None

            def mega_call() -> torch.Tensor:
                return mega_runner.forward_chunk_prefill_varlen(
                    mega_q,
                    inputs.k_history_local,
                    inputs.v_history_local,
                    inputs.k_chunk,
                    inputs.v_chunk,
                    inputs.cu_q,
                    inputs.cu_history_local,
                    max(inputs.q_lengths),
                    max(inputs.local_history_lengths),
                    cu_seqlens_q_host=inputs.cu_q_host,
                    cu_seqlens_history_local_host=inputs.cu_history_local_host,
                    num_splits=args.num_splits,
                    return_lse=False,
                )

            stages, execution = measure_mega(
                mega_runner, mega_call, args, device, reference_output
            )
            reports[METHOD_MEGA] = {
                "stages_ms": stages,
                "output_collective_kind": mega_runner.output_collective_kind,
                "workspace_policy": mega_runner.workspace_policy,
                "communication": communication_report(
                    METHOD_MEGA,
                    sum(inputs.q_lengths),
                    topology.q_heads_local,
                    args.headdim,
                    args.dcp_size,
                ),
                "execution": execution,
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
        for method, report in reports.items():
            kv_bytes_by_rank = (
                [float(full_kv_bytes)] * topology.tp_size
                if method == METHOD_FULL
                else [float(value) for value in local_kv_bytes_by_rank]
            )
            average_kv_bytes = sum(kv_bytes_by_rank) / len(kv_bytes_by_rank)
            report["logical_kv_read"] = {
                "bytes_by_tp_rank": kv_bytes_by_rank,
                "average_bytes_per_gpu": average_kv_bytes,
                "effective_bandwidth_gbps_per_gpu": (
                    effective_kv_bandwidth_gbps_per_gpu(
                        kv_bytes_by_rank,
                        report["stages_ms"]["attention_end_to_end_ms"]["p50"],
                    )
                ),
                "latency_basis": "p50_max_across_ranks",
                "traffic_model": "logical BF16 K+V input bytes counted once",
            }
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
            print_results(args, topology, inputs, reports)
            print(f"Wrote benchmark JSON to {output_path}", flush=True)
        case_completed = True
        return result
    finally:
        if mega_runner is not None:
            mega_runner.close()
        if case_completed and dist.is_initialized():
            dist.barrier()
        if manage_process_group and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
