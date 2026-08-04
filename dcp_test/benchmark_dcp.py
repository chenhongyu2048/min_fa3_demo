"""Compare DCP orchestration while holding the local min FA3 kernel fixed."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Callable, Iterable

import torch
import torch.distributed as dist

import min_fa3_op
from dcp_test.benchmark_output import (
    DCPBenchmarkRow,
    effective_kv_bandwidth_gbps_per_gpu,
    print_benchmark_results,
)
from min_fa3_dcp import (
    DCPAttentionCUDAGraph,
    DCPAttentionRunner,
    DCPTopology,
    SGLangDCPAttentionRunner,
    VLLMA2ADCPAttentionRunner,
    VLLMDCPAttentionRunner,
    _dcp_a2a_payload_bytes,
    make_topology,
    validate_topology,
)


VLLM_COMMIT = "a89015c6df8eeb37a843b717c97a5be1355de83d"
SGLANG_COMMIT = "8d6549bc4039d33635844495d86684677a4f0df8"
METHOD_OURS_OVERLAP = "ours_overlap"
METHOD_OURS_NO_OVERLAP = "ours_no_overlap"
METHOD_VLLM = "vllm_ag_rs_min_fa3"
METHOD_VLLM_A2A = "vllm_a2a_min_fa3"
METHOD_SGLANG = "sglang_mha_ag_ar_min_fa3"
METHOD_FULL = "full_kv_min_fa3"
PHASE_NAMES = (
    "q_allgather_and_reorder_ms",
    "local_chunk_attention_ms",
    "overlapped_ag_chunk_window_ms",
    "local_history_attention_ms",
    "lse_allgather_correct_ms",
    "output_collective_ms",
    "output_reduce_scatter_ms",
    "a2a_pack_ms",
    "a2a_all_to_all_ms",
    "a2a_unpack_combine_ms",
    "state_merge_ms",
    "attention_end_to_end_ms",
)
CAPTURE_EAGER_WARMUP = 3


@dataclass(frozen=True)
class DCPGroup:
    size: int
    start_rank: int
    ranks: tuple[int, ...]
    process_group: dist.ProcessGroup

    @property
    def rank(self) -> int:
        return dist.get_rank(self.process_group)


@dataclass
class CaseInputs:
    q_local: torch.Tensor
    k_history_full: torch.Tensor
    v_history_full: torch.Tensor
    k_history_local: torch.Tensor
    v_history_local: torch.Tensor
    history_lengths_full: torch.Tensor
    history_lengths_local: torch.Tensor
    k_chunk: torch.Tensor | None = None
    v_chunk: torch.Tensor | None = None
    k_reference: torch.Tensor | None = None
    v_reference: torch.Tensor | None = None
    reference_lengths: torch.Tensor | None = None


def parse_int_list(spec: str, name: str) -> list[int]:
    try:
        values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    except ValueError as error:
        raise SystemExit(f"{name} must be a comma-separated integer list") from error
    if not values:
        raise SystemExit(f"{name} must contain at least one integer")
    return values


def parse_implementations(spec: str) -> tuple[str, ...]:
    values = tuple(token.strip().lower() for token in spec.split(",") if token.strip())
    allowed = {"ours", "vllm", "sglang"}
    unknown = sorted(set(values) - allowed)
    if not values or unknown:
        raise SystemExit(
            "--implementations must contain one or more of ours,vllm,sglang; "
            f"unknown={unknown}"
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
    methods.append(METHOD_FULL)
    return tuple(methods)


def make_dcp_groups(sizes: Iterable[int], device: torch.device) -> dict[int, DCPGroup]:
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    local_groups: dict[int, DCPGroup] = {}
    for size in sorted(set(sizes)):
        if size <= 0 or size > world_size or world_size % size:
            raise SystemExit(
                f"every DCP size must divide torchrun world size {world_size}, got {size}"
            )
        for start_rank in range(0, world_size, size):
            ranks = tuple(range(start_rank, start_rank + size))
            group = dist.new_group(list(ranks), backend="nccl", device_id=device)
            if global_rank in ranks:
                local_groups[size] = DCPGroup(size, start_rank, ranks, group)
    return local_groups


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def randn_bf16(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)


def case_seed(
    topology: DCPTopology,
    workload: str,
    batch_size: int,
    sq: int,
    sk: int,
) -> int:
    return (
        17011
        + topology.q_heads * 1009
        + topology.kv_heads * 503
        + topology.dcp_size * 211
        + batch_size * 97
        + sq * 53
        + sk * 7
        + (1 if workload == "chunk" else 0)
    )


def shard_interleaved(tensor: torch.Tensor, dcp_rank: int, dcp_size: int) -> torch.Tensor:
    return tensor[:, dcp_rank::dcp_size].contiguous()


def interleaved_local_length(global_length: int, dcp_rank: int, dcp_size: int) -> int:
    return max(0, (global_length + dcp_size - 1 - dcp_rank) // dcp_size)


def build_case_inputs(
    topology: DCPTopology,
    workload: str,
    batch_size: int,
    sq: int,
    sk: int,
    tp_rank: int,
    device: torch.device,
) -> CaseInputs:
    seed = case_seed(topology, workload, batch_size, sq, sk)
    q_generator = make_generator(seed + tp_rank * 100_003, device)
    q_local = randn_bf16(
        (batch_size, sq, topology.q_heads_local, 128), q_generator, device
    )

    kv_head = topology.kv_head_for_rank(tp_rank)
    kv_generator = make_generator(seed + kv_head * 1_000_003, device)
    k_history_full = randn_bf16((batch_size, sk, 1, 128), kv_generator, device)
    v_history_full = randn_bf16((batch_size, sk, 1, 128), kv_generator, device)
    dcp_rank = topology.dcp_rank(tp_rank)
    k_history_local = shard_interleaved(
        k_history_full, dcp_rank, topology.dcp_size
    )
    v_history_local = shard_interleaved(
        v_history_full, dcp_rank, topology.dcp_size
    )
    local_sk = interleaved_local_length(sk, dcp_rank, topology.dcp_size)
    history_lengths_full = torch.full(
        (batch_size,), sk, device=device, dtype=torch.int32
    )
    history_lengths_local = torch.full(
        (batch_size,), local_sk, device=device, dtype=torch.int32
    )
    inputs = CaseInputs(
        q_local=q_local,
        k_history_full=k_history_full,
        v_history_full=v_history_full,
        k_history_local=k_history_local,
        v_history_local=v_history_local,
        history_lengths_full=history_lengths_full,
        history_lengths_local=history_lengths_local,
    )
    if workload == "decode":
        inputs.k_reference = k_history_full
        inputs.v_reference = v_history_full
        inputs.reference_lengths = history_lengths_full
        return inputs

    k_chunk = randn_bf16((batch_size, sq, 1, 128), kv_generator, device)
    v_chunk = randn_bf16((batch_size, sq, 1, 128), kv_generator, device)
    inputs.k_chunk = k_chunk
    inputs.v_chunk = v_chunk
    inputs.k_reference = torch.cat((k_history_full, k_chunk), dim=1)
    inputs.v_reference = torch.cat((v_history_full, v_chunk), dim=1)
    inputs.reference_lengths = torch.full(
        (batch_size,), sk + sq, device=device, dtype=torch.int32
    )
    return inputs


def global_rank_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def global_rank_max_dict(
    values: dict[str, float], device: torch.device
) -> dict[str, float]:
    names = sorted(values)
    tensor = torch.tensor(
        [values[name] for name in names], device=device, dtype=torch.float64
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return {name: float(value) for name, value in zip(names, tensor.tolist())}


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
    }


def all_rank_quantiles(
    values: list[float], device: torch.device
) -> dict[str, list[float]]:
    local = quantiles(values)
    local_tensor = torch.tensor(
        [local["p50"], local["p90"]], device=device, dtype=torch.float64
    )
    gathered = [torch.empty_like(local_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_tensor)
    return {
        "p50": [float(item[0].item()) for item in gathered],
        "p90": [float(item[1].item()) for item in gathered],
    }


def synchronize_before_samples(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    dist.barrier()


def benchmark_full_kv(
    call: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
    device: torch.device,
    *,
    cuda_graph: bool,
    static_tensors: dict[str, torch.Tensor],
    static_scalars: dict[str, object],
) -> tuple[dict[str, list[float]], dict[str, object]]:
    graph: torch.cuda.CUDAGraph | None = None
    capture_stream: torch.cuda.Stream | None = None

    def replay() -> torch.Tensor:
        if graph is None or capture_stream is None:
            return call()
        current_stream = torch.cuda.current_stream(device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            graph.replay()
        current_stream.wait_stream(capture_stream)
        return static_output

    if cuda_graph:
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

    try:
        for _ in range(warmup):
            replay()
        synchronize_before_samples(device)
        samples: list[float] = []
        local_samples: list[float] = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            replay()
            end.record()
            end.synchronize()
            local_elapsed_ms = start.elapsed_time(end)
            local_samples.append(local_elapsed_ms)
            samples.append(global_rank_max(local_elapsed_ms, device))
        signature = {
            "operation": "full_kv",
            "bindings": {
                **{
                    name: DCPAttentionRunner._tensor_signature(tensor)
                    for name, tensor in static_tensors.items()
                },
                **static_scalars,
            },
        }
        execution = {
            "execution_mode": "cuda_graph" if cuda_graph else "eager",
            "capture_eager_warmup": CAPTURE_EAGER_WARMUP if cuda_graph else 0,
            "post_capture_warmup": warmup,
            "stream_policy": "single_stream",
            "overlap_q_allgather": False,
            "graph_static_signature": signature if cuda_graph else None,
            "rank_latency_ms": all_rank_quantiles(local_samples, device),
        }
        phase_samples = {
            name: [0.0] * iterations for name in PHASE_NAMES
        }
        phase_samples["attention_end_to_end_ms"] = samples
        return phase_samples, execution
    finally:
        if graph is not None:
            torch.cuda.synchronize(device)
            graph.reset()


def benchmark_runner(
    runner: DCPAttentionRunner,
    call: Callable[[bool], torch.Tensor],
    warmup: int,
    iterations: int,
    device: torch.device,
    *,
    cuda_graph: bool,
    capture: Callable[[], DCPAttentionCUDAGraph],
    overlap_q_allgather: bool,
) -> tuple[dict[str, list[float]], dict[str, object]]:
    captured: DCPAttentionCUDAGraph | None = capture() if cuda_graph else None
    try:
        for _ in range(warmup):
            captured.replay() if captured is not None else call(False)
        synchronize_before_samples(device)
        samples: dict[str, list[float]] = {}
        local_latency_samples: list[float] = []
        for _ in range(iterations):
            if captured is not None:
                captured.replay()
            else:
                call(True)
            local_timing = runner.last_timing_ms(synchronize=True)
            local_latency_samples.append(local_timing["attention_end_to_end_ms"])
            timing = global_rank_max_dict(local_timing, device)
            for name, value in timing.items():
                samples.setdefault(name, []).append(value)
        execution = {
            "execution_mode": "cuda_graph" if cuda_graph else "eager",
            "capture_eager_warmup": CAPTURE_EAGER_WARMUP if cuda_graph else 0,
            "post_capture_warmup": warmup,
            "stream_policy": (
                "compute_plus_communication"
                if overlap_q_allgather
                else "single_stream"
            ),
            "overlap_q_allgather": overlap_q_allgather,
            "graph_static_signature": (
                captured.signature if captured is not None else None
            ),
            "rank_latency_ms": all_rank_quantiles(
                local_latency_samples, device
            ),
        }
        return samples, execution
    finally:
        if captured is not None:
            captured.close()


def useful_flops(
    topology: DCPTopology,
    workload: str,
    batch_size: int,
    sq: int,
    sk: int,
    head_dim: int,
) -> float:
    if workload == "decode":
        valid_pairs = sq * sk
    else:
        valid_pairs = sq * sk + sq * (sq + 1) / 2.0
    return 4.0 * batch_size * topology.q_heads * head_dim * valid_pairs


def memory_report(
    topology: DCPTopology,
    workload: str,
    batch_size: int,
    sq: int,
    sk: int,
    head_dim: int,
) -> dict[str, float]:
    current_tokens = sq if workload == "chunk" else 0
    bytes_per_token = head_dim * 2 * 2
    full_per_rank = batch_size * (sk + current_tokens) * bytes_per_token
    local_per_rank = batch_size * (
        math.ceil(sk / topology.dcp_size) + current_tokens
    ) * bytes_per_token
    return {
        "full_kv_bytes_per_rank": float(full_per_rank),
        "dcp_kv_bytes_per_rank": float(local_per_rank),
        "full_kv_bytes_all_tp_ranks": float(full_per_rank * topology.tp_size),
        "dcp_kv_bytes_all_tp_ranks": float(local_per_rank * topology.tp_size),
        "actual_kv_memory_reduction": full_per_rank / local_per_rank,
        "theoretical_history_kv_reduction": float(topology.dcp_size),
    }


def logical_kv_bytes_by_tp_rank(
    topology: DCPTopology,
    workload: str,
    batch_size: int,
    sq: int,
    sk: int,
    head_dim: int,
    *,
    full_kv: bool,
) -> list[float]:
    current_tokens = sq if workload == "chunk" else 0
    bytes_per_kv_token = head_dim * 2 * 2
    if full_kv:
        bytes_per_rank = batch_size * (sk + current_tokens) * bytes_per_kv_token
        return [float(bytes_per_rank)] * topology.tp_size
    return [
        float(
            batch_size
            * (
                interleaved_local_length(
                    sk, topology.dcp_rank(tp_rank), topology.dcp_size
                )
                + current_tokens
            )
            * bytes_per_kv_token
        )
        for tp_rank in range(topology.tp_size)
    ]


def communication_report(
    method: str,
    topology: DCPTopology,
    batch_size: int,
    sq: int,
    head_dim: int,
) -> dict[str, object]:
    if method == METHOD_FULL:
        return {
            "output_collective": "none",
            "q_allgather_receive_bytes_per_rank": 0.0,
            "lse_allgather_receive_bytes_per_rank": 0.0,
            "output_collective_buffer_bytes_per_rank": 0.0,
            "output_collective_remote_bytes_per_rank": 0.0,
            "output_collective_payload_bytes_per_rank": 0.0,
            "collective_payload_bytes_per_rank_total": 0.0,
        }
    n = topology.dcp_size
    h_local = topology.q_heads_local
    h_group = h_local * n
    q_local_bytes = batch_size * sq * h_local * head_dim * 2
    lse_local_bytes = batch_size * sq * h_group * 4
    q_receive = (n - 1) * q_local_bytes
    lse_receive = (n - 1) * lse_local_bytes
    if method == METHOD_VLLM_A2A:
        output_kind = "bf16_packed_all_to_all"
        output_buffer, output_payload = _dcp_a2a_payload_bytes(
            batch_size * sq, h_local, head_dim, n
        )
        lse_receive = 0
    elif method == METHOD_SGLANG:
        output_kind = "fp32_all_reduce"
        output_tensor_bytes = batch_size * sq * h_group * head_dim * 4
        output_buffer = output_tensor_bytes
        output_payload = 2.0 * (n - 1) / n * output_tensor_bytes
    else:
        output_kind = "bf16_reduce_scatter"
        output_tensor_bytes = batch_size * sq * h_group * head_dim * 2
        output_buffer = output_tensor_bytes
        output_payload = (n - 1) / n * output_tensor_bytes
    return {
        "output_collective": output_kind,
        "q_allgather_receive_bytes_per_rank": float(q_receive),
        "lse_allgather_receive_bytes_per_rank": float(lse_receive),
        "output_collective_buffer_bytes_per_rank": float(output_buffer),
        "output_collective_remote_bytes_per_rank": float(output_payload),
        "output_collective_payload_bytes_per_rank": float(output_payload),
        "collective_payload_bytes_per_rank_total": float(
            q_receive + lse_receive + output_payload
        ),
        "payload_model": (
            "Per-rank logical receive/reduction payload. FP32 ring all-reduce is "
            "modeled as reduce-scatter plus all-gather; BF16 RS as its reduce-scatter leg."
        ),
    }


def method_metadata(method: str) -> dict[str, str]:
    metadata = {
        METHOD_OURS_OVERLAP: {
            "source": "min_fa3_dcp.DCPAttentionRunner",
            "workspace_policy": "persistent streams/events/grow-only buffers",
            "chunk_schedule": "Q all-gather overlaps local causal chunk attention",
        },
        METHOD_OURS_NO_OVERLAP: {
            "source": "min_fa3_dcp.DCPAttentionRunner",
            "workspace_policy": (
                "persistent grow-only buffers; single compute stream; "
                "no intra-forward dependency events"
            ),
            "chunk_schedule": (
                "single stream: Q all-gather, local causal chunk attention, "
                "then history attention"
            ),
        },
        METHOD_VLLM: {
            "source": f"vLLM {VLLM_COMMIT} default ag_rs, copied and trimmed",
            "workspace_policy": "framework-style per-call layout/collective tensors",
            "chunk_schedule": "context path then local causal chunk attention",
        },
        METHOD_VLLM_A2A: {
            "source": f"vLLM {VLLM_COMMIT} a2a, copied and trimmed",
            "workspace_policy": (
                "per-call graph-private send/recv tensors; no growable workspace"
            ),
            "chunk_schedule": "context A2A then local causal chunk attention",
        },
        METHOD_SGLANG: {
            "source": f"SGLang {SGLANG_COMMIT} MHA DCP path, copied and trimmed",
            "workspace_policy": "framework-style per-call layout/collective tensors",
            "chunk_schedule": "local causal chunk attention then context path",
        },
        METHOD_FULL: {
            "source": "min_fa3_op.forward_kvcache",
            "workspace_policy": "operator-managed",
            "chunk_schedule": "single full-KV causal attention",
        },
    }
    return metadata[method]


def build_method_report(
    method: str,
    samples: dict[str, list[float]],
    topology: DCPTopology,
    workload: str,
    batch_size: int,
    sq: int,
    sk: int,
    head_dim: int,
) -> dict[str, object]:
    latency = samples["attention_end_to_end_ms"]
    flops = useful_flops(topology, workload, batch_size, sq, sk, head_dim)
    tflops_samples = [flops / max(ms, 1.0e-12) / 1.0e9 for ms in latency]
    stages = {
        name: quantiles(samples[name])
        for name in PHASE_NAMES
        if name in samples and name != "attention_end_to_end_ms"
    }
    report: dict[str, object] = {
        "method": method,
        "metadata": method_metadata(method),
        "latency_ms": quantiles(latency),
        "stage_latency_ms": stages,
        "effective_tflops": quantiles(tflops_samples),
        "communication": communication_report(
            method, topology, batch_size, sq, head_dim
        ),
        "raw_samples_ms": {name: list(values) for name, values in samples.items()},
    }
    if (
        method == METHOD_OURS_OVERLAP
        and workload == "chunk"
        and all(
            name in samples
            for name in (
                "q_allgather_and_reorder_ms",
                "local_chunk_attention_ms",
                "overlapped_ag_chunk_window_ms",
            )
        )
    ):
        hidden_times = [
            max(0.0, q_ms + chunk_ms - window_ms)
            for q_ms, chunk_ms, window_ms in zip(
                samples["q_allgather_and_reorder_ms"],
                samples["local_chunk_attention_ms"],
                samples["overlapped_ag_chunk_window_ms"],
            )
        ]
        hidden_fractions = [
            hidden / max(min(q_ms, chunk_ms), 1.0e-12)
            for hidden, q_ms, chunk_ms in zip(
                hidden_times,
                samples["q_allgather_and_reorder_ms"],
                samples["local_chunk_attention_ms"],
            )
        ]
        report["overlap"] = {
            "hidden_time_ms": quantiles(hidden_times),
            "hidden_fraction": quantiles(hidden_fractions),
        }
    return report


def make_method_calls(
    implementations: tuple[str, ...],
    workload: str,
    inputs: CaseInputs,
    runners: dict[str, DCPAttentionRunner],
    num_splits: int,
) -> list[tuple[str, DCPAttentionRunner | None, Callable[[bool], torch.Tensor]]]:
    calls: list[
        tuple[str, DCPAttentionRunner | None, Callable[[bool], torch.Tensor]]
    ] = []
    if "ours" in implementations:
        overlap_runner = runners[METHOD_OURS_OVERLAP]
        no_overlap_runner = runners[METHOD_OURS_NO_OVERLAP]
        if workload == "decode":
            calls.extend(
                (
                    (
                        METHOD_OURS_NO_OVERLAP,
                        no_overlap_runner,
                        lambda timing: no_overlap_runner.forward_decode(
                            inputs.q_local,
                            inputs.k_history_local,
                            inputs.v_history_local,
                            inputs.history_lengths_local,
                            num_splits=num_splits,
                            overlap_q_allgather=False,
                            _record_timing=timing,
                        ),
                    ),
                    (
                        METHOD_OURS_OVERLAP,
                        overlap_runner,
                        lambda timing: overlap_runner.forward_decode(
                            inputs.q_local,
                            inputs.k_history_local,
                            inputs.v_history_local,
                            inputs.history_lengths_local,
                            num_splits=num_splits,
                            overlap_q_allgather=True,
                            _record_timing=timing,
                        ),
                    ),
                )
            )
        else:
            assert inputs.k_chunk is not None and inputs.v_chunk is not None
            calls.extend(
                (
                    (
                        METHOD_OURS_NO_OVERLAP,
                        no_overlap_runner,
                        lambda timing: no_overlap_runner.forward_chunk_prefill(
                            inputs.q_local,
                            inputs.k_history_local,
                            inputs.v_history_local,
                            inputs.history_lengths_local,
                            inputs.k_chunk,
                            inputs.v_chunk,
                            num_splits=num_splits,
                            overlap_q_allgather=False,
                            _record_timing=timing,
                        ),
                    ),
                    (
                        METHOD_OURS_OVERLAP,
                        overlap_runner,
                        lambda timing: overlap_runner.forward_chunk_prefill(
                            inputs.q_local,
                            inputs.k_history_local,
                            inputs.v_history_local,
                            inputs.history_lengths_local,
                            inputs.k_chunk,
                            inputs.v_chunk,
                            num_splits=num_splits,
                            overlap_q_allgather=True,
                            _record_timing=timing,
                        ),
                    ),
                )
            )

    for implementation, method in (
        ("vllm", METHOD_VLLM),
        ("vllm", METHOD_VLLM_A2A),
        ("sglang", METHOD_SGLANG),
    ):
        if implementation not in implementations:
            continue
        runner = runners[method]
        if workload == "decode":
            call = lambda timing, runner=runner: runner.forward_decode(
                inputs.q_local,
                inputs.k_history_local,
                inputs.v_history_local,
                inputs.history_lengths_local,
                num_splits=num_splits,
                overlap_q_allgather=False,
                _record_timing=timing,
            )
        else:
            assert inputs.k_chunk is not None and inputs.v_chunk is not None
            call = lambda timing, runner=runner: runner.forward_chunk_prefill(
                inputs.q_local,
                inputs.k_history_local,
                inputs.v_history_local,
                inputs.history_lengths_local,
                inputs.k_chunk,
                inputs.v_chunk,
                num_splits=num_splits,
                overlap_q_allgather=False,
                _record_timing=timing,
            )
        calls.append((method, runner, call))
    return calls


def capture_method(
    runner: DCPAttentionRunner,
    workload: str,
    inputs: CaseInputs,
    num_splits: int,
    overlap_q_allgather: bool,
) -> DCPAttentionCUDAGraph:
    if workload == "decode":
        return runner.capture_decode(
            inputs.q_local,
            inputs.k_history_local,
            inputs.v_history_local,
            inputs.history_lengths_local,
            num_splits=num_splits,
            return_lse=False,
            overlap_q_allgather=overlap_q_allgather,
            record_timing=True,
            capture_warmup=CAPTURE_EAGER_WARMUP,
        )
    assert inputs.k_chunk is not None and inputs.v_chunk is not None
    return runner.capture_chunk_prefill(
        inputs.q_local,
        inputs.k_history_local,
        inputs.v_history_local,
        inputs.history_lengths_local,
        inputs.k_chunk,
        inputs.v_chunk,
        num_splits=num_splits,
        return_lse=False,
        overlap_q_allgather=overlap_q_allgather,
        record_timing=True,
        capture_warmup=CAPTURE_EAGER_WARMUP,
    )


def check_output(
    name: str, actual: torch.Tensor, expected: torch.Tensor
) -> None:
    close = torch.tensor(
        int(
            torch.isclose(
                actual.float(), expected.float(), atol=3.0e-2, rtol=3.0e-2
            ).all()
        ),
        device=actual.device,
        dtype=torch.int32,
    )
    dist.all_reduce(close, op=dist.ReduceOp.MIN)
    if not close.item():
        raise RuntimeError(f"{name} failed the pre-benchmark correctness check")


def run_case(
    topology: DCPTopology,
    case_kind: str,
    workload: str,
    batch_size: int,
    sq: int,
    sk: int,
    implementations: tuple[str, ...],
    runners: dict[str, DCPAttentionRunner],
    num_splits: int,
    warmup: int,
    iterations: int,
    device: torch.device,
    cuda_graph: bool,
) -> dict[str, object]:
    tp_rank = dist.get_rank()
    inputs = build_case_inputs(
        topology, workload, batch_size, sq, sk, tp_rank, device
    )
    assert inputs.k_reference is not None
    assert inputs.v_reference is not None
    assert inputs.reference_lengths is not None

    def full_call() -> torch.Tensor:
        return min_fa3_op.forward_kvcache(
            inputs.q_local,
            inputs.k_reference,
            inputs.v_reference,
            inputs.reference_lengths,
            num_splits=num_splits,
            return_lse=False,
            is_causal=workload == "chunk",
        )

    full_samples, full_execution = benchmark_full_kv(
        full_call,
        warmup,
        iterations,
        device,
        cuda_graph=cuda_graph,
        static_tensors={
            "q_local": inputs.q_local,
            "k_reference": inputs.k_reference,
            "v_reference": inputs.v_reference,
            "reference_lengths": inputs.reference_lengths,
        },
        static_scalars={
            "num_splits": num_splits,
            "return_lse": False,
            "is_causal": workload == "chunk",
        },
    )
    reference_output = full_call()
    method_samples: dict[str, dict[str, list[float]]] = {
        METHOD_FULL: full_samples
    }
    method_execution = {METHOD_FULL: full_execution}
    for method, runner, call in make_method_calls(
        implementations, workload, inputs, runners, num_splits
    ):
        assert runner is not None
        check_output(method, call(False), reference_output)
        overlap = method == METHOD_OURS_OVERLAP
        samples, execution = benchmark_runner(
            runner,
            call,
            warmup,
            iterations,
            device,
            cuda_graph=cuda_graph,
            capture=lambda runner=runner, overlap=overlap: capture_method(
                runner, workload, inputs, num_splits, overlap
            ),
            overlap_q_allgather=overlap,
        )
        method_samples[method] = samples
        method_execution[method] = execution

    methods = {
        method: build_method_report(
            method,
            samples,
            topology,
            workload,
            batch_size,
            sq,
            sk,
            128,
        )
        for method, samples in method_samples.items()
    }
    for method, report in methods.items():
        report["execution"] = method_execution[method]
        kv_bytes_by_rank = logical_kv_bytes_by_tp_rank(
            topology,
            workload,
            batch_size,
            sq,
            sk,
            128,
            full_kv=method == METHOD_FULL,
        )
        average_kv_bytes = sum(kv_bytes_by_rank) / len(kv_bytes_by_rank)
        report["logical_kv_read"] = {
            "bytes_by_tp_rank": kv_bytes_by_rank,
            "average_bytes_per_gpu": average_kv_bytes,
            "effective_bandwidth_gbps_per_gpu": (
                effective_kv_bandwidth_gbps_per_gpu(
                    kv_bytes_by_rank, report["latency_ms"]["p50"]
                )
            ),
            "latency_basis": "p50_max_across_ranks",
            "traffic_model": "logical BF16 K+V input bytes counted once",
        }
    ours_report = methods.get(METHOD_OURS_NO_OVERLAP)
    full_report = methods[METHOD_FULL]
    for method, report in methods.items():
        report["speedup_vs_full_kv"] = {
            quantile: full_report["latency_ms"][quantile]
            / max(report["latency_ms"][quantile], 1.0e-12)
            for quantile in ("p50", "p90")
        }
        if ours_report is not None:
            report["ours_no_overlap_speedup_vs_method"] = {
                quantile: report["latency_ms"][quantile]
                / max(ours_report["latency_ms"][quantile], 1.0e-12)
                for quantile in ("p50", "p90")
            }

    return {
        "status": "ok",
        "case_kind": case_kind,
        "topology": topology.to_dict(),
        "workload": workload,
        "execution_mode": "cuda_graph" if cuda_graph else "eager",
        "shape": {
            "batch_size": batch_size,
            "sq": sq,
            "sk_history_or_cache": sk,
            "head_dim": 128,
            "dtype": "bfloat16",
            "num_splits": num_splits,
        },
        "memory": memory_report(topology, workload, batch_size, sq, sk, 128),
        "methods": methods,
    }


def geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(value, 1.0e-12)) for value in values) / len(values))


def summarize_group(
    group_by: str,
    group: dict[str, object],
    cases: list[dict[str, object]],
) -> dict[str, object]:
    method_names = sorted(
        {method for case in cases for method in case["methods"]}  # type: ignore[index]
    )
    methods: dict[str, object] = {}
    for method in method_names:
        reports = [
            case["methods"][method]  # type: ignore[index]
            for case in cases
            if method in case["methods"]  # type: ignore[operator]
        ]
        p50_values = [report["latency_ms"]["p50"] for report in reports]
        p90_values = [report["latency_ms"]["p90"] for report in reports]
        speedups = [
            report["ours_no_overlap_speedup_vs_method"]["p50"]
            for report in reports
            if "ours_no_overlap_speedup_vs_method" in report
        ]
        methods[method] = {
            "case_count": len(reports),
            "median_case_p50_ms": median(p50_values),
            "median_case_p90_ms": median(p90_values),
            "geomean_ours_no_overlap_speedup_vs_method": geometric_mean(speedups),
        }
    return {"group_by": group_by, "group": group, "methods": methods}


def build_summaries(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    gqa_cases = [case for case in cases if case["case_kind"] == "gqa"]
    groupers: tuple[tuple[str, Callable[[dict[str, object]], tuple[object, ...]]], ...] = (
        (
            "topology",
            lambda case: (
                case["topology"]["q_heads"],  # type: ignore[index]
                case["topology"]["kv_heads"],  # type: ignore[index]
                case["topology"]["dcp_size"],  # type: ignore[index]
            ),
        ),
        ("workload", lambda case: (case["workload"],)),
        ("batch_size", lambda case: (case["shape"]["batch_size"],)),  # type: ignore[index]
        (
            "context_length",
            lambda case: (case["shape"]["sk_history_or_cache"],),  # type: ignore[index]
        ),
    )
    summaries: list[dict[str, object]] = []
    for group_by, key_fn in groupers:
        grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for case in gqa_cases:
            grouped.setdefault(key_fn(case), []).append(case)
        for key, selected in sorted(grouped.items(), key=lambda item: str(item[0])):
            if group_by == "topology":
                group = {"q_heads": key[0], "kv_heads": key[1], "dcp_size": key[2]}
            else:
                group = {group_by: key[0]}
            summaries.append(summarize_group(group_by, group, selected))
    return summaries


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_report(device: torch.device) -> dict[str, object]:
    props = torch.cuda.get_device_properties(device)
    nccl_version = torch.cuda.nccl.version() if torch.cuda.nccl.is_available([]) else None
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl": nccl_version,
        "gpu_name": props.name,
        "gpu_count": torch.cuda.device_count(),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "memory_bytes_per_gpu": props.total_memory,
        "world_size": dist.get_world_size(),
        "repository_commit": git_commit(),
        "vllm_source_commit": VLLM_COMMIT,
        "sglang_source_commit": SGLANG_COMMIT,
    }


def default_output_path() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return Path("benchmarks/results") / f"dcp_gqa_compare_h100_8gpu_{timestamp}.json"


def print_case(case: dict[str, object], case_index: int, case_total: int) -> None:
    topology = case["topology"]
    shape = case["shape"]
    methods = case["methods"]
    print(
        f"\n[{case_index}/{case_total}] Running {case['case_kind']} "
        f"{case['workload']}",
        flush=True,
    )
    mode = "causal" if case["workload"] == "chunk" else "decode"
    title = (
        f"B={shape['batch_size']}, Sq={shape['sq']}, "
        f"Sk_history_or_cache={shape['sk_history_or_cache']}, "
        f"QH={topology['q_heads']}, KVH={topology['kv_heads']}, "
        f"D={shape['head_dim']}, TP={topology['tp_size']}, "
        f"DCP={topology['dcp_size']}, mode={mode}"
    )
    rows = []
    for name, report in methods.items():
        execution = report["execution"]
        metadata = report["metadata"]
        output_kind = report["communication"]["output_collective"]
        check = "reference" if name == METHOD_FULL else "ok"
        rows.append(
            DCPBenchmarkRow(
                method=name,
                p50_ms=report["latency_ms"]["p50"],
                p90_ms=report["latency_ms"]["p90"],
                aggregate_tflops=report["effective_tflops"]["p50"],
                avg_gpu_tflops=(
                    report["effective_tflops"]["p50"] / topology["tp_size"]
                ),
                kv_bandwidth_gbps_per_gpu=report["logical_kv_read"][
                    "effective_bandwidth_gbps_per_gpu"
                ],
                check=check,
                note=(
                    f"{execution['execution_mode']}; output={output_kind}; "
                    f"{metadata['source']}"
                ),
                rank_p50_ms=execution["rank_latency_ms"]["p50"],
            )
        )
    print_benchmark_results(title, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare vLLM, SGLang, and local DCP orchestration with the same "
            "min_fa3_op.forward_kvcache kernel."
        )
    )
    parser.add_argument("--qhead", type=str, default="32,64")
    parser.add_argument("--kvhead", type=str, default="2,4")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--dcp-sizes", type=str, default="2,4,8")
    parser.add_argument(
        "--implementations", type=str, default="ours,vllm,sglang"
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--workload", choices=("decode", "chunk", "both"), default="both")
    parser.add_argument("--decode-b", type=str, default="1,8,32")
    parser.add_argument("--chunk-b", type=str, default="1,4,16")
    parser.add_argument("--seqlen", type=str, default="4096,16384,65536")
    parser.add_argument("--sq", type=str, default="8,32,128")
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--num-splits", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture each method and full-KV reference before timed replay",
    )
    parser.add_argument(
        "--mqa-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the two fixed Hq=64,Hkv=1,TP=8,DCP=8 control cases",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise SystemExit("This benchmark requires SM90 Hopper")
        if args.tp_size != world_size:
            raise SystemExit(
                f"--tp-size ({args.tp_size}) must equal torchrun world size ({world_size})"
            )
        if args.headdim != 128:
            raise SystemExit("--headdim must be 128")
        if not 0 <= args.num_splits <= 128:
            raise SystemExit("--num-splits must be in [0, 128]")
        if args.warmup < 0 or args.iters <= 0:
            raise SystemExit("--warmup must be nonnegative and --iters must be positive")

        q_heads = parse_int_list(args.qhead, "--qhead")
        kv_heads = parse_int_list(args.kvhead, "--kvhead")
        dcp_sizes = parse_int_list(args.dcp_sizes, "--dcp-sizes")
        implementations = parse_implementations(args.implementations)
        decode_batches = parse_int_list(args.decode_b, "--decode-b")
        chunk_batches = parse_int_list(args.chunk_b, "--chunk-b")
        seqlens = parse_int_list(args.seqlen, "--seqlen")
        chunk_sqs = parse_int_list(args.sq, "--sq")
        if any(value <= 0 for value in q_heads + kv_heads + decode_batches + chunk_batches + seqlens + chunk_sqs):
            raise SystemExit("all head counts and workload dimensions must be positive")

        local_groups = make_dcp_groups(dcp_sizes, device)
        runner_cache: dict[int, dict[str, DCPAttentionRunner]] = {}

        def runners_for(dcp_size: int) -> dict[str, DCPAttentionRunner]:
            cached = runner_cache.get(dcp_size)
            if cached is not None:
                return cached
            group = local_groups[dcp_size].process_group
            cached = {}
            if "ours" in implementations:
                cached[METHOD_OURS_OVERLAP] = DCPAttentionRunner(group)
                cached[METHOD_OURS_NO_OVERLAP] = DCPAttentionRunner(group)
            if "vllm" in implementations:
                cached[METHOD_VLLM] = VLLMDCPAttentionRunner(group)
                cached[METHOD_VLLM_A2A] = VLLMA2ADCPAttentionRunner(group)
            if "sglang" in implementations:
                cached[METHOD_SGLANG] = SGLangDCPAttentionRunner(group)
            runner_cache[dcp_size] = cached
            return cached

        topology_records: list[dict[str, object]] = []
        topologies: list[DCPTopology] = []
        for hq in q_heads:
            for hkv in kv_heads:
                for dcp_size in dcp_sizes:
                    issues = validate_topology(hq, hkv, args.tp_size, dcp_size)
                    record: dict[str, object] = {
                        "q_heads": hq,
                        "kv_heads": hkv,
                        "tp_size": args.tp_size,
                        "dcp_size": dcp_size,
                    }
                    if issues:
                        record.update(
                            status="skipped_topology",
                            reasons=[issue.to_dict() for issue in issues],
                        )
                    else:
                        topology = make_topology(hq, hkv, args.tp_size, dcp_size)
                        record.update(status="valid", derived=topology.to_dict())
                        topologies.append(topology)
                    topology_records.append(record)

        workloads = (
            ("decode", "chunk") if args.workload == "both" else (args.workload,)
        )
        main_shapes = 0
        for workload in workloads:
            if workload == "decode":
                main_shapes += len(decode_batches) * len(seqlens)
            else:
                main_shapes += len(chunk_batches) * len(chunk_sqs) * len(seqlens)
        case_total = len(topologies) * main_shapes
        run_mqa = (
            args.mqa_control
            and args.tp_size == 8
            and 8 in dcp_sizes
        )
        if run_mqa:
            case_total += sum(workload in workloads for workload in ("decode", "chunk"))

        if global_rank == 0:
            execution_mode = "cuda_graph" if args.cuda_graph else "eager"
            valid_count = len(topologies)
            skipped_count = len(topology_records) - valid_count
            print(
                f"Config: world_size={world_size}, "
                f"methods={list(expanded_method_labels(implementations))}, "
                f"QH={q_heads}, KVH={kv_heads}, D={args.headdim}, "
                f"TP={args.tp_size}, DCP={dcp_sizes}, workload={args.workload}, "
                f"execution={execution_mode}, warmup={args.warmup}, "
                f"iters={args.iters}, check=True"
            )
            print(
                f"Workload matrix: cases={case_total}, "
                f"valid_topologies={valid_count}, "
                f"skipped_topologies={skipped_count}, "
                f"decode_B={decode_batches}, chunk_B={chunk_batches}, "
                f"Sq={chunk_sqs}, Sk={seqlens}"
            )
            print(
                "Agg TFLOPS uses useful attention work across all TP ranks and "
                "p50(max_across_ranks); Avg/GPU divides it by world_size."
            )
            print(
                "KV GB/s/GPU is average logical BF16 K+V bytes read per TP rank "
                "divided by p50(max_across_ranks); it is not hardware-counter HBM traffic."
            )

        cases: list[dict[str, object]] = []
        case_index = 0
        for topology in topologies:
            expected_group = topology.dcp_group_ranks(global_rank)
            actual_group = local_groups[topology.dcp_size].ranks
            if actual_group != expected_group:
                raise RuntimeError(
                    f"DCP group {actual_group} crosses the topology's KV replica "
                    f"boundary; expected {expected_group} for TP rank {global_rank}"
                )
            runners = runners_for(topology.dcp_size)
            for workload in workloads:
                batches = decode_batches if workload == "decode" else chunk_batches
                sqs = [1] if workload == "decode" else chunk_sqs
                for batch_size in batches:
                    for sq in sqs:
                        for sk in seqlens:
                            case = run_case(
                                topology,
                                "gqa",
                                workload,
                                batch_size,
                                sq,
                                sk,
                                implementations,
                                runners,
                                args.num_splits,
                                args.warmup,
                                args.iters,
                                device,
                                args.cuda_graph,
                            )
                            cases.append(case)
                            case_index += 1
                            if global_rank == 0:
                                print_case(case, case_index, case_total)

        if run_mqa:
            topology = make_topology(64, 1, 8, 8)
            expected_group = topology.dcp_group_ranks(global_rank)
            actual_group = local_groups[8].ranks
            if actual_group != expected_group:
                raise RuntimeError(
                    f"MQA DCP group mismatch: actual={actual_group}, "
                    f"expected={expected_group}"
                )
            runners = runners_for(8)
            control_shapes = {
                "decode": (8, 1, 16_384),
                "chunk": (4, 128, 16_384),
            }
            for workload in workloads:
                batch_size, sq, sk = control_shapes[workload]
                case = run_case(
                    topology,
                    "mqa_control",
                    workload,
                    batch_size,
                    sq,
                    sk,
                    implementations,
                    runners,
                    args.num_splits,
                    args.warmup,
                    args.iters,
                    device,
                    args.cuda_graph,
                )
                cases.append(case)
                case_index += 1
                if global_rank == 0:
                    print_case(case, case_index, case_total)

        result = {
            "schema_version": 4,
            "comparison_scope": (
                "DCP orchestration comparison under the same "
                "min_fa3_op.forward_kvcache kernel; not end-to-end vLLM or "
                "SGLang serving-engine/backend performance."
            ),
            "environment": environment_report(device),
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
            "topologies": topology_records,
            "cases": cases,
            "summaries": build_summaries(cases),
        }
        if global_rank == 0:
            output_path = args.output_json or default_output_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            valid_count = sum(record["status"] == "valid" for record in topology_records)
            skipped_count = len(topology_records) - valid_count
            print(
                f"Wrote {len(cases)} cases, {valid_count} valid GQA topologies, "
                f"and {skipped_count} skipped topology records to {output_path}",
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
