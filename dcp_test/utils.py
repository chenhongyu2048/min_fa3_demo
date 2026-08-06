"""Shared distributed, input, runner, and timing helpers for DCP tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

import torch
import torch.distributed as dist

from min_fa3_dcp import DCPAttentionRunner


CAPTURE_EAGER_WARMUP = 3

PHASE_EVENT_NAMES = (
    "attention_start",
    "attention_end",
    "q_ag_start",
    "q_ag_end",
    "chunk_start",
    "chunk_end",
    "ag_chunk_end",
    "history_start",
    "history_end",
    "lse_correct_start",
    "lse_correct_end",
    "reduce_scatter_start",
    "reduce_scatter_end",
    "a2a_pack_start",
    "a2a_pack_end",
    "a2a_all_to_all_start",
    "a2a_all_to_all_end",
    "a2a_unpack_combine_start",
    "a2a_unpack_combine_end",
    "merge_start",
    "merge_end",
)


@dataclass(frozen=True)
class DCPGroup:
    size: int
    start_rank: int
    ranks: tuple[int, ...]
    process_group: dist.ProcessGroup

    @property
    def rank(self) -> int:
        return dist.get_rank(self.process_group)


class BenchmarkPhaseRecorder:
    """Own host-queryable CUDA events for one benchmark runner."""

    def __init__(
        self,
        *,
        world_size: int,
        output_collective_kind: str,
        phase_timing: bool = True,
    ) -> None:
        self.world_size = world_size
        self.output_collective_kind = output_collective_kind
        self.phase_timing = phase_timing
        event_names = (
            PHASE_EVENT_NAMES
            if phase_timing
            else ("attention_start", "attention_end")
        )
        self.events = {
            name: torch.cuda.Event(enable_timing=True, external=True)
            for name in event_names
        }
        self.last_kind: str | None = None

    def begin(self, kind: str, stream: torch.cuda.Stream) -> None:
        if kind not in ("decode", "chunk"):
            raise ValueError(f"unknown DCP timing kind: {kind}")
        self.last_kind = kind
        self.record("attention_start", stream)

    def record(self, name: str, stream: torch.cuda.Stream) -> None:
        event = self.events.get(name)
        if event is None:
            if getattr(self, "phase_timing", True):
                raise KeyError(name)
            return
        event.record(stream)

    def elapsed_ms(self, synchronize: bool = True) -> dict[str, float]:
        if self.last_kind is None:
            raise RuntimeError("No timed DCP forward has been recorded")
        if synchronize:
            self.events["attention_end"].synchronize()
        events = self.events
        end_to_end_ms = events["attention_start"].elapsed_time(
            events["attention_end"]
        )
        if not getattr(self, "phase_timing", True):
            return {"attention_end_to_end_ms": end_to_end_ms}
        values = {
            "q_allgather_and_reorder_ms": events["q_ag_start"].elapsed_time(
                events["q_ag_end"]
            ),
            "local_history_attention_ms": events["history_start"].elapsed_time(
                events["history_end"]
            ),
            "attention_end_to_end_ms": end_to_end_ms,
        }
        if self.last_kind == "chunk":
            values.update(
                local_chunk_attention_ms=events["chunk_start"].elapsed_time(
                    events["chunk_end"]
                ),
                overlapped_ag_chunk_window_ms=events["q_ag_start"].elapsed_time(
                    events["ag_chunk_end"]
                ),
                state_merge_ms=events["merge_start"].elapsed_time(
                    events["merge_end"]
                ),
            )
        else:
            values.update(
                local_chunk_attention_ms=0.0,
                overlapped_ag_chunk_window_ms=values[
                    "q_allgather_and_reorder_ms"
                ],
                state_merge_ms=0.0,
            )

        is_a2a = self.output_collective_kind == "bf16_packed_all_to_all"
        if self.world_size > 1 and not is_a2a:
            values.update(
                lse_allgather_correct_ms=events[
                    "lse_correct_start"
                ].elapsed_time(events["lse_correct_end"]),
                output_reduce_scatter_ms=events[
                    "reduce_scatter_start"
                ].elapsed_time(events["reduce_scatter_end"]),
            )
        else:
            values.update(
                lse_allgather_correct_ms=0.0,
                output_reduce_scatter_ms=0.0,
            )
        if self.world_size > 1 and is_a2a:
            values.update(
                a2a_pack_ms=events["a2a_pack_start"].elapsed_time(
                    events["a2a_pack_end"]
                ),
                a2a_all_to_all_ms=events["a2a_all_to_all_start"].elapsed_time(
                    events["a2a_all_to_all_end"]
                ),
                a2a_unpack_combine_ms=events[
                    "a2a_unpack_combine_start"
                ].elapsed_time(events["a2a_unpack_combine_end"]),
            )
        else:
            values.update(
                a2a_pack_ms=0.0,
                a2a_all_to_all_ms=0.0,
                a2a_unpack_combine_ms=0.0,
            )
        values["output_collective_ms"] = (
            values["a2a_all_to_all_ms"]
            if is_a2a
            else values["output_reduce_scatter_ms"]
        )
        if self.output_collective_kind == "fp32_all_reduce":
            values["output_allreduce_ms"] = values["output_collective_ms"]
            values["output_reduce_scatter_ms"] = 0.0
        values["sequential_ag_plus_chunk_ms"] = (
            values["q_allgather_and_reorder_ms"]
            + values["local_chunk_attention_ms"]
        )
        return values


class BenchmarkTimingMixin:
    """Install benchmark phase events without adding them to runtime runners."""

    def __init__(
        self, *args, benchmark_phase_timing: bool = True, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        recorder = BenchmarkPhaseRecorder(
            world_size=self.world_size,
            output_collective_kind=self.output_collective_kind,
            phase_timing=benchmark_phase_timing,
        )
        self._benchmark_phase_recorder = recorder
        self._install_phase_recorder(recorder)

    def last_timing_ms(self, synchronize: bool = True) -> dict[str, float]:
        return self._benchmark_phase_recorder.elapsed_ms(synchronize=synchronize)


class TimedDCPAttentionRunner(BenchmarkTimingMixin, DCPAttentionRunner):
    pass


def initialize_distributed_sm90(purpose: str) -> torch.device:
    """Initialize the torchrun NCCL process and validate the supported GPU."""
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError) as error:
        raise SystemExit("LOCAL_RANK must be set by torchrun") from error
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise SystemExit(f"This {purpose} requires SM90 Hopper")
    return device


def require_world_size(expected: int, option: str = "--tp-size") -> None:
    actual = dist.get_world_size()
    if expected != actual:
        raise SystemExit(f"{option} ({expected}) must equal torchrun world size ({actual})")


def parse_int_list(spec: str, name: str) -> list[int]:
    try:
        values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    except ValueError as error:
        raise SystemExit(f"{name} must be a comma-separated integer list") from error
    if not values:
        raise SystemExit(f"{name} must contain at least one integer")
    return values


def parse_lengths(spec: str, batch_size: int, name: str) -> list[int]:
    values = parse_int_list(spec, name)
    if len(values) == 1:
        values *= batch_size
    if len(values) != batch_size:
        raise SystemExit(
            f"{name} must contain one value or exactly B={batch_size} values"
        )
    if any(value <= 0 for value in values):
        raise SystemExit(f"{name} values must be positive")
    return values


def make_dcp_groups(
    sizes: Iterable[int], device: torch.device
) -> dict[int, DCPGroup]:
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


def make_dcp_group(dcp_size: int, device: torch.device) -> DCPGroup:
    return make_dcp_groups((dcp_size,), device)[dcp_size]


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def randn_bf16(
    shape: tuple[int, ...],
    source: torch.device | torch.Generator | int,
    device: torch.device | None = None,
    *,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if isinstance(source, torch.Generator):
        if generator is not None or seed is not None or device is None:
            raise ValueError("legacy generator form requires shape, generator, device")
        generator = source
    elif isinstance(source, int):
        if generator is not None or seed is not None or device is None:
            raise ValueError("legacy seed form requires shape, seed, device")
        seed = source
    else:
        if device is not None:
            raise ValueError("device must not be repeated")
        device = source
    if device is None or (seed is None) == (generator is None):
        raise ValueError("provide exactly one of seed or generator")
    if generator is None:
        generator = make_generator(seed, device)  # type: ignore[arg-type]
    return torch.randn(
        shape, generator=generator, device=device, dtype=torch.bfloat16
    )


def make_cu_seqlens(
    lengths: Iterable[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    for length in lengths:
        if length <= 0:
            raise ValueError("sequence lengths must be positive")
        offsets.append(offsets[-1] + length)
    host = torch.tensor(offsets, dtype=torch.int32)
    return host.to(device), host


def interleaved_local_length(
    global_length: int, dcp_rank: int, dcp_size: int
) -> int:
    return max(0, (global_length + dcp_size - 1 - dcp_rank) // dcp_size)


def shard_dense_interleaved(
    tensor: torch.Tensor, dcp_rank: int, dcp_size: int
) -> torch.Tensor:
    return tensor[:, dcp_rank::dcp_size].contiguous()


def shard_packed_interleaved(
    tensor: torch.Tensor,
    lengths: Iterable[int],
    dcp_rank: int,
    dcp_size: int,
) -> tuple[torch.Tensor, list[int]]:
    pieces: list[torch.Tensor] = []
    local_lengths: list[int] = []
    start = 0
    for length in lengths:
        piece = tensor[start : start + length][dcp_rank::dcp_size]
        pieces.append(piece)
        local_lengths.append(piece.shape[0])
        start += length
    return torch.cat(pieces, dim=0).contiguous(), local_lengths


def append_packed_chunk(
    history: torch.Tensor,
    chunk: torch.Tensor,
    history_lengths: Iterable[int],
    q_lengths: Iterable[int],
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    history_start = 0
    chunk_start = 0
    for history_length, q_length in zip(history_lengths, q_lengths):
        pieces.append(history[history_start : history_start + history_length])
        pieces.append(chunk[chunk_start : chunk_start + q_length])
        history_start += history_length
        chunk_start += q_length
    return torch.cat(pieces, dim=0).contiguous()


def global_rank_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def global_rank_max_dict(
    values: dict[str, float],
    device: torch.device,
    names: Iterable[str] | None = None,
) -> dict[str, float]:
    selected = sorted(values) if names is None else list(names)
    tensor = torch.tensor(
        [values.get(name, 0.0) for name in selected],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return {
        name: float(value) for name, value in zip(selected, tensor.tolist())
    }


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


_T = TypeVar("_T")


def capture_cuda_graph_callable(
    call: Callable[[], _T],
    device: torch.device,
    *,
    capture_warmup: int = CAPTURE_EAGER_WARMUP,
) -> tuple[Callable[[], _T], Callable[[], None], _T]:
    """Capture a single-stream callable and return replay, close, and output."""
    capture_stream = torch.cuda.Stream(device=device)
    caller_stream = torch.cuda.current_stream(device)
    capture_stream.wait_stream(caller_stream)
    with torch.cuda.stream(capture_stream):
        for _ in range(capture_warmup):
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
            output = call()
    except Exception:
        torch.cuda.synchronize(device)
        graph.reset()
        raise

    def replay() -> _T:
        current_stream = torch.cuda.current_stream(device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            graph.replay()
        current_stream.wait_stream(capture_stream)
        return output

    def close() -> None:
        torch.cuda.synchronize(device)
        graph.reset()

    return replay, close, output


def measure_timed_runner(
    runner: DCPAttentionRunner,
    call: Callable[[], torch.Tensor],
    capture: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
    cuda_graph: bool,
    overlap_q_allgather: bool,
    phase_timing: bool = True,
    phase_names: Iterable[str] | None = None,
    transform_timing: Callable[[dict[str, float]], None] | None = None,
    captured_graph: object | None = None,
) -> tuple[dict[str, list[float]], dict[str, object]]:
    """Measure eager calls or graph replays using an installed phase recorder."""
    captured = captured_graph
    if cuda_graph and captured is None:
        captured = capture()
    if not cuda_graph and captured is not None:
        raise ValueError("captured_graph requires cuda_graph=True")
    try:
        for _ in range(warmup):
            captured.replay() if captured is not None else call()  # type: ignore[attr-defined]
        synchronize_before_samples(device)
        samples: dict[str, list[float]] = {}
        local_latency_samples: list[float] = []
        for _ in range(iterations):
            if captured is not None:
                captured.replay()  # type: ignore[attr-defined]
            else:
                call()
            local_timing = runner.last_timing_ms(synchronize=True)  # type: ignore[attr-defined]
            if transform_timing is not None:
                transform_timing(local_timing)
            local_latency_samples.append(local_timing["attention_end_to_end_ms"])
            timing = global_rank_max_dict(local_timing, device, phase_names)
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
            "cuda_event_phase_timing_enabled": phase_timing,
            "timing_source": (
                "runner_cuda_event_phase_breakdown"
                if phase_timing
                else "runner_cuda_event_end_to_end_only"
            ),
            "graph_static_signature": (
                captured.signature if captured is not None else None  # type: ignore[attr-defined]
            ),
            "rank_latency_ms": all_rank_quantiles(
                local_latency_samples, device
            ),
        }
        return samples, execution
    finally:
        if captured is not None:
            captured.close()  # type: ignore[attr-defined]


def make_runner_set(
    process_group: dist.ProcessGroup,
    implementations: Iterable[str],
    *,
    timed: bool,
    varlen: bool,
    phase_timing: bool = True,
) -> dict[str, DCPAttentionRunner]:
    """Construct ours and selected comparison runners with stable labels."""
    from dcp_test.baselines import (
        SGLangDCPAttentionRunner,
        TimedSGLangDCPAttentionRunner,
        TimedVLLMA2ADCPAttentionRunner,
        TimedVLLMDCPAttentionRunner,
        VLLMA2ADCPAttentionRunner,
        VLLMDCPAttentionRunner,
    )

    selected = set(implementations)
    ours_type = TimedDCPAttentionRunner if timed else DCPAttentionRunner
    vllm_type = TimedVLLMDCPAttentionRunner if timed else VLLMDCPAttentionRunner
    a2a_type = TimedVLLMA2ADCPAttentionRunner if timed else VLLMA2ADCPAttentionRunner
    sglang_type = (
        TimedSGLangDCPAttentionRunner if timed else SGLangDCPAttentionRunner
    )
    runners: dict[str, DCPAttentionRunner] = {}
    runner_kwargs = (
        {"benchmark_phase_timing": phase_timing} if timed else {}
    )
    if "ours" in selected:
        ours_no_overlap = ours_type(process_group, **runner_kwargs)
        ours_overlap = ours_type(process_group, **runner_kwargs)
        no_overlap_label = (
            ours_no_overlap.varlen_method_name
            if varlen
            else ours_no_overlap.method_name
        )
        overlap_label = (
            ours_overlap.varlen_overlap_method_name
            if varlen
            else ours_overlap.overlap_method_name
        )
        runners[no_overlap_label] = ours_no_overlap
        runners[overlap_label] = ours_overlap
    for name, runner_type in (
        ("vllm", vllm_type),
        ("vllm_a2a", a2a_type),
        ("sglang", sglang_type),
    ):
        if name == "vllm_a2a":
            enabled = "vllm" in selected or "vllm_a2a" in selected
        else:
            enabled = name in selected
        if not enabled:
            continue
        runner = runner_type(process_group, **runner_kwargs)
        label = runner.varlen_method_name if varlen else runner.method_name
        runners[label] = runner
    return runners


__all__ = [
    "BenchmarkPhaseRecorder",
    "BenchmarkTimingMixin",
    "CAPTURE_EAGER_WARMUP",
    "DCPGroup",
    "TimedDCPAttentionRunner",
    "all_rank_quantiles",
    "append_packed_chunk",
    "capture_cuda_graph_callable",
    "global_rank_max",
    "global_rank_max_dict",
    "initialize_distributed_sm90",
    "interleaved_local_length",
    "make_cu_seqlens",
    "make_dcp_group",
    "make_dcp_groups",
    "make_generator",
    "make_runner_set",
    "measure_timed_runner",
    "parse_int_list",
    "parse_lengths",
    "quantiles",
    "randn_bf16",
    "require_world_size",
    "shard_dense_interleaved",
    "shard_packed_interleaved",
    "synchronize_before_samples",
]
