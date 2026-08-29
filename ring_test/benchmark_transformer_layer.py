"""Dataset benchmark for one Megatron Transformer layer with eight CP methods."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch
import torch.distributed as dist


THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
DEFAULT_MEGATRON = DEMO_DIR / "third_party" / "Megatron-LM"
for _path in (THIS_DIR, DEMO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@dataclass(frozen=True)
class Case:
    case_index: int
    global_lengths: tuple[int, ...]
    ring_sizes: tuple[int, ...]
    ring_starts: tuple[int, ...]


@dataclass(frozen=True)
class Timing:
    forward_cuda_critical_rank_avg_ms: float
    backward_cuda_critical_rank_avg_ms: float
    total_cuda_critical_rank_avg_ms: float
    self_attn_forward_cuda_critical_rank_avg_ms: float
    others_forward_cuda_critical_rank_avg_ms: float
    self_attn_backward_cuda_critical_rank_avg_ms: float
    others_backward_cuda_critical_rank_avg_ms: float
    total_wall_max_avg_ms: float
    per_rank_total_wall_avg_ms: tuple[float, ...]
    cuda_critical_rank_counts: tuple[int, ...]


@dataclass(frozen=True)
class CriticalRankTiming:
    forward_ms: float
    backward_ms: float
    self_attn_forward_ms: float
    self_attn_backward_ms: float
    wall_max_ms: float
    rank: int


class SelfAttentionTimingProbe:
    """CUDA-event instrumentation around one Megatron SelfAttention module.

    Forward hooks delimit the complete SelfAttention call. Tensor-gradient
    hooks delimit its backward without adding a synchronization point or
    changing the autograd graph with a full-module backward hook.
    """

    def __init__(self, self_attention: torch.nn.Module) -> None:
        self._active = False
        self._backward_start_recorded = False
        self._backward_end_recorded = False
        self._forward_pre_handle = self_attention.register_forward_pre_hook(
            self._forward_pre_hook
        )
        self._forward_handle = self_attention.register_forward_hook(
            self._forward_hook
        )

    def begin(self) -> None:
        if self._active:
            raise RuntimeError("self-attention timing probe is already active")
        self._forward_start = torch.cuda.Event(enable_timing=True)
        self._forward_end = torch.cuda.Event(enable_timing=True)
        self._backward_start = torch.cuda.Event(enable_timing=True)
        self._backward_end = torch.cuda.Event(enable_timing=True)
        self._backward_start_recorded = False
        self._backward_end_recorded = False
        self._active = True

    def _forward_pre_hook(
        self, module: torch.nn.Module, inputs: tuple[Any, ...]
    ) -> None:
        del module
        if not self._active:
            return
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("SelfAttention timing requires a tensor input")
        hidden_states = inputs[0]
        if not hidden_states.requires_grad:
            raise RuntimeError("SelfAttention timing input must require gradients")
        self._forward_start.record()
        hidden_states.register_hook(self._input_gradient_hook)

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, inputs
        if not self._active:
            return
        self._forward_end.record()
        output_tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(output_tensor, torch.Tensor):
            raise RuntimeError("SelfAttention timing requires a tensor output")
        if not output_tensor.requires_grad:
            raise RuntimeError("SelfAttention timing output must require gradients")
        output_tensor.register_hook(self._output_gradient_hook)

    def _output_gradient_hook(self, gradient: torch.Tensor) -> torch.Tensor:
        if self._active:
            self._backward_start.record()
            self._backward_start_recorded = True
        return gradient

    def _input_gradient_hook(self, gradient: torch.Tensor) -> torch.Tensor:
        if self._active:
            self._backward_end.record()
            self._backward_end_recorded = True
        return gradient

    def finish(self) -> tuple[float, float]:
        if not self._active:
            raise RuntimeError("self-attention timing probe is not active")
        if not self._backward_start_recorded or not self._backward_end_recorded:
            raise RuntimeError(
                "SelfAttention backward timing hooks did not both execute"
            )
        forward_ms = self._forward_start.elapsed_time(self._forward_end)
        backward_ms = self._backward_start.elapsed_time(self._backward_end)
        self._active = False
        return forward_ms, backward_ms

    def close(self) -> None:
        self._active = False
        self._forward_pre_handle.remove()
        self._forward_handle.remove()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one Llama-3-8B-style Megatron Transformer layer forward+backward "
            "with CP=8 and TP=PP=DP=1"
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("arxiv", "github", "pile", "freelaw", "prolong"),
    )
    parser.add_argument("--target-tokens", type=_positive_int, default=128 * 1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-cases", type=_positive_int, default=20)
    parser.add_argument(
        "--world-size",
        type=int,
        choices=(4, 8),
        default=8,
        help="CP/rank count; CP=4 is a functional smoke mode and CP=8 is formal",
    )
    parser.add_argument("--methods", default="all")
    parser.add_argument("--warmup-iters", type=_nonnegative_int, default=10)
    parser.add_argument("--num-iters", type=_positive_int, default=40)
    parser.add_argument(
        "--sm-configs",
        help="Comma-separated COMP:COMM MegaRing points; default is (device SMs - 8):8",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--megatron-path", type=Path, default=DEFAULT_MEGATRON)
    parser.add_argument("--compute-balance-tolerance", type=float, default=0.05)
    parser.add_argument("--token-balance-tolerance", type=float, default=0.05)
    parser.add_argument("--beam-width", type=_positive_int, default=64)
    parser.add_argument("--finalist-count", type=_positive_int, default=8)
    parser.add_argument("--structure-threshold", type=float, default=0.5)
    parser.add_argument("--max-repair-iterations", type=_nonnegative_int, default=32)
    parser.add_argument("--megatron-max-seqlen-per-rank", type=_positive_int, default=8192)
    parser.add_argument("--zeppelin-threshold", type=_positive_int, default=4096)
    parser.add_argument("--magi-overlap-degree", type=_positive_int, default=2)
    parser.add_argument(
        "--allgather-heads-k-stride", type=_positive_int, default=4
    )
    parser.add_argument(
        "--dry-run-layout",
        action="store_true",
        help="Generate workloads and CPU layout metadata without distributed/GPU imports",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    from transformer_layer_cp import parse_methods

    parse_methods(args.methods)
    if not 0.0 <= args.compute_balance_tolerance <= 1.0:
        raise SystemExit("--compute-balance-tolerance must be in [0, 1]")
    if not 0.0 <= args.token_balance_tolerance <= 1.0:
        raise SystemExit("--token-balance-tolerance must be in [0, 1]")
    if not 0.0 <= args.structure_threshold <= 1.0:
        raise SystemExit("--structure-threshold must be in [0, 1]")
    if not 1 <= args.magi_overlap_degree <= 8:
        raise SystemExit("--magi-overlap-degree must be in [1, 8]")
    if 8 % args.allgather_heads_k_stride:
        raise SystemExit("--allgather-heads-k-stride must divide KVH=8")
    if not args.megatron_path.is_dir() and not args.dry_run_layout:
        raise SystemExit(f"Megatron path does not exist: {args.megatron_path}")


def _generate_cases(args: argparse.Namespace) -> list[Case]:
    import balancer

    workloads = balancer.make_workloads(
        dataset=args.dataset,
        target_tokens=args.target_tokens,
        seed=args.seed,
        num_cases=args.num_cases,
        world_size=args.world_size,
        mode="causal",
        compute_balance_tolerance=args.compute_balance_tolerance,
        token_balance_tolerance=args.token_balance_tolerance,
        beam_width=args.beam_width,
        finalist_count=args.finalist_count,
        structure_threshold=args.structure_threshold,
        max_repair_iterations=args.max_repair_iterations,
    )
    return [
        Case(
            case_index=index,
            global_lengths=tuple(workload.global_lengths),
            ring_sizes=tuple(workload.ring_sizes),
            ring_starts=tuple(workload.ring_starts),
        )
        for index, workload in enumerate(workloads)
    ]


def _broadcast_cases(args: argparse.Namespace, rank: int) -> list[Case]:
    payload: list[Any] = [_generate_cases(args) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    cases = payload[0]
    if not isinstance(cases, list) or len(cases) != args.num_cases:
        raise RuntimeError("rank 0 did not broadcast the requested dataset cases")
    return cases


def _dry_run(args: argparse.Namespace) -> None:
    from transformer_layer_cp import build_physical_layout, parse_methods

    methods = parse_methods(args.methods)
    cases = _generate_cases(args)
    for case in cases:
        payload: dict[str, Any] = {
            "dataset": args.dataset,
            "case_index": case.case_index,
            "original_tokens": sum(case.global_lengths),
            "global_lengths": case.global_lengths,
            "ring_sizes": case.ring_sizes,
            "ring_starts": case.ring_starts,
            "methods": {},
        }
        for method in methods:
            layout = build_physical_layout(
                method,
                case.global_lengths,
                case.ring_sizes,
                case.ring_starts,
                args.world_size,
                megatron_max_seqlen_per_rank=args.megatron_max_seqlen_per_rank,
                zeppelin_threshold=args.zeppelin_threshold,
            )
            payload["methods"][method] = {
                "execution_tokens": layout.execution_tokens,
                "padding_tokens": layout.padding_tokens,
                "rank_token_loads": layout.rank_token_loads,
                "note": layout.note,
            }
        print(json.dumps(payload, ensure_ascii=False), flush=True)


def _init_distributed(expected_world_size: int) -> tuple[int, int, torch.device]:
    required = ("LOCAL_RANK", "LOCAL_WORLD_SIZE", "RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise SystemExit(f"run with torchrun; missing environment variables {missing}")
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    world_size = int(os.environ["WORLD_SIZE"])
    if local_world_size != expected_world_size or world_size != expected_world_size:
        raise SystemExit(
            "single-layer CP benchmark requires exactly one node and the requested "
            f"{expected_world_size} local ranks: LOCAL_WORLD_SIZE={local_world_size}, "
            f"WORLD_SIZE={world_size}"
        )
    if int(os.environ.get("GROUP_RANK", "0")) != 0:
        raise SystemExit("MegaRing TKParallelTensor requires a single node")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < expected_world_size
    ):
        raise SystemExit(
            f"{expected_world_size} visible CUDA devices are required, "
            f"got {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    return dist.get_rank(), world_size, device


def _all_rank_preflight(local_error: str | None) -> None:
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
    if failures:
        _destroy_distributed_state()
        raise SystemExit("strict preflight failed: " + "; ".join(failures))


def _destroy_distributed_state() -> None:
    """Best-effort teardown for collective preflight and normal completion."""
    try:
        from megatron.core import parallel_state

        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
    except (ImportError, RuntimeError):
        pass
    if dist.is_initialized():
        dist.destroy_process_group()


def _collect_device_inventory(
    rank: int, local_rank: int, device: torch.device
) -> list[dict[str, Any]]:
    """Collect physical/MIG CUDA properties before rank-dependent validation."""
    properties = torch.cuda.get_device_properties(device)
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "cuda_ordinal": device.index,
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": int(properties.multi_processor_count),
        "total_memory_bytes": int(properties.total_memory),
        "uuid": str(getattr(properties, "uuid", "")) or None,
    }
    gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if any(item is None for item in gathered):
        raise RuntimeError("device inventory collection returned an empty rank entry")
    return [dict(item) for item in gathered if item is not None]


def _format_device_inventory(inventory: Sequence[dict[str, Any]]) -> str:
    return "; ".join(
        (
            f"rank {item['rank']}/local {item['local_rank']}: "
            f"cuda:{item['cuda_ordinal']} {item['name']}, "
            f"cc={item['capability'][0]}.{item['capability'][1]}, "
            f"SMs={item['sm_count']}, "
            f"memory={item['total_memory_bytes'] / (1024**3):.1f} GiB, "
            f"uuid={item['uuid'] or '-'}"
        )
        for item in inventory
    )


def _resolve_sm_configs_collectively(
    spec: str | None, inventory: Sequence[dict[str, Any]]
) -> list[Any]:
    """Resolve one SM split against the smallest visible rank without divergence."""
    from transformer_layer_cp import parse_sm_configs

    minimum_sm_count = min(int(item["sm_count"]) for item in inventory)
    try:
        return parse_sm_configs(spec, minimum_sm_count)
    except (TypeError, ValueError) as exc:
        _destroy_distributed_state()
        raise SystemExit(
            f"collective SM-config preflight failed: {exc}. "
            f"All-rank CUDA inventory: {_format_device_inventory(inventory)}"
        ) from exc


def _module_version(distributions: Sequence[str]) -> str | None:
    for distribution in distributions:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _preflight(
    args: argparse.Namespace,
    methods: Sequence[str],
    device: torch.device,
    device_inventory: Sequence[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    local_error: str | None = None
    try:
        capability = torch.cuda.get_device_capability(device)
        if capability != (9, 0):
            raise RuntimeError(f"Hopper SM90 is required, got capability {capability}")
        import transformer_engine.pytorch  # noqa: F401

        if str(args.megatron_path) not in sys.path:
            sys.path.insert(0, str(args.megatron_path))
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_submodules,
        )

        get_gpt_layer_with_transformer_engine_submodules()
        import min_fa3_op  # noqa: F401

        if "magi_attention" in methods:
            from baseline.magi_attention import probe_magi_attention

            available, reason = probe_magi_attention()
            if not available:
                raise RuntimeError(
                    "MagiAttention is mandatory when requested: "
                    + (reason or "unknown import failure")
                )
    except Exception as exc:  # preflight reports rank-specific optional imports
        local_error = f"{type(exc).__name__}: {exc}"
    _all_rank_preflight(local_error)

    from ring_test.allgather_attention import select_fa3_backend

    block_backend = select_fa3_backend(dist.group.WORLD, require_backward=True)
    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformer_engine": _module_version(("transformer-engine",)),
        "magi_attention": _module_version(("magi_attention", "magi-attention")),
        "fa3_block_backend": block_backend,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "device_sm_count": torch.cuda.get_device_properties(device).multi_processor_count,
        "device_inventory": list(device_inventory),
    }
    return block_backend, info


def _megatron_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _initialize_megatron(args: argparse.Namespace) -> None:
    if str(args.megatron_path) not in sys.path:
        sys.path.insert(0, str(args.megatron_path))
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=args.world_size,
        expert_model_parallel_size=1,
        order="tp-cp-ep-dp-pp",
        create_gloo_process_groups=False,
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)


def _zero_grads(layer: torch.nn.Module, hidden_states: torch.Tensor) -> None:
    layer.zero_grad(set_to_none=True)
    hidden_states.grad = None


def _one_iteration(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    dout: torch.Tensor,
    packed_seq_params: Any,
    self_attn_timing_probe: SelfAttentionTimingProbe,
) -> tuple[float, float, float, float, float]:
    _zero_grads(layer, hidden_states)
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()

    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    self_attn_timing_probe.begin()
    wall_start = time.perf_counter()
    forward_start.record()
    output, _context = layer(
        hidden_states,
        attention_mask=None,
        packed_seq_params=packed_seq_params,
    )
    forward_end.record()
    backward_start.record()
    output.backward(dout)
    backward_end.record()
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - wall_start) * 1000.0
    self_attn_forward_ms, self_attn_backward_ms = self_attn_timing_probe.finish()
    return (
        forward_start.elapsed_time(forward_end),
        backward_start.elapsed_time(backward_end),
        total_ms,
        self_attn_forward_ms,
        self_attn_backward_ms,
    )


def _validate_complete_gradients(
    layer: torch.nn.Module, hidden_states: torch.Tensor
) -> None:
    local_error: str | None = None
    if hidden_states.grad is None:
        local_error = "input hidden-state gradient was not produced"
    else:
        missing = [
            name
            for name, parameter in layer.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = " ..." if len(missing) > 8 else ""
            local_error = (
                f"{len(missing)} trainable layer parameters have no local gradient: "
                f"{preview}{suffix}"
            )
    _all_rank_preflight(local_error)


def select_cuda_critical_rank_timing(
    rank_samples: torch.Tensor,
) -> CriticalRankTiming:
    """Select all CUDA components from the rank with maximum full FWD+BWD.

    ``rank_samples`` has one row per rank and columns ``[forward, backward,
    total_wall, self_attn_forward, self_attn_backward]``. Wall time is
    deliberately not used to choose the CUDA critical rank; it remains an
    independent host/synchronization diagnostic.
    """
    if rank_samples.ndim != 2 or rank_samples.size(1) != 5:
        raise ValueError(
            "rank timing samples must have shape [world_size, 5], got "
            f"{tuple(rank_samples.shape)}"
        )
    cuda_totals = rank_samples[:, 0] + rank_samples[:, 1]
    critical_rank = int(torch.argmax(cuda_totals).item())
    return CriticalRankTiming(
        forward_ms=float(rank_samples[critical_rank, 0].item()),
        backward_ms=float(rank_samples[critical_rank, 1].item()),
        self_attn_forward_ms=float(rank_samples[critical_rank, 3].item()),
        self_attn_backward_ms=float(rank_samples[critical_rank, 4].item()),
        wall_max_ms=float(rank_samples[:, 2].max().item()),
        rank=critical_rank,
    )


def _measure(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    dout: torch.Tensor,
    packed_seq_params: Any,
    warmup_iters: int,
    num_iters: int,
) -> Timing:
    self_attn_timing_probe = SelfAttentionTimingProbe(layer.self_attention)
    try:
        gradients_validated = False
        for _ in range(warmup_iters):
            _one_iteration(
                layer,
                hidden_states,
                dout,
                packed_seq_params,
                self_attn_timing_probe,
            )
            if not gradients_validated:
                _validate_complete_gradients(layer, hidden_states)
                gradients_validated = True

        critical_cuda_samples: list[CriticalRankTiming] = []
        critical_rank_counts = [0] * dist.get_world_size()
        local_total_samples: list[float] = []
        for _ in range(num_iters):
            (
                forward_ms,
                backward_ms,
                total_ms,
                self_attn_forward_ms,
                self_attn_backward_ms,
            ) = _one_iteration(
                layer,
                hidden_states,
                dout,
                packed_seq_params,
                self_attn_timing_probe,
            )
            local_total_samples.append(total_ms)
            if not gradients_validated:
                _validate_complete_gradients(layer, hidden_states)
                gradients_validated = True
            local_values = torch.tensor(
                [
                    forward_ms,
                    backward_ms,
                    total_ms,
                    self_attn_forward_ms,
                    self_attn_backward_ms,
                ],
                dtype=torch.float64,
                device=hidden_states.device,
            )
            gathered_values = torch.empty(
                dist.get_world_size() * local_values.numel(),
                dtype=local_values.dtype,
                device=local_values.device,
            )
            dist.all_gather_into_tensor(gathered_values, local_values)
            rank_samples = gathered_values.view(dist.get_world_size(), 5)
            critical_sample = select_cuda_critical_rank_timing(rank_samples)
            critical_cuda_samples.append(critical_sample)
            critical_rank_counts[critical_sample.rank] += 1
    finally:
        self_attn_timing_probe.close()

    local_avg = mean(local_total_samples)
    per_rank: list[float | None] = [None] * dist.get_world_size()
    dist.all_gather_object(per_rank, local_avg)
    measured_forward_avg_ms = mean(
        sample.forward_ms for sample in critical_cuda_samples
    )
    measured_backward_avg_ms = mean(
        sample.backward_ms for sample in critical_cuda_samples
    )
    self_attn_forward_avg_ms = mean(
        sample.self_attn_forward_ms for sample in critical_cuda_samples
    )
    self_attn_backward_avg_ms = mean(
        sample.self_attn_backward_ms for sample in critical_cuda_samples
    )
    others_forward_avg_ms = measured_forward_avg_ms - self_attn_forward_avg_ms
    others_backward_avg_ms = measured_backward_avg_ms - self_attn_backward_avg_ms
    # Build the stored parent phases from their stored components. This keeps
    # the JSON arithmetic identities exact while changing the measured parent
    # averages by at most the final floating-point rounding bit.
    forward_avg_ms = self_attn_forward_avg_ms + others_forward_avg_ms
    backward_avg_ms = self_attn_backward_avg_ms + others_backward_avg_ms
    return Timing(
        forward_cuda_critical_rank_avg_ms=forward_avg_ms,
        backward_cuda_critical_rank_avg_ms=backward_avg_ms,
        total_cuda_critical_rank_avg_ms=forward_avg_ms + backward_avg_ms,
        self_attn_forward_cuda_critical_rank_avg_ms=self_attn_forward_avg_ms,
        others_forward_cuda_critical_rank_avg_ms=others_forward_avg_ms,
        self_attn_backward_cuda_critical_rank_avg_ms=self_attn_backward_avg_ms,
        others_backward_cuda_critical_rank_avg_ms=others_backward_avg_ms,
        total_wall_max_avg_ms=mean(
            sample.wall_max_ms for sample in critical_cuda_samples
        ),
        per_rank_total_wall_avg_ms=tuple(float(value) for value in per_rank),
        cuda_critical_rank_counts=tuple(critical_rank_counts),
    )


def _make_inputs(
    local_tokens: int, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    hidden = torch.randn(
        (local_tokens, 1, 4096),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    dout = torch.randn(
        hidden.shape,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    return hidden, dout


def _record(
    *,
    args: argparse.Namespace,
    case: Case,
    prepared: Any,
    timing: Timing,
    sm_config: Any,
    dependency_info: dict[str, Any],
    megatron_commit: str | None,
) -> dict[str, Any]:
    return {
        "schema": "min_fa3.megatron_transformer_layer_cp.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "case_index": case.case_index,
        "num_cases": args.num_cases,
        "seed": args.seed,
        "target_tokens": args.target_tokens,
        "method": prepared.method,
        "causal": True,
        "sm_config": None if sm_config is None else asdict(sm_config),
        "model": {
            "profile": "llama3_8b_single_layer",
            "dtype": "bfloat16",
            "hidden_size": 4096,
            "q_heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "ffn_hidden_size": 14336,
            "activation": "swiglu",
            "normalization": "rmsnorm",
            "bias": False,
            "dropout": 0.0,
            "rope": False,
        },
        "parallelism": {"cp": args.world_size, "tp": 1, "pp": 1, "dp": 1},
        "formal_cp8_result": args.world_size == 8,
        "iterations": {"warmup": args.warmup_iters, "measure": args.num_iters},
        "tokens": {
            "original_global": prepared.layout.original_tokens,
            "execution_global": prepared.layout.execution_tokens,
            "padding_global": prepared.layout.padding_tokens,
            "per_rank_local": prepared.layout.rank_token_loads,
        },
        "workload": {
            "global_lengths": case.global_lengths,
            "ring_sizes": case.ring_sizes,
            "ring_starts": case.ring_starts,
            "execution_lengths": prepared.layout.execution_lengths,
        },
        "timing": asdict(timing),
        "method_metadata": {
            "note": prepared.layout.note,
            "adapter_note": prepared.adapter.note,
            **prepared.metadata,
        },
        "dependencies": dependency_info,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "NCCL_CGA_CLUSTER_SIZE",
                "TORCH_NCCL_HIGH_PRIORITY",
                "MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE",
                "OMP_NUM_THREADS",
            )
        },
        "planner": {
            "compute_balance_tolerance": args.compute_balance_tolerance,
            "token_balance_tolerance": args.token_balance_tolerance,
            "beam_width": args.beam_width,
            "finalist_count": args.finalist_count,
            "structure_threshold": args.structure_threshold,
            "max_repair_iterations": args.max_repair_iterations,
            "megatron_max_seqlen_per_rank": args.megatron_max_seqlen_per_rank,
            "zeppelin_threshold": args.zeppelin_threshold,
            "magi_overlap_degree": args.magi_overlap_degree,
            "allgather_heads_k_stride": args.allgather_heads_k_stride,
        },
        "megatron_commit": megatron_commit,
        "timing_boundary": {
            "includes": (
                "complete Transformer layer forward/backward; MegaRing K/V population, "
                "synchronization, and backward accumulator resets"
            ),
            "excludes": (
                "sampling, planning, physical-layout allocation, process-group creation, "
                "workspace construction, gradient clearing, and pre-iteration synchronization"
            ),
            "primary": (
                "for each iteration, select one rank by maximum CUDA forward+backward; "
                "average that same rank's full and self-attention components"
            ),
            "self_attention": (
                "complete Megatron SelfAttention module: QKV projection, CP core "
                "attention, and output projection"
            ),
            "others": (
                "algebraic full-phase remainder after subtracting self-attention: "
                "RMSNorm, residual/BDA, pre-MLP RMSNorm, SwiGLU MLP, and MLP residual/BDA"
            ),
            "diagnostic": (
                "total_wall_max_avg_ms independently averages each iteration's "
                "maximum rank wall time and is not additively decomposed"
            ),
        },
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()


def _print_result(record: dict[str, Any]) -> None:
    timing = record["timing"]
    tokens = record["tokens"]
    sm = record["sm_config"]
    sm_label = "-" if sm is None else f"{sm['num_comp_sm']}:{sm['num_comm_sm']}"
    forward_ms = float(timing["forward_cuda_critical_rank_avg_ms"])
    backward_ms = float(timing["backward_cuda_critical_rank_avg_ms"])
    self_attn_forward_ms = float(
        timing["self_attn_forward_cuda_critical_rank_avg_ms"]
    )
    self_attn_backward_ms = float(
        timing["self_attn_backward_cuda_critical_rank_avg_ms"]
    )
    # Construct displayed remainders and total from the displayed parent and
    # self-attention components so all RESULT identities are exact at 3 decimals.
    forward_display_ms = round(forward_ms, 3)
    backward_display_ms = round(backward_ms, 3)
    self_attn_forward_display_ms = round(self_attn_forward_ms, 3)
    self_attn_backward_display_ms = round(self_attn_backward_ms, 3)
    others_forward_display_ms = (
        forward_display_ms - self_attn_forward_display_ms
    )
    others_backward_display_ms = (
        backward_display_ms - self_attn_backward_display_ms
    )
    total_display_ms = forward_display_ms + backward_display_ms
    critical_rank_counts = "/".join(
        str(count) for count in timing["cuda_critical_rank_counts"]
    )
    print(
        f"RESULT dataset={record['dataset']} case={record['case_index'] + 1}/"
        f"{record['num_cases']} method={record['method']} SM={sm_label} "
        f"tokens={tokens['original_global']}/{tokens['execution_global']} "
        f"forward_cuda_critical_rank_avg_ms={forward_display_ms:.3f} "
        f"backward_cuda_critical_rank_avg_ms={backward_display_ms:.3f} "
        f"total_cuda_critical_rank_avg_ms={total_display_ms:.3f} "
        f"self_attn_forward_cuda_critical_rank_avg_ms="
        f"{self_attn_forward_display_ms:.3f} "
        f"others_forward_cuda_critical_rank_avg_ms="
        f"{others_forward_display_ms:.3f} "
        f"self_attn_backward_cuda_critical_rank_avg_ms="
        f"{self_attn_backward_display_ms:.3f} "
        f"others_backward_cuda_critical_rank_avg_ms="
        f"{others_backward_display_ms:.3f} "
        f"cuda_critical_rank_counts={critical_rank_counts} "
        f"wall_max_avg_ms={timing['total_wall_max_avg_ms']:.3f}",
        flush=True,
    )


def _run(args: argparse.Namespace) -> None:
    from transformer_layer_cp import (
        MEGA_RING_METHODS,
        build_megatron_layer,
        build_physical_layout,
        make_packed_seq_params,
        parse_methods,
        prepare_method,
    )

    methods = parse_methods(args.methods)
    rank, world_size, device = _init_distributed(args.world_size)
    local_rank = int(os.environ["LOCAL_RANK"])
    device_inventory = _collect_device_inventory(rank, local_rank, device)
    sm_configs = _resolve_sm_configs_collectively(args.sm_configs, device_inventory)
    block_backend, dependency_info = _preflight(
        args, methods, device, device_inventory
    )

    _initialize_megatron(args)
    from baseline.megatron_hybrid_cp import create_hybrid_cp_process_groups
    from megatron.core import parallel_state

    hybrid_groups = create_hybrid_cp_process_groups(dist.group.WORLD)
    cases = _broadcast_cases(args, rank)

    # Fail the complete method/case matrix before writing any timing records.
    # Magi's exact dispatched token counts remain a runtime property, but its
    # mandatory extensions were already checked collectively in _preflight.
    layout_error: str | None = None
    try:
        for case in cases:
            for method in methods:
                build_physical_layout(
                    method,
                    case.global_lengths,
                    case.ring_sizes,
                    case.ring_starts,
                    world_size,
                    megatron_max_seqlen_per_rank=args.megatron_max_seqlen_per_rank,
                    zeppelin_threshold=args.zeppelin_threshold,
                )
    except Exception as exc:
        layout_error = f"{type(exc).__name__}: {exc}"
    _all_rank_preflight(layout_error)

    layer = build_megatron_layer(args.megatron_path, device, world_size)
    dispatcher = layer.self_attention.core_attention
    commit = _megatron_commit(args.megatron_path) if rank == 0 else None
    commit_payload = [commit]
    dist.broadcast_object_list(commit_payload, src=0)
    commit = commit_payload[0]

    if rank == 0:
        sm_counts = {int(item["sm_count"]) for item in device_inventory}
        memory_sizes = {int(item["total_memory_bytes"]) for item in device_inventory}
        if len(sm_counts) > 1 or len(memory_sizes) > 1:
            print(
                "WARNING: heterogeneous full-GPU/MIG CUDA inventory; this run is "
                "suitable for functional smoke testing only. "
                + _format_device_inventory(device_inventory),
                flush=True,
            )
        print(
            f"Single-layer benchmark: CP={world_size} TP=PP=DP=1, "
            f"mode={'formal CP=8' if world_size == 8 else 'functional CP=4 smoke'}, "
            "BF16 Llama-3-8B profile, "
            f"dataset={args.dataset}, cases={len(cases)}, methods={methods}, "
            f"MegaRing SM={[config.label for config in sm_configs]}, "
            f"FA3 block backend={block_backend}",
            flush=True,
        )

    try:
        for case in cases:
            for method_index, method in enumerate(methods):
                method_configs = sm_configs if method in MEGA_RING_METHODS else [None]
                for config_index, sm_config in enumerate(method_configs):
                    prepared = prepare_method(
                        method=method,
                        global_lengths=case.global_lengths,
                        ring_sizes=case.ring_sizes,
                        ring_starts=case.ring_starts,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        block_backend=block_backend,
                        hybrid_groups=hybrid_groups,
                        sm_config=sm_config,
                        allgather_heads_k_stride=args.allgather_heads_k_stride,
                        megatron_max_seqlen_per_rank=args.megatron_max_seqlen_per_rank,
                        zeppelin_threshold=args.zeppelin_threshold,
                        magi_overlap_degree=args.magi_overlap_degree,
                        seed=args.seed + case.case_index,
                    )
                    local_tokens = sum(prepared.local_packed_lengths)
                    hidden, dout = _make_inputs(
                        local_tokens,
                        device,
                        args.seed
                        + 1009 * rank
                        + 100_003 * case.case_index
                        + 1_000_003 * method_index
                        + config_index,
                    )
                    packed = make_packed_seq_params(
                        prepared.local_packed_lengths, device
                    )
                    dispatcher.set_adapter(
                        prepared.adapter,
                        native_autograd=prepared.native_autograd,
                    )
                    timing = _measure(
                        layer,
                        hidden,
                        dout,
                        packed,
                        args.warmup_iters,
                        args.num_iters,
                    )
                    record = _record(
                        args=args,
                        case=case,
                        prepared=prepared,
                        timing=timing,
                        sm_config=sm_config,
                        dependency_info=dependency_info,
                        megatron_commit=commit,
                    )
                    if rank == 0:
                        _append_jsonl(args.output_jsonl, record)
                        _print_result(record)
                    dispatcher.set_adapter(None)
                    _zero_grads(layer, hidden)
                    del packed, dout, hidden, prepared
                    torch.cuda.empty_cache()
                    dist.barrier()
    finally:
        dispatcher.set_adapter(None)
        del layer
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()
        _destroy_distributed_state()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    if args.dry_run_layout:
        _dry_run(args)
        return
    _run(args)


if __name__ == "__main__":
    main()
