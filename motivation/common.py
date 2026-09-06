"""Shared helpers for motivation experiments."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.distributed as dist


def init_distributed_sm90(purpose: str) -> tuple[int, int, torch.device]:
    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit(f"{purpose} must be launched with torchrun")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise SystemExit(f"{purpose} requires SM90 Hopper")
    if dist.get_world_size() != world_size:
        raise SystemExit("LOCAL_WORLD_SIZE and torch.distributed world size disagree")
    return rank, world_size, device


def require_homogeneous_devices(purpose: str, world_size: int, device: torch.device) -> int:
    """Validate the full-device assumption required by TK VMM arenas."""
    visible = torch.cuda.device_count()
    local_props = torch.cuda.get_device_properties(device)
    local = {
        "ordinal": int(device.index),
        "name": local_props.name,
        "capability": tuple(torch.cuda.get_device_capability(device)),
        "sm_count": int(local_props.multi_processor_count),
        "memory_bytes": int(local_props.total_memory),
    }
    entries: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(entries, {"visible": visible, **local})
    inventories = [entry for entry in entries if entry is not None]
    reference = inventories[0] if inventories else None
    mismatches = [
        entry for entry in inventories
        if reference is not None and (
            entry["name"] != reference["name"]
            or entry["capability"] != reference["capability"]
            or entry["sm_count"] != reference["sm_count"]
            or entry["memory_bytes"] != reference["memory_bytes"]
        )
    ]
    if visible != world_size or len(mismatches) != 0:
        summary = "; ".join(
            f"rank {idx}: visible={entry['visible']} ordinal={entry['ordinal']} "
            f"name={entry['name']} cc={entry['capability']} SM={entry['sm_count']} "
            f"memory={entry['memory_bytes'] / (1024**3):.1f}GiB"
            for idx, entry in enumerate(inventories)
        )
        raise SystemExit(
            f"{purpose} requires {world_size} visible, homogeneous full GPUs for "
            f"ThunderKittens VMM; inventory: {summary}. Disable MIG and choose an "
            "unoccupied homogeneous allocation before retrying."
        )
    return int(local_props.multi_processor_count)


def cuda_barrier() -> None:
    torch.cuda.synchronize()
    try:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    except TypeError:
        dist.barrier()


def make_cu_seqlens(lengths: Iterable[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = [int(length) for length in lengths]
    if not values or any(length <= 0 for length in values):
        raise ValueError("sequence lengths must be positive")
    offsets = [0]
    for length in values:
        offsets.append(offsets[-1] + length)
    host = torch.tensor(offsets, dtype=torch.int32)
    return host.to(device=device), host


def randn_bf16(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32).mul_(0.25).to(torch.bfloat16).contiguous()


def timed_call(fn: Callable[[], Any], warmup: int, iters: int, device: torch.device) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    cuda_barrier()
    local: list[float] = []
    rank_max: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        value = float(start.elapsed_time(end))
        tensor = torch.tensor(value, device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        local.append(value)
        rank_max.append(float(tensor.item()))
    cuda_barrier()
    local_average = sum(local) / len(local)
    gathered = [torch.empty((), device=device, dtype=torch.float64) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, torch.tensor(local_average, device=device, dtype=torch.float64))
    return {
        "local_ms": local,
        "rank_max_ms": rank_max,
        "p50_rank_max_ms": _quantile(rank_max, 0.50),
        "p90_rank_max_ms": _quantile(rank_max, 0.90),
        "rank_average_ms": [float(value.item()) for value in gathered],
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * probability))))
    return ordered[index]


def repository_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": props.name,
        "sm_count": int(props.multi_processor_count),
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "repository_commit": repository_commit(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
