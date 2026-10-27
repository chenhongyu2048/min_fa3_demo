"""Measurement boundaries and CPU-only sample aggregation."""

import json
from pathlib import Path
import subprocess


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rank_samples):
    if not rank_samples or not rank_samples[0]:
        raise ValueError("samples must be nonempty")
    if any(len(row) != len(rank_samples[0]) for row in rank_samples):
        raise ValueError("all ranks must provide the same number of samples")
    maxima = [max(sample) for sample in zip(*rank_samples)]
    return {"per_rank_ms": rank_samples, "rank_max_ms": maxima,
            "p50_ms": quantile(maxima, 0.5), "p90_ms": quantile(maxima, 0.9),
            "min_ms": min(maxima), "max_ms": max(maxima)}


def synchronize_before_sample(device):
    import torch
    import torch.distributed as dist

    torch.cuda.synchronize(device)
    dist.barrier(device_ids=[device.index])
    torch.cuda.synchronize(device)


def measure(call, device, warmup, iters, observe=None):
    """One replay/sample; all global barriers and observations are untimed."""
    import torch
    import torch.distributed as dist

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Materialize CUDA events before sampling, including when warmup == 0.
    start.record()
    end.record()
    for _ in range(warmup):
        call()
    samples, observations = [], []
    for _ in range(iters):
        synchronize_before_sample(device)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        if observe is not None:
            observations.append(observe())
    per_rank = [None] * dist.get_world_size()
    dist.all_gather_object(per_rank, samples)
    return summarize(per_rank), observations


def check_all_ranks(check):
    """Propagate correctness failures before ranks enter the next collective."""
    import torch.distributed as dist

    error = None
    try:
        check()
    except (AssertionError, ValueError, RuntimeError) as exc:
        error = str(exc)
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(item is not None for item in errors):
        raise RuntimeError(f"distributed validation failed: {errors}")


def environment(device):
    import os
    import torch
    import torch.distributed as dist
    from .config import ROOT

    props = torch.cuda.get_device_properties(device)
    return {"commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "gpu": props.name, "sm_count": props.multi_processor_count,
            "world_size": dist.get_world_size(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "aggregation": "quantiles of per-sample rank maximum",
            "pre_sample_sync": "CUDA synchronize -> WORLD barrier -> CUDA synchronize; untimed"}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
