"""Shared console formatting for the dense and varlen DCP benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DCPBenchmarkRow:
    method: str
    p50_ms: float
    p90_ms: float
    aggregate_tflops: float
    avg_gpu_tflops: float
    kv_bandwidth_gbps_per_gpu: float
    check: str
    note: str
    rank_p50_ms: Sequence[float] | None = None


def effective_kv_bandwidth_gbps_per_gpu(
    kv_bytes_by_rank: Sequence[float], p50_ms: float
) -> float:
    """Return average logical per-rank K+V bytes divided by critical latency."""
    if not kv_bytes_by_rank:
        raise ValueError("kv_bytes_by_rank must not be empty")
    average_bytes = sum(kv_bytes_by_rank) / len(kv_bytes_by_rank)
    return average_bytes / max(p50_ms, 1.0e-12) / 1.0e6


def print_benchmark_results(title: str, rows: Sequence[DCPBenchmarkRow]) -> None:
    """Print one result table using the ring benchmark's console layout."""
    print(f"\n{title}")
    formatted: list[tuple[str, str, str, str, str, str, str]] = []
    for row in rows:
        if row.rank_p50_ms is None:
            rank_times = ""
        else:
            rank_times = ", ".join(
                f"t{rank}={time_ms:.3f}"
                for rank, time_ms in enumerate(row.rank_p50_ms)
            )
            rank_times += " | "
        time_s = (
            f"{rank_times}p50(max_across_ranks)={row.p50_ms:.3f}, "
            f"p90(max_across_ranks)={row.p90_ms:.3f}"
        )
        formatted.append(
            (
                row.method,
                time_s,
                f"{row.aggregate_tflops:.1f}",
                f"{row.avg_gpu_tflops:.1f}",
                f"{row.kv_bandwidth_gbps_per_gpu:.1f}",
                row.check,
                row.note,
            )
        )

    method_width = max((24, *(len(row[0]) for row in formatted)))
    time_width = max((64, *(len(row[1]) for row in formatted)))
    print(
        f"{'Method':<{method_width}} {'Time ms':<{time_width}} "
        f"{'Agg TFLOPS':>12} {'Avg/GPU':>10} {'KV GB/s/GPU':>12} "
        f"{'Check':>10}  Note"
    )
    for (
        method,
        time_s,
        aggregate_s,
        per_gpu_s,
        kv_bandwidth_s,
        check,
        note,
    ) in formatted:
        print(
            f"{method:<{method_width}} {time_s:<{time_width}} "
            f"{aggregate_s:>12} {per_gpu_s:>10} {kv_bandwidth_s:>12} "
            f"{check:>10}  {note}"
        )
