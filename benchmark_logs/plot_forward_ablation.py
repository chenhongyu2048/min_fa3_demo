#!/usr/bin/env python3
"""Plot 8-GPU Mega Ring forward ablation summaries at 64K, 128K, and 256K."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "experiment_queue" / "20260726-002324"
DEFAULT_TOKEN_COUNTS = (65536, 131072, 262144)
DEFAULT_OUTPUT = DEFAULT_RUN_DIR / "forward_ablation_arxiv.png"
GPU_COUNT = 8

PROFILE_LABELS = {
    "step_external_reduce": "Step +\nexternal reduce",
    "step_fused_reduce": "Step +\nfused reduce",
    "linear_queue_no_recycle": "Linear queue\nno recycle",
    "linear_queue_recycle": "Linear queue\nrecycle",
    "dynamic_segment_recycle": "Dynamic segment\nrecycle",
    "hybrid_br_pbs": "Hybrid\nBR-PBS",
}
COLORS = ("#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B", "#B279A2")


@dataclass(frozen=True)
class AblationRecord:
    level: int
    profile: str
    sm_config: str
    mean_ms: float
    mean_gpu_tflops: float


def parse_records(path: Path) -> list[AblationRecord]:
    """Select the best-SM L1-L6 rows from a cross-case ablation summary."""

    if not path.is_file():
        raise FileNotFoundError(f"ablation log does not exist: {path}")

    records_by_level: dict[int, AblationRecord] = {}
    in_summary = False
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        if line == "Cross-case forward ablation summary":
            in_summary = True
            continue
        if not in_summary:
            continue

        fields = line.split()
        if not fields or re.fullmatch(r"L[1-6]", fields[0]) is None:
            continue
        if len(fields) != 11:
            raise ValueError(f"malformed summary row at {path}:{line_number}: {line}")
        level = int(fields[0][1:])
        if re.fullmatch(r"\d+:\d+", fields[2]) is None or re.fullmatch(
            r"\d+/\d+", fields[3]
        ) is None:
            raise ValueError(f"malformed summary metadata at {path}:{line_number}: {line}")
        try:
            mean_ms = float(fields[5])
            mean_gpu_tflops = float(fields[8]) / GPU_COUNT
        except ValueError as exc:
            raise ValueError(f"invalid summary row at {path}:{line_number}: {line}") from exc

        record = AblationRecord(
            level=level,
            profile=fields[1],
            sm_config=fields[2],
            mean_ms=mean_ms,
            mean_gpu_tflops=mean_gpu_tflops,
        )
        current = records_by_level.get(level)
        if current is not None and current.profile != record.profile:
            raise ValueError(
                f"inconsistent profile for L{level} at {path}:{line_number}: "
                f"{current.profile} versus {record.profile}"
            )
        if current is None or record.mean_gpu_tflops > current.mean_gpu_tflops:
            records_by_level[level] = record

    expected_levels = set(range(1, 7))
    found_levels = set(records_by_level)
    if found_levels != expected_levels:
        missing = ", ".join(f"L{level}" for level in sorted(expected_levels - found_levels))
        extra = ", ".join(f"L{level}" for level in sorted(found_levels - expected_levels))
        details = ", ".join(part for part in (f"missing {missing}" if missing else "", f"unexpected {extra}" if extra else "") if part)
        raise ValueError(f"expected exactly L1-L6 performance rows in {path}; {details}")
    return [records_by_level[level] for level in sorted(records_by_level)]


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.85)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.margins(x=0.06, y=0.16)


def make_figure(
    records_by_token_count: Sequence[tuple[int, Sequence[AblationRecord]]],
) -> plt.Figure:
    figure, axes = plt.subplots(
        len(records_by_token_count),
        2,
        figsize=(14.5, 5.0 * len(records_by_token_count) + 1.8),
        squeeze=False,
    )

    for row, (token_count, records) in enumerate(records_by_token_count):
        levels = [f"L{record.level}" for record in records]
        profiles = [
            PROFILE_LABELS.get(record.profile, record.profile.replace("_", "\n"))
            for record in records
        ]
        mean_ms = [record.mean_ms for record in records]
        mean_gpu_tflops = [record.mean_gpu_tflops for record in records]
        positions = list(range(len(records)))
        time_axis, tflops_axis = axes[row]

        time_axis.plot(
            positions,
            mean_ms,
            color="#1F77B4",
            marker="o",
            markersize=7.5,
            linewidth=2.5,
            markerfacecolor="white",
            markeredgewidth=2.0,
        )
        for position, value in zip(positions, mean_ms):
            time_axis.annotate(
                f"{value:.3f}",
                (position, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#1F4E79",
            )
        time_axis.set_title(
            f"{token_count // 1024}K Mean Latency",
            loc="left",
            fontsize=13,
            fontweight="bold",
        )
        time_axis.set_ylabel("Mean time (ms)")

        bars = tflops_axis.bar(
            positions,
            mean_gpu_tflops,
            color=COLORS,
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value in zip(bars, mean_gpu_tflops):
            tflops_axis.annotate(
                f"{value:,.0f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#333333",
            )
        tflops_axis.set_title(
            f"{token_count // 1024}K Mean Throughput per GPU",
            loc="left",
            fontsize=13,
            fontweight="bold",
        )
        tflops_axis.set_ylabel("Mean TFLOPS per GPU")

        tick_labels = [
            f"{level}\n{profile}\nSM {record.sm_config}"
            for level, profile, record in zip(levels, profiles, records)
        ]
        for axis in (time_axis, tflops_axis):
            _style_axis(axis)
            axis.set_xticks(positions)
            axis.set_xticklabels(tick_labels, fontsize=8.4)
            axis.set_xlabel("Ablation level")

    figure.suptitle(
        f"{GPU_COUNT}-GPU ArXiv Mega Ring Forward Ablation",
        y=0.995,
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Causal BF16, QH=16, KVH=8, D=128. Each level selects its highest mean per-GPU throughput SM configuration.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=(
            "directory containing forward_ablation_arxiv_<tokens>_20cases.console.log "
            f"(default: {DEFAULT_RUN_DIR})"
        ),
    )
    parser.add_argument(
        "--token-count",
        type=int,
        action="append",
        default=None,
        help="token count to plot; repeat to select several (default: 65536, 131072, 262144)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    token_counts = tuple(args.token_count or DEFAULT_TOKEN_COUNTS)
    if not token_counts or any(token_count <= 0 for token_count in token_counts):
        raise ValueError("--token-count values must be positive")
    if len(set(token_counts)) != len(token_counts):
        raise ValueError("--token-count values must be unique")

    records_by_token_count = []
    for token_count in token_counts:
        path = args.run_dir / f"forward_ablation_arxiv_{token_count}_20cases.console.log"
        records_by_token_count.append((token_count, parse_records(path)))
    figure = make_figure(records_by_token_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
