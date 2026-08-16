#!/usr/bin/env python3
"""Plot decode-only (no chunks) and mixed-prefill ablations in a 2x9 grid."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "wave_ablation_summary.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "wave_ablation_2x9.png"

ARRIVALS = ("1", "2", "4")
DCP_SIZES = (2, 4, 8)
BATCH_TYPES = ("decode_only", "mixed_prefill")
BATCH_TYPE_LABELS = {
    "decode_only": "Decode-only",
    "mixed_prefill": "Mixed-prefill",
}
STRATEGIES = ("fa3_native_fifo", "critical_wave_auto_lpt")
STRATEGY_LABELS = ("FA3 Native\nFIFO", "Critical Wave\nAuto-LPT")
STRATEGY_COLORS = ("#4C78A8", "#E45756")


@dataclass(frozen=True)
class Metric:
    mean_column: str
    median_column: str
    ylabel: str
    scale: float
    decimals: int
    higher_is_better: bool


METRICS = {
    "p50": Metric(
        "p50_ms_mean",
        "p50_ms_median",
        "Mean per-case p50 latency (us)",
        1000.0,
        1,
        False,
    ),
    "p90": Metric(
        "p90_ms_mean",
        "p90_ms_median",
        "Mean per-case p90 latency (us)",
        1000.0,
        1,
        False,
    ),
    "tflops": Metric(
        "effective_tflops_mean",
        "effective_tflops_median",
        "Mean effective TFLOPS",
        1.0,
        0,
        True,
    ),
    "bandwidth": Metric(
        "effective_kv_bandwidth_gbps_per_gpu_mean",
        "effective_kv_bandwidth_gbps_per_gpu_median",
        "Mean effective KV bandwidth (GB/s/GPU)",
        1.0,
        0,
        True,
    ),
}


@dataclass(frozen=True)
class SummaryValue:
    mean: float
    median: float
    case_count: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 2x9 decode-only/mixed-prefill plot comparing the two "
            "wave-scheduler strategies for arrival scales 1/2/4 and DCP "
            "sizes 2/4/8."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"plot-ready summary CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output image path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--metric",
        choices=tuple(METRICS),
        default="p50",
        help="metric shown by the bars (default: p50)",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--title",
        default="Mega DCP Wave-Scheduler Ablation",
        help="figure title; pass an empty string to omit it",
    )
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def load_values(
    path: Path, metric: Metric
) -> dict[tuple[str, str, int, str], SummaryValue]:
    required = {
        "arrival_time_scale",
        "dcp_size",
        "strategy",
        "batch_type",
        "case_count",
        metric.mean_column,
        metric.median_column,
    }
    grouped: dict[tuple[str, str, int, str], SummaryValue] = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            arrival = row["arrival_time_scale"].strip()
            try:
                dcp_size = int(row["dcp_size"])
                case_count = int(row["case_count"])
                mean_value = float(row[metric.mean_column]) * metric.scale
                median_value = float(row[metric.median_column]) * metric.scale
            except ValueError as error:
                raise ValueError(f"invalid numeric value at {path}:{line_number}") from error
            strategy = row["strategy"].strip()
            batch_type = row["batch_type"].strip()
            if arrival not in ARRIVALS or dcp_size not in DCP_SIZES:
                continue
            if strategy not in STRATEGIES:
                continue
            if batch_type not in BATCH_TYPES:
                raise ValueError(f"invalid batch_type at {path}:{line_number}")
            if case_count <= 0:
                raise ValueError(f"invalid case_count at {path}:{line_number}")
            if any(
                not math.isfinite(value) or value < 0
                for value in (mean_value, median_value)
            ):
                raise ValueError(
                    f"invalid metric value at {path}:{line_number}"
                )
            key = (batch_type, arrival, dcp_size, strategy)
            if key in grouped:
                raise ValueError(
                    f"duplicate row for arrival={arrival}, DCP={dcp_size}, "
                    f"batch_type={batch_type}, strategy={strategy}"
                )
            grouped[key] = SummaryValue(mean_value, median_value, case_count)
    return grouped


def validate_matrix(
    grouped: dict[tuple[str, str, int, str], SummaryValue],
) -> dict[tuple[str, str, int], int]:
    case_counts: dict[tuple[str, str, int], int] = {}
    for batch_type in BATCH_TYPES:
        for arrival in ARRIVALS:
            arrival_case_count: int | None = None
            for dcp_size in DCP_SIZES:
                native = grouped.get(
                    (batch_type, arrival, dcp_size, STRATEGIES[0])
                )
                critical = grouped.get(
                    (batch_type, arrival, dcp_size, STRATEGIES[1])
                )
                if not native or not critical:
                    raise ValueError(
                        f"missing strategy data for batch_type={batch_type}, "
                        f"arrival={arrival}, DCP={dcp_size}"
                    )
                if native.case_count != critical.case_count:
                    raise ValueError(
                        f"case count mismatch for batch_type={batch_type}, "
                        f"arrival={arrival}, DCP={dcp_size}"
                    )
                if arrival_case_count is None:
                    arrival_case_count = native.case_count
                elif native.case_count != arrival_case_count:
                    raise ValueError(
                        f"DCP case count mismatch for batch_type={batch_type}, "
                        f"arrival={arrival}: expected {arrival_case_count}, "
                        f"got {native.case_count} at DCP={dcp_size}"
                    )
                case_counts[(batch_type, arrival, dcp_size)] = native.case_count
    return case_counts


def comparison_ratio(native: float, critical: float, higher_is_better: bool) -> float:
    if native <= 0 or critical <= 0:
        return math.nan
    return critical / native if higher_is_better else native / critical


def plot(
    grouped: dict[tuple[str, str, int, str], SummaryValue],
    metric: Metric,
    metric_name: str,
    case_counts: dict[tuple[str, str, int], int],
    output: Path,
    dpi: int,
    title: str,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        2,
        9,
        figsize=(25.5, 8.2),
        sharey="row",
    )
    figure.subplots_adjust(
        left=0.055, right=0.995, bottom=0.11, top=0.80, hspace=0.42, wspace=0.06
    )

    columns = tuple(
        (arrival, dcp_size) for arrival in ARRIVALS for dcp_size in DCP_SIZES
    )
    for row, batch_type in enumerate(BATCH_TYPES):
        for axis, (arrival, dcp_size) in zip(axes[row], columns, strict=True):
            summaries = [
                grouped[(batch_type, arrival, dcp_size, strategy)]
                for strategy in STRATEGIES
            ]
            means = [summary.mean for summary in summaries]
            medians = [summary.median for summary in summaries]
            bars = axis.bar(
                (0, 1),
                means,
                width=0.68,
                color=STRATEGY_COLORS,
                edgecolor="#333333",
                linewidth=0.65,
                zorder=2,
            )
            axis.scatter(
                (0, 1),
                medians,
                marker="D",
                s=22,
                color="white",
                edgecolor="#222222",
                linewidth=0.8,
                zorder=3,
            )
            axis.set_title(f"Arrival {arrival}x\nDCP {dcp_size}", pad=7)
            axis.set_xticks((0, 1), STRATEGY_LABELS)
            axis.grid(
                axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8, zorder=0
            )
            axis.tick_params(axis="x", length=0, pad=5)
            axis.margins(y=0.28)
            if axis is axes[row, 0]:
                axis.set_ylabel(
                    f"{BATCH_TYPE_LABELS[batch_type]}\n{metric.ylabel}",
                    fontweight="bold",
                )

            for bar, value in zip(bars, means, strict=True):
                axis.annotate(
                    f"{value:.{metric.decimals}f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

            ratio = comparison_ratio(means[0], means[1], metric.higher_is_better)
            ratio_label = "n/a" if not math.isfinite(ratio) else f"{ratio:.2f}x"
            case_count = case_counts[(batch_type, arrival, dcp_size)]
            axis.text(
                0.5,
                0.98,
                f"{ratio_label}\nn={case_count}",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=8.5,
                fontweight="bold",
                color="#2F6B3C" if ratio >= 1 else "#A33A32",
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="#333333")
        for color in STRATEGY_COLORS
    ]
    labels = ("FA3 Native FIFO", "Critical Wave Auto-LPT")
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.89),
    )
    if title:
        figure.suptitle(
            f"{title} - {metric_name} (mean bars, median diamonds)",
            fontsize=13,
            y=0.97,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metric = METRICS[args.metric]
    grouped = load_values(args.input, metric)
    case_counts = validate_matrix(grouped)
    plot(
        grouped,
        metric,
        args.metric,
        case_counts,
        args.output,
        args.dpi,
        args.title,
    )
    for batch_type in BATCH_TYPES:
        counts = [
            case_counts[(batch_type, arrival, DCP_SIZES[0])]
            for arrival in ARRIVALS
        ]
        print(
            f"{BATCH_TYPE_LABELS[batch_type]} paired cases by arrival "
            f"{ARRIVALS}: {counts}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
