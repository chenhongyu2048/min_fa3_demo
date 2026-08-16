#!/usr/bin/env python3
"""Plot the logged Mega DCP kernel_done p50 and p90 timestamps."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


RUN_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = RUN_DIR / "kernel_done_summary.csv"
ARRIVALS = (1, 2, 4)
DCP_SIZES = (2, 4, 8)
STRATEGIES = ("fa3_native_fifo", "critical_wave_auto_lpt")
STRATEGY_LABELS = ("FA3 Native\nFIFO", "Critical Wave\nAuto-LPT")
STRATEGY_COLORS = ("#4C78A8", "#E45756")
BATCH_TYPES = ("decode_only", "mixed_prefill")
BATCH_TYPE_LABELS = {
    "decode_only": "Decode-only",
    "mixed_prefill": "Mixed-prefill",
}

@dataclass(frozen=True)
class SummaryRecord:
    batch_type: str
    arrival: int
    dcp_size: int
    strategy: str
    comm_sm: int
    case_count: int
    p50_mean_us: float
    p90_mean_us: float


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot kernel_done p50/p90 timestamps from summary CSV for "
            "FA3 Native FIFO and Critical Wave Auto-LPT"
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="plot-ready kernel_done summary CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: figures next to INPUT)",
    )
    parser.add_argument("--dpi", type=positive_int, default=220)
    return parser.parse_args(argv)


def load_summary(
    path: Path,
) -> dict[tuple[str, int, int, str], SummaryRecord]:
    required = {
        "batch_type",
        "arrival_time_scale",
        "dcp_size",
        "strategy",
        "comm_sm",
        "case_count",
        "kernel_done_p50_us_mean",
        "kernel_done_p90_us_mean",
    }
    grouped: dict[tuple[str, int, int, str], SummaryRecord] = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                arrival = int(row["arrival_time_scale"])
                dcp_size = int(row["dcp_size"])
                comm_sm = int(row["comm_sm"])
                case_count = int(row["case_count"])
                values = {
                    name: float(row[column])
                    for name, column in (
                        ("p50_mean_us", "kernel_done_p50_us_mean"),
                        ("p90_mean_us", "kernel_done_p90_us_mean"),
                    )
                }
            except ValueError as error:
                raise ValueError(f"invalid numeric value at {path}:{line_number}") from error
            batch_type = row["batch_type"].strip()
            strategy = row["strategy"].strip()
            if batch_type not in BATCH_TYPES:
                raise ValueError(f"invalid batch type at {path}:{line_number}")
            if arrival not in ARRIVALS or dcp_size not in DCP_SIZES:
                raise ValueError(f"invalid matrix key at {path}:{line_number}")
            if strategy not in STRATEGIES or case_count <= 0:
                raise ValueError(f"invalid strategy or case count at {path}:{line_number}")
            if any(not math.isfinite(value) or value <= 0 for value in values.values()):
                raise ValueError(f"invalid timestamp at {path}:{line_number}")
            key = (batch_type, arrival, dcp_size, strategy)
            if key in grouped:
                raise ValueError(f"duplicate summary row at {path}:{line_number}: {key}")
            grouped[key] = SummaryRecord(
                batch_type=batch_type,
                arrival=arrival,
                dcp_size=dcp_size,
                strategy=strategy,
                comm_sm=comm_sm,
                case_count=case_count,
                **values,
            )

    for batch_type in BATCH_TYPES:
        for arrival in ARRIVALS:
            for dcp_size in DCP_SIZES:
                native = grouped.get((batch_type, arrival, dcp_size, STRATEGIES[0]))
                critical = grouped.get((batch_type, arrival, dcp_size, STRATEGIES[1]))
                if native is None or critical is None:
                    raise ValueError(
                        f"missing summary pair for {batch_type}, arrival={arrival}, "
                        f"DCP={dcp_size}"
                    )
                if native.case_count != critical.case_count:
                    raise ValueError(
                        f"case count mismatch for {batch_type}, arrival={arrival}, "
                        f"DCP={dcp_size}"
                    )
    return grouped


def plot_stat(
    grouped: dict[tuple[str, int, int, str], SummaryRecord],
    stat: str,
    output: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 9, figsize=(15.5, 7.2), sharey="row")
    figure.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.10,
        top=0.79,
        hspace=0.52,
        wspace=0.05,
    )
    columns = tuple(
        (arrival, dcp_size) for arrival in ARRIVALS for dcp_size in DCP_SIZES
    )
    mean_attribute = f"{stat}_mean_us"

    for row, batch_type in enumerate(BATCH_TYPES):
        for axis, (arrival, dcp_size) in zip(axes[row], columns, strict=True):
            summaries = [
                grouped[(batch_type, arrival, dcp_size, strategy)]
                for strategy in STRATEGIES
            ]
            means = [getattr(record, mean_attribute) for record in summaries]
            bars = axis.bar(
                (0, 1),
                means,
                width=0.68,
                color=STRATEGY_COLORS,
                edgecolor="#333333",
                linewidth=0.65,
                zorder=2,
            )
            axis.set_title(f"Arrival {arrival}x\nDCP {dcp_size}", pad=4)
            axis.set_xticks((0, 1), STRATEGY_LABELS)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
            axis.tick_params(axis="x", length=0, pad=3)
            axis.margins(y=0.25)
            if axis is axes[row, 0]:
                axis.set_ylabel(
                    f"{BATCH_TYPE_LABELS[batch_type]}\n"
                    f"Mean per-case {stat} (us)",
                    fontweight="bold",
                )
            for bar, value in zip(bars, means, strict=True):
                axis.annotate(
                    f"{value:.1f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )
            speedup = means[0] / means[1]
            case_count = summaries[0].case_count
            axis.text(
                0.5,
                0.98,
                f"{speedup:.2f}x\nn={case_count}",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=9.5,
                fontweight="bold",
                color="#2F6B3C" if speedup >= 1 else "#A33A32",
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="#333333")
        for color in STRATEGY_COLORS
    ]
    figure.legend(
        handles,
        ("FA3 Native FIFO", "Critical Wave Auto-LPT"),
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.88),
    )
    figure.suptitle(
        f"Mega DCP Kernel-Done Timestamp - {stat.upper()} (mean bars)",
        fontsize=14,
        y=0.97,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or args.input.parent / "figures"
    try:
        grouped = load_summary(args.input)
        outputs = []
        for stat in ("p50", "p90"):
            output = output_dir / f"kernel_done_{stat}.png"
            plot_stat(grouped, stat, output, args.dpi)
            outputs.append(output)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    counts = {
        batch_type: grouped[
            (batch_type, ARRIVALS[0], DCP_SIZES[0], STRATEGIES[0])
        ].case_count
        for batch_type in BATCH_TYPES
    }
    print(f"Loaded {len(grouped)} plotting rows from {args.input}")
    print(
        "Per-log case mix: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
