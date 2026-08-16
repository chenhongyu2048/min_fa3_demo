#!/usr/bin/env python3
"""Plot DCP wave latency separately for decode-only and mixed batches."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_RUN_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = DEFAULT_RUN_DIR / "wave_ablation_summary.csv"
DEFAULT_OUTPUT = DEFAULT_RUN_DIR / "figures" / "dcp_wave_latency_by_batch_type.png"

STRATEGIES = (
    "fa3_native_fifo",
    "critical_wave_fifo",
    "critical_wave_release_lpt",
)
STRATEGY_STYLE = {
    "fa3_native_fifo": ("FA3 native FIFO", "#4C78A8", "o", "-"),
    "critical_wave_fifo": ("Critical-wave FIFO", "#F28E2B", "s", "--"),
    "critical_wave_release_lpt": (
        "Critical-wave release LPT",
        "#59A14F",
        "^",
        "-.",
    ),
}

BATCH_TYPES = ("decode_only", "mixed_prefill")
BATCH_TYPE_LABELS = {
    "decode_only": "Decode-only\n(chunk requests = 0)",
    "mixed_prefill": "Mixed batch\n(chunk requests > 0)",
}


@dataclass(frozen=True)
class SummaryRecord:
    arrival_scale: Decimal
    dcp_size: int
    strategy: str
    batch_type: str
    case_count: int
    latency_ms: float


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
            "Plot mean per-case latency by DCP size and arrival-time scale, "
            "with decode-only and mixed batches in separate rows"
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="plot-ready summary CSV (default: next to this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output PNG path",
    )
    parser.add_argument(
        "--latency-stat",
        choices=("p50", "p90"),
        default="p50",
        help="per-case latency statistic to average (default: p50)",
    )
    parser.add_argument("--dpi", type=positive_int, default=220)
    parser.add_argument(
        "--title",
        default="DCP wave-scheduling latency by batch type",
        help="figure title",
    )
    return parser.parse_args(argv)


def finite_positive(value: str | None, *, column: str, line: int) -> float:
    try:
        parsed = float(value) if value is not None else math.nan
    except ValueError as error:
        raise ValueError(f"line {line}: invalid {column}: {value!r}") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"line {line}: {column} must be finite and positive")
    return parsed


def load_records(path: Path, latency_stat: str) -> list[SummaryRecord]:
    latency_column = f"{latency_stat}_ms_mean"
    required = {
        "arrival_time_scale",
        "dcp_size",
        "strategy",
        "batch_type",
        "case_count",
        latency_column,
    }
    records: list[SummaryRecord] = []
    seen: set[tuple[Decimal, int, str, str]] = set()
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot open {path}: {error}") from error
    with handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            try:
                arrival = Decimal(row["arrival_time_scale"])
                dcp_size = int(row["dcp_size"])
                case_count = int(row["case_count"])
            except (InvalidOperation, ValueError) as error:
                raise ValueError(
                    f"line {line}: invalid arrival scale, DCP size, or case count"
                ) from error
            strategy = row["strategy"]
            batch_type = row["batch_type"]
            if not arrival.is_finite() or arrival <= 0 or dcp_size <= 0:
                raise ValueError(
                    f"line {line}: arrival scale and DCP size must be positive"
                )
            if case_count <= 0:
                raise ValueError(f"line {line}: case_count must be positive")
            if strategy not in STRATEGY_STYLE:
                raise ValueError(f"line {line}: unknown strategy {strategy!r}")
            if batch_type not in BATCH_TYPES:
                raise ValueError(f"line {line}: unknown batch_type {batch_type!r}")
            key = (arrival, dcp_size, batch_type, strategy)
            if key in seen:
                raise ValueError(f"line {line}: duplicate summary record {key}")
            seen.add(key)
            records.append(
                SummaryRecord(
                    arrival_scale=arrival,
                    dcp_size=dcp_size,
                    strategy=strategy,
                    batch_type=batch_type,
                    case_count=case_count,
                    latency_ms=finite_positive(
                        row[latency_column], column=latency_column, line=line
                    ),
                )
            )
    if not records:
        raise ValueError(f"{path} contains no data rows")
    return records


def decimal_label(value: Decimal) -> str:
    return format(value.normalize(), "f")


def aggregate_records(
    records: Sequence[SummaryRecord],
    arrivals: Sequence[Decimal],
    dcp_sizes: Sequence[int],
) -> dict[tuple[Decimal, int, str, str], float]:
    grouped: dict[tuple[Decimal, int, str, str], SummaryRecord] = {}
    for record in records:
        key = (
            record.arrival_scale,
            record.dcp_size,
            record.batch_type,
            record.strategy,
        )
        grouped[key] = record

    missing = [
        (decimal_label(arrival), dcp_size, batch_type, strategy)
        for arrival in arrivals
        for dcp_size in dcp_sizes
        for batch_type in BATCH_TYPES
        for strategy in STRATEGIES
        if (arrival, dcp_size, batch_type, strategy) not in grouped
    ]
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(f"incomplete case matrix; missing {preview}{suffix}")

    for arrival in arrivals:
        for dcp_size in dcp_sizes:
            for batch_type in BATCH_TYPES:
                case_counts = {
                    strategy: grouped[
                        (arrival, dcp_size, batch_type, strategy)
                    ].case_count
                    for strategy in STRATEGIES
                }
                reference = case_counts[STRATEGIES[0]]
                mismatched = [
                    strategy
                    for strategy in STRATEGIES[1:]
                    if case_counts[strategy] != reference
                ]
                if mismatched:
                    raise ValueError(
                        "strategy case counts differ for "
                        f"arrival={decimal_label(arrival)}, DCP={dcp_size}, "
                        f"batch_type={batch_type}: {mismatched}"
                    )

    return {
        key: record.latency_ms for key, record in grouped.items()
    }


def plot(args: argparse.Namespace) -> None:
    records = load_records(args.input, args.latency_stat)
    arrivals = sorted({record.arrival_scale for record in records})
    dcp_sizes = sorted({record.dcp_size for record in records})
    latency_by_group = aggregate_records(records, arrivals, dcp_sizes)

    fig, axes = plt.subplots(
        len(BATCH_TYPES),
        len(dcp_sizes),
        figsize=(4.8 * len(dcp_sizes), 3.7 * len(BATCH_TYPES)),
        sharex=True,
        squeeze=False,
    )
    x = list(range(len(arrivals)))
    legend_handles = []
    for row_index, batch_type in enumerate(BATCH_TYPES):
        for column_index, dcp_size in enumerate(dcp_sizes):
            axis = axes[row_index][column_index]
            for strategy in STRATEGIES:
                label, color, marker, linestyle = STRATEGY_STYLE[strategy]
                values_us = [
                    latency_by_group[(arrival, dcp_size, batch_type, strategy)]
                    * 1000.0
                    for arrival in arrivals
                ]
                (line,) = axis.plot(
                    x,
                    values_us,
                    color=color,
                    marker=marker,
                    markersize=6.0,
                    linewidth=2.0,
                    linestyle=linestyle,
                    label=label,
                )
                if row_index == 0 and column_index == 0:
                    legend_handles.append(line)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
            axis.set_axisbelow(True)
            if row_index == 0:
                axis.set_title(f"DCP size = {dcp_size}", fontsize=12)
            if column_index == 0:
                axis.set_ylabel(
                    f"{BATCH_TYPE_LABELS[batch_type]}\n"
                    f"Mean per-case {args.latency_stat} latency (us)"
                )
            if row_index < len(BATCH_TYPES) - 1:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("Arrival-time scale")
                axis.set_xticks(x, [decimal_label(value) for value in arrivals])

    fig.suptitle(args.title, fontsize=16, y=0.995)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(STRATEGIES),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plot(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
