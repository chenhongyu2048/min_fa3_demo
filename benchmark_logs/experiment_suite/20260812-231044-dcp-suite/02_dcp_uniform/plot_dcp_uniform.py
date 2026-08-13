#!/usr/bin/env python3
"""Plot the uniform DCP matrix, one workload grid per DCP size."""

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


DEFAULT_RUN_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = DEFAULT_RUN_DIR / "uniform_summary.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "figures"

METHODS = (
    ("best_mega_p50_ms", "Best Mega DCP", "#E15759", "o", "-"),
    (
        "vllm_ag_rs_graph_p50_ms",
        "vLLM AG+RS graph",
        "#4C78A8",
        "s",
        "--",
    ),
    (
        "vllm_a2a_graph_p50_ms",
        "vLLM A2A graph",
        "#59A14F",
        "^",
        "-.",
    ),
    (
        "sglang_graph_p50_ms",
        "SGLang MHA AG+AR graph",
        "#F28E2B",
        "D",
        ":",
    ),
)


@dataclass(frozen=True)
class Record:
    decode_requests: int
    chunk_requests: int
    kv_cache_tokens: int
    dcp_size: int
    topology: str
    best_mega_comm_sm: int
    latencies_ms: dict[str, float]


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def integer_list(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated integer list"
        ) from error
    if not parsed or any(item < 0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(
            "must contain unique nonnegative integers"
        )
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot best Mega DCP and three CUDA-Graph baselines from "
            "uniform_summary.csv"
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="summary CSV (default: the CSV next to this script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for dcp_uniform_latency_dcpN.png files",
    )
    parser.add_argument(
        "--dcp-sizes",
        type=integer_list,
        default=None,
        metavar="N,N,...",
        help="DCP sizes to plot (default: all)",
    )
    parser.add_argument(
        "--decode-requests",
        type=integer_list,
        default=None,
        metavar="N,N,...",
        help="decode-request counts to plot (default: all)",
    )
    parser.add_argument(
        "--chunk-requests",
        type=integer_list,
        default=None,
        metavar="N,N,...",
        help="chunk-request counts to plot (default: all)",
    )
    parser.add_argument(
        "--unit",
        choices=("us", "ms"),
        default="us",
        help="latency display unit (default: us)",
    )
    parser.add_argument("--dpi", type=positive_int, default=220)
    parser.add_argument(
        "--title-prefix",
        default="Uniform DCP latency matrix",
        help="figure title prefix",
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


def load_records(path: Path) -> list[Record]:
    latency_columns = {method[0] for method in METHODS}
    required = {
        "decode_requests",
        "chunk_requests",
        "kv_cache_tokens",
        "dcp_size",
        "topology",
        "best_mega_comm_sm",
        *latency_columns,
    }
    records: list[Record] = []
    seen: set[tuple[int, int, int, int]] = set()
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
                decode = int(row["decode_requests"])
                chunk = int(row["chunk_requests"])
                kv_tokens = int(row["kv_cache_tokens"])
                dcp_size = int(row["dcp_size"])
                comm_sm = int(row["best_mega_comm_sm"])
            except ValueError as error:
                raise ValueError(f"line {line}: invalid integer dimension") from error
            if min(decode, chunk) < 0 or min(kv_tokens, dcp_size, comm_sm) <= 0:
                raise ValueError(f"line {line}: invalid nonpositive dimension")
            key = (decode, chunk, kv_tokens, dcp_size)
            if key in seen:
                raise ValueError(f"line {line}: duplicate matrix point {key}")
            seen.add(key)
            records.append(
                Record(
                    decode_requests=decode,
                    chunk_requests=chunk,
                    kv_cache_tokens=kv_tokens,
                    dcp_size=dcp_size,
                    topology=row["topology"],
                    best_mega_comm_sm=comm_sm,
                    latencies_ms={
                        column: finite_positive(
                            row[column], column=column, line=line
                        )
                        for column in latency_columns
                    },
                )
            )
    if not records:
        raise ValueError(f"{path} contains no data rows")
    return records


def selected_or_all(
    requested: tuple[int, ...] | None,
    available: set[int],
    *,
    name: str,
) -> tuple[int, ...]:
    if requested is None:
        return tuple(sorted(available))
    unknown = set(requested).difference(available)
    if unknown:
        raise ValueError(f"unknown {name}: {sorted(unknown)}")
    return requested


def kv_label(tokens: int) -> str:
    if tokens % 1024 == 0:
        return f"{tokens // 1024}K"
    return str(tokens)


def validate_submatrix(
    by_key: dict[tuple[int, int, int, int], Record],
    dcp_sizes: Sequence[int],
    decodes: Sequence[int],
    chunks: Sequence[int],
    kv_lengths: Sequence[int],
) -> None:
    missing = [
        (decode, chunk, kv_tokens, dcp_size)
        for dcp_size in dcp_sizes
        for chunk in chunks
        for decode in decodes
        for kv_tokens in kv_lengths
        if (decode, chunk, kv_tokens, dcp_size) not in by_key
    ]
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(f"incomplete workload matrix; missing {preview}{suffix}")


def plot_dcp_size(
    *,
    args: argparse.Namespace,
    dcp_size: int,
    by_key: dict[tuple[int, int, int, int], Record],
    decodes: Sequence[int],
    chunks: Sequence[int],
    kv_lengths: Sequence[int],
) -> Path:
    scale = 1000.0 if args.unit == "us" else 1.0
    unit_label = "us" if args.unit == "us" else "ms"
    fig, axes = plt.subplots(
        len(chunks),
        len(decodes),
        figsize=(4.0 * len(decodes), 2.9 * len(chunks)),
        sharex=True,
        squeeze=False,
    )
    x = list(range(len(kv_lengths)))
    legend_handles = []
    topology = by_key[(decodes[0], chunks[0], kv_lengths[0], dcp_size)].topology

    for row_index, chunk in enumerate(chunks):
        for column_index, decode in enumerate(decodes):
            axis = axes[row_index][column_index]
            records = [
                by_key[(decode, chunk, kv_tokens, dcp_size)]
                for kv_tokens in kv_lengths
            ]
            for latency_column, label, color, marker, linestyle in METHODS:
                values = [
                    record.latencies_ms[latency_column] * scale
                    for record in records
                ]
                (line,) = axis.plot(
                    x,
                    values,
                    color=color,
                    marker=marker,
                    markersize=4.5,
                    linewidth=1.8,
                    linestyle=linestyle,
                    label=label,
                )
                if row_index == 0 and column_index == 0:
                    legend_handles.append(line)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
            axis.set_axisbelow(True)
            axis.set_title(f"decode={decode}, chunk={chunk}", fontsize=10)
            if column_index == 0:
                axis.set_ylabel(f"p50 latency ({unit_label})")
            if row_index < len(chunks) - 1:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("KV cache tokens")
                axis.set_xticks(x, [kv_label(value) for value in kv_lengths])

    fig.suptitle(
        f"{args.title_prefix}: DCP={dcp_size} ({topology})",
        fontsize=15,
        y=0.995,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(METHODS),
        frameon=False,
    )
    fig.text(
        0.5,
        0.004,
        "Best Mega selects the lowest-p50 comm-SM setting independently for each workload.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.925))
    output = args.output_dir / f"dcp_uniform_latency_dcp{dcp_size}.png"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def plot(args: argparse.Namespace) -> list[Path]:
    records = load_records(args.input)
    dcp_sizes = selected_or_all(
        args.dcp_sizes,
        {record.dcp_size for record in records},
        name="DCP sizes",
    )
    decodes = selected_or_all(
        args.decode_requests,
        {record.decode_requests for record in records},
        name="decode-request counts",
    )
    chunks = selected_or_all(
        args.chunk_requests,
        {record.chunk_requests for record in records},
        name="chunk-request counts",
    )
    kv_lengths = tuple(sorted({record.kv_cache_tokens for record in records}))
    by_key = {
        (
            record.decode_requests,
            record.chunk_requests,
            record.kv_cache_tokens,
            record.dcp_size,
        ): record
        for record in records
    }
    validate_submatrix(by_key, dcp_sizes, decodes, chunks, kv_lengths)
    return [
        plot_dcp_size(
            args=args,
            dcp_size=dcp_size,
            by_key=by_key,
            decodes=decodes,
            chunks=chunks,
            kv_lengths=kv_lengths,
        )
        for dcp_size in dcp_sizes
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outputs = plot(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for output in outputs:
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
