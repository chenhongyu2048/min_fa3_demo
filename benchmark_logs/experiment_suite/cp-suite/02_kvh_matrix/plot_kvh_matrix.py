#!/usr/bin/env python3
"""Plot the causal 128K dataset benchmark matrix from summary CSV."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "kvh_matrix_summary.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "kvh_matrix_weighted_gpu_tflops.png"

DIRECTIONS = ("forward", "backward")
KVHEADS = (1, 2, 4, 8)
DATASETS = ("prolong", "arxiv", "freelaw", "github", "pile")
DATASET_LABELS = {
    "arxiv": "ArXiv",
    "github": "GitHub",
    "pile": "Pile-CC",
    "freelaw": "FreeLaw",
    "prolong": "ProLong",
}
METHODS = (
    "allgather_attention",
    "llama3_allgather_attention",
    "fa3_ring",
    "megatron_hybrid_cp",
    "magi_attention",
    "zeppelin",
    "mega_ring_all_cp",
    "mega_ring_hybrid",
)
TUNED_METHODS = frozenset(("mega_ring_all_cp", "mega_ring_hybrid"))
HYBRID_COMPARISON_METHODS = tuple(
    method for method in METHODS if method not in TUNED_METHODS
)
METHOD_LABELS = {
    "allgather_attention": "AllGather",
    "llama3_allgather_attention": "Llama3 AllGather",
    "fa3_ring": "FA3 Ring",
    "megatron_hybrid_cp": "Megatron Hybrid CP",
    "magi_attention": "MagiAttention",
    "zeppelin": "Zeppelin",
    "mega_ring_all_cp": "Mega-Ring All-CP",
    "mega_ring_hybrid": "Mega-Ring Hybrid",
}
METHOD_COLORS = {
    "allgather_attention": "#4C78A8",
    "llama3_allgather_attention": "#59A14F",
    "fa3_ring": "#9C755F",
    "megatron_hybrid_cp": "#F28E2B",
    "magi_attention": "#17A2B8",
    "zeppelin": "#ECA82C",
    "mega_ring_all_cp": "#E15759",
    "mega_ring_hybrid": "#B07AA1",
}

SM_RE = re.compile(r"^(?:-|\d+:\d+)$")


@dataclass(frozen=True)
class SummaryRecord:
    direction: str
    kvhead: int
    dataset: str
    method: str
    sm_config: str
    cases_done: int
    cases_total: int
    weighted_tflops: float
    weighted_gpu_tflops: float
    source: Path


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
        description="Plot the 4x2 KV-head forward/backward dataset matrix"
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
        help="output PNG",
    )
    parser.add_argument("--dpi", type=positive_int, default=220)
    return parser.parse_args(argv)


def load_summary(path: Path) -> tuple[list[SummaryRecord], int]:
    required = {
        "direction",
        "kvhead",
        "dataset",
        "method",
        "sm_config",
        "cases_done",
        "cases_total",
        "weighted_tflops",
        "weighted_gpu_tflops",
        "world_size",
    }
    records: list[SummaryRecord] = []
    world_sizes: set[int] = set()
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                direction = row["direction"].strip()
                kvhead = int(row["kvhead"])
                cases_done = int(row["cases_done"])
                cases_total = int(row["cases_total"])
                world_size = int(row["world_size"])
                weighted_tflops = float(row["weighted_tflops"])
                weighted_gpu_tflops = float(row["weighted_gpu_tflops"])
            except ValueError as error:
                raise ValueError(f"invalid numeric value at {path}:{line_number}") from error
            dataset = row["dataset"].strip()
            method = row["method"].strip()
            sm_config = row["sm_config"].strip()
            if direction not in DIRECTIONS or kvhead not in KVHEADS:
                raise ValueError(f"invalid matrix key at {path}:{line_number}")
            if dataset not in DATASETS or method not in METHODS:
                raise ValueError(f"invalid dataset or method at {path}:{line_number}")
            if SM_RE.fullmatch(sm_config) is None:
                raise ValueError(f"invalid SM config at {path}:{line_number}")
            if cases_done != cases_total or cases_done <= 0:
                raise ValueError(f"incomplete cases at {path}:{line_number}")
            if any(
                not math.isfinite(value) or value <= 0.0
                for value in (weighted_tflops, weighted_gpu_tflops)
            ):
                raise ValueError(f"invalid throughput at {path}:{line_number}")
            world_sizes.add(world_size)
            records.append(
                SummaryRecord(
                    direction=direction,
                    kvhead=kvhead,
                    dataset=dataset,
                    method=method,
                    sm_config=sm_config,
                    cases_done=cases_done,
                    cases_total=cases_total,
                    weighted_tflops=weighted_tflops,
                    weighted_gpu_tflops=weighted_gpu_tflops,
                    source=path,
                )
            )
    if not records:
        raise ValueError(f"{path}: no summary rows found")
    if len(world_sizes) != 1:
        raise ValueError(f"{path}: expected one world size, found {sorted(world_sizes)}")
    return records, world_sizes.pop()


def select_results(
    records: Sequence[SummaryRecord],
) -> dict[tuple[str, int, str, str], SummaryRecord]:
    grouped: dict[tuple[str, int, str, str], list[SummaryRecord]] = defaultdict(list)
    exact_keys: set[tuple[str, int, str, str, str]] = set()
    for record in records:
        exact_key = (
            record.direction,
            record.kvhead,
            record.dataset,
            record.method,
            record.sm_config,
        )
        if exact_key in exact_keys:
            raise ValueError(f"duplicate summary row for {exact_key} in {record.source}")
        exact_keys.add(exact_key)
        grouped[exact_key[:-1]].append(record)

    selected: dict[tuple[str, int, str, str], SummaryRecord] = {}
    expected_keys = {
        (direction, kvhead, dataset, method)
        for direction in DIRECTIONS
        for kvhead in KVHEADS
        for dataset in DATASETS
        for method in METHODS
    }
    missing = expected_keys.difference(grouped)
    extra = set(grouped).difference(expected_keys)
    if missing or extra:
        raise ValueError(
            f"incomplete result matrix: missing={len(missing)}, extra={len(extra)}"
        )

    for key, candidates in grouped.items():
        method = key[-1]
        if method in TUNED_METHODS:
            selected[key] = max(
                candidates, key=lambda item: item.weighted_gpu_tflops
            )
        elif len(candidates) == 1:
            selected[key] = candidates[0]
        else:
            raise ValueError(
                f"expected one untuned result for {key}, found {len(candidates)}"
            )
    return selected


def make_figure(
    selected: dict[tuple[str, int, str, str], SummaryRecord], world_size: int
) -> plt.Figure:
    fig, axes = plt.subplots(
        len(KVHEADS),
        len(DIRECTIONS),
        figsize=(18.5, 18.0),
        sharey=True,
        squeeze=False,
    )
    group_width = 0.86
    bar_width = group_width / len(METHODS)
    x_positions = list(range(len(DATASETS)))
    all_cp_method = "mega_ring_all_cp"
    hybrid_method = "mega_ring_hybrid"
    all_cp_index = METHODS.index(all_cp_method)
    hybrid_index = METHODS.index(hybrid_method)
    all_cp_offset = (all_cp_index - (len(METHODS) - 1) / 2) * bar_width
    hybrid_offset = (hybrid_index - (len(METHODS) - 1) / 2) * bar_width

    for row, kvhead in enumerate(KVHEADS):
        for column, direction in enumerate(DIRECTIONS):
            axis = axes[row, column]
            for method_index, method in enumerate(METHODS):
                offset = (method_index - (len(METHODS) - 1) / 2) * bar_width
                values = [
                    selected[(direction, kvhead, dataset, method)].weighted_gpu_tflops
                    for dataset in DATASETS
                ]
                axis.bar(
                    [x + offset for x in x_positions],
                    values,
                    width=bar_width * 0.92,
                    color=METHOD_COLORS[method],
                    edgecolor="white",
                    linewidth=0.55,
                    label=METHOD_LABELS[method],
                    zorder=3,
                )

            for x, dataset in zip(x_positions, DATASETS):
                all_cp = selected[(direction, kvhead, dataset, all_cp_method)]
                hybrid = selected[(direction, kvhead, dataset, hybrid_method)]
                best_baseline = max(
                    selected[
                        (direction, kvhead, dataset, method)
                    ].weighted_gpu_tflops
                    for method in HYBRID_COMPARISON_METHODS
                )

                for offset, record in (
                    (all_cp_offset, all_cp),
                    (hybrid_offset, hybrid),
                ):
                    axis.annotate(
                        record.sm_config,
                        xy=(x + offset, record.weighted_gpu_tflops),
                        xytext=(0, -4),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        rotation=90,
                        color="white",
                        fontsize=6.4,
                        fontweight="bold",
                        clip_on=True,
                        zorder=4,
                    )

                axis.annotate(
                    f"{hybrid.weighted_gpu_tflops / best_baseline:.2f}x",
                    xy=(x + hybrid_offset, hybrid.weighted_gpu_tflops),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color="#704E6F",
                    fontsize=7.8,
                    fontweight="bold",
                    zorder=5,
                )

            axis.set_title(
                f"KVH = {kvhead} | {direction.capitalize()}",
                fontsize=13,
                fontweight="bold",
            )
            axis.set_xticks(
                x_positions,
                [DATASET_LABELS[dataset] for dataset in DATASETS],
            )
            axis.set_xlabel("Dataset")
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.8)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.margins(x=0.025, y=0.14)
            if column == 0:
                axis.set_ylabel("Weighted average TFLOPS / GPU")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
        fontsize=10,
        columnspacing=1.5,
    )
    fig.suptitle(
        f"128K Causal Dataset Benchmark by KV Heads ({world_size} GPUs)",
        y=0.995,
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Bars show cross-case weighted throughput. White labels are the best Comp:Comm settings; Hybrid labels show speedup over the best non-Mega-Ring baseline.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.925), h_pad=2.2, w_pad=1.5)
    return fig


def print_summary(
    records: Sequence[SummaryRecord],
    selected: dict[tuple[str, int, str, str], SummaryRecord],
    world_size: int,
) -> None:
    print(
        f"Loaded {len(records)} rows from summary CSV; selected "
        f"{len(selected)} bars for {world_size} GPUs."
    )
    for method in TUNED_METHODS:
        counts = Counter(
            record.sm_config for record in selected.values() if record.method == method
        )
        rendered = ", ".join(
            f"{sm}={count}" for sm, count in sorted(counts.items())
        )
        print(f"Best-SM selections for {METHOD_LABELS[method]}: {rendered}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records, world_size = load_summary(args.input.resolve())
    selected = select_results(records)
    figure = make_figure(selected, world_size)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print_summary(records, selected, world_size)
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
