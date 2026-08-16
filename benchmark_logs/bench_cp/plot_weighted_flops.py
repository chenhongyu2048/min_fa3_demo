#!/usr/bin/env python3
"""Plot dataset-weighted forward/backward throughput from benchmark logs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


LOG_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = LOG_DIR / "20260726-002324"
DEFAULT_INPUT = DEFAULT_RUN_DIR / "weighted_flops_summary.csv"
DEFAULT_TOKEN_COUNTS = (65536, 131072, 262144)

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
METHOD_ALIASES = {
    "zepplin": "zeppelin",
}
HYBRID_COMPARISON_METHODS = tuple(
    method
    for method in METHODS
    if method not in {"mega_ring_all_cp", "mega_ring_hybrid"}
)
TUNED_METHODS = frozenset(("mega_ring_all_cp", "mega_ring_hybrid"))
METHOD_LABELS = {
    "allgather_attention": "allgather_attention",
    "llama3_allgather_attention": "llama3_allgather_attention",
    "fa3_ring": "ring_flash_attention",
    "megatron_hybrid_cp": "megatron_hybrid_cp",
    "magi_attention": "magi_attention",
    "zeppelin": "Zeppelin",
    "mega_ring_all_cp": "mega_ring_all_cp",
    "mega_ring_hybrid": "mega_ring_hybrid",
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
DATASET_LABELS = {
    "prolong": "ProLong",
    "arxiv": "ArXiv",
    "freelaw": "FreeLaw",
    "github": "GitHub",
    "pile": "Pile-CC",
}
DATASET_ORDER = {name: index for index, name in enumerate(DATASET_LABELS)}

SECTION_RE = re.compile(
    r"^\[(?:hybrid_dataset_|dataset_)(forward|backward)\]\s+dataset=(\S+)\s+GPUs=(\d+)"
)
PLANNER_RE = re.compile(
    r"^Planner workload:\s+dataset=(\S+),.*?world_size=(\d+)"
)
SUMMARY_RE = re.compile(r"^Cross-case (forward|backward) summary$")
CASES_RE = re.compile(r"^\d+/\d+$")
SM_RE = re.compile(r"^(?:-|\d+:\d+)$")


@dataclass(frozen=True)
class SummaryRecord:
    token_count: int
    dataset: str
    direction: str
    mode: str
    world_size: int
    method: str
    sm_config: str
    weighted_tflops: float
    weighted_gpu_tflops: float
    source: Path


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def load_summary(path: Path) -> list[SummaryRecord]:
    required = {
        "token_count", "dataset", "direction", "mode", "world_size", "method",
        "sm_config", "weighted_tflops", "weighted_gpu_tflops",
    }
    records: list[SummaryRecord] = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                records.append(
                    SummaryRecord(
                        token_count=int(row["token_count"]),
                        dataset=row["dataset"].strip(),
                        direction=row["direction"].strip(),
                        mode=row["mode"].strip(),
                        world_size=int(row["world_size"]),
                        method=canonical_method(row["method"].strip()),
                        sm_config=row["sm_config"].strip(),
                        weighted_tflops=float(row["weighted_tflops"]),
                        weighted_gpu_tflops=float(row["weighted_gpu_tflops"]),
                        source=path,
                    )
                )
            except ValueError as error:
                raise ValueError(f"invalid summary row at {path}:{line_number}") from error
    if not records:
        raise ValueError(f"{path}: no summary rows found")
    return records


def parse_summary_row(
    line: str,
    *,
    dataset: str,
    direction: str,
    world_size: int,
    source: Path,
) -> SummaryRecord | None:
    fields = line.split()
    if not fields:
        return None
    method = canonical_method(fields[0])
    if method not in METHODS:
        return None

    prefix_length = 4 if direction == "forward" else 3
    if len(fields) != prefix_length + 7:
        raise ValueError(f"malformed {direction} summary row in {source}: {line}")

    if direction == "forward":
        mode, sm_config, cases = fields[1:4]
    else:
        mode = "causal"
        sm_config, cases = fields[1:3]
    if not SM_RE.fullmatch(sm_config) or not CASES_RE.fullmatch(cases):
        raise ValueError(f"malformed {direction} summary row in {source}: {line}")

    try:
        metrics = [float(value) for value in fields[-7:]]
    except ValueError as exc:
        raise ValueError(f"non-numeric summary metric in {source}: {line}") from exc

    return SummaryRecord(
        token_count=0,
        dataset=dataset,
        direction=direction,
        mode=mode,
        world_size=world_size,
        method=method,
        sm_config=sm_config,
        weighted_tflops=metrics[-2],
        weighted_gpu_tflops=metrics[-1],
        source=source,
    )


def parse_log(path: Path) -> list[SummaryRecord]:
    records: list[SummaryRecord] = []
    dataset: str | None = None
    world_size: int | None = None
    summary_direction: str | None = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        section_match = SECTION_RE.match(line)
        if section_match is not None:
            _direction, dataset, world_size_text = section_match.groups()
            world_size = int(world_size_text)
            summary_direction = None
            continue

        planner_match = PLANNER_RE.match(line)
        if planner_match is not None:
            dataset, world_size_text = planner_match.groups()
            world_size = int(world_size_text)
            continue

        summary_match = SUMMARY_RE.match(line)
        if summary_match is not None:
            summary_direction = summary_match.group(1)
            if dataset is None or world_size is None:
                raise ValueError(
                    f"summary in {path} has no preceding dataset/world-size metadata"
                )
            continue

        if summary_direction is None or dataset is None or world_size is None:
            continue
        record = parse_summary_row(
            line,
            dataset=dataset,
            direction=summary_direction,
            world_size=world_size,
            source=path,
        )
        if record is not None:
            records.append(record)

    return records


def load_records(paths: Iterable[Path]) -> list[SummaryRecord]:
    records: list[SummaryRecord] = []
    for path in paths:
        records.extend(parse_log(path))
    if not records:
        raise ValueError("no cross-case forward/backward summary rows were found")
    return records


def load_direction_records(
    path: Path, direction: str, token_count: int
) -> list[SummaryRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"{direction} log does not exist: {path}")

    records = load_records([path.resolve()])
    unexpected_directions = sorted({record.direction for record in records} - {direction})
    if unexpected_directions:
        raise ValueError(
            f"{direction} log {path} also contains "
            + ", ".join(unexpected_directions)
            + " summary rows"
        )
    return [replace(record, token_count=token_count) for record in records]


def choose_world_size(records: Sequence[SummaryRecord], requested: int | None) -> int:
    available = sorted({record.world_size for record in records})
    if requested is not None:
        if requested not in available:
            raise ValueError(
                f"world size {requested} is unavailable; found: "
                + ", ".join(str(value) for value in available)
            )
        return requested
    if len(available) != 1:
        raise ValueError(
            "logs contain multiple GPU counts; select one with --world-size: "
            + ", ".join(str(value) for value in available)
        )
    return available[0]


def select_best(
    records: Sequence[SummaryRecord], *, world_size: int, mode: str
) -> dict[tuple[int, str, str, str], SummaryRecord]:
    grouped: dict[tuple[int, str, str, str], list[SummaryRecord]] = defaultdict(list)
    exact_keys: dict[tuple[int, str, str, str, str, str], SummaryRecord] = {}

    for record in records:
        if record.world_size != world_size:
            continue
        if record.direction == "forward" and record.mode != mode:
            continue
        key = (
            record.token_count,
            record.dataset,
            record.direction,
            record.mode,
            record.method,
            record.sm_config,
        )
        previous = exact_keys.get(key)
        if previous is not None:
            raise ValueError(
                "duplicate benchmark summary for "
                f"dataset={record.dataset}, direction={record.direction}, "
                f"method={record.method}, SM={record.sm_config}: "
                f"{previous.source} and {record.source}"
            )
        exact_keys[key] = record
        grouped[(record.token_count, record.dataset, record.direction, record.method)].append(record)

    selected: dict[tuple[int, str, str, str], SummaryRecord] = {}
    for key, candidates in grouped.items():
        method = key[3]
        if method in TUNED_METHODS:
            selected[key] = max(candidates, key=lambda item: item.weighted_gpu_tflops)
        elif len(candidates) == 1:
            selected[key] = candidates[0]
        else:
            sources = ", ".join(str(item.source) for item in candidates)
            raise ValueError(f"multiple untuned baseline rows for {key}: {sources}")

    return selected


def weighted_gpu_tflops_or_zero(
    selected: dict[tuple[int, str, str, str], SummaryRecord],
    token_count: int,
    dataset: str,
    direction: str,
    method: str,
) -> float:
    record = selected.get((token_count, dataset, direction, method))
    return record.weighted_gpu_tflops if record is not None else 0.0


def make_figure(
    selected: dict[tuple[int, str, str, str], SummaryRecord],
    world_size: int,
    token_counts: Sequence[int],
) -> plt.Figure:
    datasets = sorted(
        {dataset for _tokens, dataset, _direction, _method in selected},
        key=lambda name: (DATASET_ORDER.get(name, len(DATASET_ORDER)), name),
    )
    figure_width = max(15.5, 5.0 + 2.4 * len(datasets))
    fig, axes = plt.subplots(
        len(token_counts),
        2,
        figsize=(figure_width, 4.8 * len(token_counts) + 1.8),
        sharey="row",
        squeeze=False,
    )

    group_width = 0.84
    bar_width = group_width / len(METHODS)
    x_positions = list(range(len(datasets)))
    hybrid_method = "mega_ring_hybrid"
    hybrid_index = METHODS.index(hybrid_method)
    hybrid_offset = (hybrid_index - (len(METHODS) - 1) / 2) * bar_width
    for row, token_count in enumerate(token_counts):
        for column, direction in enumerate(("forward", "backward")):
            axis = axes[row, column]
            for method_index, method in enumerate(METHODS):
                offset = (method_index - (len(METHODS) - 1) / 2) * bar_width
                values = [
                    weighted_gpu_tflops_or_zero(
                        selected, token_count, dataset, direction, method
                    )
                    for dataset in datasets
                ]
                axis.bar(
                    [x + offset for x in x_positions],
                    values,
                    width=bar_width * 0.92,
                    color=METHOD_COLORS[method],
                    edgecolor="white",
                    linewidth=0.6,
                    label=METHOD_LABELS[method],
                )

            for x, dataset in zip(x_positions, datasets):
                hybrid_value = weighted_gpu_tflops_or_zero(
                    selected, token_count, dataset, direction, hybrid_method
                )
                best_baseline = max(
                    weighted_gpu_tflops_or_zero(
                        selected, token_count, dataset, direction, method
                    )
                    for method in HYBRID_COMPARISON_METHODS
                )
                speedup = (
                    f"{hybrid_value / best_baseline:.2f}x"
                    if best_baseline > 0.0
                    else "N/A"
                )
                axis.annotate(
                    speedup,
                    xy=(x + hybrid_offset, hybrid_value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color="#704E6F",
                    fontsize=8.5,
                    fontweight="bold",
                )

            axis.set_title(
                f"{token_count // 1024}K {direction.capitalize()}",
                loc="left",
                fontsize=13,
                fontweight="bold",
            )
            axis.set_xticks(
                x_positions,
                [DATASET_LABELS.get(dataset, dataset) for dataset in datasets],
            )
            axis.set_xlabel("Dataset")
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.8)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.margins(y=0.14)
        axes[row, 0].set_ylabel("Weighted average TFLOPS per GPU")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        f"Dataset-weighted Attention Throughput per GPU ({world_size} GPUs)",
        y=0.995,
        fontsize=16,
    )
    token_text = ", ".join(f"{token_count // 1024}K" for token_count in token_counts)
    fig.text(
        0.5,
        0.01,
        f"Rows show {token_text} workloads. Mega-ring uses the best per-GPU SM configuration; hybrid labels show speedup over the best baseline excluding mega_ring_all_cp.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    return fig


def print_selection(
    selected: dict[tuple[int, str, str, str], SummaryRecord],
    world_size: int,
    token_counts: Sequence[int],
) -> None:
    datasets = sorted(
        {dataset for _tokens, dataset, _direction, _method in selected},
        key=lambda name: (DATASET_ORDER.get(name, len(DATASET_ORDER)), name),
    )
    print(f"Selected weighted throughput (world_size={world_size})")
    print(
        f"{'Tokens':>8} {'Dataset':<10} {'Direction':<10} {'Method':<29} "
        f"{'SM':>7} {'Weighted TFLOPS/GPU':>20}"
    )
    for token_count in token_counts:
        for dataset in datasets:
            for direction in ("forward", "backward"):
                for method in METHODS:
                    record = selected.get((token_count, dataset, direction, method))
                    if record is None:
                        print(
                            f"{token_count:>8} {dataset:<10} {direction:<10} "
                            f"{METHOD_LABELS[method]:<29} {'-':>7} {0.0:>20.2f}"
                        )
                        continue
                    print(
                        f"{token_count:>8} {dataset:<10} {direction:<10} "
                        f"{METHOD_LABELS[method]:<29} {record.sm_config:>7} "
                        f"{record.weighted_gpu_tflops:>20.2f}"
                    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 64K, 128K, and 256K weighted forward/backward TFLOPS"
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
        "--token-count",
        type=int,
        action="append",
        default=None,
        help="token count to plot; repeat to select several (default: 65536, 131072, 262144)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN_DIR / "weighted_flops.png",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="GPU count to plot; required only when logs contain multiple counts",
    )
    parser.add_argument(
        "--forward-mode",
        choices=("causal", "noncausal"),
        default="causal",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    token_counts = tuple(args.token_count or DEFAULT_TOKEN_COUNTS)
    if not token_counts or any(token_count <= 0 for token_count in token_counts):
        raise ValueError("--token-count values must be positive")
    if len(set(token_counts)) != len(token_counts):
        raise ValueError("--token-count values must be unique")
    records = [
        record for record in load_summary(args.input)
        if record.token_count in token_counts
    ]
    world_size = choose_world_size(records, args.world_size)
    selected = select_best(records, world_size=world_size, mode=args.forward_mode)
    figure = make_figure(selected, world_size, token_counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print_selection(selected, world_size, token_counts)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
