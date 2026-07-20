#!/usr/bin/env python3
"""Plot dataset-weighted forward/backward throughput from benchmark logs."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


LOG_DIR = Path(__file__).resolve().parent
DEFAULT_FORWARD_LOG = LOG_DIR / "20260720-105750" / "benchmark_dataset.log"
DEFAULT_BACKWARD_LOG = (LOG_DIR / "20260720-110739" / "benchmark_dataset_backward.log")

METHODS = (
    "allgather_attention",
    "llama3_allgather_attention",
    "fa3_ring",
    "zepplin",
    "mega_ring_all_cp",
    "mega_ring_hybrid",
)
TUNED_METHODS = frozenset(("mega_ring_all_cp", "mega_ring_hybrid"))
METHOD_LABELS = {
    "allgather_attention": "allgather_attention",
    "llama3_allgather_attention": "llama3_allgather_attention",
    "fa3_ring": "ring_flash_attention",
    "zepplin": "zepplin",
    "mega_ring_all_cp": "mega_ring_all_cp",
    "mega_ring_hybrid": "mega_ring_hybrid",
}
METHOD_COLORS = {
    "allgather_attention": "#4C78A8",
    "llama3_allgather_attention": "#59A14F",
    "fa3_ring": "#9C755F",
    "zepplin": "#ECA82C",
    "mega_ring_all_cp": "#E15759",
    "mega_ring_hybrid": "#B07AA1",
}
DATASET_LABELS = {
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
    dataset: str
    direction: str
    mode: str
    world_size: int
    method: str
    sm_config: str
    weighted_tflops: float
    weighted_gpu_tflops: float
    source: Path


def parse_summary_row(
    line: str,
    *,
    dataset: str,
    direction: str,
    world_size: int,
    source: Path,
) -> SummaryRecord | None:
    fields = line.split()
    if not fields or fields[0] not in METHODS:
        return None

    prefix_length = 4 if direction == "forward" else 3
    if len(fields) != prefix_length + 7:
        raise ValueError(f"malformed {direction} summary row in {source}: {line}")

    method = fields[0]
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


def load_direction_records(path: Path, direction: str) -> list[SummaryRecord]:
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
    return records


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
) -> dict[tuple[str, str, str], SummaryRecord]:
    grouped: dict[tuple[str, str, str], list[SummaryRecord]] = defaultdict(list)
    exact_keys: dict[tuple[str, str, str, str, str], SummaryRecord] = {}

    for record in records:
        if record.world_size != world_size:
            continue
        if record.direction == "forward" and record.mode != mode:
            continue
        key = (
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
        grouped[(record.dataset, record.direction, record.method)].append(record)

    selected: dict[tuple[str, str, str], SummaryRecord] = {}
    for key, candidates in grouped.items():
        method = key[2]
        if method in TUNED_METHODS:
            selected[key] = max(candidates, key=lambda item: item.weighted_gpu_tflops)
        elif len(candidates) == 1:
            selected[key] = candidates[0]
        else:
            sources = ", ".join(str(item.source) for item in candidates)
            raise ValueError(f"multiple untuned baseline rows for {key}: {sources}")

    datasets = sorted(
        {dataset for dataset, _direction, _method in selected},
        key=lambda name: (DATASET_ORDER.get(name, len(DATASET_ORDER)), name),
    )
    missing = [
        f"{dataset}/{direction}/{method}"
        for dataset in datasets
        for direction in ("forward", "backward")
        for method in METHODS
        if (dataset, direction, method) not in selected
    ]
    if missing:
        raise ValueError("missing required summary rows: " + ", ".join(missing))
    return selected


def make_figure(
    selected: dict[tuple[str, str, str], SummaryRecord], world_size: int
) -> plt.Figure:
    datasets = sorted(
        {dataset for dataset, _direction, _method in selected},
        key=lambda name: (DATASET_ORDER.get(name, len(DATASET_ORDER)), name),
    )
    figure_width = max(14.0, 5.0 + 2.2 * len(datasets))
    fig, axes = plt.subplots(1, 2, figsize=(figure_width, 6.2), sharey=True)

    group_width = 0.84
    bar_width = group_width / len(METHODS)
    x_positions = list(range(len(datasets)))
    for ax, direction in zip(axes, ("forward", "backward")):
        for method_index, method in enumerate(METHODS):
            offset = (method_index - (len(METHODS) - 1) / 2) * bar_width
            values = [
                selected[(dataset, direction, method)].weighted_gpu_tflops
                for dataset in datasets
            ]
            ax.bar(
                [x + offset for x in x_positions],
                values,
                width=bar_width * 0.92,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.6,
                label=METHOD_LABELS[method],
            )
        ax.set_title(direction.capitalize(), fontsize=14)
        ax.set_xticks(
            x_positions,
            [DATASET_LABELS.get(dataset, dataset) for dataset in datasets],
        )
        ax.set_xlabel("Dataset")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Weighted aggregate TFLOPS")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        f"Dataset-weighted Attention Throughput ({world_size} GPUs)",
        y=1.055,
        fontsize=16,
    )
    fig.text(
        0.5,
        0.01,
        "Mega-ring methods use the best Weighted TFLOPS/GPU SM configuration for each dataset and direction.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.89))
    return fig


def print_selection(
    selected: dict[tuple[str, str, str], SummaryRecord], world_size: int
) -> None:
    datasets = sorted(
        {dataset for dataset, _direction, _method in selected},
        key=lambda name: (DATASET_ORDER.get(name, len(DATASET_ORDER)), name),
    )
    print(f"Selected weighted throughput (world_size={world_size})")
    print(
        f"{'Dataset':<10} {'Direction':<10} {'Method':<29} "
        f"{'SM':>7} {'Weighted TFLOPS':>17} {'Weighted/GPU':>14}"
    )
    for dataset in datasets:
        for direction in ("forward", "backward"):
            for method in METHODS:
                record = selected[(dataset, direction, method)]
                print(
                    f"{dataset:<10} {direction:<10} {METHOD_LABELS[method]:<29} "
                    f"{record.sm_config:>7} {record.weighted_gpu_tflops:>17.2f} "
                    f"{record.weighted_gpu_tflops:>14.2f}"
                )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot weighted forward/backward TFLOPS from the specified benchmark logs"
        )
    )
    parser.add_argument(
        "--forward-log",
        type=Path,
        default=DEFAULT_FORWARD_LOG,
        help=f"forward benchmark log (default: {DEFAULT_FORWARD_LOG})",
    )
    parser.add_argument(
        "--backward-log",
        type=Path,
        default=DEFAULT_BACKWARD_LOG,
        help=f"backward benchmark log (default: {DEFAULT_BACKWARD_LOG})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LOG_DIR / "weighted_flops.png",
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
    records = [
        *load_direction_records(args.forward_log, "forward"),
        *load_direction_records(args.backward_log, "backward"),
    ]
    world_size = choose_world_size(records, args.world_size)
    selected = select_best(records, world_size=world_size, mode=args.forward_mode)
    figure = make_figure(selected, world_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print_selection(selected, world_size)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
