#!/usr/bin/env python3
"""Plot recorded load-balancing-suite throughput by dataset and direction."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "20260726-002324"
DEFAULT_INPUT = DEFAULT_RUN_DIR / "load_balance_algorithms_summary.csv"
DEFAULT_FORWARD_LOG = DEFAULT_RUN_DIR / "load_balance_algorithms_131072_forward.results.log"
DEFAULT_BACKWARD_LOG = DEFAULT_RUN_DIR / "load_balance_algorithms_131072_backward.results.log"
DEFAULT_OUTPUT = DEFAULT_RUN_DIR / "load_balance_algorithms_131072.png"

METHOD_ORDER = (
    "native_megatron_hybrid_cp",
    "native_zeppelin",
    "mega_ring_hybrid_br_pbs",
    "mega_ring_hybrid_megatron_cp",
    "mega_ring_hybrid_zeppelin",
)
METHOD_ALIASES = {
    "native_zepplin": "native_zeppelin",
    "mega_ring_hybrid_zepplin": "mega_ring_hybrid_zeppelin",
}
METHOD_LABELS = {
    "native_megatron_hybrid_cp": "Megatron",
    "native_zeppelin": "Zeppelin",
    "mega_ring_hybrid_br_pbs": "Mega Ring\nBR-PBS",
    "mega_ring_hybrid_megatron_cp": "Mega Ring\nMegatron-CP",
    "mega_ring_hybrid_zeppelin": "Mega Ring\nZeppelin",
}
METHOD_COLORS = {
    "native_megatron_hybrid_cp": "#4C78A8",
    "native_zeppelin": "#F28E2B",
    "mega_ring_hybrid_br_pbs": "#54A24B",
    "mega_ring_hybrid_megatron_cp": "#E15759",
    "mega_ring_hybrid_zeppelin": "#B279A2",
}
DATASET_LABELS = {
    "prolong": "ProLong",
    "arxiv": "ArXiv",
    "freelaw": "FreeLaw",
    "github": "GitHub",
    "pile": "Pile",
}
DATASET_ORDER = {dataset: index for index, dataset in enumerate(DATASET_LABELS)}

SECTION_RE = re.compile(
    r"^\[load_balance_(?P<direction>forward|backward)\]\s+"
    r"dataset=(?P<dataset>\S+)\s+GPUs=(?P<world_size>\d+)"
)
SUMMARY_RE = re.compile(r"^Cross-case (?P<direction>forward|backward) summary$")


@dataclass(frozen=True)
class SummaryRecord:
    dataset: str
    direction: str
    world_size: int
    method: str
    sm_config: str
    weighted_tflops: float
    weighted_gpu_tflops: float


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def load_summary(path: Path) -> list[SummaryRecord]:
    required = {
        "dataset", "direction", "world_size", "method", "sm_config",
        "weighted_tflops", "weighted_gpu_tflops",
    }
    records: list[SummaryRecord] = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                record = SummaryRecord(
                    dataset=row["dataset"].strip(),
                    direction=row["direction"].strip(),
                    world_size=int(row["world_size"]),
                    method=canonical_method(row["method"].strip()),
                    sm_config=row["sm_config"].strip(),
                    weighted_tflops=float(row["weighted_tflops"]),
                    weighted_gpu_tflops=float(row["weighted_gpu_tflops"]),
                )
            except ValueError as error:
                raise ValueError(f"invalid summary row at {path}:{line_number}") from error
            if record.direction not in {"forward", "backward"} or record.method not in METHOD_ORDER:
                raise ValueError(f"invalid summary key at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no summary rows found")
    return records


def parse_log(path: Path, expected_direction: str) -> list[SummaryRecord]:
    """Extract the per-dataset five-method cross-case summary rows."""

    if not path.is_file():
        raise FileNotFoundError(f"{expected_direction} log does not exist: {path}")

    records: list[SummaryRecord] = []
    dataset: str | None = None
    world_size: int | None = None
    in_summary = False
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        section_match = SECTION_RE.match(line)
        if section_match is not None:
            section_direction = section_match.group("direction")
            if section_direction != expected_direction:
                raise ValueError(
                    f"expected {expected_direction} section, found {section_direction} at "
                    f"{path}:{line_number}"
                )
            dataset = section_match.group("dataset")
            world_size = int(section_match.group("world_size"))
            in_summary = False
            continue

        summary_match = SUMMARY_RE.match(line)
        if summary_match is not None:
            summary_direction = summary_match.group("direction")
            if summary_direction != expected_direction:
                raise ValueError(
                    f"expected {expected_direction} summary, found {summary_direction} at "
                    f"{path}:{line_number}"
                )
            if dataset is None or world_size is None:
                raise ValueError(f"summary lacks dataset metadata at {path}:{line_number}")
            in_summary = True
            continue

        if not in_summary:
            continue
        fields = line.split()
        if not fields:
            continue
        method = canonical_method(fields[0])
        if method not in METHOD_ORDER:
            continue
        if dataset is None or world_size is None:
            raise AssertionError("summary context was lost")
        expected_field_count = 11 if expected_direction == "forward" else 10
        if len(fields) != expected_field_count:
            raise ValueError(f"malformed summary row at {path}:{line_number}: {line}")
        sm_index = 2 if expected_direction == "forward" else 1
        case_index = sm_index + 1
        if not re.fullmatch(r"\d+:\d+|-", fields[sm_index]) or not re.fullmatch(
            r"\d+/\d+", fields[case_index]
        ):
            raise ValueError(f"malformed summary metadata at {path}:{line_number}: {line}")
        try:
            metrics = [float(value) for value in fields[case_index + 1 :]]
        except ValueError as exc:
            raise ValueError(f"non-numeric summary row at {path}:{line_number}: {line}") from exc
        if len(metrics) != 7:
            raise ValueError(f"malformed summary metrics at {path}:{line_number}: {line}")
        records.append(
            SummaryRecord(
                dataset=dataset,
                direction=expected_direction,
                world_size=world_size,
                method=method,
                sm_config=fields[sm_index],
                weighted_tflops=metrics[-2],
                weighted_gpu_tflops=metrics[-1],
            )
        )

    if not records:
        raise ValueError(f"no {expected_direction} five-method summaries found in {path}")
    return records


def select_best_records(records: Iterable[SummaryRecord]) -> dict[tuple[str, str], SummaryRecord]:
    """Select the highest-throughput SM sweep result for every dataset/method."""

    grouped: dict[tuple[str, str], list[SummaryRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.dataset, record.method)].append(record)

    selected: dict[tuple[str, str], SummaryRecord] = {}
    for key, candidates in grouped.items():
        selected[key] = max(candidates, key=lambda record: record.weighted_gpu_tflops)
    return selected


def _ordered_datasets(selected: dict[tuple[str, str], SummaryRecord]) -> list[str]:
    return sorted(
        {dataset for dataset, _method in selected},
        key=lambda dataset: (DATASET_ORDER.get(dataset, len(DATASET_ORDER)), dataset),
    )


def _validate_selection(
    forward: dict[tuple[str, str], SummaryRecord],
    backward: dict[tuple[str, str], SummaryRecord]
) -> tuple[list[str], int]:
    forward_datasets = _ordered_datasets(forward)
    backward_datasets = _ordered_datasets(backward)
    if forward_datasets != backward_datasets:
        raise ValueError(
            "forward/backward datasets differ: "
            f"forward={forward_datasets}, backward={backward_datasets}"
        )
    for direction, selected in (("forward", forward), ("backward", backward)):
        missing = [
            f"{dataset}/{method}"
            for dataset in forward_datasets
            for method in METHOD_ORDER
            if (dataset, method) not in selected
        ]
        if missing:
            raise ValueError(f"{direction} summary is missing: {', '.join(missing)}")
    world_sizes = {record.world_size for record in [*forward.values(), *backward.values()]}
    if len(world_sizes) != 1:
        raise ValueError(f"expected one GPU count, found {sorted(world_sizes)}")
    return forward_datasets, world_sizes.pop()


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.85)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.margins(y=0.14)


def make_figure(
    forward: dict[tuple[str, str], SummaryRecord],
    backward: dict[tuple[str, str], SummaryRecord],
) -> plt.Figure:
    datasets, world_size = _validate_selection(forward, backward)
    positions = list(range(len(datasets)))
    bar_width = 0.82 / len(METHOD_ORDER)
    figure, axes = plt.subplots(1, 2, figsize=(17.5, 6.5), sharey=True)

    for axis, title, selected in zip(axes, ("Forward", "Backward"), (forward, backward)):
        for method_index, method in enumerate(METHOD_ORDER):
            offset = (method_index - (len(METHOD_ORDER) - 1) / 2) * bar_width
            values = [selected[(dataset, method)].weighted_gpu_tflops for dataset in datasets]
            axis.bar(
                [position + offset for position in positions],
                values,
                width=bar_width * 0.92,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.7,
                label=METHOD_LABELS[method],
            )
        _style_axis(axis)
        axis.set_title(title, loc="left", fontsize=13, fontweight="bold")
        axis.set_xticks(positions)
        axis.set_xticklabels([DATASET_LABELS.get(dataset, dataset) for dataset in datasets])
        axis.set_xlabel("Dataset")

    axes[0].set_ylabel("Weighted average TFLOPS per GPU")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        fontsize=9.3,
    )
    figure.suptitle(
        f"128K Load-Balancing Algorithm Comparison ({world_size} GPUs)",
        y=1.055,
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Each Mega Ring placement uses its highest per-GPU throughput SM configuration from the recorded sweep.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.89))
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    records = load_summary(args.input)
    forward = select_best_records(record for record in records if record.direction == "forward")
    backward = select_best_records(record for record in records if record.direction == "backward")
    figure = make_figure(forward, backward)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
