"""Plot grouped sequence-length frequencies for the five MegaRing datasets.

Bins use (lower, upper] intervals and 1K = 1024 tokens. The final >64K
bin includes all remaining samples, including lengths clipped by the input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "sequence_length_buckets.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "dataset_length_distribution.pdf"
# First five METHOD_COLORS from benchmark_logs/bench_cp/plot_weighted_flops.py.
COLORS = ("#4C78A8", "#59A14F", "#9C755F", "#F28E2B", "#17A2B8")
HATCHES = ("--", "\\", "//", "..", "xx")
BIN_UPPER_EDGES = tuple(1024 * 2**index for index in range(7))
BIN_LABELS = ("≤1K", "1–2K", "2–4K", "4–8K", "8–16K", "16–32K", "32–64K", ">64K")
DATASET_ORDER = ("prolong", "arxiv", "freelaw", "github", "pile-cc")
DATASET_ALIASES = {"pile-cc": "pile"}
DISPLAY_NAMES = {
    "arxiv": "ArXiv",
    "github": "GitHub",
    "pile": "Pile-CC",
    "pile-cc": "Pile-CC",
    "freelaw": "FreeLaw",
    "prolong": "ProLong",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot sequence-length bucket frequencies as a grouped bar chart."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"input JSON file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output image file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def load_statistics(path: Path) -> tuple[int, int, dict[str, dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc

    bucket_size = payload.get("bucket_size")
    max_sequence_tokens = payload.get("max_sequence_tokens")
    datasets = payload.get("datasets")
    if not isinstance(bucket_size, int) or bucket_size <= 0:
        raise ValueError("bucket_size must be a positive integer")
    if not isinstance(max_sequence_tokens, int) or max_sequence_tokens <= 0:
        raise ValueError("max_sequence_tokens must be a positive integer")
    if max_sequence_tokens % bucket_size != 0:
        raise ValueError("max_sequence_tokens must be divisible by bucket_size")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("datasets must contain at least one dataset")

    expected_bins = max_sequence_tokens // bucket_size
    for name, dataset in datasets.items():
        if not isinstance(dataset, dict):
            raise ValueError(f"dataset {name!r} must be an object")
        counts = dataset.get("bucket_counts")
        sample_count = dataset.get("sample_count")
        if (
            not isinstance(counts, list)
            or len(counts) != expected_bins
            or any(not isinstance(count, int) or count < 0 for count in counts)
        ):
            raise ValueError(
                f"dataset {name!r} must have {expected_bins} non-negative "
                "integer bucket counts"
            )
        if (
            not isinstance(sample_count, int)
            or sample_count <= 0
            or sample_count != sum(counts)
        ):
            raise ValueError(
                f"dataset {name!r} sample_count must be positive and equal "
                "the sum of its bins"
            )

    return bucket_size, max_sequence_tokens, datasets


def grouped_counts(counts: list[int], bucket_size: int) -> list[int]:
    """Merge the existing fine bins without dropping the long-sequence tail."""
    if any(edge % bucket_size for edge in BIN_UPPER_EDGES):
        raise ValueError("input bucket size must divide every plotted bin edge")
    boundaries = [0, *(edge // bucket_size for edge in BIN_UPPER_EDGES), len(counts)]
    return [sum(counts[left:right]) for left, right in zip(boundaries, boundaries[1:])]


def plot_statistics(
    bucket_size: int,
    max_sequence_tokens: int,
    datasets: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    if max_sequence_tokens <= BIN_UPPER_EDGES[-1]:
        raise ValueError("input length limit must exceed 64K to preserve the tail bin")
    normalized_datasets = {
        name.lower(): (name, dataset) for name, dataset in datasets.items()
    }
    dataset_items = [
        normalized_datasets[DATASET_ALIASES.get(name, name)]
        for name in DATASET_ORDER
        if DATASET_ALIASES.get(name, name) in normalized_datasets
    ]
    ordered_dataset_names = {name for name, _dataset in dataset_items}
    dataset_items.extend(
        item for item in datasets.items() if item[0] not in ordered_dataset_names
    )
    figure, axis = plt.subplots(figsize=(11, 4.64), constrained_layout=True)
    bar_width = 0.8 / len(dataset_items)

    for index, (name, dataset) in enumerate(dataset_items):
        color = COLORS[index % len(COLORS)]
        counts = grouped_counts(dataset["bucket_counts"], bucket_size)
        sample_count = dataset["sample_count"]
        frequencies = [count / sample_count for count in counts]
        positions = [
            group + (index - (len(dataset_items) - 1) / 2) * bar_width
            for group in range(len(BIN_LABELS))
        ]
        axis.bar(
            positions,
            frequencies,
            width=bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            hatch=HATCHES[index % len(HATCHES)],
            label=DISPLAY_NAMES.get(name.lower(), name),
            zorder=3,
        )

    axis.set_xlim(-0.6, len(BIN_LABELS) - 0.4)
    axis.set_ylim(0, 1)
    axis.set_xticks(range(len(BIN_LABELS)), BIN_LABELS)
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axis.set_xlabel("Sequence Length (tokens)", fontsize=23)
    axis.set_ylabel("Sequence Frequency", fontsize=23)
    axis.tick_params(axis="both", labelsize=20)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        ncol=5, fontsize=20, frameon=True,
        columnspacing=1.0, handlelength=1.6, handletextpad=0.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    bucket_size, max_sequence_tokens, datasets = load_statistics(args.input)
    plot_statistics(bucket_size, max_sequence_tokens, datasets, args.output)
    print(f"Saved sequence-length bucket chart: {args.output}")


if __name__ == "__main__":
    main()
