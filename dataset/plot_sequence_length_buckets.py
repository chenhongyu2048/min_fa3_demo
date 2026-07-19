"""Plot per-bin sequence-length frequencies for the sampled datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "sequence_length_buckets.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "sequence_length_buckets_frequency.png"
COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
DISPLAY_NAMES = {
    "arxiv": "ArXiv",
    "github": "GitHub",
    "pile": "Pile",
    "freelaw": "FreeLaw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot sequence-length bucket frequencies as vertically "
            "stacked bar charts."
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


def format_tokens(value: float, _position: float) -> str:
    if value == 0:
        return "0"
    return f"{value / 1024:g}K"


def plot_statistics(
    bucket_size: int,
    max_sequence_tokens: int,
    datasets: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    dataset_items = list(datasets.items())
    figure, axes = plt.subplots(
        nrows=len(dataset_items),
        ncols=1,
        figsize=(15, 3 * len(dataset_items) + 1),
        sharex=True,
        constrained_layout=True,
    )
    if len(dataset_items) == 1:
        axes = [axes]
    bin_left_edges = range(0, max_sequence_tokens, bucket_size)

    for index, (axis, (name, dataset)) in enumerate(zip(axes, dataset_items)):
        color = COLORS[index % len(COLORS)]
        counts = dataset["bucket_counts"]
        sample_count = dataset["sample_count"]
        frequencies = [count / sample_count for count in counts]
        axis.bar(
            bin_left_edges,
            frequencies,
            width=bucket_size,
            align="edge",
            color=color,
            linewidth=0,
        )
        axis.set_xlim(0, max_sequence_tokens)
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axis.set_ylabel("Frequency")
        axis.set_title(
            f"{DISPLAY_NAMES.get(name.lower(), name)} "
            f"(n={sample_count:,})",
            loc="left",
        )
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.set_axisbelow(True)

    major_tick_spacing = max(16 * 1024, bucket_size)
    axes[-1].xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    axes[-1].xaxis.set_major_formatter(FuncFormatter(format_tokens))
    axes[-1].set_xlabel(
        f"Sequence length (tokens; one bar per {bucket_size}-token bin)"
    )
    figure.suptitle("Sequence length distribution by dataset", fontsize=15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    bucket_size, max_sequence_tokens, datasets = load_statistics(args.input)
    plot_statistics(bucket_size, max_sequence_tokens, datasets, args.output)
    print(f"Saved sequence-length bucket chart: {args.output}")


if __name__ == "__main__":
    main()
