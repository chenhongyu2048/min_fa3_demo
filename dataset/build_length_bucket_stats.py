"""Build shared 256-token bucket statistics from sampled document lengths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


BUCKET_SIZE = 256
MAX_SEQUENCE_TOKENS = 128 * 1024
DATASET_NAMES = ("arxiv", "github", "pile")
TOKENIZER_NAME = "gpt2"
OUTPUT_NAME = "sequence_length_buckets.json"


def bucket_counts(lengths: Sequence[int] | np.ndarray) -> list[int]:
    """Count lengths in (lower, upper] buckets, clamping overflow to 128K."""

    values = np.asarray(lengths)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("lengths must be a non-empty one-dimensional array")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("lengths must contain integers")
    if np.any(values <= 0):
        raise ValueError("lengths must contain only positive integers")

    clipped = np.minimum(values, MAX_SEQUENCE_TOKENS)
    bucket_indices = (clipped - 1) // BUCKET_SIZE
    counts = np.bincount(
        bucket_indices,
        minlength=MAX_SEQUENCE_TOKENS // BUCKET_SIZE,
    )
    return [int(count) for count in counts]


def build_statistics(dataset_dir: Path) -> dict[str, object]:
    datasets: dict[str, object] = {}
    for dataset in DATASET_NAMES:
        lengths_path = dataset_dir / f"{dataset}_doc_lengths.npy"
        try:
            lengths = np.load(lengths_path, allow_pickle=False)
        except OSError as exc:
            raise RuntimeError(
                f"failed to load sampled lengths from {lengths_path}"
            ) from exc
        counts = bucket_counts(lengths)
        datasets[dataset] = {
            "sample_count": int(lengths.size),
            "bucket_counts": counts,
        }

    return {
        "bucket_size": BUCKET_SIZE,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "interval": "(lower, upper]",
        "overflow_policy": "clamp_to_last_bucket",
        "tokenizer": TOKENIZER_NAME,
        "datasets": datasets,
    }


def main() -> None:
    dataset_dir = Path(__file__).resolve().parent
    output_path = dataset_dir / OUTPUT_NAME
    statistics = build_statistics(dataset_dir)
    output_path.write_text(
        json.dumps(statistics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved 256-token bucket statistics: {output_path}")


if __name__ == "__main__":
    main()
