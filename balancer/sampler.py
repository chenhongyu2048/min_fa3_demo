"""Dataset length sampling and ring-aware sequence alignment."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence


MAX_SEQUENCE_TOKENS = 128 * 1024
LENGTH_BUCKET_SIZE = 256
_BUCKET_STATS_PATH = (
    Path(__file__).resolve().parent.parent
    / "dataset"
    / "sequence_length_buckets.json"
)
_ALIGNMENT_TIERS = (
    (4 * 1024, 256 * 2),
    (8 * 1024, 256 * 4),
    (16 * 1024, 256 * 8),
)
_LARGE_SEQUENCE_ALIGNMENT = 256 * 8


def _load_bucket_counts(path: Path) -> dict[str, tuple[int, ...]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"failed to load dataset bucket statistics from {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"bucket statistics in {path} must be a JSON object")
    if payload.get("bucket_size") != LENGTH_BUCKET_SIZE:
        raise RuntimeError(
            f"bucket_size in {path} must be {LENGTH_BUCKET_SIZE}, "
            f"got {payload.get('bucket_size')!r}"
        )
    if payload.get("max_sequence_tokens") != MAX_SEQUENCE_TOKENS:
        raise RuntimeError(
            f"max_sequence_tokens in {path} must be {MAX_SEQUENCE_TOKENS}, "
            f"got {payload.get('max_sequence_tokens')!r}"
        )
    if payload.get("interval") != "(lower, upper]":
        raise RuntimeError(f"interval in {path} must be '(lower, upper]'")
    if payload.get("overflow_policy") != "clamp_to_last_bucket":
        raise RuntimeError(
            f"overflow_policy in {path} must be 'clamp_to_last_bucket'"
        )

    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise RuntimeError(f"datasets in {path} must be a non-empty JSON object")

    expected_bucket_count = MAX_SEQUENCE_TOKENS // LENGTH_BUCKET_SIZE
    result: dict[str, tuple[int, ...]] = {}
    for dataset, statistics in datasets.items():
        if not isinstance(dataset, str) or not dataset:
            raise RuntimeError(f"dataset names in {path} must be non-empty strings")
        if not isinstance(statistics, dict):
            raise RuntimeError(
                f"statistics for {dataset!r} in {path} must be an object"
            )
        sample_count = statistics.get("sample_count")
        counts = statistics.get("bucket_counts")
        if type(sample_count) is not int or sample_count <= 0:
            raise RuntimeError(
                f"sample_count for {dataset!r} in {path} must be a positive integer"
            )
        if not isinstance(counts, list) or len(counts) != expected_bucket_count:
            raise RuntimeError(
                f"bucket_counts for {dataset!r} in {path} must contain "
                f"{expected_bucket_count} entries"
            )
        if any(type(count) is not int or count < 0 for count in counts):
            raise RuntimeError(
                f"bucket_counts for {dataset!r} in {path} must be non-negative integers"
            )
        if sum(counts) != sample_count:
            raise RuntimeError(
                f"bucket_counts for {dataset!r} in {path} sum to {sum(counts)}, "
                f"expected sample_count={sample_count}"
            )
        result[dataset] = tuple(counts)
    return result


LENGTH_BUCKETS = tuple(
    range(LENGTH_BUCKET_SIZE, MAX_SEQUENCE_TOKENS + 1, LENGTH_BUCKET_SIZE)
)
_DATASET_BUCKET_COUNTS = _load_bucket_counts(_BUCKET_STATS_PATH)
DATASET_WEIGHTS = {
    dataset: tuple(count / sum(counts) for count in counts)
    for dataset, counts in _DATASET_BUCKET_COUNTS.items()
}


def _weighted_bucket(rng: random.Random, counts: Sequence[int]) -> int:
    draw = rng.randrange(sum(counts))
    cumulative = 0
    for idx, count in enumerate(counts):
        cumulative += count
        if draw < cumulative:
            return idx
    raise RuntimeError("bucket counts must contain at least one sample")


def _alignment_for_length(length: int) -> int:
    for upper_bound, alignment in _ALIGNMENT_TIERS:
        if length < upper_bound:
            return alignment
    return _LARGE_SEQUENCE_ALIGNMENT


def _align_sequence_length(length: int) -> int:
    alignment = _alignment_for_length(length)
    return min(
        MAX_SEQUENCE_TOKENS,
        ((length + alignment - 1) // alignment) * alignment,
    )


def generate_dataset_lengths(
    dataset: str,
    target_tokens: int,
    seed: int,
) -> list[int]:
    """Sample and pack aligned sequence lengths up to a target token budget."""

    if dataset not in _DATASET_BUCKET_COUNTS:
        raise ValueError(f"unknown dataset {dataset!r}")
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")
    counts = _DATASET_BUCKET_COUNTS[dataset]

    rng = random.Random(seed)
    lengths: list[int] = []
    remaining = target_tokens
    while remaining > 0:
        sampled = LENGTH_BUCKETS[_weighted_bucket(rng, counts)]
        length = min(sampled, remaining, MAX_SEQUENCE_TOKENS)
        length = _align_sequence_length(length)
        lengths.append(length)
        remaining = max(remaining - length, 0)
    return lengths
