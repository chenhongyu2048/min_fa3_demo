"""Dataset length sampling and ring-aware sequence alignment."""

from __future__ import annotations

import random
from typing import Sequence


MAX_SEQUENCE_TOKENS = 128 * 1024
LENGTH_BUCKETS = (
    512,
    1536,
    3072,
    6144,
    12288,
    24576,
    49152,
    98304,
    MAX_SEQUENCE_TOKENS,
)
DATASET_WEIGHTS = {
    "arxiv": (0.032, 0.030, 0.080, 0.219, 0.338, 0.224, 0.077, 0.0, 0.0),
    # The supplied bins total 0.945. The remaining 0.055 is the >256K tail;
    # together with the 0.045 128-256K bin it is clamped to 128K.
    "github": (0.0, 0.340, 0.095, 0.104, 0.107, 0.102, 0.088, 0.064, 0.100),
}
_ALIGNMENT_TIERS = (
    (4 * 1024, 256 * 2),
    (8 * 1024, 256 * 4),
    (16 * 1024, 256 * 8),
)
_LARGE_SEQUENCE_ALIGNMENT = 256 * 8


def _weighted_bucket(rng: random.Random, weights: Sequence[float]) -> int:
    draw = rng.random() * sum(weights)
    cumulative = 0.0
    for idx, weight in enumerate(weights):
        cumulative += weight
        if draw < cumulative:
            return idx
    return len(weights) - 1


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

    if dataset not in DATASET_WEIGHTS:
        raise ValueError(f"unknown dataset {dataset!r}")
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")
    weights = DATASET_WEIGHTS[dataset]
    if abs(sum(weights) - 1.0) > 1e-9:
        raise RuntimeError(f"{dataset} weights must sum to 1, got {sum(weights)}")

    rng = random.Random(seed)
    lengths: list[int] = []
    remaining = target_tokens
    while remaining > 0:
        sampled = LENGTH_BUCKETS[_weighted_bucket(rng, weights)]
        length = min(sampled, remaining, MAX_SEQUENCE_TOKENS)
        length = _align_sequence_length(length)
        lengths.append(length)
        remaining = max(remaining - length, 0)
    return lengths
