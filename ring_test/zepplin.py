"""Deterministic placement helpers for the hybrid ``zepplin`` baseline."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_ZEPPLIN_THRESHOLD = 4096


def zepplin_attention_weight(length: int, is_causal: bool) -> int:
    """Return the exact score count used by the LPT placement heuristic."""
    if length <= 0:
        raise ValueError("sequence lengths must be positive")
    if is_causal:
        return length * (length + 1) // 2
    return length * length


def zepplin_incompatibility(
    global_lengths: list[int],
    world_size: int,
    is_causal: bool,
    threshold: int,
) -> str | None:
    if threshold <= 0:
        return f"zepplin threshold must be positive, got {threshold}"
    if world_size <= 0:
        return f"world_size must be positive, got {world_size}"
    for batch_idx, global_len in enumerate(global_lengths):
        if global_len <= 0:
            return (
                "zepplin requires positive global lengths: "
                f"batch={batch_idx}, global_len={global_len}"
            )
        if global_len < threshold:
            continue
        if global_len % world_size:
            return (
                "zepplin Gworld sequences must be divisible by world_size: "
                f"batch={batch_idx}, global_len={global_len}, "
                f"world_size={world_size}"
            )
        local_len = global_len // world_size
        if is_causal and local_len % 2:
            return (
                "causal zepplin Gworld sequences require even rank-local shards: "
                f"batch={batch_idx}, local_len={local_len}"
            )
    return None


@dataclass(frozen=True)
class ZepplinPlan:
    """G1/Gworld placement and rank-local packed-layout metadata."""

    global_lengths: tuple[int, ...]
    world_size: int
    is_causal: bool
    threshold: int
    short_indices: tuple[int, ...]
    long_indices: tuple[int, ...]
    short_owners: tuple[int, ...]
    short_loads: tuple[int, ...]

    @property
    def packed_global_lengths(self) -> list[int]:
        return [
            *(self.global_lengths[idx] for idx in self.short_indices),
            *(self.global_lengths[idx] for idx in self.long_indices),
        ]

    @property
    def ring_sizes(self) -> list[int]:
        return [1] * len(self.short_indices) + [self.world_size] * len(
            self.long_indices
        )

    @property
    def ring_starts(self) -> list[int]:
        return [*self.short_owners, *([0] * len(self.long_indices))]

    def short_lengths_for_rank(self, rank: int) -> list[int]:
        self._validate_rank(rank)
        return [
            self.global_lengths[batch_idx]
            for batch_idx, owner in zip(self.short_indices, self.short_owners)
            if owner == rank
        ]

    def long_local_lengths(self) -> list[int]:
        return [
            self.global_lengths[batch_idx] // self.world_size
            for batch_idx in self.long_indices
        ]

    def topology_lengths_for_rank(self, rank: int) -> list[int]:
        """Lengths aligned with the packed G1/Gworld topology, including zeros."""
        self._validate_rank(rank)
        return [
            *(
                self.global_lengths[batch_idx] if owner == rank else 0
                for batch_idx, owner in zip(self.short_indices, self.short_owners)
            ),
            *self.long_local_lengths(),
        ]

    def packed_lengths_for_rank(self, rank: int) -> list[int]:
        """Physical local layout: owned short sequences, then all long shards."""
        return [*self.short_lengths_for_rank(rank), *self.long_local_lengths()]

    def _validate_rank(self, rank: int) -> None:
        if not 0 <= rank < self.world_size:
            raise ValueError(
                f"rank must be in [0, {self.world_size}), got {rank}"
            )


def make_zepplin_plan(
    global_lengths: list[int],
    world_size: int,
    is_causal: bool,
    threshold: int = DEFAULT_ZEPPLIN_THRESHOLD,
) -> ZepplinPlan:
    reason = zepplin_incompatibility(
        global_lengths, world_size, is_causal, threshold
    )
    if reason is not None:
        raise ValueError(reason)

    short_indices = tuple(
        idx for idx, length in enumerate(global_lengths) if length < threshold
    )
    long_indices = tuple(
        idx for idx, length in enumerate(global_lengths) if length >= threshold
    )
    short_loads = [0] * world_size
    owner_by_index: dict[int, int] = {}
    work = sorted(
        (
            (zepplin_attention_weight(global_lengths[idx], is_causal), idx)
            for idx in short_indices
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for weight, batch_idx in work:
        owner = min(range(world_size), key=lambda rank: (short_loads[rank], rank))
        owner_by_index[batch_idx] = owner
        short_loads[owner] += weight

    return ZepplinPlan(
        global_lengths=tuple(global_lengths),
        world_size=world_size,
        is_causal=is_causal,
        threshold=threshold,
        short_indices=short_indices,
        long_indices=long_indices,
        short_owners=tuple(owner_by_index[idx] for idx in short_indices),
        short_loads=tuple(short_loads),
    )


def zepplin_note(plan: ZepplinPlan, backend_name: str) -> str:
    loads = ",".join(str(load) for load in plan.short_loads)
    return (
        f"threshold={plan.threshold}; G1={len(plan.short_indices)}, "
        f"Gworld={len(plan.long_indices)}; G1 loads=[{loads}]; {backend_name}"
    )


__all__ = [
    "DEFAULT_ZEPPLIN_THRESHOLD",
    "ZepplinPlan",
    "make_zepplin_plan",
    "zepplin_attention_weight",
    "zepplin_incompatibility",
    "zepplin_note",
]
