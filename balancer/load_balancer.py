"""Buddy-Ring Pareto Beam Scheduling for dataset workloads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .sampler import generate_dataset_length_cases


RING_SIZES = (1, 2, 4, 8)
LENGTH_BUCKET_LABELS = ("<=2K", "2K-4K", "4K-8K", "8K-16K", ">16K")

_LENGTH_BUCKET_UPPER_BOUNDS = (2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024)
_LOAD_QUANTIZATION = 0.02
_SMOOTHMAX_LAMBDA = 8.0
_REPAIR_RANK_COUNT = 2
_REPAIR_SEQUENCE_COUNT = 4
_EPSILON = 1e-12


@dataclass(frozen=True)
class SequencePlacement:
    original_index: int
    length: int
    compute: int
    ring_size: int
    ring_start: int


@dataclass(frozen=True)
class HybridWorkload:
    global_lengths: list[int]
    ring_sizes: list[int]
    ring_starts: list[int]
    rank_compute: list[float]
    rank_tokens: list[int]
    rank_communication: list[float]
    average_compute: float
    peak_compute: float
    average_tokens: float
    peak_tokens: int
    compute_balance_tolerance: float
    token_balance_tolerance: float
    load_violation: float
    feasible: bool
    relaxation_level: int
    relaxation_label: str
    split_counts: list[int]
    split_penalties: list[int]
    communication_cost: float
    active_ring_count: int
    repair_moves: int

    @property
    def compute_deviation(self) -> float:
        return _max_relative_deviation(self.rank_compute, self.average_compute)

    @property
    def token_deviation(self) -> float:
        return _max_relative_deviation(self.rank_tokens, self.average_tokens)

    @property
    def communication_amplification(self) -> float:
        return sum(self.rank_communication) / sum(self.global_lengths)


@dataclass(frozen=True)
class _Job:
    original_index: int
    length: int
    compute: int
    legal_sizes: tuple[int, ...]
    candidates: tuple[SequencePlacement, ...]
    bucket: int
    kappa: float
    minimum_ring_size: int


@dataclass(frozen=True)
class _BeamState:
    rank_compute: tuple[float, ...]
    rank_tokens: tuple[int, ...]
    split_counts: tuple[int, ...]
    split_penalties: tuple[int, ...]
    communication_cost: float
    active_rings: frozenset[tuple[int, int]]
    parent: _BeamState | None
    placement: SequencePlacement | None


@dataclass(frozen=True)
class _Solution:
    placements: tuple[SequencePlacement, ...]
    rank_compute: tuple[float, ...]
    rank_tokens: tuple[int, ...]
    rank_communication: tuple[float, ...]
    split_counts: tuple[int, ...]
    split_penalties: tuple[int, ...]
    communication_cost: float
    active_ring_count: int
    compute_deviation: float
    token_deviation: float
    violation: float
    objective: tuple[float | int, ...]


def attention_compute(length: int, is_causal: bool) -> int:
    if is_causal:
        return length * (length + 1) // 2
    return length * length


def ring_communication_per_rank(length: int, ring_size: int) -> float:
    if ring_size == 1:
        return 0.0
    return (ring_size - 1) * length / ring_size


def eligible_ring_sizes(length: int, world_size: int, is_causal: bool) -> list[int]:
    eligible: list[int] = []
    for ring_size in RING_SIZES:
        if ring_size > world_size:
            continue
        if is_causal:
            if ring_size == 1 or length % (256 * ring_size) == 0:
                eligible.append(ring_size)
        elif length % ring_size == 0:
            eligible.append(ring_size)
    return eligible


def _length_bucket(length: int) -> int:
    for bucket, upper_bound in enumerate(_LENGTH_BUCKET_UPPER_BOUNDS):
        if length <= upper_bound:
            return bucket
    return len(_LENGTH_BUCKET_UPPER_BOUNDS)


def _max_relative_deviation(values: Sequence[float], average: float) -> float:
    return max(abs(value / average - 1.0) for value in values)


def _split_key(
    split_counts: Sequence[int], split_penalties: Sequence[int]
) -> tuple[int, ...]:
    return tuple(
        value
        for pair in zip(split_counts, split_penalties)
        for value in pair
    )


def _placement_key(placements: Sequence[SequencePlacement]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(placements, key=lambda placement: placement.original_index)
    return tuple((placement.ring_size, placement.ring_start) for placement in ordered)


def _candidate_key(candidate: SequencePlacement) -> tuple[int, int]:
    return candidate.ring_size, candidate.ring_start


def _add_to_beam_state(
    state: _BeamState,
    job: _Job,
    placement: SequencePlacement,
) -> _BeamState:
    rank_compute = list(state.rank_compute)
    rank_tokens = list(state.rank_tokens)
    compute_increment = placement.compute / placement.ring_size
    token_increment = placement.length // placement.ring_size
    for rank in range(placement.ring_start, placement.ring_start + placement.ring_size):
        rank_compute[rank] += compute_increment
        rank_tokens[rank] += token_increment

    split_counts = list(state.split_counts)
    split_penalties = list(state.split_penalties)
    if placement.ring_size > 1:
        split_counts[job.bucket] += 1
        split_penalties[job.bucket] += int(math.log2(placement.ring_size))

    return _BeamState(
        rank_compute=tuple(rank_compute),
        rank_tokens=tuple(rank_tokens),
        split_counts=tuple(split_counts),
        split_penalties=tuple(split_penalties),
        communication_cost=(
            state.communication_cost
            + ring_communication_per_rank(placement.length, placement.ring_size)
        ),
        active_rings=(
            state.active_rings | {(placement.ring_size, placement.ring_start)}
        ),
        parent=state,
        placement=placement,
    )


def _prefix_metrics(
    state: _BeamState,
    average_compute: float,
    average_tokens: float,
    compute_tolerance: float,
    token_tolerance: float,
) -> tuple[float | int, ...]:
    token_overload = max(
        0.0,
        (max(state.rank_tokens) - (1.0 + token_tolerance) * average_tokens)
        / average_tokens,
    )
    compute_overload = max(
        0.0,
        (max(state.rank_compute) - (1.0 + compute_tolerance) * average_compute)
        / average_compute,
    )
    token_spread = (max(state.rank_tokens) - min(state.rank_tokens)) / average_tokens
    compute_spread = (
        max(state.rank_compute) - min(state.rank_compute)
    ) / average_compute
    return (
        token_overload,
        compute_overload,
        token_spread,
        compute_spread,
        *_split_key(state.split_counts, state.split_penalties),
        state.communication_cost,
        len(state.active_rings),
    )


def _dominates(
    left: Sequence[float | int], right: Sequence[float | int]
) -> bool:
    no_worse = all(a <= b + _EPSILON for a, b in zip(left, right))
    strictly_better = any(a < b - _EPSILON for a, b in zip(left, right))
    return no_worse and strictly_better


def _remove_dominated(
    states: Sequence[_BeamState],
    metric,
) -> list[_BeamState]:
    frontier: list[_BeamState] = []
    frontier_metrics: list[tuple[float | int, ...]] = []
    for state in sorted(states, key=metric):
        state_metrics = metric(state)
        if any(
            incumbent == state_metrics or _dominates(incumbent, state_metrics)
            for incumbent in frontier_metrics
        ):
            continue
        keep_states: list[_BeamState] = []
        keep_metrics: list[tuple[float | int, ...]] = []
        for incumbent_state, incumbent_metrics in zip(frontier, frontier_metrics):
            if not _dominates(state_metrics, incumbent_metrics):
                keep_states.append(incumbent_state)
                keep_metrics.append(incumbent_metrics)
        keep_states.append(state)
        keep_metrics.append(state_metrics)
        frontier = keep_states
        frontier_metrics = keep_metrics
    return frontier


def _quantized_signature(
    state: _BeamState,
    average_compute: float,
    average_tokens: float,
) -> tuple[int, ...]:
    signature: list[int] = []
    token_step = _LOAD_QUANTIZATION * average_tokens
    compute_step = _LOAD_QUANTIZATION * average_compute
    for token_load, compute_load in zip(state.rank_tokens, state.rank_compute):
        signature.append(math.floor(token_load / token_step))
        signature.append(math.floor(compute_load / compute_step))
    return tuple(signature)


def _merge_equivalent_states(
    states: Sequence[_BeamState],
    metric,
    average_compute: float,
    average_tokens: float,
) -> list[_BeamState]:
    merged: dict[tuple[int, ...], _BeamState] = {}
    for state in states:
        signature = _quantized_signature(state, average_compute, average_tokens)
        incumbent = merged.get(signature)
        state_key = (
            _split_key(state.split_counts, state.split_penalties),
            state.communication_cost,
            len(state.active_rings),
            metric(state),
        )
        if incumbent is None:
            merged[signature] = state
            continue
        incumbent_key = (
            _split_key(incumbent.split_counts, incumbent.split_penalties),
            incumbent.communication_cost,
            len(incumbent.active_rings),
            metric(incumbent),
        )
        if state_key < incumbent_key:
            merged[signature] = state
    return list(merged.values())


def _select_representatives(
    states: Sequence[_BeamState], beam_width: int, metric
) -> list[_BeamState]:
    ordered = sorted(states, key=metric)
    if len(ordered) <= beam_width:
        return ordered

    metrics = [metric(state) for state in ordered]
    selected: list[int] = []

    def select(index: int) -> None:
        if index not in selected and len(selected) < beam_width:
            selected.append(index)

    select(min(range(len(ordered)), key=lambda idx: (max(metrics[idx][0:2]), metrics[idx])))
    select(min(range(len(ordered)), key=lambda idx: (metrics[idx][2], metrics[idx])))
    select(min(range(len(ordered)), key=lambda idx: (metrics[idx][3], metrics[idx])))
    select(
        min(
            range(len(ordered)),
            key=lambda idx: (ordered[idx].communication_cost, metrics[idx]),
        )
    )
    select(
        min(
            range(len(ordered)),
            key=lambda idx: (
                _split_key(ordered[idx].split_counts, ordered[idx].split_penalties),
                metrics[idx],
            ),
        )
    )
    select(
        min(
            range(len(ordered)),
            key=lambda idx: (len(ordered[idx].active_rings), metrics[idx]),
        )
    )

    distances = [0.0] * len(ordered)
    for dimension in range(len(metrics[0])):
        indices = sorted(
            range(len(ordered)), key=lambda idx: (metrics[idx][dimension], metrics[idx])
        )
        minimum = float(metrics[indices[0]][dimension])
        maximum = float(metrics[indices[-1]][dimension])
        distances[indices[0]] = math.inf
        distances[indices[-1]] = math.inf
        if maximum <= minimum + _EPSILON:
            continue
        scale = maximum - minimum
        for position in range(1, len(indices) - 1):
            index = indices[position]
            if math.isinf(distances[index]):
                continue
            previous_value = float(metrics[indices[position - 1]][dimension])
            next_value = float(metrics[indices[position + 1]][dimension])
            distances[index] += (next_value - previous_value) / scale

    remaining = sorted(
        (idx for idx in range(len(ordered)) if idx not in selected),
        key=lambda idx: (-distances[idx], metrics[idx]),
    )
    for index in remaining:
        select(index)
    return sorted((ordered[index] for index in selected), key=metric)


def _recover_placements(state: _BeamState) -> list[SequencePlacement]:
    placements: list[SequencePlacement] = []
    cursor: _BeamState | None = state
    while cursor is not None and cursor.placement is not None:
        placements.append(cursor.placement)
        cursor = cursor.parent
    placements.reverse()
    return placements


def _pareto_beam_search(
    structural_jobs: Sequence[_Job],
    allowed_candidates: dict[int, tuple[SequencePlacement, ...]],
    world_size: int,
    average_compute: float,
    average_tokens: float,
    compute_tolerance: float,
    token_tolerance: float,
    beam_width: int,
) -> list[_BeamState]:
    empty = _BeamState(
        rank_compute=(0.0,) * world_size,
        rank_tokens=(0,) * world_size,
        split_counts=(0,) * len(LENGTH_BUCKET_LABELS),
        split_penalties=(0,) * len(LENGTH_BUCKET_LABELS),
        communication_cost=0.0,
        active_rings=frozenset(),
        parent=None,
        placement=None,
    )
    beam = [empty]

    def metric(state: _BeamState) -> tuple[float | int, ...]:
        return _prefix_metrics(
            state,
            average_compute,
            average_tokens,
            compute_tolerance,
            token_tolerance,
        )

    for job in structural_jobs:
        candidates = [
            _add_to_beam_state(state, job, placement)
            for state in beam
            for placement in allowed_candidates[job.original_index]
        ]
        candidates = _remove_dominated(candidates, metric)
        candidates = _merge_equivalent_states(
            candidates, metric, average_compute, average_tokens
        )
        beam = _select_representatives(candidates, beam_width, metric)
    return beam


def _smooth_max(values: Sequence[float]) -> float:
    scaled = [_SMOOTHMAX_LAMBDA * value for value in values]
    maximum = max(scaled)
    return (
        maximum
        + math.log(sum(math.exp(value - maximum) for value in scaled))
    ) / _SMOOTHMAX_LAMBDA


def _residual_fill(
    state: _BeamState,
    filler_jobs: Sequence[_Job],
    average_compute: float,
    average_tokens: float,
    compute_tolerance: float,
    token_tolerance: float,
) -> list[SequencePlacement]:
    placements = _recover_placements(state)
    rank_compute = list(state.rank_compute)
    rank_tokens = list(state.rank_tokens)

    for job in filler_jobs:
        scores: list[tuple[tuple[float, float, int], SequencePlacement]] = []
        for placement in job.candidates:
            if placement.ring_size != 1:
                continue
            rank = placement.ring_start
            projected_compute = rank_compute.copy()
            projected_tokens = rank_tokens.copy()
            projected_compute[rank] += job.compute
            projected_tokens[rank] += job.length
            violation = max(
                0.0,
                max(projected_tokens) / average_tokens - (1.0 + token_tolerance),
                max(projected_compute) / average_compute - (1.0 + compute_tolerance),
            )
            potential = _smooth_max(
                [value / average_tokens for value in projected_tokens]
            ) + _smooth_max(
                [value / average_compute for value in projected_compute]
            )
            scores.append(((violation, potential, rank), placement))
        _, best = min(scores, key=lambda item: item[0])
        placements.append(best)
        rank_compute[best.ring_start] += job.compute
        rank_tokens[best.ring_start] += job.length
    return placements


def _evaluate_solution(
    placements: Sequence[SequencePlacement],
    jobs: Sequence[_Job],
    world_size: int,
    average_compute: float,
    average_tokens: float,
    compute_tolerance: float,
    token_tolerance: float,
) -> _Solution:
    rank_compute = [0.0] * world_size
    rank_tokens = [0] * world_size
    rank_communication = [0.0] * world_size
    split_counts = [0] * len(LENGTH_BUCKET_LABELS)
    split_penalties = [0] * len(LENGTH_BUCKET_LABELS)
    communication_cost = 0.0
    active_rings: set[tuple[int, int]] = set()
    jobs_by_index = {job.original_index: job for job in jobs}

    for placement in placements:
        job = jobs_by_index[placement.original_index]
        compute_increment = placement.compute / placement.ring_size
        token_increment = placement.length // placement.ring_size
        communication_increment = ring_communication_per_rank(
            placement.length, placement.ring_size
        )
        for rank in range(
            placement.ring_start, placement.ring_start + placement.ring_size
        ):
            rank_compute[rank] += compute_increment
            rank_tokens[rank] += token_increment
            rank_communication[rank] += communication_increment
        if placement.ring_size > 1:
            split_counts[job.bucket] += 1
            split_penalties[job.bucket] += int(math.log2(placement.ring_size))
        communication_cost += communication_increment
        active_rings.add((placement.ring_size, placement.ring_start))

    compute_deviation = _max_relative_deviation(rank_compute, average_compute)
    token_deviation = _max_relative_deviation(rank_tokens, average_tokens)
    violation = max(
        0.0,
        compute_deviation - compute_tolerance,
        token_deviation - token_tolerance,
    )
    objective: tuple[float | int, ...] = (
        violation,
        *_split_key(split_counts, split_penalties),
        communication_cost,
        len(active_rings),
        compute_deviation,
        token_deviation,
    )
    return _Solution(
        placements=tuple(placements),
        rank_compute=tuple(rank_compute),
        rank_tokens=tuple(rank_tokens),
        rank_communication=tuple(rank_communication),
        split_counts=tuple(split_counts),
        split_penalties=tuple(split_penalties),
        communication_cost=communication_cost,
        active_ring_count=len(active_rings),
        compute_deviation=compute_deviation,
        token_deviation=token_deviation,
        violation=violation,
        objective=objective,
    )


def _selected_jobs_for_ranks(
    placements: Sequence[SequencePlacement],
    jobs_by_index: dict[int, _Job],
    ranks: set[int],
) -> set[int]:
    selected: set[int] = set()
    for rank in sorted(ranks):
        candidates = [
            placement
            for placement in placements
            if placement.ring_start <= rank < placement.ring_start + placement.ring_size
        ]
        candidates.sort(
            key=lambda placement: (
                -jobs_by_index[placement.original_index].kappa,
                -placement.compute,
                placement.original_index,
            )
        )
        selected.update(
            placement.original_index
            for placement in candidates[:_REPAIR_SEQUENCE_COUNT]
        )
    return selected


def _repair_neighbors(
    solution: _Solution,
    jobs: Sequence[_Job],
    allowed_candidates: dict[int, tuple[SequencePlacement, ...]],
) -> Iterable[tuple[tuple[int, SequencePlacement], ...]]:
    placements = solution.placements
    by_index = {placement.original_index: placement for placement in placements}
    jobs_by_index = {job.original_index: job for job in jobs}
    ranks = range(len(solution.rank_tokens))
    high_token = set(
        sorted(ranks, key=lambda rank: (-solution.rank_tokens[rank], rank))[
            :_REPAIR_RANK_COUNT
        ]
    )
    low_token = set(
        sorted(ranks, key=lambda rank: (solution.rank_tokens[rank], rank))[
            :_REPAIR_RANK_COUNT
        ]
    )
    high_compute = set(
        sorted(ranks, key=lambda rank: (-solution.rank_compute[rank], rank))[
            :_REPAIR_RANK_COUNT
        ]
    )
    low_compute = set(
        sorted(ranks, key=lambda rank: (solution.rank_compute[rank], rank))[
            :_REPAIR_RANK_COUNT
        ]
    )
    high_ranks = high_token | high_compute
    low_ranks = low_token | low_compute
    high_jobs = _selected_jobs_for_ranks(
        placements, jobs_by_index, high_ranks
    )
    low_jobs = _selected_jobs_for_ranks(placements, jobs_by_index, low_ranks)
    allowed_keys = {
        index: {_candidate_key(candidate): candidate for candidate in candidates}
        for index, candidates in allowed_candidates.items()
    }
    emitted: set[tuple[tuple[int, int, int], ...]] = set()

    def emit(changes: Sequence[tuple[int, SequencePlacement]]):
        normalized = tuple(sorted(changes, key=lambda change: change[0]))
        signature = tuple(
            (index, placement.ring_size, placement.ring_start)
            for index, placement in normalized
        )
        if signature in emitted:
            return None
        if all(by_index[index] == placement for index, placement in normalized):
            return None
        emitted.add(signature)
        return normalized

    for index in sorted(high_jobs):
        current = by_index[index]
        if current.ring_size == 1:
            for target_rank in sorted(low_ranks):
                candidate = allowed_keys[index].get((1, target_rank))
                if candidate is not None:
                    neighbor = emit(((index, candidate),))
                    if neighbor is not None:
                        yield neighbor

    high_locals = [
        index for index in sorted(high_jobs) if by_index[index].ring_size == 1
    ]
    low_locals = [
        index for index in sorted(low_jobs) if by_index[index].ring_size == 1
    ]
    for left in high_locals:
        for right in low_locals:
            if left == right:
                continue
            left_target = allowed_keys[left].get((1, by_index[right].ring_start))
            right_target = allowed_keys[right].get((1, by_index[left].ring_start))
            if left_target is not None and right_target is not None:
                neighbor = emit(((left, left_target), (right, right_target)))
                if neighbor is not None:
                    yield neighbor

    for index in sorted(high_jobs):
        current = by_index[index]
        if current.ring_size > 1:
            for candidate in allowed_candidates[index]:
                if candidate.ring_size != current.ring_size:
                    continue
                candidate_ranks = set(
                    range(
                        candidate.ring_start,
                        candidate.ring_start + candidate.ring_size,
                    )
                )
                if candidate_ranks & low_ranks:
                    neighbor = emit(((index, candidate),))
                    if neighbor is not None:
                        yield neighbor

        parent_size = current.ring_size * 2
        if parent_size <= len(solution.rank_tokens):
            parent_start = (current.ring_start // parent_size) * parent_size
            candidate = allowed_keys[index].get((parent_size, parent_start))
            if candidate is not None:
                neighbor = emit(((index, candidate),))
                if neighbor is not None:
                    yield neighbor

        if current.ring_size > 1:
            child_size = current.ring_size // 2
            for child_start in (
                current.ring_start,
                current.ring_start + child_size,
            ):
                child_ranks = set(range(child_start, child_start + child_size))
                if not child_ranks & low_ranks:
                    continue
                candidate = allowed_keys[index].get((child_size, child_start))
                if candidate is not None:
                    neighbor = emit(((index, candidate),))
                    if neighbor is not None:
                        yield neighbor

    parent_groups: dict[tuple[int, int], list[int]] = {}
    for placement in placements:
        if placement.ring_size > 1:
            parent_groups.setdefault(
                (placement.ring_size, placement.ring_start), []
            ).append(placement.original_index)
    for (parent_size, parent_start), indices in sorted(parent_groups.items()):
        parent_ranks = set(range(parent_start, parent_start + parent_size))
        if not parent_ranks & (high_ranks | low_ranks):
            continue
        indices.sort(
            key=lambda index: (
                -jobs_by_index[index].kappa,
                -jobs_by_index[index].compute,
                index,
            )
        )
        indices = indices[:_REPAIR_SEQUENCE_COUNT]
        child_size = parent_size // 2
        child_starts = (parent_start, parent_start + child_size)
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                for left_start, right_start in (
                    child_starts,
                    tuple(reversed(child_starts)),
                ):
                    left_candidate = allowed_keys[left].get((child_size, left_start))
                    right_candidate = allowed_keys[right].get((child_size, right_start))
                    if left_candidate is not None and right_candidate is not None:
                        neighbor = emit(
                            ((left, left_candidate), (right, right_candidate))
                        )
                        if neighbor is not None:
                            yield neighbor


def _hierarchical_local_repair(
    solution: _Solution,
    jobs: Sequence[_Job],
    allowed_candidates: dict[int, tuple[SequencePlacement, ...]],
    world_size: int,
    average_compute: float,
    average_tokens: float,
    compute_tolerance: float,
    token_tolerance: float,
    max_iterations: int,
) -> tuple[_Solution, int]:
    moves = 0
    current = solution
    for _ in range(max_iterations):
        best = current
        best_key = (current.objective, _placement_key(current.placements))
        for changes in _repair_neighbors(current, jobs, allowed_candidates):
            placements = list(current.placements)
            positions = {
                placement.original_index: position
                for position, placement in enumerate(placements)
            }
            for original_index, replacement in changes:
                placements[positions[original_index]] = replacement
            candidate = _evaluate_solution(
                placements,
                jobs,
                world_size,
                average_compute,
                average_tokens,
                compute_tolerance,
                token_tolerance,
            )
            candidate_key = (candidate.objective, _placement_key(candidate.placements))
            if candidate.objective < current.objective and candidate_key < best_key:
                best = candidate
                best_key = candidate_key
        if best is current:
            break
        current = best
        moves += 1
    return current, moves


def _build_jobs(
    lengths: Sequence[int],
    world_size: int,
    is_causal: bool,
    average_compute: float,
    average_tokens: float,
    compute_tolerance: float,
    token_tolerance: float,
) -> list[_Job]:
    jobs: list[_Job] = []
    for original_index, length in enumerate(lengths):
        compute = attention_compute(length, is_causal)
        legal_sizes = tuple(eligible_ring_sizes(length, world_size, is_causal))
        if not legal_sizes:
            raise RuntimeError(f"sequence {original_index} has no legal ring size")
        candidates = tuple(
            SequencePlacement(
                original_index=original_index,
                length=length,
                compute=compute,
                ring_size=ring_size,
                ring_start=ring_start,
            )
            for ring_size in legal_sizes
            for ring_start in range(0, world_size, ring_size)
        )
        target_sizes = [
            ring_size
            for ring_size in legal_sizes
            if length / ring_size <= (1.0 + token_tolerance) * average_tokens
            and compute / ring_size
            <= (1.0 + compute_tolerance) * average_compute
        ]
        minimum_ring_size = min(target_sizes) if target_sizes else max(legal_sizes)
        jobs.append(
            _Job(
                original_index=original_index,
                length=length,
                compute=compute,
                legal_sizes=legal_sizes,
                candidates=candidates,
                bucket=_length_bucket(length),
                kappa=max(length / average_tokens, compute / average_compute),
                minimum_ring_size=minimum_ring_size,
            )
        )
    return jobs


def _relaxation_levels(
    jobs: Sequence[_Job], world_size: int
) -> list[tuple[str, dict[int, frozenset[int]], bool]]:
    levels: list[tuple[str, dict[int, frozenset[int]], bool]] = [
        ("initial", {}, False)
    ]
    unlocked: dict[int, set[int]] = {
        bucket: set() for bucket in range(len(LENGTH_BUCKET_LABELS))
    }
    present_buckets = {job.bucket for job in jobs}
    for bucket in reversed(range(len(LENGTH_BUCKET_LABELS))):
        if bucket not in present_buckets:
            continue
        for ring_size in RING_SIZES[1:]:
            if ring_size > world_size:
                continue
            unlocked[bucket].add(ring_size)
            snapshot = {
                key: frozenset(values)
                for key, values in unlocked.items()
                if values
            }
            levels.append(
                (
                    f"unlock {LENGTH_BUCKET_LABELS[bucket]} G{ring_size}",
                    snapshot,
                    False,
                )
            )
    levels.append(("all hard-legal candidates", levels[-1][1], True))
    return levels


def _allowed_for_level(
    jobs: Sequence[_Job],
    unlocked: dict[int, frozenset[int]],
    structure_threshold: float,
    all_hard_candidates: bool,
) -> tuple[list[_Job], list[_Job], dict[int, tuple[SequencePlacement, ...]]]:
    structural: list[_Job] = []
    fillers: list[_Job] = []
    allowed: dict[int, tuple[SequencePlacement, ...]] = {}
    for job in jobs:
        base_structural = (
            job.kappa >= structure_threshold or job.minimum_ring_size > 1
        )
        unlocked_sizes = unlocked.get(job.bucket, frozenset())
        if base_structural or unlocked_sizes or all_hard_candidates:
            structural.append(job)
            if all_hard_candidates:
                candidates = job.candidates
            elif base_structural:
                candidates = tuple(
                    candidate
                    for candidate in job.candidates
                    if candidate.ring_size >= job.minimum_ring_size
                )
            else:
                candidates = tuple(
                    candidate
                    for candidate in job.candidates
                    if candidate.ring_size == 1
                    or candidate.ring_size in unlocked_sizes
                )
            allowed[job.original_index] = candidates
        else:
            fillers.append(job)
            allowed[job.original_index] = tuple(
                candidate for candidate in job.candidates if candidate.ring_size == 1
            )
    structural.sort(key=lambda job: (-job.kappa, -job.compute, -job.length, job.original_index))
    fillers.sort(key=lambda job: (-job.kappa, -job.compute, -job.length, job.original_index))
    return structural, fillers, allowed


def assign_hierarchical_rings(
    lengths: list[int],
    world_size: int,
    is_causal: bool,
    compute_balance_tolerance: float = 0.05,
    token_balance_tolerance: float = 0.10,
    beam_width: int = 64,
    finalist_count: int = 8,
    structure_threshold: float = 0.5,
    max_repair_iterations: int = 32,
) -> HybridWorkload:
    if world_size not in (2, 4, 8):
        raise ValueError(f"world_size must be 2, 4, or 8, got {world_size}")
    if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("lengths must contain positive integers")
    float_options = {
        "compute_balance_tolerance": compute_balance_tolerance,
        "token_balance_tolerance": token_balance_tolerance,
        "structure_threshold": structure_threshold,
    }
    for name, value in float_options.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}")
    integer_options = {
        "beam_width": beam_width,
        "finalist_count": finalist_count,
    }
    for name, value in integer_options.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")
    if type(max_repair_iterations) is not int or max_repair_iterations < 0:
        raise ValueError(
            "max_repair_iterations must be a non-negative integer, "
            f"got {max_repair_iterations}"
        )

    total_compute = float(
        sum(attention_compute(length, is_causal) for length in lengths)
    )
    average_compute = total_compute / world_size
    average_tokens = sum(lengths) / world_size
    jobs = _build_jobs(
        lengths,
        world_size,
        is_causal,
        average_compute,
        average_tokens,
        compute_balance_tolerance,
        token_balance_tolerance,
    )

    best_solution: _Solution | None = None
    best_level = 0
    best_label = "initial"
    best_moves = 0
    previous_signature: tuple[tuple[int, tuple[tuple[int, int], ...]], ...] | None = None

    for level, (label, unlocked, all_hard_candidates) in enumerate(
        _relaxation_levels(jobs, world_size)
    ):
        structural, fillers, allowed = _allowed_for_level(
            jobs, unlocked, structure_threshold, all_hard_candidates
        )
        structural_indices = {job.original_index for job in structural}
        signature = tuple(
            sorted(
                (
                    index,
                    tuple(_candidate_key(candidate) for candidate in candidates),
                )
                for index, candidates in allowed.items()
                if index in structural_indices
            )
        )
        if signature == previous_signature:
            continue
        previous_signature = signature

        beam = _pareto_beam_search(
            structural,
            allowed,
            world_size,
            average_compute,
            average_tokens,
            compute_balance_tolerance,
            token_balance_tolerance,
            beam_width,
        )
        completed = [
            _evaluate_solution(
                _residual_fill(
                    state,
                    fillers,
                    average_compute,
                    average_tokens,
                    compute_balance_tolerance,
                    token_balance_tolerance,
                ),
                jobs,
                world_size,
                average_compute,
                average_tokens,
                compute_balance_tolerance,
                token_balance_tolerance,
            )
            for state in beam
        ]
        completed.sort(
            key=lambda solution: (
                solution.objective,
                _placement_key(solution.placements),
            )
        )

        repaired: list[tuple[_Solution, int]] = []
        for solution in completed[:finalist_count]:
            repaired.append(
                _hierarchical_local_repair(
                    solution,
                    jobs,
                    allowed,
                    world_size,
                    average_compute,
                    average_tokens,
                    compute_balance_tolerance,
                    token_balance_tolerance,
                    max_repair_iterations,
                )
            )
        level_solution, level_moves = min(
            repaired,
            key=lambda item: (item[0].objective, _placement_key(item[0].placements)),
        )
        if best_solution is None or (
            level_solution.objective,
            _placement_key(level_solution.placements),
        ) < (best_solution.objective, _placement_key(best_solution.placements)):
            best_solution = level_solution
            best_level = level
            best_label = label
            best_moves = level_moves
        if level_solution.violation <= _EPSILON:
            best_solution = level_solution
            best_level = level
            best_label = label
            best_moves = level_moves
            break

    if best_solution is None:
        raise RuntimeError("BR-PBS did not produce a placement")

    placements = sorted(
        best_solution.placements,
        key=lambda placement: (
            -placement.ring_size,
            placement.ring_start,
            placement.original_index,
        ),
    )
    return HybridWorkload(
        global_lengths=[placement.length for placement in placements],
        ring_sizes=[placement.ring_size for placement in placements],
        ring_starts=[placement.ring_start for placement in placements],
        rank_compute=list(best_solution.rank_compute),
        rank_tokens=list(best_solution.rank_tokens),
        rank_communication=list(best_solution.rank_communication),
        average_compute=average_compute,
        peak_compute=max(best_solution.rank_compute),
        average_tokens=average_tokens,
        peak_tokens=max(best_solution.rank_tokens),
        compute_balance_tolerance=compute_balance_tolerance,
        token_balance_tolerance=token_balance_tolerance,
        load_violation=best_solution.violation,
        feasible=best_solution.violation <= _EPSILON,
        relaxation_level=best_level,
        relaxation_label=best_label,
        split_counts=list(best_solution.split_counts),
        split_penalties=list(best_solution.split_penalties),
        communication_cost=best_solution.communication_cost,
        active_ring_count=best_solution.active_ring_count,
        repair_moves=best_moves,
    )


def make_workload(
    dataset: str,
    target_tokens: int,
    seed: int,
    world_size: int,
    mode: str,
    compute_balance_tolerance: float = 0.05,
    token_balance_tolerance: float = 0.10,
    beam_width: int = 64,
    finalist_count: int = 8,
    structure_threshold: float = 0.5,
    max_repair_iterations: int = 32,
) -> HybridWorkload:
    return make_workloads(
        dataset=dataset,
        target_tokens=target_tokens,
        seed=seed,
        num_cases=1,
        world_size=world_size,
        mode=mode,
        compute_balance_tolerance=compute_balance_tolerance,
        token_balance_tolerance=token_balance_tolerance,
        beam_width=beam_width,
        finalist_count=finalist_count,
        structure_threshold=structure_threshold,
        max_repair_iterations=max_repair_iterations,
    )[0]


def make_workloads(
    dataset: str,
    target_tokens: int,
    seed: int,
    num_cases: int,
    world_size: int,
    mode: str,
    compute_balance_tolerance: float = 0.05,
    token_balance_tolerance: float = 0.10,
    beam_width: int = 64,
    finalist_count: int = 8,
    structure_threshold: float = 0.5,
    max_repair_iterations: int = 32,
) -> list[HybridWorkload]:
    """Build multiple placements from one continuously advanced sampler RNG."""

    length_cases = generate_dataset_length_cases(
        dataset, target_tokens, seed, num_cases
    )
    return [
        assign_hierarchical_rings(
            lengths,
            world_size,
            is_causal=mode in ("causal", "both"),
            compute_balance_tolerance=compute_balance_tolerance,
            token_balance_tolerance=token_balance_tolerance,
            beam_width=beam_width,
            finalist_count=finalist_count,
            structure_threshold=structure_threshold,
            max_repair_iterations=max_repair_iterations,
        )
        for lengths in length_cases
    ]
