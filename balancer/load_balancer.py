"""Hierarchical ring placement and load balancing for dataset workloads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .sampler import generate_dataset_length_cases


RING_SIZES = (1, 2, 4, 8)


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
    average_compute: float
    peak_compute: float
    requested_cap: float
    final_cap: float
    topology_lower_bound: float
    topology_limited: bool
    placement_relaxed: bool
    rank_communication: list[float]
    average_tokens: float
    peak_tokens: int
    requested_token_cap: float
    final_token_cap: float
    token_topology_lower_bound: float
    token_topology_limited: bool
    compute_cap_relaxed: bool
    token_cap_relaxed: bool
    emergency_relaxed: bool
    local_search_moves: int

    @property
    def imbalance(self) -> float:
        return self.peak_compute / self.average_compute - 1.0

    @property
    def token_imbalance(self) -> float:
        return self.peak_tokens / self.average_tokens - 1.0

    @property
    def communication_amplification(self) -> float:
        return sum(self.rank_communication) / sum(self.global_lengths)


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


def _normalized_mse(values: Sequence[float], average: float) -> float:
    return sum(((value - average) / average) ** 2 for value in values) / len(values)


def _placement_objective(
    rank_compute: Sequence[float],
    rank_tokens: Sequence[int],
    rank_communication: Sequence[float],
    average_compute: float,
    average_tokens: float,
    total_tokens: int,
    communication_weight: float,
) -> float:
    token_peak_ratio = max(rank_tokens) / average_tokens
    token_mse = _normalized_mse(rank_tokens, average_tokens)
    compute_peak_ratio = max(rank_compute) / average_compute
    compute_mse = _normalized_mse(rank_compute, average_compute)
    communication_peak_ratio = max(rank_communication) / average_tokens
    communication_total_ratio = sum(rank_communication) / total_tokens
    return (
        token_peak_ratio
        + 0.25 * token_mse
        + 0.05 * compute_peak_ratio
        + 0.05 * compute_mse
        + communication_weight
        * (0.5 * communication_peak_ratio + 0.5 * communication_total_ratio)
    )


def _placement_score(
    rank_compute: Sequence[float],
    rank_tokens: Sequence[int],
    rank_communication: Sequence[float],
    average_compute: float,
    average_tokens: float,
    total_tokens: int,
    communication_weight: float,
    ring_size: int,
    ring_start: int,
) -> tuple[float, ...]:
    token_peak_ratio = max(rank_tokens) / average_tokens
    token_mse = _normalized_mse(rank_tokens, average_tokens)
    compute_peak_ratio = max(rank_compute) / average_compute
    compute_mse = _normalized_mse(rank_compute, average_compute)
    communication_peak_ratio = max(rank_communication) / average_tokens
    communication_total_ratio = sum(rank_communication) / total_tokens
    objective = _placement_objective(
        rank_compute,
        rank_tokens,
        rank_communication,
        average_compute,
        average_tokens,
        total_tokens,
        communication_weight,
    )
    return (
        objective,
        token_peak_ratio,
        token_mse,
        communication_peak_ratio,
        communication_total_ratio,
        compute_peak_ratio,
        compute_mse,
        float(ring_size),
        float(ring_start),
    )


def _cap_ladder(initial: float, maximum: float, increment: float) -> list[float]:
    values = [initial]
    while values[-1] + increment < maximum - 1e-9:
        values.append(values[-1] + increment)
    if values[-1] < maximum - 1e-9:
        values.append(maximum)
    return values


def _apply_placement_delta(
    placement: SequencePlacement,
    rank_compute: list[float],
    rank_tokens: list[int],
    rank_communication: list[float],
    sign: int,
) -> None:
    # Causal zigzag gives every ring member exactly L(L+1)/(2G) score work.
    compute_increment = placement.compute / placement.ring_size
    token_increment = placement.length // placement.ring_size
    communication_increment = ring_communication_per_rank(
        placement.length, placement.ring_size
    )
    for rank in range(
        placement.ring_start, placement.ring_start + placement.ring_size
    ):
        rank_compute[rank] += sign * compute_increment
        rank_tokens[rank] += sign * token_increment
        rank_communication[rank] += sign * communication_increment


def _try_place_sequences(
    jobs: list[tuple[int, int, int, list[int]]],
    world_size: int,
    average_compute: float,
    average_tokens: float,
    total_tokens: int,
    compute_cap: float,
    token_cap: float,
    communication_weight: float,
) -> tuple[list[SequencePlacement], list[float], list[int], list[float]] | None:
    rank_compute = [0.0] * world_size
    rank_tokens = [0] * world_size
    rank_communication = [0.0] * world_size
    placements: list[SequencePlacement] = []

    for original_index, length, compute, eligible_sizes in jobs:
        candidates: list[
            tuple[tuple[float, ...], int, int, float, int, float]
        ] = []
        for ring_size in eligible_sizes:
            # The causal zigzag layout equalizes exact visible-score work across members.
            compute_increment = compute / ring_size
            token_increment = length // ring_size
            communication_increment = ring_communication_per_rank(length, ring_size)
            for ring_start in range(0, world_size, ring_size):
                members = range(ring_start, ring_start + ring_size)
                if any(
                    rank_compute[rank] + compute_increment > compute_cap + 1e-6
                    or rank_tokens[rank] + token_increment > token_cap + 1e-6
                    for rank in members
                ):
                    continue
                projected_compute = rank_compute.copy()
                projected_tokens = rank_tokens.copy()
                projected_communication = rank_communication.copy()
                for rank in members:
                    projected_compute[rank] += compute_increment
                    projected_tokens[rank] += token_increment
                    projected_communication[rank] += communication_increment
                score = _placement_score(
                    projected_compute,
                    projected_tokens,
                    projected_communication,
                    average_compute,
                    average_tokens,
                    total_tokens,
                    communication_weight,
                    ring_size,
                    ring_start,
                )
                candidates.append(
                    (
                        score,
                        ring_size,
                        ring_start,
                        compute_increment,
                        token_increment,
                        communication_increment,
                    )
                )
        if not candidates:
            return None

        (
            _,
            ring_size,
            ring_start,
            compute_increment,
            token_increment,
            communication_increment,
        ) = min(candidates)
        for rank in range(ring_start, ring_start + ring_size):
            rank_compute[rank] += compute_increment
            rank_tokens[rank] += token_increment
            rank_communication[rank] += communication_increment
        placements.append(
            SequencePlacement(
                original_index,
                length,
                compute,
                ring_size,
                ring_start,
            )
        )
    return placements, rank_compute, rank_tokens, rank_communication


def _improve_placements(
    jobs: list[tuple[int, int, int, list[int]]],
    placements: list[SequencePlacement],
    rank_compute: list[float],
    rank_tokens: list[int],
    rank_communication: list[float],
    world_size: int,
    average_compute: float,
    average_tokens: float,
    total_tokens: int,
    compute_cap: float,
    token_cap: float,
    communication_weight: float,
    max_passes: int,
) -> tuple[list[SequencePlacement], list[float], list[int], list[float], int]:
    placements = placements.copy()
    rank_compute = rank_compute.copy()
    rank_tokens = rank_tokens.copy()
    rank_communication = rank_communication.copy()
    moves = 0

    for _ in range(max_passes):
        pass_moves = 0
        for position, job in enumerate(jobs):
            original_index, length, compute, eligible_sizes = job
            current = placements[position]
            if current.original_index != original_index:
                raise RuntimeError("placement order does not match planner job order")
            current_objective = _placement_objective(
                rank_compute,
                rank_tokens,
                rank_communication,
                average_compute,
                average_tokens,
                total_tokens,
                communication_weight,
            )
            _apply_placement_delta(
                current,
                rank_compute,
                rank_tokens,
                rank_communication,
                -1,
            )

            candidates: list[tuple[tuple[float, ...], SequencePlacement]] = []
            for ring_size in eligible_sizes:
                compute_increment = compute / ring_size
                token_increment = length // ring_size
                communication_increment = ring_communication_per_rank(length, ring_size)
                for ring_start in range(0, world_size, ring_size):
                    members = range(ring_start, ring_start + ring_size)
                    if any(
                        rank_compute[rank] + compute_increment > compute_cap + 1e-6
                        or rank_tokens[rank] + token_increment > token_cap + 1e-6
                        for rank in members
                    ):
                        continue
                    projected_compute = rank_compute.copy()
                    projected_tokens = rank_tokens.copy()
                    projected_communication = rank_communication.copy()
                    for rank in members:
                        projected_compute[rank] += compute_increment
                        projected_tokens[rank] += token_increment
                        projected_communication[rank] += communication_increment
                    objective = _placement_objective(
                        projected_compute,
                        projected_tokens,
                        projected_communication,
                        average_compute,
                        average_tokens,
                        total_tokens,
                        communication_weight,
                    )
                    placement = SequencePlacement(
                        original_index,
                        length,
                        compute,
                        ring_size,
                        ring_start,
                    )
                    candidates.append(
                        (
                            (
                                objective,
                                float(ring_size),
                                float(ring_start),
                                float(original_index),
                            ),
                            placement,
                        )
                    )

            if not candidates:
                raise RuntimeError("local repair could not restore a valid placement")
            best_score, best = min(candidates, key=lambda candidate: candidate[0])
            if best_score[0] < current_objective - 1e-12:
                placements[position] = best
                _apply_placement_delta(
                    best,
                    rank_compute,
                    rank_tokens,
                    rank_communication,
                    1,
                )
                moves += 1
                pass_moves += 1
            else:
                _apply_placement_delta(
                    current,
                    rank_compute,
                    rank_tokens,
                    rank_communication,
                    1,
                )
        if pass_moves == 0:
            break

    return placements, rank_compute, rank_tokens, rank_communication, moves


def assign_hierarchical_rings(
    lengths: list[int],
    world_size: int,
    is_causal: bool,
    balance_tolerance: float,
    token_balance_tolerance: float = 0.10,
    max_compute_balance_tolerance: float = 0.20,
    max_token_balance_tolerance: float = 0.50,
    communication_weight: float = 0.05,
    local_search_passes: int = 4,
) -> HybridWorkload:
    if world_size not in (2, 4, 8):
        raise ValueError(f"world_size must be 2, 4, or 8, got {world_size}")
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("lengths must contain positive integers")
    float_options = {
        "balance_tolerance": balance_tolerance,
        "token_balance_tolerance": token_balance_tolerance,
        "max_compute_balance_tolerance": max_compute_balance_tolerance,
        "max_token_balance_tolerance": max_token_balance_tolerance,
        "communication_weight": communication_weight,
    }
    for name, value in float_options.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}")
    if max_compute_balance_tolerance < balance_tolerance:
        raise ValueError(
            "max_compute_balance_tolerance must be greater than or equal to "
            f"balance_tolerance, got {max_compute_balance_tolerance} < {balance_tolerance}"
        )
    if max_token_balance_tolerance < token_balance_tolerance:
        raise ValueError(
            "max_token_balance_tolerance must be greater than or equal to "
            f"token_balance_tolerance, got {max_token_balance_tolerance} "
            f"< {token_balance_tolerance}"
        )
    if local_search_passes < 0:
        raise ValueError(
            f"local_search_passes must be non-negative, got {local_search_passes}"
        )

    jobs: list[tuple[int, int, int, list[int]]] = []
    for original_index, length in enumerate(lengths):
        eligible_sizes = eligible_ring_sizes(length, world_size, is_causal)
        if not eligible_sizes:
            raise RuntimeError(f"sequence {original_index} has no legal ring size")
        jobs.append(
            (
                original_index,
                length,
                attention_compute(length, is_causal),
                eligible_sizes,
            )
        )
    total_compute = float(sum(job[2] for job in jobs))
    average_compute = total_compute / world_size
    total_tokens = sum(lengths)
    average_tokens = total_tokens / world_size
    jobs.sort(
        key=lambda job: (
            len(job[3]),
            -max(
                (job[2] / max(job[3])) / average_compute,
                (job[1] / max(job[3])) / average_tokens,
            ),
            -job[2],
            -job[1],
            job[0],
        )
    )

    topology_lower_bound = max(
        average_compute,
        max(job[2] / max(job[3]) for job in jobs),
    )
    token_topology_lower_bound = max(
        average_tokens,
        max(job[1] / max(job[3]) for job in jobs),
    )
    requested_cap = (1.0 + balance_tolerance) * average_compute
    requested_token_cap = (1.0 + token_balance_tolerance) * average_tokens
    base_compute_cap = max(requested_cap, topology_lower_bound)
    base_token_cap = max(requested_token_cap, token_topology_lower_bound)
    max_compute_cap = max(
        base_compute_cap,
        (1.0 + max_compute_balance_tolerance) * average_compute,
    )
    max_token_cap = max(
        base_token_cap,
        (1.0 + max_token_balance_tolerance) * average_tokens,
    )
    compute_caps = _cap_ladder(
        base_compute_cap,
        max_compute_cap,
        max(0.01 * average_compute, 1.0),
    )
    token_caps = _cap_ladder(
        base_token_cap,
        max_token_cap,
        max(0.01 * average_tokens, 1.0),
    )

    selected: tuple[
        float,
        float,
        list[SequencePlacement],
        list[float],
        list[int],
        list[float],
        int,
    ] | None = None
    for token_cap in token_caps:
        feasible_results: list[
            tuple[
                tuple[float, ...],
                float,
                list[SequencePlacement],
                list[float],
                list[int],
                list[float],
                int,
            ]
        ] = []
        for compute_cap in compute_caps:
            result = _try_place_sequences(
                jobs,
                world_size,
                average_compute,
                average_tokens,
                total_tokens,
                compute_cap,
                token_cap,
                communication_weight,
            )
            if result is None:
                continue
            placements, rank_compute, rank_tokens, rank_communication = result
            (
                placements,
                rank_compute,
                rank_tokens,
                rank_communication,
                local_search_moves,
            ) = _improve_placements(
                jobs,
                placements,
                rank_compute,
                rank_tokens,
                rank_communication,
                world_size,
                average_compute,
                average_tokens,
                total_tokens,
                compute_cap,
                token_cap,
                communication_weight,
                local_search_passes,
            )
            objective = _placement_objective(
                rank_compute,
                rank_tokens,
                rank_communication,
                average_compute,
                average_tokens,
                total_tokens,
                communication_weight,
            )
            key = (
                objective,
                float(max(rank_tokens)),
                sum(rank_communication),
                max(rank_compute),
                compute_cap,
            )
            feasible_results.append(
                (
                    key,
                    compute_cap,
                    placements,
                    rank_compute,
                    rank_tokens,
                    rank_communication,
                    local_search_moves,
                )
            )
        if feasible_results:
            (
                _,
                final_compute_cap,
                placements,
                rank_compute,
                rank_tokens,
                rank_communication,
                local_search_moves,
            ) = min(feasible_results, key=lambda result: result[0])
            selected = (
                final_compute_cap,
                token_cap,
                placements,
                rank_compute,
                rank_tokens,
                rank_communication,
                local_search_moves,
            )
            break

    emergency_relaxed = False
    if selected is None:
        emergency_relaxed = True
        final_compute_cap = total_compute
        final_token_cap = float(total_tokens)
        result = _try_place_sequences(
            jobs,
            world_size,
            average_compute,
            average_tokens,
            total_tokens,
            final_compute_cap,
            final_token_cap,
            communication_weight,
        )
        if result is None:
            raise RuntimeError(
                "failed to place sequences even with fully relaxed compute and token caps"
            )
        placements, rank_compute, rank_tokens, rank_communication = result
        (
            placements,
            rank_compute,
            rank_tokens,
            rank_communication,
            local_search_moves,
        ) = _improve_placements(
            jobs,
            placements,
            rank_compute,
            rank_tokens,
            rank_communication,
            world_size,
            average_compute,
            average_tokens,
            total_tokens,
            final_compute_cap,
            final_token_cap,
            communication_weight,
            local_search_passes,
        )
    else:
        (
            final_compute_cap,
            final_token_cap,
            placements,
            rank_compute,
            rank_tokens,
            rank_communication,
            local_search_moves,
        ) = selected

    compute_cap_relaxed = final_compute_cap > base_compute_cap + 1e-6
    token_cap_relaxed = final_token_cap > base_token_cap + 1e-6
    placement_relaxed = compute_cap_relaxed or token_cap_relaxed or emergency_relaxed

    placements.sort(
        key=lambda placement: (
            -placement.ring_size,
            placement.ring_start,
            placement.original_index,
        )
    )
    return HybridWorkload(
        global_lengths=[placement.length for placement in placements],
        ring_sizes=[placement.ring_size for placement in placements],
        ring_starts=[placement.ring_start for placement in placements],
        rank_compute=rank_compute,
        rank_tokens=rank_tokens,
        average_compute=average_compute,
        peak_compute=max(rank_compute),
        requested_cap=requested_cap,
        final_cap=final_compute_cap,
        topology_lower_bound=topology_lower_bound,
        topology_limited=topology_lower_bound > requested_cap + 1e-6,
        placement_relaxed=placement_relaxed,
        rank_communication=rank_communication,
        average_tokens=average_tokens,
        peak_tokens=max(rank_tokens),
        requested_token_cap=requested_token_cap,
        final_token_cap=final_token_cap,
        token_topology_lower_bound=token_topology_lower_bound,
        token_topology_limited=(
            token_topology_lower_bound > requested_token_cap + 1e-6
        ),
        compute_cap_relaxed=compute_cap_relaxed,
        token_cap_relaxed=token_cap_relaxed,
        emergency_relaxed=emergency_relaxed,
        local_search_moves=local_search_moves,
    )


def make_workload(
    dataset: str,
    target_tokens: int,
    seed: int,
    world_size: int,
    mode: str,
    balance_tolerance: float,
    token_balance_tolerance: float = 0.10,
    max_compute_balance_tolerance: float = 0.20,
    max_token_balance_tolerance: float = 0.50,
    communication_weight: float = 0.05,
    local_search_passes: int = 4,
) -> HybridWorkload:
    return make_workloads(
        dataset=dataset,
        target_tokens=target_tokens,
        seed=seed,
        num_cases=1,
        world_size=world_size,
        mode=mode,
        balance_tolerance=balance_tolerance,
        token_balance_tolerance=token_balance_tolerance,
        max_compute_balance_tolerance=max_compute_balance_tolerance,
        max_token_balance_tolerance=max_token_balance_tolerance,
        communication_weight=communication_weight,
        local_search_passes=local_search_passes,
    )[0]


def make_workloads(
    dataset: str,
    target_tokens: int,
    seed: int,
    num_cases: int,
    world_size: int,
    mode: str,
    balance_tolerance: float,
    token_balance_tolerance: float = 0.10,
    max_compute_balance_tolerance: float = 0.20,
    max_token_balance_tolerance: float = 0.50,
    communication_weight: float = 0.05,
    local_search_passes: int = 4,
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
            balance_tolerance=balance_tolerance,
            token_balance_tolerance=token_balance_tolerance,
            max_compute_balance_tolerance=max_compute_balance_tolerance,
            max_token_balance_tolerance=max_token_balance_tolerance,
            communication_weight=communication_weight,
            local_search_passes=local_search_passes,
        )
        for lengths in length_cases
    ]

