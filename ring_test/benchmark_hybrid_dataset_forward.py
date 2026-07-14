"""Dataset-shaped frontend for the hierarchical hybrid forward benchmark."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
for path in (THIS_DIR, DEMO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


MAX_SEQUENCE_TOKENS = 128 * 1024
DEFAULT_TRUNCATED_PADDING_THRESHOLD = 32 * 1024
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


def _weighted_bucket(rng: random.Random, weights: Sequence[float]) -> int:
    draw = rng.random() * sum(weights)
    cumulative = 0.0
    for idx, weight in enumerate(weights):
        cumulative += weight
        if draw < cumulative:
            return idx
    return len(weights) - 1


def generate_dataset_lengths(
    dataset: str,
    target_tokens: int,
    seed: int,
    world_size: int | None = None,
    truncated_padding_threshold: int = DEFAULT_TRUNCATED_PADDING_THRESHOLD,
) -> list[int]:
    if dataset not in DATASET_WEIGHTS:
        raise ValueError(f"unknown dataset {dataset!r}")
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")
    if world_size is not None and world_size not in (2, 4, 8):
        raise ValueError(f"world_size must be 2, 4, or 8, got {world_size}")
    if truncated_padding_threshold < 0:
        raise ValueError(
            "truncated_padding_threshold must be non-negative, got "
            f"{truncated_padding_threshold}"
        )
    weights = DATASET_WEIGHTS[dataset]
    if abs(sum(weights) - 1.0) > 1e-9:
        raise RuntimeError(f"{dataset} weights must sum to 1, got {sum(weights)}")

    rng = random.Random(seed)
    lengths: list[int] = []
    remaining = target_tokens
    padding_alignment = 256 * world_size if world_size is not None else None
    while remaining > 0:
        sampled = LENGTH_BUCKETS[_weighted_bucket(rng, weights)]
        length = min(sampled, remaining, MAX_SEQUENCE_TOKENS)
        if (
            padding_alignment is not None
            and sampled > remaining
            and length > truncated_padding_threshold
        ):
            length = min(
                MAX_SEQUENCE_TOKENS,
                ((length + padding_alignment - 1) // padding_alignment)
                * padding_alignment,
            )
        lengths.append(length)
        remaining = max(remaining - length, 0)
    return lengths


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
    truncated_padding_threshold: int = DEFAULT_TRUNCATED_PADDING_THRESHOLD,
) -> HybridWorkload:
    lengths = generate_dataset_lengths(
        dataset,
        target_tokens,
        seed,
        world_size=world_size,
        truncated_padding_threshold=truncated_padding_threshold,
    )
    return assign_hierarchical_rings(
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


def _format_int_list(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def print_workload(
    workload: HybridWorkload,
    dataset: str,
    seed: int,
    target_tokens: int,
    world_size: int,
    mode: str = "causal",
) -> None:
    actual_tokens = sum(workload.global_lengths)
    planner_is_causal = mode in ("causal", "both")
    planner_mode = "causal-compatible" if mode == "both" else mode
    print(
        f"Planner workload: dataset={dataset}, seed={seed}, "
        f"world_size={world_size}, mode={planner_mode}"
    )
    print(
        f"Batch: B={len(workload.global_lengths)}, target_tokens={target_tokens}, "
        f"actual_tokens={actual_tokens}, "
        f"target_delta={actual_tokens - target_tokens:+d}"
    )

    print("\nSequence placement")
    print(
        f"{'Seq':>4} {'Global S':>10} {'Ring':>6} {'Ranks':>9} "
        f"{'Local S':>10} {'Est compute/member':>19}"
    )
    for index, (length, ring_size, ring_start) in enumerate(
        zip(workload.global_lengths, workload.ring_sizes, workload.ring_starts)
    ):
        ranks = (
            str(ring_start)
            if ring_size == 1
            else f"{ring_start}-{ring_start + ring_size - 1}"
        )
        compute_per_member = attention_compute(length, planner_is_causal) / ring_size
        print(
            f"{index:>4} {length:>10} {f'G{ring_size}':>6} {ranks:>9} "
            f"{length // ring_size:>10} {compute_per_member:>19.3e}"
        )

    print("\nPer-rank load")
    print(
        f"{'Rank':>4} {'Tokens':>10} {'Token/avg':>10} "
        f"{'Est compute':>14} {'Compute/avg':>12} {'Est communication':>19}"
    )
    for rank, (tokens, compute, communication) in enumerate(
        zip(
            workload.rank_tokens,
            workload.rank_compute,
            workload.rank_communication,
        )
    ):
        print(
            f"{rank:>4} {tokens:>10} "
            f"{100.0 * tokens / workload.average_tokens:>9.2f}% "
            f"{compute:>14.3e} "
            f"{100.0 * compute / workload.average_compute:>11.2f}% "
            f"{communication:>19.3e}"
        )

    cap_rows = (
        (
            "Compute",
            f"{workload.average_compute:.3e}",
            f"{workload.peak_compute:.3e}",
            f"{100.0 * workload.imbalance:.2f}%",
            f"{workload.requested_cap:.3e}",
            f"{workload.topology_lower_bound:.3e}",
            f"{workload.final_cap:.3e}",
            str(workload.topology_limited),
            str(workload.compute_cap_relaxed),
        ),
        (
            "Tokens",
            f"{workload.average_tokens:.3f}",
            str(workload.peak_tokens),
            f"{100.0 * workload.token_imbalance:.2f}%",
            f"{workload.requested_token_cap:.3f}",
            f"{workload.token_topology_lower_bound:.3f}",
            f"{workload.final_token_cap:.3f}",
            str(workload.token_topology_limited),
            str(workload.token_cap_relaxed),
        ),
    )
    print("\nBalance and caps")
    print(
        f"{'Metric':<8} {'Average':>14} {'Peak':>14} {'Imbalance':>10} "
        f"{'Requested cap':>14} {'Topology LB':>14} {'Final cap':>14} "
        f"{'Topology':>10} {'Relaxed':>9}"
    )
    for row in cap_rows:
        metric, average, peak, imbalance, requested, topology, final, limited, relaxed = row
        print(
            f"{metric:<8} {average:>14} {peak:>14} {imbalance:>10} "
            f"{requested:>14} {topology:>14} {final:>14} "
            f"{limited:>10} {relaxed:>9}"
        )

    print(
        "\nPlanner status: "
        f"placement_relaxed={workload.placement_relaxed}, "
        f"emergency_relaxed={workload.emergency_relaxed}, "
        f"local_search_moves={workload.local_search_moves}"
    )
    print(
        "Communication: "
        f"amplification={workload.communication_amplification:.6f}, "
        f"total_token_hops={sum(workload.rank_communication):.3e}"
    )

    print("\nKernel metadata")
    print(f"global_seqlens={_format_int_list(workload.global_lengths)}")
    print(f"ring_sizes={_format_int_list(workload.ring_sizes)}")
    print(
        f"ring_starts={_format_int_list(workload.ring_starts)}",
        flush=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a dataset-shaped workload and run the hybrid forward benchmark"
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_WEIGHTS), required=True)
    parser.add_argument("--target-tokens", type=int, default=MAX_SEQUENCE_TOKENS)
    parser.add_argument("--balance-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--token-balance-tolerance",
        type=float,
        default=0.10,
        help="Requested maximum token imbalance relative to average",
    )
    parser.add_argument(
        "--max-compute-balance-tolerance",
        type=float,
        default=0.20,
        help="Maximum compute imbalance explored by the planner",
    )
    parser.add_argument(
        "--max-token-balance-tolerance",
        type=float,
        default=0.50,
        help="Maximum token imbalance explored before emergency fallback",
    )
    parser.add_argument(
        "--communication-weight",
        type=float,
        default=0.05,
        help="Weight of estimated ring communication in placement scoring",
    )
    parser.add_argument(
        "--local-search-passes",
        type=int,
        default=4,
        help="Number of deterministic local-repair passes; zero disables repair",
    )
    parser.add_argument(
        "--truncated-padding-threshold",
        type=int,
        default=DEFAULT_TRUNCATED_PADDING_THRESHOLD,
        help=(
            "Pad a packing-truncated sequence to 256 * world_size alignment "
            "when its truncated length exceeds this token threshold"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8))
    parser.add_argument("--print-workload", action="store_true")
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--mode", choices=("noncausal", "causal", "both"), default="causal")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--sm-configs", default="128:4,124:8,120:12,116:16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=40)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("--rtol", type=float, default=2e-1)
    return parser.parse_args(argv)


def _world_size(args: argparse.Namespace) -> int:
    env_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if args.print_workload:
        if args.world_size is not None:
            return args.world_size
        if env_world_size is not None:
            return int(env_world_size)
        raise SystemExit("--print-workload requires --world-size outside torchrun")
    if env_world_size is None:
        raise SystemExit("Run this benchmark with torchrun")
    world_size = int(env_world_size)
    if args.world_size is not None and args.world_size != world_size:
        raise SystemExit(
            f"--world-size={args.world_size} does not match LOCAL_WORLD_SIZE={world_size}"
        )
    return world_size


def _requests_all(methods: str) -> bool:
    return any(token.strip() == "all" for token in methods.split(","))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    world_size = _world_size(args)
    try:
        workload = make_workload(
            dataset=args.dataset,
            target_tokens=args.target_tokens,
            seed=args.seed,
            world_size=world_size,
            mode=args.mode,
            balance_tolerance=args.balance_tolerance,
            token_balance_tolerance=args.token_balance_tolerance,
            max_compute_balance_tolerance=args.max_compute_balance_tolerance,
            max_token_balance_tolerance=args.max_token_balance_tolerance,
            communication_weight=args.communication_weight,
            local_search_passes=args.local_search_passes,
            truncated_padding_threshold=args.truncated_padding_threshold,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.print_workload or local_rank == 0:
        print_workload(
            workload,
            args.dataset,
            args.seed,
            args.target_tokens,
            world_size,
            args.mode,
        )
    if args.print_workload:
        return

    forwarded_argv = [
        "--global-seqlens",
        _format_int_list(workload.global_lengths),
        "--ring-sizes",
        _format_int_list(workload.ring_sizes),
        "--ring-starts",
        _format_int_list(workload.ring_starts),
        "--qhead",
        str(args.qhead),
        "--kvhead",
        str(args.kvhead),
        "--headdim",
        str(args.headdim),
        "--mode",
        args.mode,
        "--methods",
        args.methods,
        "--sm-configs",
        args.sm_configs,
        "--warmup-iters",
        str(args.warmup_iters),
        "--num-iters",
        str(args.num_iters),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--seed",
        str(args.seed),
        "--check" if args.check else "--no-check",
    ]

    import benchmark_hybrid_forward

    benchmark_hybrid_forward.main(
        forwarded_argv,
        skip_incompatible_methods=_requests_all(args.methods),
    )


if __name__ == "__main__":
    main()
