"""CPU-only adapters from existing planners to fused Mega Ring metadata.

The planners remain authoritative.  This module only records their placement
decisions in a common immutable form, orders records for the fused kernel, and
rejects metadata that would require unrequested padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Literal, Sequence

import balancer
from baseline.megatron_hybrid_cp import build_hybrid_cp_plan_for_fa3_ring
from ring_test.zepplin import DEFAULT_ZEPPLIN_THRESHOLD, make_zepplin_plan


PlannerName = Literal["br_pbs", "megatron_cp", "zepplin"]


@dataclass(frozen=True)
class PlannerControls:
    """BR-PBS controls forwarded unchanged to ``assign_hierarchical_rings``."""

    compute_balance_tolerance: float = 0.05
    token_balance_tolerance: float = 0.10
    beam_width: int = 64
    finalist_count: int = 8
    structure_threshold: float = 0.5
    max_repair_iterations: int = 32


@dataclass(frozen=True)
class TopologySample:
    """One raw sample and the placement that will be executed by Mega Ring."""

    sample_id: int
    raw_length: int
    execution_length: int
    ring_size: int
    ring_start: int

    @property
    def padding(self) -> int:
        return self.execution_length - self.raw_length


@dataclass(frozen=True)
class PlannerTopology:
    """Immutable fused-Mega-Ring view of one planner's placement."""

    planner: PlannerName
    is_causal: bool
    world_size: int
    raw_lengths: tuple[int, ...]
    samples: tuple[TopologySample, ...]
    planner_build_ms: float
    diagnostics: tuple[tuple[str, str], ...] = ()

    @property
    def global_lengths(self) -> tuple[int, ...]:
        return tuple(sample.execution_length for sample in self.samples)

    @property
    def ring_sizes(self) -> tuple[int, ...]:
        return tuple(sample.ring_size for sample in self.samples)

    @property
    def ring_starts(self) -> tuple[int, ...]:
        return tuple(sample.ring_start for sample in self.samples)

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return tuple(sample.sample_id for sample in self.samples)

    @property
    def raw_tokens(self) -> int:
        return sum(self.raw_lengths)

    @property
    def execution_tokens(self) -> int:
        return sum(self.global_lengths)

    @property
    def padding_tokens(self) -> int:
        return self.execution_tokens - self.raw_tokens

    def diagnostic(self, name: str) -> str | None:
        return dict(self.diagnostics).get(name)


def _validate_raw_lengths(raw_lengths: Sequence[int], world_size: int) -> tuple[int, ...]:
    lengths = tuple(raw_lengths)
    if world_size not in (2, 4, 8):
        raise ValueError(f"world_size must be 2, 4, or 8, got {world_size}")
    if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("raw_lengths must contain positive integers")
    return lengths


def _kernel_order(samples: Iterable[TopologySample]) -> tuple[TopologySample, ...]:
    """Use Mega Ring's G8/G4/G2/G1 order with deterministic sample ties."""

    return tuple(
        sorted(
            samples,
            key=lambda sample: (
                -sample.ring_size,
                sample.ring_start,
                sample.sample_id,
            ),
        )
    )


def validate_fused_metadata(topology: PlannerTopology) -> None:
    """Validate layout constraints without importing CUDA benchmark modules.

    Both existing runners validate the basic kernel metadata again immediately
    before launch.  Keeping this CPU validation here makes ``--print-workload``
    and the adapter tests independent of CUDA and gives planners a precise
    error when their layout cannot be represented by the fused kernel.
    """

    samples = topology.samples
    if len(samples) != len(topology.raw_lengths):
        raise ValueError("planner output must contain exactly one entry per raw sample")
    if set(topology.sample_ids) != set(range(len(topology.raw_lengths))):
        raise ValueError("planner output must preserve every original sample id exactly once")

    previous_size = 8
    for index, sample in enumerate(samples):
        if sample.raw_length != topology.raw_lengths[sample.sample_id]:
            raise ValueError(
                f"{topology.planner} changed raw length identity for sample "
                f"{sample.sample_id}"
            )
        if sample.ring_size not in (1, 2, 4, 8) or sample.ring_size > previous_size:
            raise ValueError(
                f"{topology.planner} fused metadata has invalid ring size/order "
                f"at sample {sample.sample_id} (batch {index})"
            )
        if sample.ring_size > topology.world_size:
            raise ValueError(
                f"{topology.planner} fused metadata ring G{sample.ring_size} exceeds "
                f"world_size={topology.world_size} for sample {sample.sample_id}"
            )
        if (
            sample.ring_start < 0
            or sample.ring_start % sample.ring_size
            or sample.ring_start + sample.ring_size > topology.world_size
        ):
            raise ValueError(
                f"{topology.planner} fused metadata has invalid buddy rank range "
                f"for sample {sample.sample_id}: G{sample.ring_size} at "
                f"rank {sample.ring_start}"
            )
        if sample.execution_length <= 0 or sample.execution_length % sample.ring_size:
            raise ValueError(
                f"{topology.planner} fused metadata has non-divisible execution "
                f"length for sample {sample.sample_id}: "
                f"length={sample.execution_length}, G{sample.ring_size}"
            )
        local_length = sample.execution_length // sample.ring_size
        if local_length % 128:
            raise ValueError(
                f"{topology.planner} fused Mega Ring requires 128-row local "
                f"alignment: sample={sample.sample_id}, local_length={local_length}"
            )
        if topology.is_causal and sample.ring_size > 1 and (
            local_length % 2 or (local_length // 2) % 128
        ):
            raise ValueError(
                f"{topology.planner} causal fused Mega Ring requires 128-row "
                f"local halves: sample={sample.sample_id}, "
                f"local_length={local_length}"
            )
        previous_size = sample.ring_size


def validate_with_runner(topology: PlannerTopology, direction: Literal["forward", "backward"]) -> None:
    """Invoke the existing runner validator after the CPU-only validation."""

    validate_fused_metadata(topology)
    if direction == "forward":
        from ring_test.benchmark_topology_forward import validate_metadata

        validate_metadata(
            list(topology.global_lengths),
            list(topology.ring_sizes),
            list(topology.ring_starts),
            topology.world_size,
            "causal" if topology.is_causal else "noncausal",
        )
    elif direction == "backward":
        if not topology.is_causal:
            raise ValueError("the existing backward runner supports causal metadata only")
        from ring_test.benchmark_topology_backward import validate_backward_metadata

        validate_backward_metadata(
            list(topology.global_lengths),
            list(topology.ring_sizes),
            list(topology.ring_starts),
            topology.world_size,
        )
    else:
        raise ValueError(f"unknown benchmark direction {direction!r}")


def make_br_pbs_topology(
    raw_lengths: Sequence[int],
    world_size: int,
    is_causal: bool,
    controls: PlannerControls = PlannerControls(),
) -> PlannerTopology:
    """Adapt the existing BR-PBS placement without changing its controls."""

    lengths = _validate_raw_lengths(raw_lengths, world_size)
    begin = perf_counter()
    workload = balancer.assign_hierarchical_rings(
        list(lengths),
        world_size,
        is_causal,
        compute_balance_tolerance=controls.compute_balance_tolerance,
        token_balance_tolerance=controls.token_balance_tolerance,
        beam_width=controls.beam_width,
        finalist_count=controls.finalist_count,
        structure_threshold=controls.structure_threshold,
        max_repair_iterations=controls.max_repair_iterations,
    )
    elapsed_ms = (perf_counter() - begin) * 1_000.0
    samples = _kernel_order(
        TopologySample(sample_id, lengths[sample_id], execution_length, ring_size, ring_start)
        for sample_id, execution_length, ring_size, ring_start in zip(
            workload.sample_ids,
            workload.global_lengths,
            workload.ring_sizes,
            workload.ring_starts,
        )
    )
    topology = PlannerTopology(
        "br_pbs",
        is_causal,
        world_size,
        lengths,
        samples,
        elapsed_ms,
        (
            ("feasible", str(workload.feasible)),
            ("violation", f"{workload.load_violation:.6f}"),
            ("relaxation", workload.relaxation_label),
            ("repair_moves", str(workload.repair_moves)),
        ),
    )
    validate_fused_metadata(topology)
    return topology


def make_megatron_cp_topology(
    raw_lengths: Sequence[int],
    world_size: int,
    is_causal: bool,
    max_seqlen_per_rank: int = 8192,
) -> PlannerTopology:
    """Map final padded FA3-ring CP assignments to Mega Ring metadata."""

    lengths = _validate_raw_lengths(raw_lengths, world_size)
    begin = perf_counter()
    plan = build_hybrid_cp_plan_for_fa3_ring(
        lengths, world_size, is_causal, max_seqlen_per_rank
    )
    elapsed_ms = (perf_counter() - begin) * 1_000.0
    samples = _kernel_order(
        TopologySample(
            assignment.sample_id,
            lengths[assignment.sample_id],
            assignment.global_length,
            assignment.cp_size,
            assignment.rank_start,
        )
        for assignment in plan.assignments
    )
    topology = PlannerTopology(
        "megatron_cp",
        is_causal,
        world_size,
        lengths,
        samples,
        elapsed_ms,
        (
            ("execution_groups", str(plan.num_execution_groups)),
            ("max_seqlen_per_rank", str(plan.max_seqlen_per_rank)),
        ),
    )
    validate_fused_metadata(topology)
    return topology


def make_zepplin_topology(
    raw_lengths: Sequence[int],
    world_size: int,
    is_causal: bool,
    threshold: int = DEFAULT_ZEPPLIN_THRESHOLD,
) -> PlannerTopology:
    """Map Zepllin's LPT G1/Gworld placement to sorted Mega Ring metadata."""

    lengths = _validate_raw_lengths(raw_lengths, world_size)
    begin = perf_counter()
    plan = make_zepplin_plan(list(lengths), world_size, is_causal, threshold)
    elapsed_ms = (perf_counter() - begin) * 1_000.0
    short_owners = dict(zip(plan.short_indices, plan.short_owners))
    samples = _kernel_order(
        TopologySample(
            sample_id,
            length,
            length,
            1 if sample_id in short_owners else world_size,
            short_owners.get(sample_id, 0),
        )
        for sample_id, length in enumerate(lengths)
    )
    topology = PlannerTopology(
        "zepplin",
        is_causal,
        world_size,
        lengths,
        samples,
        elapsed_ms,
        (
            ("threshold", str(plan.threshold)),
            ("short_g1", str(len(plan.short_indices))),
            ("long_gworld", str(len(plan.long_indices))),
            ("short_loads", ",".join(str(load) for load in plan.short_loads)),
        ),
    )
    validate_fused_metadata(topology)
    return topology


def make_planner_topologies(
    raw_lengths: Sequence[int],
    world_size: int,
    is_causal: bool,
    *,
    controls: PlannerControls = PlannerControls(),
    zepplin_threshold: int = DEFAULT_ZEPPLIN_THRESHOLD,
    megatron_max_seqlen_per_rank: int = 8192,
) -> tuple[PlannerTopology, PlannerTopology, PlannerTopology]:
    """Build BR-PBS, Megatron CP, and Zepllin views of one raw batch."""

    return (
        make_br_pbs_topology(raw_lengths, world_size, is_causal, controls),
        make_megatron_cp_topology(
            raw_lengths,
            world_size,
            is_causal,
            megatron_max_seqlen_per_rank,
        ),
        make_zepplin_topology(raw_lengths, world_size, is_causal, zepplin_threshold),
    )


__all__ = [
    "PlannerControls",
    "PlannerName",
    "PlannerTopology",
    "TopologySample",
    "make_br_pbs_topology",
    "make_megatron_cp_topology",
    "make_planner_topologies",
    "make_zepplin_topology",
    "validate_fused_metadata",
    "validate_with_runner",
]
