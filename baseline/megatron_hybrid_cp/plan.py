"""Compile and validate standalone Megatron hybrid-CP execution plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.distributed as dist

from .scheduler import BalancedCPScheduler


SUPPORTED_WORLD_SIZES = (2, 4, 8)
SUPPORTED_CP_SIZES = (1, 2, 4, 8)


def _needs_inter_group_barrier(group_index: int, num_groups: int) -> bool:
    if not 0 <= group_index < num_groups:
        raise ValueError(
            f"group index {group_index} is outside {num_groups} execution groups"
        )
    return group_index + 1 < num_groups


@dataclass(frozen=True)
class SampleAssignment:
    sample_id: int
    global_length: int
    cp_size: int
    rank_start: int
    execution_group_id: int

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(range(self.rank_start, self.rank_start + self.cp_size))


@dataclass(frozen=True)
class ExecutionGroup:
    group_id: int
    sample_ids_by_rank: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class HybridCPPlan:
    global_lengths: tuple[int, ...]
    world_size: int
    max_seqlen_per_rank: int
    execution_groups: tuple[ExecutionGroup, ...]
    assignments: tuple[SampleAssignment, ...]

    def assignment(self, sample_id: int) -> SampleAssignment:
        return self.assignments[sample_id]

    def sample_ids_for_rank(self, rank: int) -> tuple[int, ...]:
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} is outside world_size={self.world_size}")
        return tuple(
            sample_id
            for group in self.execution_groups
            for sample_id in group.sample_ids_by_rank[rank]
        )

    def local_lengths_for_rank(self, rank: int) -> tuple[int, ...]:
        return tuple(
            self.assignments[sample_id].global_length
            // self.assignments[sample_id].cp_size
            for sample_id in self.sample_ids_for_rank(rank)
        )

    @property
    def num_execution_groups(self) -> int:
        return len(self.execution_groups)


def _members_for_sample(
    sample_ids_by_rank: tuple[tuple[int, ...], ...], sample_id: int
) -> tuple[int, ...]:
    return tuple(
        rank
        for rank, sample_ids in enumerate(sample_ids_by_rank)
        if sample_id in sample_ids
    )


def validate_hybrid_cp_plan(plan: HybridCPPlan) -> None:
    if plan.world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLD_SIZES}, got {plan.world_size}"
        )
    if plan.max_seqlen_per_rank <= 0:
        raise ValueError("max_seqlen_per_rank must be positive")
    if not plan.global_lengths or any(length <= 0 for length in plan.global_lengths):
        raise ValueError("global_lengths must contain positive lengths")
    if len(plan.assignments) != len(plan.global_lengths):
        raise ValueError("plan must contain exactly one assignment per sample")

    seen: set[int] = set()
    for expected_group_id, group in enumerate(plan.execution_groups):
        if group.group_id != expected_group_id:
            raise ValueError("execution group ids must be contiguous and ordered")
        if len(group.sample_ids_by_rank) != plan.world_size:
            raise ValueError("every execution group must describe every rank")
        for rank, sample_ids in enumerate(group.sample_ids_by_rank):
            if len(sample_ids) != len(set(sample_ids)):
                raise ValueError(
                    f"execution group {group.group_id} rank {rank} repeats a sample"
                )
        group_samples = {
            sample_id
            for rank_samples in group.sample_ids_by_rank
            for sample_id in rank_samples
        }
        for sample_id in group_samples:
            if not 0 <= sample_id < len(plan.global_lengths):
                raise ValueError(f"invalid sample id {sample_id}")
            if sample_id in seen:
                raise ValueError(
                    f"sample {sample_id} belongs to more than one execution group"
                )
            seen.add(sample_id)
            members = _members_for_sample(group.sample_ids_by_rank, sample_id)
            assignment = plan.assignments[sample_id]
            if assignment.execution_group_id != group.group_id:
                raise ValueError(f"sample {sample_id} has the wrong group id")
            if members != assignment.ranks:
                raise ValueError(f"sample {sample_id} has inconsistent members")

        member_sets = {
            plan.assignments[sample_id].ranks for sample_id in group_samples
        }
        member_sets_list = list(member_sets)
        for index, left in enumerate(member_sets_list):
            for right in member_sets_list[index + 1 :]:
                if set(left).intersection(right):
                    raise ValueError(
                        f"execution group {group.group_id} has overlapping "
                        f"CP members {left} and {right}"
                    )
        for members in member_sets:
            schedules = [
                tuple(
                    sample_id
                    for sample_id in group.sample_ids_by_rank[rank]
                    if plan.assignments[sample_id].ranks == members
                )
                for rank in members
            ]
            if any(schedule != schedules[0] for schedule in schedules[1:]):
                raise ValueError(
                    f"members {members} execute samples in inconsistent order"
                )

    expected_samples = set(range(len(plan.global_lengths)))
    if seen != expected_samples:
        missing = sorted(expected_samples - seen)
        raise ValueError(f"plan does not schedule samples {missing}")

    for sample_id, assignment in enumerate(plan.assignments):
        if assignment.sample_id != sample_id:
            raise ValueError("assignments must be indexed by sample id")
        if assignment.global_length != plan.global_lengths[sample_id]:
            raise ValueError(f"sample {sample_id} has the wrong global length")
        if (
            assignment.cp_size not in SUPPORTED_CP_SIZES
            or assignment.cp_size > plan.world_size
        ):
            raise ValueError(
                f"sample {sample_id} has unsupported CP size {assignment.cp_size}"
            )
        if (
            assignment.rank_start < 0
            or assignment.rank_start % assignment.cp_size
            or assignment.rank_start + assignment.cp_size > plan.world_size
        ):
            raise ValueError(
                f"sample {sample_id} has invalid aligned rank range"
            )


def build_hybrid_cp_plan(
    global_lengths: list[int] | tuple[int, ...],
    world_size: int,
    max_seqlen_per_rank: int = 8192,
) -> HybridCPPlan:
    lengths = tuple(int(length) for length in global_lengths)
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    scheduler = BalancedCPScheduler(max_seqlen_per_rank, world_size)
    required = [scheduler.gpus_needed(length) for length in lengths]
    if any(cp_size > world_size for cp_size in required):
        sample_id = next(
            index for index, cp_size in enumerate(required) if cp_size > world_size
        )
        raise ValueError(
            f"sample {sample_id} length {lengths[sample_id]} requires CP{required[sample_id]}, "
            f"which exceeds world_size={world_size}"
        )

    _, raw_groups = scheduler.get_groups_and_subsamples(list(enumerate(lengths)))
    execution_groups = tuple(
        ExecutionGroup(
            group_id,
            tuple(tuple(sample_ids) for sample_ids in sample_ids_by_rank),
        )
        for group_id, sample_ids_by_rank in enumerate(raw_groups)
    )

    assignments_by_id: dict[int, SampleAssignment] = {}
    for group in execution_groups:
        group_samples = {
            sample_id
            for rank_samples in group.sample_ids_by_rank
            for sample_id in rank_samples
        }
        for sample_id in group_samples:
            members = _members_for_sample(group.sample_ids_by_rank, sample_id)
            if not members:
                raise AssertionError("scheduled sample has no member ranks")
            assignments_by_id[sample_id] = SampleAssignment(
                sample_id=sample_id,
                global_length=lengths[sample_id],
                cp_size=len(members),
                rank_start=members[0],
                execution_group_id=group.group_id,
            )
    assignments = tuple(
        assignments_by_id[sample_id] for sample_id in range(len(lengths))
    )
    plan = HybridCPPlan(
        global_lengths=lengths,
        world_size=world_size,
        max_seqlen_per_rank=max_seqlen_per_rank,
        execution_groups=execution_groups,
        assignments=assignments,
    )
    validate_hybrid_cp_plan(plan)
    return plan


@dataclass
class HybridCPProcessGroups:
    world_group: dist.ProcessGroup
    world_size: int
    groups: dict[tuple[int, int], dist.ProcessGroup]
    creation_order: tuple[tuple[int, int, tuple[int, ...]], ...]

    def group_for(
        self, cp_size: int, rank_start: int
    ) -> Optional[dist.ProcessGroup]:
        if cp_size == 1:
            return None
        if cp_size == self.world_size and rank_start == 0:
            return self.world_group
        try:
            return self.groups[(cp_size, rank_start)]
        except KeyError as exc:
            raise ValueError(
                f"no process group for CP{cp_size} ranks "
                f"{rank_start}-{rank_start + cp_size - 1}"
            ) from exc


def create_hybrid_cp_process_groups(
    world_group: dist.ProcessGroup,
) -> HybridCPProcessGroups:
    world_size = dist.get_world_size(world_group)
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    world_ranks = dist.get_process_group_ranks(world_group)
    groups: dict[tuple[int, int], dist.ProcessGroup] = {}
    creation_order: list[tuple[int, int, tuple[int, ...]]] = []
    cp_size = 2
    while cp_size < world_size:
        for rank_start in range(0, world_size, cp_size):
            global_ranks = tuple(
                world_ranks[rank_start : rank_start + cp_size]
            )
            creation_order.append((cp_size, rank_start, global_ranks))
            groups[(cp_size, rank_start)] = dist.new_group(
                ranks=list(global_ranks)
            )
        cp_size *= 2
    return HybridCPProcessGroups(
        world_group=world_group,
        world_size=world_size,
        groups=groups,
        creation_order=tuple(creation_order),
    )


def hybrid_cp_incompatibility(
    global_lengths: list[int] | tuple[int, ...],
    world_size: int,
    is_causal: bool,
    max_seqlen_per_rank: int = 8192,
) -> Optional[str]:
    try:
        plan = build_hybrid_cp_plan(
            global_lengths, world_size, max_seqlen_per_rank
        )
    except (AssertionError, ValueError) as exc:
        return str(exc)
    for assignment in plan.assignments:
        if assignment.global_length % assignment.cp_size:
            return (
                f"sample {assignment.sample_id} length {assignment.global_length} "
                f"is not divisible by CP{assignment.cp_size}"
            )
        local_length = assignment.global_length // assignment.cp_size
        if is_causal and assignment.cp_size > 1 and (
            local_length % 2 or (local_length // 2) % 128
        ):
            return (
                f"sample {assignment.sample_id} causal CP{assignment.cp_size} "
                f"local half length {local_length // 2} is not 128-aligned"
            )
    return None


__all__ = [
    "ExecutionGroup",
    "HybridCPPlan",
    "HybridCPProcessGroups",
    "SampleAssignment",
    "build_hybrid_cp_plan",
    "create_hybrid_cp_process_groups",
    "hybrid_cp_incompatibility",
    "validate_hybrid_cp_plan",
]
