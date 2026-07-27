"""Single-node Zeppelin sequence grouping copied from paper Algorithm 2."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_ZEPPELIN_THRESHOLD = 4096


@dataclass(frozen=True)
class ZeppelinAssignment:
    """One raw sequence, its padded execution length, and ordered GPU group."""

    sample_id: int
    raw_length: int
    execution_length: int
    group_members: tuple[int, ...]

    @property
    def group_size(self) -> int:
        return len(self.group_members)

    @property
    def local_length(self) -> int:
        return self.execution_length // self.group_size

    @property
    def padding(self) -> int:
        return self.execution_length - self.raw_length


@dataclass(frozen=True)
class ZeppelinPlan:
    """Deterministic single-node Algorithm 2 placement and packed layout."""

    global_lengths: tuple[int, ...]
    world_size: int
    is_causal: bool
    threshold: int
    effective_threshold: int
    iterations: int
    assignments: tuple[ZeppelinAssignment, ...]
    bucket_loads: tuple[int, ...]

    def assignment(self, sample_id: int) -> ZeppelinAssignment:
        if not 0 <= sample_id < len(self.assignments):
            raise ValueError(
                f"sample_id must be in [0, {len(self.assignments)}), got {sample_id}"
            )
        assignment = self.assignments[sample_id]
        if assignment.sample_id != sample_id:
            raise RuntimeError("Zeppelin assignments lost sample-id ordering")
        return assignment

    @property
    def distributed_group_members(self) -> tuple[tuple[int, ...], ...]:
        groups = {
            assignment.group_members
            for assignment in self.assignments
            if assignment.group_size > 1
        }
        return tuple(sorted(groups, key=lambda members: (-len(members), members)))

    def assignments_for_group(
        self, group_members: tuple[int, ...]
    ) -> tuple[ZeppelinAssignment, ...]:
        return tuple(
            assignment
            for assignment in self.assignments
            if assignment.group_members == group_members
        )

    @property
    def distributed_queues(
        self,
    ) -> tuple[tuple[tuple[int, ...], tuple[ZeppelinAssignment, ...]], ...]:
        return tuple(
            (members, self.assignments_for_group(members))
            for members in self.distributed_group_members
        )

    def local_assignments_for_rank(
        self, rank: int
    ) -> tuple[ZeppelinAssignment, ...]:
        self._validate_rank(rank)
        return tuple(
            assignment
            for assignment in self.assignments
            if assignment.group_members == (rank,)
        )

    def packed_assignments_for_rank(
        self, rank: int
    ) -> tuple[ZeppelinAssignment, ...]:
        """Physical layout: all G>1 queues, followed by the rank's G1 queue."""

        self._validate_rank(rank)
        distributed = tuple(
            assignment
            for members, assignments in self.distributed_queues
            if rank in members
            for assignment in assignments
        )
        return distributed + self.local_assignments_for_rank(rank)

    def packed_lengths_for_rank(self, rank: int) -> list[int]:
        return [
            assignment.local_length
            for assignment in self.packed_assignments_for_rank(rank)
        ]

    def packed_sample_ids_for_rank(self, rank: int) -> tuple[int, ...]:
        return tuple(
            assignment.sample_id
            for assignment in self.packed_assignments_for_rank(rank)
        )

    @property
    def execution_lengths(self) -> tuple[int, ...]:
        return tuple(
            assignment.execution_length for assignment in self.assignments
        )

    @property
    def padding_tokens(self) -> int:
        return sum(assignment.padding for assignment in self.assignments)

    @property
    def rank_token_loads(self) -> tuple[int, ...]:
        return tuple(
            sum(
                assignment.local_length
                for assignment in self.assignments
                if rank in assignment.group_members
            )
            for rank in range(self.world_size)
        )

    @property
    def raw_rank_token_loads(self) -> tuple[float, ...]:
        return tuple(
            sum(
                assignment.raw_length / assignment.group_size
                for assignment in self.assignments
                if rank in assignment.group_members
            )
            for rank in range(self.world_size)
        )

    def _validate_rank(self, rank: int) -> None:
        if not 0 <= rank < self.world_size:
            raise ValueError(
                f"rank must be in [0, {self.world_size}), got {rank}"
            )


def _input_incompatibility(
    global_lengths: list[int], world_size: int, is_causal: bool, threshold: int
) -> str | None:
    if type(world_size) is not int or world_size <= 0:
        return f"world_size must be a positive integer, got {world_size!r}"
    if type(threshold) is not int or threshold <= 0:
        return f"Zeppelin threshold must be a positive integer, got {threshold!r}"
    if type(is_causal) is not bool:
        return f"is_causal must be a bool, got {is_causal!r}"
    if not global_lengths:
        return "Zeppelin requires at least one sequence length"
    for sample_id, raw_length in enumerate(global_lengths):
        if type(raw_length) is not int or raw_length <= 0:
            return (
                "Zeppelin requires positive integer sequence lengths: "
                f"sample={sample_id}, raw_length={raw_length!r}"
            )
    return None


def _algorithm_2_plan(
    global_lengths: list[int],
    world_size: int,
    is_causal: bool,
    threshold: int,
) -> ZeppelinPlan:
    ordered = sorted(
        enumerate(global_lengths), key=lambda item: (-item[1], item[0])
    )
    total_squared_length = sum(length * length for length in global_lengths)
    effective_threshold = threshold
    iterations = 0

    while True:
        iterations += 1
        z0 = [item for item in ordered if item[1] < effective_threshold]
        z1 = [item for item in ordered if item[1] >= effective_threshold]
        assignments: dict[int, ZeppelinAssignment] = {}
        bucket_loads = [0] * world_size
        cursor = 0

        for sample_id, raw_length in z1:
            numerator = raw_length * raw_length * world_size
            group_size = min(
                world_size,
                (numerator + total_squared_length - 1) // total_squared_length,
            )
            if group_size == world_size:
                members = tuple(range(world_size))
            else:
                members = tuple(
                    (cursor + fragment) % world_size
                    for fragment in range(group_size)
                )
            cursor = (cursor + group_size) % world_size
            alignment = group_size * (2 if is_causal else 1)
            execution_length = (
                (raw_length + alignment - 1) // alignment * alignment
            )
            assignments[sample_id] = ZeppelinAssignment(
                sample_id=sample_id,
                raw_length=raw_length,
                execution_length=execution_length,
                group_members=members,
            )

        capacity_failed = False
        for sample_id, raw_length in z0:
            owner = min(
                range(world_size), key=lambda rank: (bucket_loads[rank], rank)
            )
            if bucket_loads[owner] + raw_length > effective_threshold:
                capacity_failed = True
                break
            bucket_loads[owner] += raw_length
            alignment = 2 if is_causal else 1
            execution_length = (
                (raw_length + alignment - 1) // alignment * alignment
            )
            assignments[sample_id] = ZeppelinAssignment(
                sample_id=sample_id,
                raw_length=raw_length,
                execution_length=execution_length,
                group_members=(owner,),
            )

        if not capacity_failed:
            return ZeppelinPlan(
                global_lengths=tuple(global_lengths),
                world_size=world_size,
                is_causal=is_causal,
                threshold=threshold,
                effective_threshold=effective_threshold,
                iterations=iterations,
                assignments=tuple(
                    assignments[sample_id]
                    for sample_id in range(len(global_lengths))
                ),
                bucket_loads=tuple(bucket_loads),
            )

        effective_threshold = max(raw_length for _, raw_length in z0)


def zeppelin_incompatibility(
    global_lengths: list[int],
    world_size: int,
    is_causal: bool,
    threshold: int,
) -> str | None:
    return _input_incompatibility(
        global_lengths, world_size, is_causal, threshold
    )


def make_zeppelin_plan(
    global_lengths: list[int],
    world_size: int,
    is_causal: bool,
    threshold: int = DEFAULT_ZEPPELIN_THRESHOLD,
) -> ZeppelinPlan:
    reason = _input_incompatibility(
        global_lengths, world_size, is_causal, threshold
    )
    if reason is not None:
        raise ValueError(reason)
    return _algorithm_2_plan(global_lengths, world_size, is_causal, threshold)


def zeppelin_note(plan: ZeppelinPlan, backend_name: str) -> str:
    group_counts = ",".join(
        f"G{group_size}={sum(a.group_size == group_size for a in plan.assignments)}"
        for group_size in sorted({a.group_size for a in plan.assignments})
    )
    loads = ",".join(str(load) for load in plan.rank_token_loads)
    return (
        f"L={plan.threshold}; final_s0={plan.effective_threshold}; "
        f"iterations={plan.iterations}; {group_counts}; padding={plan.padding_tokens}; "
        f"physical rank token loads=[{loads}]; {backend_name}"
    )


__all__ = [
    "DEFAULT_ZEPPELIN_THRESHOLD",
    "ZeppelinAssignment",
    "ZeppelinPlan",
    "make_zeppelin_plan",
    "zeppelin_incompatibility",
    "zeppelin_note",
]
