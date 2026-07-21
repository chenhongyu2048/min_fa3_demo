# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Standalone copy of Megatron-LM's hybrid-CP length scheduler.

Copied and trimmed from ``megatron/core/pipeline_parallel/hybrid_cp_schedule.py``
at Megatron-LM commit ``368fa88e382b274c8fc12af851331cc1d30d69cc``.
Only the length bucketing and execution-group construction are retained.  This
module deliberately has no Megatron-LM, model, dataloader, or distributed
runtime dependency.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from math import ceil, log2
from typing import Callable, Optional


class BalancedCPScheduler:
    """Form groups of samples with roughly balanced per-rank workload."""

    def __init__(self, max_seq_len_per_rank: int, world_size: int):
        if max_seq_len_per_rank <= 0:
            raise ValueError("max_seq_len_per_rank must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.max_seq_len_per_rank = max_seq_len_per_rank
        self.total_hdp_gpus = world_size

    @lru_cache(maxsize=128)
    def get_total_workload(
        self, seq_length: int, cp_size: Optional[int] = None
    ) -> float:
        if cp_size is None:
            cp_size = self.gpus_needed(seq_length)
        return (seq_length * seq_length) / cp_size

    @lru_cache(maxsize=128)
    def gpus_needed(self, seq_len: int) -> int:
        if seq_len <= 0:
            raise ValueError("sequence lengths must be positive")
        return max(
            1,
            2 ** ceil(log2(seq_len / self.max_seq_len_per_rank)),
        )

    def make_buckets_equal(
        self,
        sample_seqlens: list[tuple[int, int]],
        compute_estimator: Callable[..., float],
    ) -> list[deque[tuple[int, int]]]:
        seqlens = [seq_len for _, seq_len in sample_seqlens]
        bucket_count = len({self.gpus_needed(length) for length in seqlens})

        work = []
        for _, seq_len in sample_seqlens:
            cp_size = self.gpus_needed(seq_len)
            work.append(compute_estimator(seq_len, cp_size))
        total_work = sum(work)
        target = total_work / bucket_count
        buckets: list[deque[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        current_work = 0.0
        remaining_work = total_work
        remaining_count = bucket_count

        for index, (sample_id, seq_len) in enumerate(sample_seqlens):
            item_work = compute_estimator(seq_len)
            projected = current_work + item_work
            if current and (
                projected > target * 1.1
                or len(sample_seqlens) - index
                <= remaining_count - len(buckets)
            ):
                buckets.append(deque(current))
                current, current_work = [], 0.0
                # Preserve the source scheduler's update order and arithmetic.
                remaining_work -= sum(
                    compute_estimator(length) for _, length in current
                )
                remaining_count -= 1

            current.append((sample_id, seq_len))
            current_work += item_work

        if current:
            buckets.append(deque(current))
        del remaining_work
        return buckets

    def next_hdp_group(
        self,
        sample_seqlens: list[tuple[int, int]],
        compute_estimator: Callable[..., float],
        total_gpus: int,
        delta: float = 0.05,
        strategy: str = "dp",
        eps_bucket: float = 0.10,
    ) -> tuple[
        list[list[int]],
        list[tuple[int, int]],
        list[float],
        list[list[int]],
    ]:
        """Return one Megatron execution group and the unplaced samples."""
        del eps_bucket
        if strategy not in ("dp", "pp"):
            raise ValueError(f"unknown scheduling strategy: {strategy}")
        if not sample_seqlens:
            return (
                [[] for _ in range(total_gpus)],
                [],
                [0.0 for _ in range(total_gpus)],
                [[] for _ in range(total_gpus)],
            )

        buckets = self.make_buckets_equal(sample_seqlens, compute_estimator)
        micro_batches: list[list[int]] = [[] for _ in range(total_gpus)]
        exec_times = [0.0 for _ in range(total_gpus)]
        sample_ids_per_gpu: list[list[int]] = [
            [] for _ in range(total_gpus)
        ]

        gpu_group_id: list[Optional[int]] = [None] * total_gpus
        group_members: dict[int, list[int]] = {}
        group_size: dict[int, int] = {}
        next_gid = 0
        pp_cursor = 0
        previous_needed: Optional[int] = None
        check_balance = False

        while buckets:
            sample_seq_tuple = None
            bucket_index = None
            scan_order = (
                range(len(buckets))
                if strategy == "dp"
                else [
                    (pp_cursor + index) % len(buckets)
                    for index in range(len(buckets))
                ]
            )
            for index in scan_order:
                if not buckets[index]:
                    continue
                candidate = buckets[index][0]
                needed = self.gpus_needed(candidate[1])
                candidate_gids = [
                    gid for gid, size in group_size.items() if size == needed
                ]
                free_ranks = [
                    rank
                    for rank, gid in enumerate(gpu_group_id)
                    if gid is None
                ]
                if candidate_gids or len(free_ranks) >= needed:
                    sample_seq_tuple, bucket_index = candidate, index
                    break
            if sample_seq_tuple is None or bucket_index is None:
                break

            if strategy == "pp":
                pp_cursor = (bucket_index + 1) % len(buckets)

            sample_id, seq_len = sample_seq_tuple
            needed = self.gpus_needed(seq_len)
            if previous_needed is None:
                previous_needed = needed

            candidate_gids = [
                gid for gid, size in group_size.items() if size == needed
            ]
            if candidate_gids:
                best_gid, best_load = min(
                    (
                        (
                            gid,
                            max(exec_times[rank] for rank in group_members[gid]),
                        )
                        for gid in candidate_gids
                    ),
                    key=lambda item: item[1],
                )
            else:
                best_gid, best_load = None, float("inf")

            free_ranks = [
                rank for rank, gid in enumerate(gpu_group_id) if gid is None
            ]
            if len(free_ranks) >= needed:
                free_sorted = sorted(free_ranks, key=lambda rank: exec_times[rank])
                new_members = free_sorted[:needed]
                new_load = exec_times[new_members[-1]]
                if new_load < best_load:
                    best_gid = None
                    chosen_members = new_members
                else:
                    if best_gid is None:
                        raise AssertionError("scheduler lost an existing group")
                    chosen_members = group_members[best_gid]
            else:
                if best_gid is None:
                    raise AssertionError("scheduler could not place a sample")
                chosen_members = group_members[best_gid]

            if best_gid is None:
                best_gid = next_gid
                next_gid += 1
                group_members[best_gid] = chosen_members
                group_size[best_gid] = needed
                for rank in chosen_members:
                    gpu_group_id[rank] = best_gid

            per_gpu_cost = compute_estimator(seq_len)
            for rank in chosen_members:
                micro_batches[rank].append(seq_len)
                exec_times[rank] += per_gpu_cost
                sample_ids_per_gpu[rank].append(sample_id)
            buckets[bucket_index].popleft()

            while buckets and not buckets[0]:
                buckets.pop(0)
                pp_cursor %= max(1, len(buckets))

            if needed < previous_needed:
                check_balance = True
            if (
                check_balance
                and buckets
                and max(exec_times) - min(exec_times)
                <= delta * max(exec_times)
            ):
                break

        leftovers = [item for bucket in buckets for item in bucket]

        def trim_overload() -> None:
            while True:
                current_max = max(exec_times)
                current_min = min(exec_times)
                current_slack = current_max - current_min
                if current_slack <= delta * current_max or current_min == 0:
                    break
                max_rank = exec_times.index(current_max)
                gid = gpu_group_id[max_rank]
                if gid is None:
                    raise AssertionError("loaded rank has no scheduler group")
                members = group_members[gid]
                if (
                    not micro_batches[max_rank]
                    or len(micro_batches[max_rank]) <= 1
                ):
                    break
                seq_len = micro_batches[max_rank][-1]
                per_gpu_cost = compute_estimator(seq_len)
                projected_times = exec_times[:]
                for rank in members:
                    projected_times[rank] -= per_gpu_cost
                projected_slack = max(projected_times) - min(projected_times)
                if projected_slack >= current_slack:
                    break
                sample_id = sample_ids_per_gpu[max_rank][-1]
                for rank in members:
                    micro_batches[rank].pop()
                    exec_times[rank] -= per_gpu_cost
                    sample_ids_per_gpu[rank].pop()
                leftovers.append((sample_id, seq_len))

        trim_overload()
        total_work_before = sum(len(batch) for batch in micro_batches)

        def fill_empty_gpus() -> tuple[
            list[list[int]],
            list[float],
            list[list[int]],
        ]:
            empty_gpus = [
                rank for rank, batch in enumerate(micro_batches) if not batch
            ]
            if not empty_gpus:
                return micro_batches, exec_times, sample_ids_per_gpu
            existing_group_sizes = set(group_size.values())
            if not existing_group_sizes:
                raise AssertionError(
                    "cannot redistribute an execution group with no work"
                )
            min_group_size = min(existing_group_sizes)
            next_power = min(min_group_size * 2, total_gpus)

            for gid, size in group_size.items():
                if size != min_group_size:
                    continue
                members = group_members[gid]
                needed_count = next_power - min_group_size
                group_start = members[0]
                group_end = members[-1]
                empty_gpu = next(
                    rank
                    for rank, batch in enumerate(micro_batches)
                    if not batch
                )
                if all(
                    micro_batches[
                        empty_gpu : empty_gpu + needed_count
                    ]
                ):
                    raise AssertionError(
                        "empty GPUs were detected but the group cannot expand"
                    )
                work_to_push = micro_batches[group_end + 1 : empty_gpu]
                times_to_push = exec_times[group_end + 1 : empty_gpu]
                ids_to_push = sample_ids_per_gpu[group_end + 1 : empty_gpu]

                new_batches: list[list[int]] = [
                    [] for _ in range(len(micro_batches))
                ]
                new_times = [0.0] * len(exec_times)
                new_ids: list[list[int]] = [
                    [] for _ in range(len(sample_ids_per_gpu))
                ]
                for rank in range(group_start):
                    new_batches[rank] = micro_batches[rank]
                    new_times[rank] = exec_times[rank]
                    new_ids[rank] = sample_ids_per_gpu[rank]
                for rank in range(
                    group_start, group_end + needed_count + 1
                ):
                    new_batches[rank] = micro_batches[group_end]
                    new_times[rank] = self.get_total_workload(
                        micro_batches[group_end][0], next_power
                    )
                    new_ids[rank] = sample_ids_per_gpu[group_end]
                for index, batch in enumerate(work_to_push):
                    destination = group_end + needed_count + 1 + index
                    new_batches[destination] = batch
                    new_times[destination] = times_to_push[index]
                    new_ids[destination] = ids_to_push[index]

                group_size[gid] = next_power
                group_members[gid] = list(
                    range(members[0], members[-1] + needed_count + 1)
                )
                for pushed_gid in group_size:
                    if pushed_gid > gid:
                        group_members[pushed_gid] = [
                            rank + needed_count
                            for rank in group_members[pushed_gid]
                        ]
                return new_batches, new_times, new_ids
            raise AssertionError("no scheduler group could be expanded")

        while any(not batch for batch in micro_batches):
            (
                micro_batches,
                exec_times,
                sample_ids_per_gpu,
            ) = fill_empty_gpus()

        total_work_after = sum(len(batch) for batch in micro_batches)
        if total_work_after < total_work_before:
            raise AssertionError(
                f"samples were removed: {total_work_before} -> {total_work_after}"
            )
        return micro_batches, leftovers, exec_times, sample_ids_per_gpu

    def get_groups_and_subsamples(
        self, sample_id_seqlens: list[tuple[int, int]]
    ) -> tuple[list[list[list[int]]], list[list[list[int]]]]:
        groups: list[list[list[int]]] = []
        sample_id_groups: list[list[list[int]]] = []
        remaining = sorted(sample_id_seqlens, key=lambda item: item[1], reverse=True)
        while remaining:
            micro_batches, remaining, _, sample_ids = self.next_hdp_group(
                remaining,
                self.get_total_workload,
                self.total_hdp_gpus,
            )
            groups.append(micro_batches)
            sample_id_groups.append(sample_ids)
        return groups, sample_id_groups


__all__ = ["BalancedCPScheduler"]
