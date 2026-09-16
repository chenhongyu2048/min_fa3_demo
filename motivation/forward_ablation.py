"""Preallocated causal Mega Ring forward ablation plans for CP4/CP8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


NUM_COMP_SM = 116
NUM_COMM_SM = 16
WORLD_SIZE = 8
BLOCK_M = 128
LEVEL_RING_SIZES = (8, 4, 2, 1)


@dataclass(frozen=True)
class ProfileSpec:
    id: int
    name: str
    executor: str
    scheduler: str
    topology: str
    attention_launches: int
    reduction_launches: int
    recycle_comm: bool
    dynamic_segments: bool

    @property
    def kernel_launches(self) -> int:
        return self.attention_launches + self.reduction_launches


PROFILES = (
    ProfileSpec(
        1,
        "step_external_reduce",
        "cpp_step_runner",
        "causal_step_linear",
        "all_cp",
        8,
        8,
        False,
        False,
    ),
    ProfileSpec(
        2,
        "step_fused_reduce",
        "cpp_step_runner",
        "causal_step_linear",
        "all_cp",
        8,
        0,
        False,
        False,
    ),
    ProfileSpec(
        3,
        "linear_queue_no_recycle",
        "single_megakernel",
        "causal_linear_atomic",
        "all_cp",
        1,
        0,
        False,
        False,
    ),
    ProfileSpec(
        4,
        "linear_queue_recycle",
        "single_megakernel",
        "causal_linear_atomic",
        "all_cp",
        1,
        0,
        True,
        False,
    ),
    ProfileSpec(
        5,
        "dynamic_segment_recycle",
        "production_dynamic_megakernel",
        "dynamic_ready_segment",
        "all_cp",
        1,
        0,
        True,
        True,
    ),
    ProfileSpec(
        6,
        "hybrid_br_pbs",
        "production_dynamic_megakernel",
        "dynamic_ready_segment",
        "br_pbs_cp_hierarchy",
        1,
        0,
        True,
        True,
    ),
)
PROFILE_BY_NAME = {profile.name: profile for profile in PROFILES}
PROFILE_BY_ID = {profile.id: profile for profile in PROFILES}


def resolve_profile(profile: str | int | ProfileSpec) -> ProfileSpec:
    if isinstance(profile, ProfileSpec):
        return profile
    if isinstance(profile, str):
        try:
            return PROFILE_BY_NAME[profile]
        except KeyError as exc:
            raise ValueError(f"unknown forward ablation profile {profile!r}") from exc
    try:
        return PROFILE_BY_ID[profile]
    except KeyError as exc:
        raise ValueError(f"unknown forward ablation profile id {profile}") from exc


def profile_dispatch_manifest(
    world_size: int = WORLD_SIZE,
) -> tuple[dict[str, object], ...]:
    if world_size not in (4, 8):
        raise ValueError(f"forward ablation supports CP world_size in (4, 8), got {world_size}")
    return tuple(
        {
            "id": profile.id,
            "name": profile.name,
            "executor": profile.executor,
            "scheduler": profile.scheduler,
            "topology": profile.topology,
            "attention_launches": (
                world_size if profile.id <= 2 else profile.attention_launches
            ),
            "reduction_launches": (
                world_size if profile.id == 1 else profile.reduction_launches
            ),
            "recycle_comm": profile.recycle_comm,
            "dynamic_segments": profile.dynamic_segments,
        }
        for profile in PROFILES
    )


def canonicalize_lengths(
    lengths: Sequence[int], alignment: int = 2048
) -> tuple[int, ...]:
    if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("lengths must contain positive integers")
    return tuple((length + alignment - 1) // alignment * alignment for length in lengths)


def make_cu_seqlens(
    lengths: Sequence[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    for index, length in enumerate(lengths):
        host[index + 1] = host[index] + int(length)
    return host.to(device=device), host


def local_lengths_for_rank(
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    rank: int,
) -> tuple[int, ...]:
    return tuple(
        global_length // ring_size
        if ring_start <= rank < ring_start + ring_size
        else 0
        for global_length, ring_size, ring_start in zip(
            global_lengths, ring_sizes, ring_starts
        )
    )


def _validate_topology(
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    rank: int,
    local_lengths: Sequence[int],
    world_size: int = WORLD_SIZE,
) -> None:
    if not (
        len(global_lengths)
        == len(ring_sizes)
        == len(ring_starts)
        == len(local_lengths)
    ):
        raise ValueError("topology vectors and local lengths must have equal size")
    previous_size = world_size
    allowed_sizes = tuple(1 << index for index in range((world_size.bit_length())))
    expected = local_lengths_for_rank(global_lengths, ring_sizes, ring_starts, rank)
    for batch, (global_length, ring_size, ring_start, local_length) in enumerate(
        zip(global_lengths, ring_sizes, ring_starts, local_lengths)
    ):
        if ring_size not in allowed_sizes or ring_size > previous_size:
            raise ValueError(f"invalid ring size/order at batch {batch}")
        if ring_start < 0 or ring_start % ring_size or ring_start + ring_size > world_size:
            raise ValueError(f"invalid ring start at batch {batch}")
        if global_length <= 0 or global_length % ring_size:
            raise ValueError(f"invalid global length at batch {batch}")
        if local_length != expected[batch]:
            raise ValueError(f"local length does not match topology at batch {batch}")
        if local_length % BLOCK_M:
            raise ValueError(f"local length is not 128-aligned at batch {batch}")
        if ring_size > 1 and local_length and (local_length // 2) % BLOCK_M:
            raise ValueError(f"causal local half is not 128-aligned at batch {batch}")
        previous_size = ring_size


def build_hierarchy(
    cu_seqlens_host: torch.Tensor,
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    rank: int,
    q_heads: int,
    world_size: int = WORLD_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    if cu_seqlens_host.device.type != "cpu" or cu_seqlens_host.dtype != torch.int32:
        raise ValueError("cu_seqlens_host must be a CPU int32 tensor")
    local_lengths = tuple(
        int(cu_seqlens_host[index + 1] - cu_seqlens_host[index])
        for index in range(cu_seqlens_host.numel() - 1)
    )
    if world_size not in (4, 8):
        raise ValueError(f"forward ablation supports CP world_size in (4, 8), got {world_size}")
    _validate_topology(
        global_lengths, ring_sizes, ring_starts, rank, local_lengths, world_size
    )

    half_cu = torch.zeros_like(cu_seqlens_host)
    for batch, (local_length, ring_size) in enumerate(zip(local_lengths, ring_sizes)):
        half_rows = local_length // 2 if ring_size > 1 and local_length else 0
        half_cu[batch + 1] = half_cu[batch] + half_rows

    def tile_count(prefix: torch.Tensor, begin: int, end: int) -> int:
        return sum(
            ((int(prefix[batch + 1] - prefix[batch]) + BLOCK_M - 1) // BLOCK_M)
            * q_heads
            for batch in range(begin, end)
        )

    batch_cursor = 0
    reduction_tiles = 0
    remote_tiles = 0
    base_work_tiles = tile_count(cu_seqlens_host, 0, len(local_lengths))
    total_work_tiles = base_work_tiles
    flat: list[int] = []
    level_ring_sizes = tuple(
        max(world_size >> level_index, 1)
        for level_index in range(len(LEVEL_RING_SIZES))
    )
    ready_bases: list[int] = []
    ready_cursor = 0
    for ring_size in level_ring_sizes:
        ready_bases.append(ready_cursor)
        ready_cursor += max(ring_size - 1, 0)
    for level_index, ring_size in enumerate(level_ring_sizes):
        batch_begin = batch_cursor
        while batch_cursor < len(ring_sizes) and ring_sizes[batch_cursor] == ring_size:
            batch_cursor += 1
        batch_end = batch_cursor
        full_tiles = tile_count(cu_seqlens_host, batch_begin, batch_end)
        half_tiles = tile_count(half_cu, batch_begin, batch_end)
        row_begin = int(cu_seqlens_host[batch_begin])
        full_rows = int(cu_seqlens_host[batch_end]) - row_begin
        half_row_begin = int(half_cu[batch_begin])
        half_rows = int(half_cu[batch_end]) - half_row_begin
        level_reduction_base = reduction_tiles
        if ring_size > 1:
            ring_local_rank = rank % ring_size
            total_work_tiles += (
                ring_local_rank * full_tiles
                + (ring_size - 1 - ring_local_rank) * half_tiles
            )
            remote_tiles += half_tiles * (2 if ring_local_rank > 0 else 1)
            reduction_tiles += full_tiles
        flat.extend(
            (
                ring_size,
                batch_begin,
                batch_end,
                row_begin,
                full_rows,
                half_row_begin,
                half_rows,
                full_tiles,
                half_tiles,
                level_reduction_base,
                ready_bases[level_index],
            )
        )
    if batch_cursor != len(local_lengths):
        raise ValueError("topology batches were not partitioned into descending CP levels")
    flat.extend((base_work_tiles, total_work_tiles, reduction_tiles, remote_tiles))
    if any(value < 0 or value > 2**31 - 1 for value in flat):
        raise OverflowError("hierarchy fields must fit in non-negative int32")
    summary = {
        "base_work_tiles": base_work_tiles,
        "total_work_tiles": total_work_tiles,
        "reduction_tiles": reduction_tiles,
        "remote_tiles": remote_tiles,
    }
    return half_cu, torch.tensor(flat, dtype=torch.int32), summary


class ForwardAblationPlan:
    """All allocations and topology preparation for one steady-state profile."""

    def __init__(
        self,
        q: torch.Tensor,
        remote_k: object,
        remote_v: object,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: torch.Tensor,
        max_seqlen: int,
        global_lengths: Sequence[int],
        ring_sizes: Sequence[int],
        ring_starts: Sequence[int],
        profile: str | int | ProfileSpec,
        *,
        num_comp_sm: int = NUM_COMP_SM,
        num_comm_sm: int = NUM_COMM_SM,
        collect_stats: bool = False,
        compute_only: bool = False,
    ) -> None:
        import min_fa3_op

        self._op = min_fa3_op._forward_varlen_mega_ring_ablation
        self.profile = resolve_profile(profile)
        self.q = q
        self.remote_k = remote_k
        self.remote_v = remote_v
        self.cu_seqlens = cu_seqlens
        self.max_seqlen = int(max_seqlen)
        self.global_lengths = tuple(global_lengths)
        self.ring_sizes_host = tuple(ring_sizes)
        self.ring_starts_host = tuple(ring_starts)
        self.num_comp_sm = int(num_comp_sm)
        self.num_comm_sm = int(num_comm_sm)
        self.collect_stats = collect_stats
        self.compute_only = bool(compute_only)

        if self.compute_only and self.profile.id >= 5:
            raise ValueError("compute_only is only supported for the step and linear ablation profiles")

        if q.device.type != "cuda" or q.dtype != torch.bfloat16 or q.size(-1) != 128:
            raise ValueError("q must be CUDA BF16 [total_q, QH, 128]")
        if self.num_comp_sm <= 0:
            raise ValueError("num_comp_sm must be positive")
        if self.num_comm_sm < 0:
            raise ValueError("num_comm_sm must be non-negative")
        device_sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
        if self.num_comp_sm + self.num_comm_sm > device_sm_count:
            raise ValueError(
                "num_comp_sm + num_comm_sm must not exceed the device SM count "
                f"({device_sm_count})"
            )
        self.world_size = int(remote_k.local_world_size_)
        if self.world_size not in (4, 8):
            raise ValueError(f"forward ablation supports CP world_size in (4, 8), got {self.world_size}")
        if self.profile.id >= 5 and self.world_size != 8:
            raise ValueError(
                f"{self.profile.name} uses the production fixed 8-GPU hierarchy; "
                "CP4 motivation runs support profiles 1-4 only"
            )
        rank = q.device.index
        if self.profile.id <= 4 and (
            any(size != self.world_size for size in ring_sizes)
            or any(start != 0 for start in ring_starts)
        ):
            raise ValueError(f"{self.profile.name} requires fixed all-CP metadata")
        half_host, hierarchy_host, hierarchy = build_hierarchy(
            cu_seqlens_host,
            global_lengths,
            ring_sizes,
            ring_starts,
            rank,
            q.size(1),
            self.world_size,
        )
        self.hierarchy = hierarchy
        if (
            not self.compute_only
            and self.num_comm_sm == 0
            and hierarchy["reduction_tiles"] > 0
        ):
            raise ValueError(
                "num_comm_sm must be positive when the topology has "
                "hierarchical CP replay work"
            )
        self.unique_ring_sizes = tuple(sorted(set(ring_sizes), reverse=True))
        self.ring_sizes = torch.tensor(
            ring_sizes, device=q.device, dtype=torch.int32
        )
        self.half_cu_seqlens = half_host.to(device=q.device)
        self.hierarchy_host = hierarchy_host

        b_rounded = (len(global_lengths) + 3) // 4 * 4
        self.scheduler_metadata = torch.empty(
            1 + 4 * b_rounded, device=q.device, dtype=torch.int32
        )
        self.kv_ready_counts = torch.empty(
            11, device=q.device, dtype=torch.int32
        )
        self.step_ready = torch.empty(
            max(hierarchy["reduction_tiles"], 1),
            device=q.device,
            dtype=torch.int32,
        )
        self.scan_cursor = torch.empty(1, device=q.device, dtype=torch.int32)
        self.completed_tiles = torch.empty(
            4 if collect_stats else 1, device=q.device, dtype=torch.int32
        )
        self.out = torch.empty_like(q)
        self.lse = torch.empty(
            (q.size(1), q.size(0)), device=q.device, dtype=torch.float32
        )
        self.scratch_out = torch.empty_like(q)
        self.scratch_lse = torch.empty_like(self.lse)
        self.stats = (
            torch.empty(2, device=q.device, dtype=torch.int64)
            if collect_stats
            else None
        )

        self._invoke(scheduler_prepared=False, prepare_only=True)
        torch.cuda.synchronize(q.device)

    def _invoke(
        self, *, scheduler_prepared: bool, prepare_only: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._op(
            self.q,
            self.remote_k.data_,
            self.remote_v.data_,
            self.remote_k,
            self.remote_v,
            self.cu_seqlens,
            self.cu_seqlens,
            self.max_seqlen,
            self.max_seqlen,
            self.profile.id,
            self.num_comp_sm,
            self.num_comm_sm,
            self.ring_sizes,
            self.half_cu_seqlens,
            self.hierarchy_host,
            self.scheduler_metadata,
            self.kv_ready_counts,
            self.step_ready,
            self.scan_cursor,
            self.completed_tiles,
            self.out,
            self.lse,
            self.scratch_out,
            self.scratch_lse,
            scheduler_prepared,
            prepare_only,
            self.stats,
            compute_only=self.compute_only,
        )

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._invoke(scheduler_prepared=True, prepare_only=False)

    def probe(self) -> dict[str, int | tuple[int, ...]]:
        if not self.collect_stats or self.stats is None:
            raise RuntimeError("probe requires collect_stats=True")
        self.run()
        torch.cuda.synchronize(self.q.device)
        qo_visits, kv_tile_reads = (int(value) for value in self.stats.cpu())
        completed = [int(value) for value in self.completed_tiles.cpu()]
        return {
            "attention_launches": (
                self.world_size if self.profile.id <= 2 else self.profile.attention_launches
            ),
            "reduction_launches": (
                self.world_size if self.profile.id == 1 else self.profile.reduction_launches
            ),
            "kernel_launches": (
                self.world_size * 2 if self.profile.id == 1
                else self.world_size if self.profile.id == 2
                else self.profile.kernel_launches
            ),
            "recycled_cta_work": completed[1]
            if self.profile.id in (3, 4)
            else 0,
            "segment_span_sum": completed[1]
            if self.profile.dynamic_segments
            else self.hierarchy["total_work_tiles"],
            "segment_span_max": completed[2]
            if self.profile.dynamic_segments
            else 1,
            "segment_claims": completed[3]
            if self.profile.dynamic_segments
            else self.hierarchy["total_work_tiles"],
            "qo_visits": qo_visits,
            "kv_tile_reads": kv_tile_reads,
            "ring_sizes": self.unique_ring_sizes,
            "world_size": self.world_size,
        }


__all__ = [
    "ForwardAblationPlan",
    "NUM_COMM_SM",
    "NUM_COMP_SM",
    "PROFILES",
    "ProfileSpec",
    "build_hierarchy",
    "canonicalize_lengths",
    "local_lengths_for_rank",
    "make_cu_seqlens",
    "profile_dispatch_manifest",
    "resolve_profile",
]
