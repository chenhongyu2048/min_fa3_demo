"""Static forward load accounting for the distributed attention baselines.

This module intentionally models only metadata that already exists in the
forward benchmark paths.  It does not allocate Q/K/V tensors, launch an
attention kernel, or estimate latency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Iterable, Sequence

from baseline.megatron_hybrid_cp import build_hybrid_cp_plan_for_fa3_ring
from ring_test.utils import (
    align_mega_ring_all_cp_lengths,
    hybrid_cp_saturation_note,
)
from ring_test.zepplin import DEFAULT_ZEPPLIN_THRESHOLD, make_zepplin_plan


METHOD_ORDER = (
    "allgather_attention",
    "llama3_allgather_attention",
    "fa3_ring",
    "megatron_hybrid_cp",
    "magi_attention",
    "zepplin",
    "mega_ring_all_cp",
    "mega_ring_hybrid",
)

TILE_TOKENS = 128
MEGA_RING_NONCAUSAL_KV_TILE_TOKENS = 176
BF16_BYTES = 2
FP32_BYTES = 4

FULL = "full"
CAUSAL = "causal"
INV_CAUSAL = "inverse-causal"
BI_CAUSAL = "bi-causal"
MASK_TYPES = (FULL, CAUSAL, INV_CAUSAL, BI_CAUSAL)


@dataclass(frozen=True)
class AttentionTask:
    """One logical attention segment with bottom-right mask alignment."""

    q_tokens: int
    kv_tokens: int
    mask_type: str

    def __post_init__(self) -> None:
        if self.q_tokens < 0 or self.kv_tokens < 0:
            raise ValueError("attention task lengths must be non-negative")
        if self.mask_type not in MASK_TYPES:
            raise ValueError(f"unsupported attention mask type {self.mask_type!r}")


@dataclass(frozen=True)
class RankLoadRecord:
    method: str
    mode: str
    rank: int
    effective_tokens: float = 0
    physical_tokens: float = 0
    effective_scores: float = 0
    physical_scores: float = 0
    effective_flops: float = 0
    physical_flops: float = 0
    comm_tx_bytes: float = 0
    comm_rx_bytes: float = 0
    comm_total_bytes: float = 0
    kv_tile_reads: int = 0
    qo_visits_worst: int = 0
    qo_visits_best: int = 0
    kv_tiles_per_qo_lower: float = 0
    kv_tiles_per_qo_upper: float = 0
    note: str = ""


@dataclass(frozen=True)
class MethodLoadResult:
    method: str
    mode: str
    records: tuple[RankLoadRecord, ...]
    note: str


@dataclass(frozen=True)
class Placement:
    global_length: int
    ring_size: int
    ring_start: int


@dataclass
class _MutableRankLoad:
    effective_tokens: float = 0
    physical_tokens: float = 0
    effective_scores: float = 0
    physical_scores: float = 0
    comm_tx_bytes: float = 0
    comm_rx_bytes: float = 0
    kv_tile_reads: int = 0
    qo_visits_worst: int = 0
    qo_visits_best: int = 0


def mode_name(is_causal: bool) -> str:
    return "causal" if is_causal else "noncausal"


def score_count(global_lengths: Sequence[int], is_causal: bool) -> int:
    if is_causal:
        return sum(length * (length + 1) // 2 for length in global_lengths)
    return sum(length * length for length in global_lengths)


def attention_area(task: AttentionTask) -> int:
    """Return the exact visible scalar score count for one task."""

    q_tokens, kv_tokens = task.q_tokens, task.kv_tokens
    if q_tokens == 0 or kv_tokens == 0:
        return 0
    if task.mask_type == FULL:
        return q_tokens * kv_tokens
    if task.mask_type in (CAUSAL, INV_CAUSAL):
        if kv_tokens >= q_tokens:
            return (2 * kv_tokens - q_tokens + 1) * q_tokens // 2
        return kv_tokens * (kv_tokens + 1) // 2
    return max(kv_tokens - q_tokens + 1, 0) * q_tokens


def _kv_tiles_for_q_tile(
    q_begin: int,
    q_end: int,
    q_tokens: int,
    kv_tokens: int,
    mask_type: str,
    kv_tile_tokens: int = TILE_TOKENS,
) -> int:
    num_kv_tiles = ceil(kv_tokens / kv_tile_tokens)
    if q_begin >= q_tokens or num_kv_tiles == 0:
        return 0
    q_end = min(q_end, q_tokens)
    if mask_type == FULL:
        return num_kv_tiles
    if mask_type == CAUSAL:
        max_k = q_end - 1 + kv_tokens - q_tokens
        return min(max(max_k // kv_tile_tokens + 1, 0), num_kv_tiles)
    if mask_type == INV_CAUSAL:
        if q_begin >= kv_tokens:
            return 0
        return num_kv_tiles - q_begin // kv_tile_tokens

    delta = kv_tokens - q_tokens
    if delta < 0:
        return 0
    first_k = q_begin
    last_k = min(q_end - 1 + delta, kv_tokens - 1)
    if first_k > last_k:
        return 0
    return last_k // kv_tile_tokens - first_k // kv_tile_tokens + 1


def task_tile_counters(
    task: AttentionTask,
    q_heads: int = 1,
    *,
    kv_tile_tokens: int = TILE_TOKENS,
) -> tuple[int, int]:
    """Return ``(KV tile reads, Q/O visits)`` at the selected KV tile size."""

    visits = ceil(task.q_tokens / TILE_TOKENS) * q_heads
    reads = 0
    for q_begin in range(0, task.q_tokens, TILE_TOKENS):
        reads += _kv_tiles_for_q_tile(
            q_begin,
            q_begin + TILE_TOKENS,
            task.q_tokens,
            task.kv_tokens,
            task.mask_type,
            kv_tile_tokens,
        )
    return reads * q_heads, visits


def _add_task(load: _MutableRankLoad, task: AttentionTask, q_heads: int) -> None:
    reads, visits = task_tile_counters(task, q_heads)
    load.kv_tile_reads += reads
    load.qo_visits_worst += visits
    load.qo_visits_best += visits


def _finalize(
    method: str,
    is_causal: bool,
    loads: Sequence[_MutableRankLoad],
    q_heads: int,
    head_dim: int,
    note: str,
) -> MethodLoadResult:
    mode = mode_name(is_causal)
    records: list[RankLoadRecord] = []
    for rank, load in enumerate(loads):
        lower = (
            load.kv_tile_reads / load.qo_visits_worst
            if load.qo_visits_worst
            else 0.0
        )
        upper = (
            load.kv_tile_reads / load.qo_visits_best
            if load.qo_visits_best
            else 0.0
        )
        records.append(
            RankLoadRecord(
                method=method,
                mode=mode,
                rank=rank,
                effective_tokens=load.effective_tokens,
                physical_tokens=load.physical_tokens,
                effective_scores=load.effective_scores,
                physical_scores=load.physical_scores,
                effective_flops=4
                * load.effective_scores
                * q_heads
                * head_dim,
                physical_flops=4
                * load.physical_scores
                * q_heads
                * head_dim,
                comm_tx_bytes=load.comm_tx_bytes,
                comm_rx_bytes=load.comm_rx_bytes,
                comm_total_bytes=load.comm_tx_bytes,
                kv_tile_reads=load.kv_tile_reads,
                qo_visits_worst=load.qo_visits_worst,
                qo_visits_best=load.qo_visits_best,
                kv_tiles_per_qo_lower=lower,
                kv_tiles_per_qo_upper=upper,
                note=note,
            )
        )
    result = MethodLoadResult(method, mode, tuple(records), note)
    validate_result(result)
    return result


def validate_result(result: MethodLoadResult) -> None:
    expected_ranks = list(range(len(result.records)))
    actual_ranks = [record.rank for record in result.records]
    if actual_ranks != expected_ranks:
        raise ValueError(
            f"{result.method} records have ranks {actual_ranks}, expected {expected_ranks}"
        )
    total_tx = sum(record.comm_tx_bytes for record in result.records)
    total_rx = sum(record.comm_rx_bytes for record in result.records)
    if total_tx != total_rx:
        raise ValueError(
            f"{result.method} communication is not conserved: TX={total_tx}, RX={total_rx}"
        )
    total_communication = sum(
        record.comm_total_bytes for record in result.records
    )
    if total_communication != total_tx:
        raise ValueError(
            f"{result.method} sent communication is inconsistent: "
            f"communication={total_communication}, TX={total_tx}"
        )
    for record in result.records:
        if record.qo_visits_best > record.qo_visits_worst:
            raise ValueError("best-case Q/O visits cannot exceed worst-case visits")
        if record.kv_tiles_per_qo_lower > record.kv_tiles_per_qo_upper + 1e-12:
            raise ValueError("KV/QO lower bound cannot exceed upper bound")


def _allgather_tasks(
    global_length: int,
    world_size: int,
    rank: int,
    is_causal: bool,
) -> list[AttentionTask]:
    local_length = global_length // world_size
    if not is_causal:
        return [AttentionTask(local_length, global_length, FULL)]
    half = local_length // 2
    return [
        AttentionTask(half, (rank + 1) * half, CAUSAL),
        AttentionTask(half, (2 * world_size - rank) * half, CAUSAL),
    ]


def _llama3_tasks_for_interval(
    global_lengths: Sequence[int],
    interval_begin: int,
    interval_end: int,
    is_causal: bool,
) -> list[AttentionTask]:
    tasks: list[AttentionTask] = []
    sequence_begin = 0
    for sequence_length in global_lengths:
        sequence_end = sequence_begin + sequence_length
        q_begin = max(sequence_begin, interval_begin)
        q_end = min(sequence_end, interval_end)
        if q_begin < q_end:
            q_tokens = q_end - q_begin
            kv_tokens = q_end - sequence_begin if is_causal else sequence_length
            tasks.append(
                AttentionTask(q_tokens, kv_tokens, CAUSAL if is_causal else FULL)
            )
        sequence_begin = sequence_end
    return tasks


def analyze_allgather(
    global_lengths: Sequence[int],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
    *,
    llama3: bool = False,
    heads_k_stride: int = 1,
) -> MethodLoadResult:
    method = (
        "llama3_allgather_attention" if llama3 else "allgather_attention"
    )
    loads = [_MutableRankLoad() for _ in range(world_size)]
    local_tokens = sum(global_lengths) // world_size
    payload = local_tokens * kv_heads * head_dim * BF16_BYTES * 2
    for rank, load in enumerate(loads):
        load.effective_tokens = local_tokens
        load.physical_tokens = local_tokens
        load.comm_tx_bytes = payload * (world_size - 1)
        load.comm_rx_bytes = payload * (world_size - 1)

        if llama3:
            total_tokens = sum(global_lengths)
            chunk = total_tokens // (2 * world_size)
            back_block = 2 * world_size - 1 - rank
            tasks = [
                *_llama3_tasks_for_interval(
                    global_lengths,
                    rank * chunk,
                    (rank + 1) * chunk,
                    is_causal,
                ),
                *_llama3_tasks_for_interval(
                    global_lengths,
                    back_block * chunk,
                    (back_block + 1) * chunk,
                    is_causal,
                ),
            ]
        else:
            tasks = [
                task
                for length in global_lengths
                for task in _allgather_tasks(length, world_size, rank, is_causal)
            ]
        for task in tasks:
            area = attention_area(task)
            load.effective_scores += area
            load.physical_scores += area
            _add_task(load, task, q_heads)

    layout = "whole-packed" if llama3 else "per-sequence"
    mode = "zigzag causal" if is_causal else "noncausal"
    note = (
        f"{layout} KV-head-sliced all-gather; {heads_k_stride} KVH/chunk; "
        f"{mode}; communication excludes setup repartition"
    )
    return _finalize(method, is_causal, loads, q_heads, head_dim, note)


def _ring_tasks(
    local_length: int,
    ring_size: int,
    ring_local_rank: int,
    is_causal: bool,
) -> list[AttentionTask]:
    if ring_size == 1:
        return [AttentionTask(local_length, local_length, CAUSAL if is_causal else FULL)]
    if not is_causal:
        return [AttentionTask(local_length, local_length, FULL)] * ring_size
    half = local_length // 2
    return [
        AttentionTask(local_length, local_length, CAUSAL),
        *([AttentionTask(local_length, half, FULL)] * ring_local_rank),
        *(
            [AttentionTask(half, local_length, FULL)]
            * (ring_size - 1 - ring_local_rank)
        ),
    ]


def _placements_loads(
    placements: Sequence[Placement],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
    *,
    communication: str,
    mega_ratio_bounds: bool = False,
) -> list[_MutableRankLoad]:
    loads = [_MutableRankLoad() for _ in range(world_size)]
    row_bytes = kv_heads * head_dim * BF16_BYTES * 2
    kv_tile_tokens = (
        MEGA_RING_NONCAUSAL_KV_TILE_TOKENS
        if communication == "mega-ring" and not is_causal
        else TILE_TOKENS
    )

    for placement in placements:
        length = placement.global_length
        ring_size = placement.ring_size
        ring_start = placement.ring_start
        local_length = length // ring_size
        for ring_local_rank in range(ring_size):
            rank = ring_start + ring_local_rank
            load = loads[rank]
            load.effective_tokens += local_length
            load.physical_tokens += local_length
            tasks = _ring_tasks(
                local_length, ring_size, ring_local_rank, is_causal
            )
            sequence_reads = 0
            sequence_worst = 0
            for task in tasks:
                area = attention_area(task)
                load.effective_scores += area
                load.physical_scores += area
                reads, visits = task_tile_counters(
                    task, q_heads, kv_tile_tokens=kv_tile_tokens
                )
                sequence_reads += reads
                sequence_worst += visits
            load.kv_tile_reads += sequence_reads
            load.qo_visits_worst += sequence_worst

            if mega_ratio_bounds and is_causal and ring_size > 1:
                half = local_length // 2
                front_q_tiles = ceil(half / TILE_TOKENS)
                back_q_tiles = ceil((local_length - half) / TILE_TOKENS)
                best_visits = ceil(local_length / TILE_TOKENS)
                if ring_local_rank > 0:
                    best_visits += front_q_tiles
                best_visits += back_q_tiles
                load.qo_visits_best += best_visits * q_heads
            else:
                load.qo_visits_best += sequence_worst

        if ring_size == 1:
            continue
        if communication == "python-ring":
            per_rank_bytes = (ring_size - 1) * local_length * row_bytes
            for rank in range(ring_start, ring_start + ring_size):
                loads[rank].comm_tx_bytes += per_rank_bytes
                loads[rank].comm_rx_bytes += per_rank_bytes
        elif communication == "mega-ring":
            half = local_length // 2
            for ring_local_rank in range(ring_size):
                receiver = ring_start + ring_local_rank
                for step in range(1, ring_size):
                    rows = (
                        half
                        if is_causal and step <= ring_local_rank
                        else local_length
                    )
                    source_local = (ring_local_rank - step) % ring_size
                    source = ring_start + source_local
                    transfer_bytes = rows * row_bytes
                    loads[receiver].comm_rx_bytes += transfer_bytes
                    loads[source].comm_tx_bytes += transfer_bytes
        else:
            raise ValueError(f"unsupported communication model {communication!r}")
    return loads


def _replace_token_and_communication_with_original_lengths(
    loads: Sequence[_MutableRankLoad],
    execution_placements: Sequence[Placement],
    original_global_lengths: Sequence[int],
    world_size: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
    *,
    communication: str,
) -> None:
    """Report input tokens and traffic without alignment-only padding.

    The forward implementations may pad an execution sequence so FA3's ring
    layout is valid.  Their score/FLOP and tile counters must continue to
    describe that physical execution.  Load-balance token and communication
    comparisons, however, should describe the caller's original workload.
    """

    if len(execution_placements) != len(original_global_lengths):
        raise ValueError("execution and original placement lengths must match")
    if len(loads) != world_size:
        raise ValueError("load count must match world size")

    physical_tokens = [0.0] * world_size
    comm_tx_bytes = [0.0] * world_size
    comm_rx_bytes = [0.0] * world_size
    row_bytes = kv_heads * head_dim * BF16_BYTES * 2

    for placement, original_length in zip(execution_placements, original_global_lengths):
        if original_length <= 0:
            raise ValueError("original global lengths must be positive")
        ring_size = placement.ring_size
        ring_start = placement.ring_start
        if ring_size <= 0 or not 0 <= ring_start <= world_size - ring_size:
            raise ValueError("invalid execution placement")

        local_length = original_length / ring_size
        for rank in range(ring_start, ring_start + ring_size):
            physical_tokens[rank] += local_length

        if ring_size == 1:
            continue
        if communication == "python-ring":
            per_rank_bytes = (ring_size - 1) * local_length * row_bytes
            for rank in range(ring_start, ring_start + ring_size):
                comm_tx_bytes[rank] += per_rank_bytes
                comm_rx_bytes[rank] += per_rank_bytes
        elif communication == "mega-ring":
            half = local_length / 2
            for ring_local_rank in range(ring_size):
                receiver = ring_start + ring_local_rank
                for step in range(1, ring_size):
                    rows = (
                        half
                        if is_causal and step <= ring_local_rank
                        else local_length
                    )
                    source_local = (ring_local_rank - step) % ring_size
                    source = ring_start + source_local
                    transfer_bytes = rows * row_bytes
                    comm_rx_bytes[receiver] += transfer_bytes
                    comm_tx_bytes[source] += transfer_bytes
        else:
            raise ValueError(f"unsupported communication model {communication!r}")

    for rank, load in enumerate(loads):
        load.physical_tokens = physical_tokens[rank]
        load.comm_tx_bytes = comm_tx_bytes[rank]
        load.comm_rx_bytes = comm_rx_bytes[rank]


def analyze_fa3_ring(
    global_lengths: Sequence[int],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
) -> MethodLoadResult:
    placements = [Placement(length, world_size, 0) for length in global_lengths]
    loads = _placements_loads(
        placements,
        world_size,
        q_heads,
        kv_heads,
        head_dim,
        is_causal,
        communication="python-ring",
    )
    note = "whole-packed Python P2P ring; zigzag causal" if is_causal else "whole-packed Python P2P ring; noncausal"
    return _finalize("fa3_ring", is_causal, loads, q_heads, head_dim, note)


def analyze_megatron(
    global_lengths: Sequence[int],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
    *,
    max_seqlen_per_rank: int = 8192,
) -> MethodLoadResult:
    plan = build_hybrid_cp_plan_for_fa3_ring(
        list(global_lengths), world_size, is_causal, max_seqlen_per_rank
    )
    placements = [
        Placement(
            assignment.global_length,
            assignment.cp_size,
            assignment.rank_start,
        )
        for assignment in plan.assignments
    ]
    loads = _placements_loads(
        placements,
        world_size,
        q_heads,
        kv_heads,
        head_dim,
        is_causal,
        communication="python-ring",
    )
    _replace_token_and_communication_with_original_lengths(
        loads,
        placements,
        global_lengths,
        world_size,
        kv_heads,
        head_dim,
        is_causal,
        communication="python-ring",
    )
    for load in loads:
        load.effective_tokens = 0
        load.effective_scores = 0
    for sample_id, original_length in enumerate(global_lengths):
        assignment = plan.assignment(sample_id)
        effective_tokens = original_length / assignment.cp_size
        effective_scores = (
            score_count([original_length], is_causal) / assignment.cp_size
        )
        for rank in assignment.ranks:
            loads[rank].effective_tokens += effective_tokens
            loads[rank].effective_scores += effective_scores
    padding = sum(plan.global_lengths) - sum(global_lengths)
    saturation = hybrid_cp_saturation_note(list(global_lengths), plan)
    note = (
        f"Megatron scheduler with post-schedule FA3 ring padding; "
        f"{plan.num_execution_groups} execution groups; padding={padding} tokens; "
        f"max_seqlen_per_rank={max_seqlen_per_rank}; token and communication "
        "counters use original sequence lengths; FLOPs use execution lengths"
    )
    if saturation:
        note = f"{note}; {saturation}"
    return _finalize(
        "megatron_hybrid_cp", is_causal, loads, q_heads, head_dim, note
    )


def analyze_zepplin(
    global_lengths: Sequence[int],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
    *,
    threshold: int = DEFAULT_ZEPPLIN_THRESHOLD,
) -> MethodLoadResult:
    plan = make_zepplin_plan(
        list(global_lengths), world_size, is_causal, threshold
    )
    owner_by_index = dict(zip(plan.short_indices, plan.short_owners))
    placements = [
        Placement(
            length,
            1 if index in owner_by_index else world_size,
            owner_by_index.get(index, 0),
        )
        for index, length in enumerate(global_lengths)
    ]
    loads = _placements_loads(
        placements,
        world_size,
        q_heads,
        kv_heads,
        head_dim,
        is_causal,
        communication="python-ring",
    )
    _replace_token_and_communication_with_original_lengths(
        loads,
        placements,
        global_lengths,
        world_size,
        kv_heads,
        head_dim,
        is_causal,
        communication="python-ring",
    )
    note = (
        f"threshold={threshold}; G1={len(plan.short_indices)}, "
        f"Gworld={len(plan.long_indices)}; LPT short-sequence owners; token and "
        "communication counters use original sequence lengths"
    )
    return _finalize("zepplin", is_causal, loads, q_heads, head_dim, note)


def analyze_mega_ring_hybrid(
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
) -> MethodLoadResult:
    placements = [
        Placement(length, ring_size, ring_start)
        for length, ring_size, ring_start in zip(
            global_lengths, ring_sizes, ring_starts
        )
    ]
    loads = _placements_loads(
        placements,
        world_size,
        q_heads,
        kv_heads,
        head_dim,
        is_causal,
        communication="mega-ring",
        mega_ratio_bounds=True,
    )
    note = (
        "hierarchical fused mega-ring; causal remote segments shown as "
        "worst/best scheduler bounds"
        if is_causal
        else "hierarchical fused mega-ring; noncausal per-step segments"
    )
    return _finalize(
        "mega_ring_hybrid",
        is_causal,
        loads,
        q_heads,
        head_dim,
        note,
    )


def analyze_mega_ring_all_cp(
    global_lengths: Sequence[int],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
) -> MethodLoadResult:
    aligned = align_mega_ring_all_cp_lengths(list(global_lengths))
    placements = [Placement(length, world_size, 0) for length in aligned]
    loads = _placements_loads(
        placements,
        world_size,
        q_heads,
        kv_heads,
        head_dim,
        is_causal,
        communication="mega-ring",
        mega_ratio_bounds=True,
    )
    _replace_token_and_communication_with_original_lengths(
        loads,
        placements,
        global_lengths,
        world_size,
        kv_heads,
        head_dim,
        is_causal,
        communication="mega-ring",
    )
    effective_tokens = sum(global_lengths) / world_size
    effective_scores = score_count(global_lengths, is_causal) / world_size
    for load in loads:
        load.effective_tokens = effective_tokens
        load.effective_scores = effective_scores
    padded = sum(aligned) - sum(global_lengths)
    note = (
        f"all-CP fused mega-ring; 2048-token alignment; padding={padded} tokens; "
        "token and communication counters use original sequence lengths; FLOPs use "
        "execution lengths; "
        + (
            "causal remote segments shown as worst/best scheduler bounds"
            if is_causal
            else "noncausal per-step segments"
        )
    )
    return _finalize(
        "mega_ring_all_cp",
        is_causal,
        loads,
        q_heads,
        head_dim,
        note,
    )


def analyze_method(
    method: str,
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
    *,
    heads_k_stride: int = 4,
    zepplin_threshold: int = DEFAULT_ZEPPLIN_THRESHOLD,
    megatron_max_seqlen_per_rank: int = 8192,
) -> MethodLoadResult:
    if method == "allgather_attention":
        return analyze_allgather(
            global_lengths,
            world_size,
            q_heads,
            kv_heads,
            head_dim,
            is_causal,
            heads_k_stride=heads_k_stride,
        )
    if method == "llama3_allgather_attention":
        return analyze_allgather(
            global_lengths,
            world_size,
            q_heads,
            kv_heads,
            head_dim,
            is_causal,
            llama3=True,
            heads_k_stride=heads_k_stride,
        )
    if method == "fa3_ring":
        return analyze_fa3_ring(
            global_lengths,
            world_size,
            q_heads,
            kv_heads,
            head_dim,
            is_causal,
        )
    if method == "megatron_hybrid_cp":
        return analyze_megatron(
            global_lengths,
            world_size,
            q_heads,
            kv_heads,
            head_dim,
            is_causal,
            max_seqlen_per_rank=megatron_max_seqlen_per_rank,
        )
    if method == "zepplin":
        return analyze_zepplin(
            global_lengths,
            world_size,
            q_heads,
            kv_heads,
            head_dim,
            is_causal,
            threshold=zepplin_threshold,
        )
    if method == "mega_ring_all_cp":
        return analyze_mega_ring_all_cp(
            global_lengths,
            world_size,
            q_heads,
            kv_heads,
            head_dim,
            is_causal,
        )
    if method == "mega_ring_hybrid":
        return analyze_mega_ring_hybrid(
            global_lengths,
            ring_sizes,
            ring_starts,
            world_size,
            q_heads,
            kv_heads,
            head_dim,
            is_causal,
        )
    if method == "magi_attention":
        raise ValueError("MagiAttention requires its distributed metadata adapter")
    raise ValueError(f"unknown method {method!r}")


def _collective_cast_endpoints(arg: object) -> tuple[int, int]:
    kwargs = arg.to_group_cast_args()  # type: ignore[attr-defined]
    alignment = int(kwargs.get("split_alignment", 1))
    tx = sum(
        split_size * len(destinations)
        for split_size, destinations in zip(
            kwargs["input_split_sizes"], kwargs["dst_indices"]
        )
    )
    rx = sum(kwargs["output_split_sizes"])
    return tx * alignment, rx * alignment


def _collective_reduce_endpoints(arg: object) -> tuple[int, int]:
    kwargs = arg.to_group_reduce_args()  # type: ignore[attr-defined]
    alignment = int(kwargs.get("split_alignment", 1))
    tx = sum(kwargs["input_split_sizes"])
    rx = sum(
        split_size * len(sources)
        for split_size, sources in zip(
            kwargs["output_split_sizes"], kwargs["src_indices"]
        )
    )
    return tx * alignment, rx * alignment


def _magi_mask_name(mask: object) -> str:
    if isinstance(mask, int):
        value = mask
    elif hasattr(mask, "to_int_type"):
        value = int(mask.to_int_type())  # type: ignore[attr-defined]
    elif hasattr(mask, "value"):
        names = {
            "full": FULL,
            "causal": CAUSAL,
            "inv_causal": INV_CAUSAL,
            "bi_causal": BI_CAUSAL,
        }
        try:
            return names[str(mask.value)]  # type: ignore[attr-defined]
        except KeyError as exc:
            raise ValueError(f"unknown Magi mask {mask!r}") from exc
    else:
        value = int(mask)
    if not 0 <= value < len(MASK_TYPES):
        raise ValueError(f"unknown Magi mask integer {value}")
    return (FULL, CAUSAL, INV_CAUSAL, BI_CAUSAL)[value]


def _range_bounds(attn_range: object) -> tuple[int, int]:
    return int(attn_range.start), int(attn_range.end)  # type: ignore[attr-defined]


def _magi_attn_tasks(attn_arg: object) -> list[AttentionTask]:
    tasks: list[AttentionTask] = []
    for q_range, k_range, mask in zip(
        attn_arg.q_ranges,  # type: ignore[attr-defined]
        attn_arg.k_ranges,  # type: ignore[attr-defined]
        attn_arg.attn_type_map,  # type: ignore[attr-defined]
    ):
        q_begin, q_end = _range_bounds(q_range)
        k_begin, k_end = _range_bounds(k_range)
        tasks.append(
            AttentionTask(
                q_end - q_begin,
                k_end - k_begin,
                _magi_mask_name(mask),
            )
        )
    return tasks


def analyze_magi_rank(
    metadata: object,
    global_lengths: Sequence[int],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
) -> RankLoadRecord:
    """Consume one rank's Magi DispatchMeta, CalcMeta, and CommMeta."""

    dispatch_q = metadata.dispatch_meta_q  # type: ignore[attr-defined]
    rank = int(dispatch_q.cp_rank)
    host_ranges = dispatch_q.host_ranges_per_rank[rank]
    valid_ranges = [_range_bounds(attn_range) for attn_range in host_ranges]
    effective_tokens, effective_scores = iter_global_q_score(
        global_lengths, valid_ranges, is_causal
    )
    physical_tokens = int(dispatch_q.shard_seqlen)

    calc_meta = metadata.calc_meta  # type: ignore[attr-defined]
    merged_arg = getattr(calc_meta, "merged_attn_arg", None)
    if getattr(calc_meta, "no_overlap", False) and merged_arg is not None:
        attn_args = [merged_arg]
    else:
        attn_args = [
            calc_meta.local_attn_arg,
            *calc_meta.remote_attn_args_list,
        ]

    physical_scores = 0
    kv_tile_reads = 0
    qo_visits = 0
    for attn_arg in attn_args:
        tasks = _magi_attn_tasks(attn_arg)
        known_area = int(getattr(attn_arg, "total_area", -1))
        physical_scores += (
            known_area
            if known_area >= 0
            else sum(attention_area(task) for task in tasks)
        )
        for task in tasks:
            reads, visits = task_tile_counters(task, q_heads)
            kv_tile_reads += reads
            qo_visits += visits

    comm_meta = metadata.comm_meta  # type: ignore[attr-defined]
    tx_bytes = 0
    rx_bytes = 0
    for arg in comm_meta.kv_group_collective_args_list:
        tx_tokens, rx_tokens = _collective_cast_endpoints(arg)
        # A2AV metadata has already packed K and V by duplicating split lists.
        # Native collectives retain one token list and launch two tensors.
        tensor_count = 2 if metadata.use_native_grpcoll else 1  # type: ignore[attr-defined]
        bytes_per_token = kv_heads * head_dim * BF16_BYTES * tensor_count
        tx_bytes += tx_tokens * bytes_per_token
        rx_bytes += rx_tokens * bytes_per_token

    if metadata.enable_qo_comm:  # type: ignore[attr-defined]
        for arg in comm_meta.qo_group_collective_args_list:
            tx_tokens, rx_tokens = _collective_cast_endpoints(arg)
            bytes_per_token = q_heads * head_dim * BF16_BYTES
            tx_bytes += tx_tokens * bytes_per_token
            rx_bytes += rx_tokens * bytes_per_token

        reduce_args = (
            comm_meta.qo_group_collective_args_list
            if metadata.use_native_grpcoll  # type: ignore[attr-defined]
            else comm_meta.out_lse_group_collective_args_list
        )
        # Magi downcasts partial O to the input dtype unless high-precision
        # forward reduction is enabled. LSE remains FP32 in either path.
        out_bytes = (
            FP32_BYTES
            if metadata.fwd_hp_reduce  # type: ignore[attr-defined]
            else BF16_BYTES
        )
        bytes_per_token = q_heads * (
            head_dim * out_bytes + FP32_BYTES
        )
        for arg in reduce_args:
            tx_tokens, rx_tokens = _collective_reduce_endpoints(arg)
            tx_bytes += tx_tokens * bytes_per_token
            rx_bytes += rx_tokens * bytes_per_token

    ratio = kv_tile_reads / qo_visits if qo_visits else 0.0
    note = (
        f"metadata-only Magi plan; chunk_size={metadata.chunk_size}; "  # type: ignore[attr-defined]
        f"tokens(original/padded)={metadata.original_tokens}/{metadata.padded_tokens}; "  # type: ignore[attr-defined]
        f"overlap_degree={metadata.overlap_degree}; dispatch excluded"  # type: ignore[attr-defined]
    )
    return RankLoadRecord(
        method="magi_attention",
        mode=mode_name(is_causal),
        rank=rank,
        effective_tokens=effective_tokens,
        physical_tokens=physical_tokens,
        effective_scores=effective_scores,
        physical_scores=physical_scores,
        effective_flops=4 * effective_scores * q_heads * head_dim,
        physical_flops=4 * physical_scores * q_heads * head_dim,
        comm_tx_bytes=tx_bytes,
        comm_rx_bytes=rx_bytes,
        comm_total_bytes=tx_bytes,
        kv_tile_reads=kv_tile_reads,
        qo_visits_worst=qo_visits,
        qo_visits_best=qo_visits,
        kv_tiles_per_qo_lower=ratio,
        kv_tiles_per_qo_upper=ratio,
        note=note,
    )


def magi_result_from_records(
    records: Sequence[RankLoadRecord],
) -> MethodLoadResult:
    ordered = tuple(sorted(records, key=lambda record: record.rank))
    if not ordered:
        raise ValueError("MagiAttention result requires at least one rank record")
    note = ordered[0].note
    result = MethodLoadResult("magi_attention", ordered[0].mode, ordered, note)
    validate_result(result)
    return result


def cumulative_result(results: Sequence[MethodLoadResult]) -> MethodLoadResult:
    if not results:
        raise ValueError("cannot accumulate an empty result list")
    first = results[0]
    world_size = len(first.records)
    if any(
        result.method != first.method
        or result.mode != first.mode
        or len(result.records) != world_size
        for result in results
    ):
        raise ValueError("cumulative results must share method, mode, and world size")

    records: list[RankLoadRecord] = []
    summed_fields = (
        "effective_tokens",
        "physical_tokens",
        "effective_scores",
        "physical_scores",
        "effective_flops",
        "physical_flops",
        "comm_tx_bytes",
        "comm_rx_bytes",
        "comm_total_bytes",
        "kv_tile_reads",
        "qo_visits_worst",
        "qo_visits_best",
    )
    stable_notes = {
        "megatron_hybrid_cp": (
            "Megatron scheduler; execution groups vary by case; token and "
            "communication counters use original sequence lengths; FLOPs use "
            "execution lengths"
        ),
        "magi_attention": "metadata-only Magi plans; dispatch excluded; chunking/padding vary by case",
        "zepplin": (
            "Zeppelin LPT G1/Gworld placement varies by case; token and "
            "communication counters use original sequence lengths"
        ),
        "mega_ring_all_cp": (
            "all-CP fused mega-ring; per-case 2048-token alignment; token and "
            "communication counters use original sequence lengths; FLOPs use "
            "execution lengths"
        ),
    }
    note = (
        f"cumulative over {len(results)} cases; "
        f"{stable_notes.get(first.method, first.note)}"
    )
    for rank in range(world_size):
        values = {
            field: sum(getattr(result.records[rank], field) for result in results)
            for field in summed_fields
        }
        lower = (
            values["kv_tile_reads"] / values["qo_visits_worst"]
            if values["qo_visits_worst"]
            else 0.0
        )
        upper = (
            values["kv_tile_reads"] / values["qo_visits_best"]
            if values["qo_visits_best"]
            else 0.0
        )
        records.append(
            replace(
                first.records[rank],
                **values,
                kv_tiles_per_qo_lower=lower,
                kv_tiles_per_qo_upper=upper,
                note=note,
            )
        )
    cumulative = MethodLoadResult(
        first.method, first.mode, tuple(records), note
    )
    validate_result(cumulative)
    return cumulative


def global_ratio(result: MethodLoadResult) -> tuple[float, float]:
    reads = sum(record.kv_tile_reads for record in result.records)
    worst = sum(record.qo_visits_worst for record in result.records)
    best = sum(record.qo_visits_best for record in result.records)
    return (reads / worst if worst else 0.0, reads / best if best else 0.0)


def global_score_conserved(
    result: MethodLoadResult,
    global_lengths: Sequence[int],
    is_causal: bool,
) -> bool:
    expected = score_count(global_lengths, is_causal)
    actual = sum(record.effective_scores for record in result.records)
    return abs(actual - expected) <= max(1e-6, abs(expected) * 1e-12)


def iter_global_q_score(
    global_lengths: Sequence[int],
    ranges: Iterable[tuple[int, int]],
    is_causal: bool,
) -> tuple[int, int]:
    """Count valid packed Q tokens and their original-workload scores."""

    total_tokens = sum(global_lengths)
    valid_tokens = 0
    scores = 0
    sequence_begin = 0
    sequence_intervals: list[tuple[int, int, int]] = []
    for length in global_lengths:
        sequence_intervals.append(
            (sequence_begin, sequence_begin + length, length)
        )
        sequence_begin += length
    for begin, end in ranges:
        begin = max(begin, 0)
        end = min(end, total_tokens)
        if begin >= end:
            continue
        for sequence_begin, sequence_end, length in sequence_intervals:
            left = max(begin, sequence_begin)
            right = min(end, sequence_end)
            if left >= right:
                continue
            count = right - left
            valid_tokens += count
            if is_causal:
                first = left - sequence_begin + 1
                last = right - sequence_begin
                scores += (first + last) * count // 2
            else:
                scores += count * length
    return valid_tokens, scores


__all__ = [
    "AttentionTask",
    "BF16_BYTES",
    "BI_CAUSAL",
    "CAUSAL",
    "FP32_BYTES",
    "FULL",
    "INV_CAUSAL",
    "METHOD_ORDER",
    "MethodLoadResult",
    "Placement",
    "RankLoadRecord",
    "TILE_TOKENS",
    "analyze_allgather",
    "analyze_fa3_ring",
    "analyze_mega_ring_all_cp",
    "analyze_mega_ring_hybrid",
    "analyze_magi_rank",
    "analyze_megatron",
    "analyze_method",
    "analyze_zepplin",
    "attention_area",
    "cumulative_result",
    "global_ratio",
    "global_score_conserved",
    "iter_global_q_score",
    "magi_result_from_records",
    "mode_name",
    "score_count",
    "task_tile_counters",
    "validate_result",
]
