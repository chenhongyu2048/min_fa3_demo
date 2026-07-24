"""Static backward load accounting for the distributed attention baselines.

The model follows the execution boundaries of ``benchmark_topology_backward``.
It consumes placement and runtime metadata only: no tensors are dispatched, no
attention kernel is launched, and no autograd graph is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Sequence

from baseline.megatron_hybrid_cp import build_hybrid_cp_plan_for_fa3_ring
from ring_test.forward_load_model import (
    AttentionTask,
    BF16_BYTES,
    BI_CAUSAL,
    CAUSAL,
    FP32_BYTES,
    FULL,
    INV_CAUSAL,
    METHOD_ORDER,
    Placement,
    TILE_TOKENS,
    _allgather_tasks,
    _collective_cast_endpoints,
    _collective_reduce_endpoints,
    _llama3_tasks_for_interval,
    _magi_mask_name,
    _range_bounds,
    _ring_tasks,
    attention_area,
    iter_global_q_score,
    score_count,
)
from ring_test.utils import (
    align_mega_ring_all_cp_lengths,
    hybrid_cp_saturation_note,
)
from ring_test.zepplin import DEFAULT_ZEPPLIN_THRESHOLD, make_zepplin_plan


BACKWARD_FLOPS_PER_SCORE = 10


@dataclass(frozen=True)
class BackwardRankLoadRecord:
    direction: str
    method: str
    mode: str
    rank: int
    effective_tokens: float = 0
    physical_tokens: float = 0
    effective_scores: float = 0
    physical_scores: float = 0
    effective_flops: float = 0
    physical_flops: float = 0
    comm_tx_bytes: int = 0
    comm_rx_bytes: int = 0
    comm_total_bytes: int = 0
    q_tile_reads: int = 0
    k_dkv_visits: int = 0
    q_tiles_per_k_dkv: float = 0
    note: str = ""


@dataclass(frozen=True)
class BackwardMethodLoadResult:
    direction: str
    method: str
    mode: str
    records: tuple[BackwardRankLoadRecord, ...]
    note: str


@dataclass
class _MutableBackwardRankLoad:
    effective_tokens: float = 0
    physical_tokens: float = 0
    effective_scores: float = 0
    physical_scores: float = 0
    comm_tx_bytes: int = 0
    comm_rx_bytes: int = 0
    q_tile_reads: int = 0
    k_dkv_visits: int = 0


def _tile_pair_intersects(
    task: AttentionTask,
    q_begin: int,
    q_end: int,
    k_begin: int,
    k_end: int,
) -> bool:
    """Return whether two logical tiles contain at least one visible score."""

    if q_begin >= q_end or k_begin >= k_end:
        return False
    if task.mask_type == FULL:
        return True
    if task.mask_type == CAUSAL:
        delta = task.kv_tokens - task.q_tokens
        return k_begin <= q_end - 1 + delta
    if task.mask_type == INV_CAUSAL:
        return k_end - 1 >= q_begin
    if task.mask_type == BI_CAUSAL:
        delta = task.kv_tokens - task.q_tokens
        return (
            delta >= 0
            and q_begin <= k_end - 1
            and k_begin <= q_end - 1 + delta
        )
    raise ValueError(f"unsupported attention mask type {task.mask_type!r}")


def backward_task_tile_counters(
    task: AttentionTask, q_heads: int = 1
) -> tuple[int, int]:
    """Return ``(Q tile reads, K/dKV visits)`` for one backward task.

    A K/dKV visit is one logical 128-token K tile processed by one Q head. Its
    Q reads are the logical Q tiles that intersect that K tile under the task's
    mask. Both counters therefore match the backward scheduler's Q-head work
    granularity.
    """

    q_tiles = ceil(task.q_tokens / TILE_TOKENS)
    k_tiles = ceil(task.kv_tokens / TILE_TOKENS)
    reads = 0
    visits = 0
    for k_tile in range(k_tiles):
        k_begin = k_tile * TILE_TOKENS
        k_end = min(k_begin + TILE_TOKENS, task.kv_tokens)
        tile_reads = 0
        for q_tile in range(q_tiles):
            q_begin = q_tile * TILE_TOKENS
            q_end = min(q_begin + TILE_TOKENS, task.q_tokens)
            tile_reads += _tile_pair_intersects(
                task, q_begin, q_end, k_begin, k_end
            )
        if tile_reads:
            visits += 1
            reads += tile_reads
    return reads * q_heads, visits * q_heads


def _add_task(
    load: _MutableBackwardRankLoad, task: AttentionTask, q_heads: int
) -> None:
    reads, visits = backward_task_tile_counters(task, q_heads)
    load.q_tile_reads += reads
    load.k_dkv_visits += visits


def _finalize(
    method: str,
    loads: Sequence[_MutableBackwardRankLoad],
    q_heads: int,
    head_dim: int,
    note: str,
) -> BackwardMethodLoadResult:
    records: list[BackwardRankLoadRecord] = []
    for rank, load in enumerate(loads):
        ratio = (
            load.q_tile_reads / load.k_dkv_visits
            if load.k_dkv_visits
            else 0.0
        )
        records.append(
            BackwardRankLoadRecord(
                direction="backward",
                method=method,
                mode="causal",
                rank=rank,
                effective_tokens=load.effective_tokens,
                physical_tokens=load.physical_tokens,
                effective_scores=load.effective_scores,
                physical_scores=load.physical_scores,
                effective_flops=BACKWARD_FLOPS_PER_SCORE
                * load.effective_scores
                * q_heads
                * head_dim,
                physical_flops=BACKWARD_FLOPS_PER_SCORE
                * load.physical_scores
                * q_heads
                * head_dim,
                comm_tx_bytes=load.comm_tx_bytes,
                comm_rx_bytes=load.comm_rx_bytes,
                comm_total_bytes=load.comm_tx_bytes,
                q_tile_reads=load.q_tile_reads,
                k_dkv_visits=load.k_dkv_visits,
                q_tiles_per_k_dkv=ratio,
                note=note,
            )
        )
    result = BackwardMethodLoadResult(
        "backward", method, "causal", tuple(records), note
    )
    validate_backward_result(result)
    return result


def validate_backward_result(result: BackwardMethodLoadResult) -> None:
    if result.direction != "backward" or result.mode != "causal":
        raise ValueError("backward load results must use direction=backward and causal mode")
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
    numeric_fields = (
        "effective_tokens",
        "physical_tokens",
        "effective_scores",
        "physical_scores",
        "effective_flops",
        "physical_flops",
        "comm_tx_bytes",
        "comm_rx_bytes",
        "comm_total_bytes",
        "q_tile_reads",
        "k_dkv_visits",
        "q_tiles_per_k_dkv",
    )
    for record in result.records:
        if record.direction != "backward" or record.mode != "causal":
            raise ValueError("backward rank record has the wrong direction or mode")
        if record.method != result.method:
            raise ValueError("backward rank record has the wrong method")
        if any(getattr(record, field) < 0 for field in numeric_fields):
            raise ValueError("backward load counters must be non-negative")
        expected_ratio = (
            record.q_tile_reads / record.k_dkv_visits
            if record.k_dkv_visits
            else 0.0
        )
        if abs(record.q_tiles_per_k_dkv - expected_ratio) > 1e-12:
            raise ValueError("Q tiles/K-dKV ratio must be derived from its counters")


def _add_tasks(
    load: _MutableBackwardRankLoad,
    tasks: Sequence[AttentionTask],
    q_heads: int,
) -> None:
    for task in tasks:
        area = attention_area(task)
        load.effective_scores += area
        load.physical_scores += area
        _add_task(load, task, q_heads)


def analyze_backward_allgather(
    global_lengths: Sequence[int],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    llama3: bool = False,
    heads_k_stride: int = 1,
) -> BackwardMethodLoadResult:
    method = "llama3_allgather_attention" if llama3 else "allgather_attention"
    loads = [_MutableBackwardRankLoad() for _ in range(world_size)]
    local_tokens = sum(global_lengths) // world_size
    gather_bytes = (
        local_tokens
        * kv_heads
        * head_dim
        * 2
        * BF16_BYTES
        * (world_size - 1)
    )
    reduce_scatter_bytes = (
        local_tokens
        * kv_heads
        * head_dim
        * 2
        * FP32_BYTES
        * (world_size - 1)
    )
    for rank, load in enumerate(loads):
        load.effective_tokens = local_tokens
        load.physical_tokens = local_tokens
        load.comm_tx_bytes = gather_bytes + reduce_scatter_bytes
        load.comm_rx_bytes = gather_bytes + reduce_scatter_bytes
        if llama3:
            total_tokens = sum(global_lengths)
            chunk = total_tokens // (2 * world_size)
            back_block = 2 * world_size - 1 - rank
            tasks = [
                *_llama3_tasks_for_interval(
                    global_lengths, rank * chunk, (rank + 1) * chunk, True
                ),
                *_llama3_tasks_for_interval(
                    global_lengths,
                    back_block * chunk,
                    (back_block + 1) * chunk,
                    True,
                ),
            ]
        else:
            tasks = [
                task
                for length in global_lengths
                for task in _allgather_tasks(length, world_size, rank, True)
            ]
        _add_tasks(load, tasks, q_heads)

    layout = "whole-packed" if llama3 else "per-sequence"
    note = (
        f"{layout} KV-head-sliced backward ({heads_k_stride} KVH/chunk); "
        "BF16 K/V all-gather plus FP32 dK/dV reduce-scatter; setup and "
        "forward preparation excluded"
    )
    return _finalize(method, loads, q_heads, head_dim, note)


def _add_python_ring_communication(
    loads: Sequence[_MutableBackwardRankLoad],
    placement: Placement,
    kv_heads: int,
    head_dim: int,
) -> None:
    if placement.ring_size == 1:
        return
    local_length = placement.global_length // placement.ring_size
    kv_bytes = (
        (placement.ring_size - 1)
        * local_length
        * kv_heads
        * head_dim
        * 2
        * BF16_BYTES
    )
    dkv_bytes = (
        placement.ring_size
        * local_length
        * kv_heads
        * head_dim
        * 2
        * FP32_BYTES
    )
    for rank in range(
        placement.ring_start, placement.ring_start + placement.ring_size
    ):
        loads[rank].comm_tx_bytes += kv_bytes + dkv_bytes
        loads[rank].comm_rx_bytes += kv_bytes + dkv_bytes


def _add_fused_kv_communication(
    loads: Sequence[_MutableBackwardRankLoad],
    placement: Placement,
    kv_heads: int,
    head_dim: int,
) -> None:
    ring_size = placement.ring_size
    if ring_size == 1:
        return
    local_length = placement.global_length // ring_size
    half = local_length // 2
    row_bytes = kv_heads * head_dim * 2 * BF16_BYTES
    for ring_local_rank in range(ring_size):
        receiver = placement.ring_start + ring_local_rank
        for step in range(1, ring_size):
            rows = half if step <= ring_local_rank else local_length
            source = placement.ring_start + (ring_local_rank - step) % ring_size
            transfer_bytes = rows * row_bytes
            loads[source].comm_tx_bytes += transfer_bytes
            loads[receiver].comm_rx_bytes += transfer_bytes


def _add_fused_dkv_communication(
    loads: Sequence[_MutableBackwardRankLoad],
    placements: Sequence[Placement],
    world_size: int,
    kv_heads: int,
    head_dim: int,
) -> None:
    """Account for the kernel's per-level padded FP32 owner accumulators."""

    for ring_size in (8, 4, 2, 1):
        level = [placement for placement in placements if placement.ring_size == ring_size]
        if not level or ring_size == 1:
            continue
        for rank in range(world_size):
            local_rows = sum(
                placement.global_length // ring_size
                for placement in level
                if placement.ring_start <= rank < placement.ring_start + ring_size
            )
            if not local_rows:
                continue
            # The backward descriptor places one 128-row accumulator gap after
            # every batch in this level, including zero-length non-member slots.
            padded_rows = local_rows + len(level) * TILE_TOKENS
            transfer_bytes = (
                padded_rows
                * kv_heads
                * head_dim
                * 2
                * FP32_BYTES
            )
            ring_base = (rank // ring_size) * ring_size
            ring_local_rank = rank - ring_base
            for step in range(1, ring_size):
                owner = ring_base + (ring_local_rank - step) % ring_size
                loads[rank].comm_tx_bytes += transfer_bytes
                loads[owner].comm_rx_bytes += transfer_bytes


def _placements_backward_loads(
    placements: Sequence[Placement],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    communication: str,
) -> list[_MutableBackwardRankLoad]:
    loads = [_MutableBackwardRankLoad() for _ in range(world_size)]
    for placement in placements:
        local_length = placement.global_length // placement.ring_size
        for ring_local_rank in range(placement.ring_size):
            rank = placement.ring_start + ring_local_rank
            load = loads[rank]
            load.effective_tokens += local_length
            load.physical_tokens += local_length
            _add_tasks(
                load,
                _ring_tasks(
                    local_length, placement.ring_size, ring_local_rank, True
                ),
                q_heads,
            )
        if communication == "python-ring":
            _add_python_ring_communication(loads, placement, kv_heads, head_dim)
        elif communication == "mega-ring":
            _add_fused_kv_communication(loads, placement, kv_heads, head_dim)
        else:
            raise ValueError(f"unsupported communication model {communication!r}")
    if communication == "mega-ring":
        _add_fused_dkv_communication(
            loads, placements, world_size, kv_heads, head_dim
        )
    return loads


def analyze_backward_fa3_ring(
    global_lengths: Sequence[int],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> BackwardMethodLoadResult:
    placements = [Placement(length, world_size, 0) for length in global_lengths]
    loads = _placements_backward_loads(
        placements,
        world_size,
        q_heads,
        kv_heads,
        head_dim,
        communication="python-ring",
    )
    note = (
        "whole-packed NCCL zigzag backward; BF16 K/V ring uses p-1 steps; "
        "FP32 dK/dV owner-return ring uses p steps"
    )
    return _finalize("fa3_ring", loads, q_heads, head_dim, note)


def analyze_backward_megatron(
    global_lengths: Sequence[int],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    max_seqlen_per_rank: int = 8192,
) -> BackwardMethodLoadResult:
    plan = build_hybrid_cp_plan_for_fa3_ring(
        list(global_lengths), world_size, True, max_seqlen_per_rank
    )
    ordered_sample_ids: list[int] = []
    seen_sample_ids: set[int] = set()
    for group in plan.execution_groups:
        for rank_samples in group.sample_ids_by_rank:
            for sample_id in rank_samples:
                if sample_id not in seen_sample_ids:
                    seen_sample_ids.add(sample_id)
                    ordered_sample_ids.append(sample_id)
    placements = [
        Placement(
            plan.assignment(sample_id).global_length,
            plan.assignment(sample_id).cp_size,
            plan.assignment(sample_id).rank_start,
        )
        for sample_id in ordered_sample_ids
    ]
    loads = _placements_backward_loads(
        placements,
        world_size,
        q_heads,
        kv_heads,
        head_dim,
        communication="python-ring",
    )
    for load in loads:
        load.effective_tokens = 0
        load.effective_scores = 0
    for sample_id, original_length in enumerate(global_lengths):
        assignment = plan.assignment(sample_id)
        effective_tokens = original_length / assignment.cp_size
        effective_scores = score_count([original_length], True) / assignment.cp_size
        for rank in assignment.ranks:
            loads[rank].effective_tokens += effective_tokens
            loads[rank].effective_scores += effective_scores
    padding = sum(plan.global_lengths) - sum(global_lengths)
    saturation = hybrid_cp_saturation_note(list(global_lengths), plan)
    note = (
        f"Megatron scheduler backward with post-schedule FA3 ring padding; "
        f"{plan.num_execution_groups} execution groups; padding={padding} tokens; "
        f"max_seqlen_per_rank={max_seqlen_per_rank}; CP1 communication is zero"
    )
    if saturation:
        note = f"{note}; {saturation}"
    return _finalize("megatron_hybrid_cp", loads, q_heads, head_dim, note)


def analyze_backward_zepplin(
    global_lengths: Sequence[int],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    threshold: int = DEFAULT_ZEPPLIN_THRESHOLD,
) -> BackwardMethodLoadResult:
    plan = make_zepplin_plan(list(global_lengths), world_size, True, threshold)
    owner_by_index = dict(zip(plan.short_indices, plan.short_owners))
    placements = [
        Placement(
            length,
            1 if index in owner_by_index else world_size,
            owner_by_index.get(index, 0),
        )
        for index, length in enumerate(global_lengths)
    ]
    loads = _placements_backward_loads(
        placements,
        world_size,
        q_heads,
        kv_heads,
        head_dim,
        communication="python-ring",
    )
    note = (
        f"Zeppelin backward; threshold={threshold}; G1={len(plan.short_indices)}, "
        f"Gworld={len(plan.long_indices)}; LPT G1 communication is zero"
    )
    return _finalize("zepplin", loads, q_heads, head_dim, note)


def analyze_backward_mega_ring_hybrid(
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> BackwardMethodLoadResult:
    placements = [
        Placement(length, ring_size, ring_start)
        for length, ring_size, ring_start in zip(
            global_lengths, ring_sizes, ring_starts
        )
    ]
    loads = _placements_backward_loads(
        placements,
        world_size,
        q_heads,
        kv_heads,
        head_dim,
        communication="mega-ring",
    )
    note = (
        "hierarchical fused mega-ring backward; BF16 causal half/full remote K/V; "
        "FP32 remote dK/dV store-add includes one 128-row accumulator gap per "
        "batch and level; local owner stores excluded"
    )
    return _finalize(
        "mega_ring_hybrid",
        loads,
        q_heads,
        head_dim,
        note,
    )


def analyze_backward_mega_ring_all_cp(
    global_lengths: Sequence[int],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> BackwardMethodLoadResult:
    aligned = align_mega_ring_all_cp_lengths(list(global_lengths))
    placements = [Placement(length, world_size, 0) for length in aligned]
    loads = _placements_backward_loads(
        placements,
        world_size,
        q_heads,
        kv_heads,
        head_dim,
        communication="mega-ring",
    )
    effective_tokens = sum(global_lengths) / world_size
    effective_scores = score_count(global_lengths, True) / world_size
    for load in loads:
        load.effective_tokens = effective_tokens
        load.effective_scores = effective_scores
    padding = sum(aligned) - sum(global_lengths)
    note = (
        f"all-CP fused mega-ring backward; 2048-token alignment; padding={padding} "
        "tokens; FP32 remote dK/dV includes per-level 128-row batch padding"
    )
    return _finalize(
        "mega_ring_all_cp",
        loads,
        q_heads,
        head_dim,
        note,
    )


def analyze_backward_method(
    method: str,
    global_lengths: Sequence[int],
    ring_sizes: Sequence[int],
    ring_starts: Sequence[int],
    world_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    heads_k_stride: int = 4,
    zepplin_threshold: int = DEFAULT_ZEPPLIN_THRESHOLD,
    megatron_max_seqlen_per_rank: int = 8192,
) -> BackwardMethodLoadResult:
    if method == "allgather_attention":
        return analyze_backward_allgather(
            global_lengths,
            world_size,
            q_heads,
            kv_heads,
            head_dim,
            heads_k_stride=heads_k_stride,
        )
    if method == "llama3_allgather_attention":
        return analyze_backward_allgather(
            global_lengths,
            world_size,
            q_heads,
            kv_heads,
            head_dim,
            llama3=True,
            heads_k_stride=heads_k_stride,
        )
    if method == "fa3_ring":
        return analyze_backward_fa3_ring(
            global_lengths, world_size, q_heads, kv_heads, head_dim
        )
    if method == "megatron_hybrid_cp":
        return analyze_backward_megatron(
            global_lengths,
            world_size,
            q_heads,
            kv_heads,
            head_dim,
            max_seqlen_per_rank=megatron_max_seqlen_per_rank,
        )
    if method == "zepplin":
        return analyze_backward_zepplin(
            global_lengths,
            world_size,
            q_heads,
            kv_heads,
            head_dim,
            threshold=zepplin_threshold,
        )
    if method == "mega_ring_all_cp":
        return analyze_backward_mega_ring_all_cp(
            global_lengths, world_size, q_heads, kv_heads, head_dim
        )
    if method == "mega_ring_hybrid":
        return analyze_backward_mega_ring_hybrid(
            global_lengths,
            ring_sizes,
            ring_starts,
            world_size,
            q_heads,
            kv_heads,
            head_dim,
        )
    if method == "magi_attention":
        raise ValueError("MagiAttention requires its distributed metadata adapter")
    raise ValueError(f"unknown method {method!r}")


def _backward_magi_tasks(attn_arg: object) -> list[AttentionTask]:
    q_ranges = getattr(attn_arg, "q_ranges_bwd", attn_arg.q_ranges)
    k_ranges = getattr(attn_arg, "k_ranges_bwd", attn_arg.k_ranges)
    masks = getattr(attn_arg, "attn_type_map_bwd", attn_arg.attn_type_map)
    tasks: list[AttentionTask] = []
    for q_range, k_range, mask in zip(q_ranges, k_ranges, masks):
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


def _backend_name(backend: object) -> str:
    value = getattr(backend, "value", backend)
    return str(value).lower()


def _magi_gradient_bytes(metadata: object) -> int:
    backend = _backend_name(metadata.kernel_backend)
    if "fa4" in backend:
        return BF16_BYTES
    return (
        FP32_BYTES
        if metadata.use_native_grpcoll or metadata.bwd_hp_reduce
        else BF16_BYTES
    )


def analyze_backward_magi_rank(
    metadata: object,
    global_lengths: Sequence[int],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> BackwardRankLoadRecord:
    """Consume one rank's Magi backward metadata without running data paths."""

    dispatch_q = metadata.dispatch_meta_q
    rank = int(dispatch_q.cp_rank)
    valid_ranges = [
        _range_bounds(attn_range)
        for attn_range in dispatch_q.host_ranges_per_rank[rank]
    ]
    effective_tokens, effective_scores = iter_global_q_score(
        global_lengths, valid_ranges, True
    )
    physical_tokens = int(dispatch_q.shard_seqlen)

    calc_meta = metadata.calc_meta
    no_overlap = bool(getattr(calc_meta, "no_overlap", False))
    merged_arg = getattr(calc_meta, "merged_attn_arg", None)
    if no_overlap and merged_arg is not None:
        attn_args = [merged_arg]
    else:
        attn_args = [calc_meta.local_attn_arg, *calc_meta.remote_attn_args_list]

    physical_scores = 0
    q_tile_reads = 0
    k_dkv_visits = 0
    for attn_arg in attn_args:
        for task in _backward_magi_tasks(attn_arg):
            physical_scores += attention_area(task)
            reads, visits = backward_task_tile_counters(task, q_heads)
            q_tile_reads += reads
            k_dkv_visits += visits

    comm_meta = metadata.comm_meta
    tx_bytes = 0
    rx_bytes = 0
    remote_stage_count = len(comm_meta.kv_group_collective_args_list)
    cached_tail = bool(metadata.save_tail_stage) and not no_overlap
    fetch_stage_count = remote_stage_count - int(cached_tail and remote_stage_count > 0)

    for arg in comm_meta.kv_group_collective_args_list[:fetch_stage_count]:
        tx_tokens, rx_tokens = _collective_cast_endpoints(arg)
        tensor_count = 2 if metadata.use_native_grpcoll else 1
        bytes_per_token = kv_heads * head_dim * BF16_BYTES * tensor_count
        tx_bytes += tx_tokens * bytes_per_token
        rx_bytes += rx_tokens * bytes_per_token

    grad_bytes = _magi_gradient_bytes(metadata)
    for arg in comm_meta.kv_group_collective_args_list:
        tx_tokens, rx_tokens = _collective_reduce_endpoints(arg)
        tensor_count = 2 if metadata.use_native_grpcoll else 1
        bytes_per_token = kv_heads * head_dim * grad_bytes * tensor_count
        tx_bytes += tx_tokens * bytes_per_token
        rx_bytes += rx_tokens * bytes_per_token

    if metadata.enable_qo_comm:
        if metadata.use_native_grpcoll:
            fetch_args = comm_meta.qo_group_collective_args_list
            for arg in fetch_args:
                tx_tokens, rx_tokens = _collective_cast_endpoints(arg)
                bytes_per_token = q_heads * (
                    3 * head_dim * BF16_BYTES + FP32_BYTES
                )
                tx_bytes += tx_tokens * bytes_per_token
                rx_bytes += rx_tokens * bytes_per_token
        else:
            for arg in comm_meta.qo_do_group_collective_args_list:
                tx_tokens, rx_tokens = _collective_cast_endpoints(arg)
                bytes_per_token = q_heads * head_dim * BF16_BYTES
                tx_bytes += tx_tokens * bytes_per_token
                rx_bytes += rx_tokens * bytes_per_token
            for arg in comm_meta.qo_group_collective_args_list:
                tx_tokens, rx_tokens = _collective_cast_endpoints(arg)
                bytes_per_token = q_heads * FP32_BYTES
                tx_bytes += tx_tokens * bytes_per_token
                rx_bytes += rx_tokens * bytes_per_token

        for arg in comm_meta.qo_group_collective_args_list:
            tx_tokens, rx_tokens = _collective_reduce_endpoints(arg)
            bytes_per_token = q_heads * head_dim * grad_bytes
            tx_bytes += tx_tokens * bytes_per_token
            rx_bytes += rx_tokens * bytes_per_token

    ratio = q_tile_reads / k_dkv_visits if k_dkv_visits else 0.0
    backend = _backend_name(metadata.kernel_backend)
    note = (
        f"metadata-only Magi backward plan; backend={backend}; "
        f"bwd_hp_reduce={metadata.bwd_hp_reduce}; "
        f"save_tail_stage={metadata.save_tail_stage}; chunk_size={metadata.chunk_size}; "
        f"tokens(original/padded)={metadata.original_tokens}/{metadata.padded_tokens}; "
        f"overlap_degree={metadata.overlap_degree}; dispatch and forward preparation excluded"
    )
    record = BackwardRankLoadRecord(
        direction="backward",
        method="magi_attention",
        mode="causal",
        rank=rank,
        effective_tokens=effective_tokens,
        physical_tokens=physical_tokens,
        effective_scores=effective_scores,
        physical_scores=physical_scores,
        effective_flops=BACKWARD_FLOPS_PER_SCORE
        * effective_scores
        * q_heads
        * head_dim,
        physical_flops=BACKWARD_FLOPS_PER_SCORE
        * physical_scores
        * q_heads
        * head_dim,
        comm_tx_bytes=tx_bytes,
        comm_rx_bytes=rx_bytes,
        comm_total_bytes=tx_bytes,
        q_tile_reads=q_tile_reads,
        k_dkv_visits=k_dkv_visits,
        q_tiles_per_k_dkv=ratio,
        note=note,
    )
    return record


def backward_magi_result_from_records(
    records: Sequence[BackwardRankLoadRecord],
) -> BackwardMethodLoadResult:
    ordered = tuple(sorted(records, key=lambda record: record.rank))
    if not ordered:
        raise ValueError("MagiAttention result requires at least one rank record")
    result = BackwardMethodLoadResult(
        "backward", "magi_attention", "causal", ordered, ordered[0].note
    )
    validate_backward_result(result)
    return result


def cumulative_backward_result(
    results: Sequence[BackwardMethodLoadResult],
) -> BackwardMethodLoadResult:
    if not results:
        raise ValueError("cannot accumulate an empty result list")
    first = results[0]
    world_size = len(first.records)
    if any(
        result.direction != "backward"
        or result.method != first.method
        or result.mode != "causal"
        or len(result.records) != world_size
        for result in results
    ):
        raise ValueError("cumulative backward results must share method and world size")
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
        "q_tile_reads",
        "k_dkv_visits",
    )
    stable_notes = {
        "megatron_hybrid_cp": "Megatron backward plans vary by case",
        "magi_attention": "metadata-only Magi backward plans vary by case",
        "zepplin": "Zeppelin backward placement varies by case",
        "mega_ring_all_cp": "per-case 2048-token alignment and dKV padding",
    }
    note = (
        f"cumulative over {len(results)} cases; "
        f"{stable_notes.get(first.method, first.note)}"
    )
    records: list[BackwardRankLoadRecord] = []
    for rank in range(world_size):
        values = {
            field: sum(getattr(result.records[rank], field) for result in results)
            for field in summed_fields
        }
        ratio = (
            values["q_tile_reads"] / values["k_dkv_visits"]
            if values["k_dkv_visits"]
            else 0.0
        )
        records.append(
            replace(
                first.records[rank],
                **values,
                q_tiles_per_k_dkv=ratio,
                note=note,
            )
        )
    result = BackwardMethodLoadResult(
        "backward", first.method, "causal", tuple(records), note
    )
    validate_backward_result(result)
    return result


def global_backward_ratio(result: BackwardMethodLoadResult) -> float:
    reads = sum(record.q_tile_reads for record in result.records)
    visits = sum(record.k_dkv_visits for record in result.records)
    return reads / visits if visits else 0.0


def global_backward_score_conserved(
    result: BackwardMethodLoadResult, global_lengths: Sequence[int]
) -> bool:
    expected = score_count(global_lengths, True)
    actual = sum(record.effective_scores for record in result.records)
    return abs(actual - expected) <= max(1e-6, abs(expected) * 1e-12)


__all__ = [
    "BACKWARD_FLOPS_PER_SCORE",
    "BackwardMethodLoadResult",
    "BackwardRankLoadRecord",
    "METHOD_ORDER",
    "analyze_backward_allgather",
    "analyze_backward_fa3_ring",
    "analyze_backward_magi_rank",
    "analyze_backward_mega_ring_all_cp",
    "analyze_backward_mega_ring_hybrid",
    "analyze_backward_megatron",
    "analyze_backward_method",
    "analyze_backward_zepplin",
    "backward_magi_result_from_records",
    "backward_task_tile_counters",
    "cumulative_backward_result",
    "global_backward_ratio",
    "global_backward_score_conserved",
    "validate_backward_result",
]
