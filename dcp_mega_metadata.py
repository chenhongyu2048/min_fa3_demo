"""Host metadata builder for the batched varlen DCP mega forward path.

The split selection and dynamic per-sequence split calculation are copied and
trimmed from ``include/hopper_compat/heuristics.h`` and
``csrc/min_fa3_varlen_prepare_scheduler.cu``.  Keeping the builder pure Python
makes its queue coverage testable without a GPU and lets the runner fill a
preallocated pinned host buffer before each launch.
"""

from __future__ import annotations

import heapq
import math
from array import array
from dataclasses import dataclass
from typing import Sequence


CHUNK = 0
HISTORY = 1

ATTENTION_DESC_FIELDS = 8
Q_TASK_FIELDS = 4
PUBLISH_DESC_FIELDS = 8
HISTORY_COMBINE_DESC_FIELDS = 8
FINAL_DESC_FIELDS = 8
METADATA_HEADER_INTS = 40
METADATA_VERSION = 7
MEGA_COMPUTE_WARPS = 12
# Minimum adaptive final granularity and the runner capacity bound.
FINAL_TOKENS_PER_TASK = 4
FINAL_TOKEN_GRANULARITIES = (4, 8, 16)
HISTORY_MAX_COPY_VECTORS = 32
HISTORY_TASK_WAVE_TARGET = 0.8
SCHEDULER_POLICY_FIFO = "fifo"
SCHEDULER_POLICY_HEURISTIC = "release_lpt_critical_wave"
SCHEDULER_POLICY_CRITICAL_WAVE_FIFO = "fifo_critical_wave"
SCHEDULER_POLICY_NATIVE_RELEASE_LPT = "release_lpt_fa3_native"
SPLIT_POLICY_CRITICAL_WAVE = "critical_wave"
SPLIT_POLICY_FA3_NATIVE = "fa3_native"
HISTORY_ORDER_POLICY_FIFO = "fifo"
HISTORY_ORDER_POLICY_RELEASE_LPT = "release_lpt"
HEURISTIC_TASK_OVERHEAD = 4
HEURISTIC_COMBINE_TASK_OVERHEAD = HEURISTIC_TASK_OVERHEAD
HEURISTIC_COMBINE_PARTIAL_VECTOR_COST = 1
HEURISTIC_MIN_MAKESPAN_GAIN = 0.10
HEURISTIC_SPLIT4_EXTRA_GAIN = 0.05


@dataclass(frozen=True)
class DCPMegaDispatch:
    effective_num_splits: int
    chunk_num_splits: int
    history_num_splits: int
    pack_gqa: bool
    split: bool
    block_n: int
    history_copy_vectors_per_task: int


@dataclass(frozen=True)
class _CriticalWavePlan:
    sequence_splits: tuple[int, ...]
    source: str
    attention_makespan: int
    attention_tasks: int
    combine_tasks: int
    combine_partial_vectors: int
    combine_work: int
    combine_penalty: float
    critical_history_sequences: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class _HistoryCombineProfile:
    task_count: int
    partial_vector_count: int
    work: int


@dataclass(frozen=True)
class _AttentionScheduleProfile:
    makespan: int
    task_count: int
    critical_history_sequences: tuple[int, ...]
    cta_finish_times: tuple[int, ...] = ()
    completion_finish_times: tuple[int, ...] = ()


@dataclass(frozen=True)
class _HistoryCombineScheduleTask:
    dependencies: tuple[int, ...]
    work: int


@dataclass(frozen=True)
class DCPMegaMetadata:
    """Immutable queue image consumed by the CUDA binding.

    Descriptor rows deliberately contain only int32-compatible values.  The
    binding copies these rows verbatim into preallocated pinned tensors.
    """

    dispatch: DCPMegaDispatch
    attention: tuple[tuple[int, ...], ...]
    q_tasks: tuple[tuple[int, ...], ...]
    q_dependencies: tuple[int, ...]
    publish: tuple[tuple[int, ...], ...]
    history_combine: tuple[tuple[int, ...], ...]
    publish_dependencies: tuple[int, ...]
    final: tuple[tuple[int, ...], ...]
    final_dependencies: tuple[int, ...]
    chunk_sequence_splits: tuple[int, ...]
    history_sequence_splits: tuple[int, ...]
    total_q: int
    total_vectors: int
    final_tokens_per_task: int
    token_block_count: int
    q_ready_count: int
    receive_count: int
    tile_ready_count: int
    dcp_size: int
    scheduler_mode: str
    split_policy: str
    history_order_policy: str
    scheduler_policy: str
    heuristic_model_block_n: int | None
    heuristic_plan_source: str | None
    heuristic_history_sequence_splits: tuple[int, ...] | None
    heuristic_baseline_attention_tasks: int | None
    heuristic_selected_attention_tasks: int | None
    heuristic_baseline_combine_tasks: int | None
    heuristic_selected_combine_tasks: int | None
    heuristic_baseline_combine_partial_vectors: int | None
    heuristic_selected_combine_partial_vectors: int | None
    heuristic_baseline_combine_work: int | None
    heuristic_selected_combine_work: int | None
    heuristic_baseline_combine_penalty: float | None
    heuristic_selected_combine_penalty: float | None
    heuristic_split_sequence_idx: int | None
    heuristic_split_sequence_splits: int | None
    heuristic_baseline_makespan: int | None
    heuristic_selected_makespan: int | None
    heuristic_gain: float | None
    heuristic_q_block_order: tuple[int, ...]

    @property
    def counts(self) -> tuple[int, int, int, int, int, int, int, int]:
        return (
            len(self.attention),
            len(self.q_tasks),
            len(self.q_dependencies),
            len(self.publish),
            len(self.history_combine),
            len(self.publish_dependencies),
            len(self.final),
            len(self.final_dependencies),
        )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _choose_final_tokens_per_task(
    token_block_count: int, num_compute_ctas: int
) -> int:
    """Choose final compute granularity from 16-token communication tasks."""
    if token_block_count < num_compute_ctas:
        return 4
    if token_block_count < 2 * num_compute_ctas:
        return 8
    return 16


def _combined_scheduler_policy(
    split_policy: str, history_order_policy: str
) -> str:
    release_lpt = history_order_policy == HISTORY_ORDER_POLICY_RELEASE_LPT
    if split_policy == SPLIT_POLICY_CRITICAL_WAVE:
        return (
            SCHEDULER_POLICY_HEURISTIC
            if release_lpt
            else SCHEDULER_POLICY_CRITICAL_WAVE_FIFO
        )
    if split_policy == SPLIT_POLICY_FA3_NATIVE:
        return (
            SCHEDULER_POLICY_NATIVE_RELEASE_LPT
            if release_lpt
            else SCHEDULER_POLICY_FIFO
        )
    raise ValueError(f"unknown split policy: {split_policy}")


def _validate_cu_seqlens(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) < 2:
        raise ValueError(f"{name} must have shape [B + 1] with B >= 1")
    if result[0] != 0:
        raise ValueError(f"{name} must start with 0")
    for batch_idx, (begin, end) in enumerate(zip(result, result[1:])):
        if end <= begin:
            raise ValueError(
                f"{name} must be strictly increasing; sequence {batch_idx} "
                f"has non-positive length {end - begin}"
            )
    return result


def num_splits_heuristic(
    total_mblocks: int,
    num_sms: int,
    num_n_blocks: int,
    num_m_blocks: int,
    size_one_kv_head: int,
    is_causal: bool,
    max_splits: int = 128,
) -> int:
    """Copied host equivalent of the Hopper forward split heuristic."""
    if min(total_mblocks, num_sms, num_n_blocks, num_m_blocks, max_splits) <= 0:
        raise ValueError("split heuristic inputs must be positive")
    if total_mblocks >= 0.8 * num_sms:
        size_l2 = 50 * 1024 * 1024
        if (
            size_one_kv_head > size_l2
            and num_m_blocks >= num_sms * 2
            and not is_causal
        ):
            return min(_ceil_div(size_one_kv_head, size_l2), max_splits)
        return 1
    if num_n_blocks <= 4:
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    efficiencies: list[float] = []
    for num_splits in range(1, max_splits + 1):
        waves = total_mblocks * num_splits / num_sms
        efficiencies.append(waves / math.ceil(waves))
    threshold = 0.85 * max(efficiencies)
    return next(
        split
        for split, efficiency in enumerate(efficiencies, start=1)
        if efficiency >= threshold
    )


def choose_split_upper_bound(
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    q_heads: int,
    num_sms: int,
    block_n: int,
    is_causal: bool,
    requested_num_splits: int,
    max_num_splits: int = 128,
) -> int:
    if requested_num_splits < 0 or requested_num_splits > 128:
        raise ValueError("num_splits must be 0 (auto), 1, or in [2, 128]")
    if max_num_splits < 1 or max_num_splits > 128:
        raise ValueError("max_num_splits must be in [1, 128]")
    if requested_num_splits > max_num_splits:
        raise ValueError("num_splits cannot exceed max_num_splits")
    if requested_num_splits:
        return requested_num_splits
    num_m_blocks = _ceil_div(max_seqlen_q * q_heads, 128)
    num_n_blocks = _ceil_div(max_seqlen_k, block_n)
    return num_splits_heuristic(
        num_m_blocks,
        num_sms,
        num_n_blocks,
        num_m_blocks,
        max_seqlen_k * (128 + 128) * 2,
        is_causal,
        max_splits=max_num_splits,
    )


def choose_dispatch(
    *,
    max_seqlen_q: int,
    max_seqlen_history: int,
    hq_local: int,
    dcp_size: int,
    num_sms: int,
    requested_num_splits: int,
    block_n_override: int | None = None,
    max_num_splits: int = 128,
) -> DCPMegaDispatch:
    """Apply the existing Split/Pack rules to the two attention domains."""
    if block_n_override not in (None, 128, 176):
        raise ValueError("block_n_override must be None, 128, or 176")
    if dcp_size not in (2, 4, 8):
        raise ValueError("DCP mega only supports dcp_size in {2, 4, 8}")
    if hq_local not in (4, 8):
        raise ValueError("DCP mega requires hq_local in {4, 8}")
    if min(max_seqlen_q, max_seqlen_history, hq_local, num_sms) <= 0:
        raise ValueError("sequence lengths, hq_local, and num_sms must be positive")

    block_n = 128 if block_n_override is None else block_n_override
    chunk_splits = choose_split_upper_bound(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_q,
        q_heads=hq_local,
        num_sms=num_sms,
        block_n=block_n,
        is_causal=True,
        requested_num_splits=requested_num_splits,
        max_num_splits=max_num_splits,
    )
    history_splits = choose_split_upper_bound(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_history,
        q_heads=dcp_size * hq_local,
        num_sms=num_sms,
        block_n=block_n,
        is_causal=False,
        requested_num_splits=requested_num_splits,
        max_num_splits=max_num_splits,
    )
    effective = max(chunk_splits, history_splits)
    return DCPMegaDispatch(
        effective_num_splits=effective,
        chunk_num_splits=chunk_splits,
        history_num_splits=history_splits,
        pack_gqa=True,
        split=effective > 1,
        block_n=block_n,
        history_copy_vectors_per_task=1,
    )


def _dynamic_sequence_splits(
    q_lengths: Sequence[int],
    k_lengths: Sequence[int],
    *,
    heads: int,
    pack_gqa: bool,
    split_upper_bound: int,
    num_sms: int,
    block_n: int,
) -> tuple[int, ...]:
    if split_upper_bound == 1:
        return (1,) * len(q_lengths)
    m_blocks = [
        _ceil_div(q_len * heads if pack_gqa else q_len, 128)
        for q_len in q_lengths
    ]
    n_blocks = [_ceil_div(k_len, block_n) for k_len in k_lengths]
    total_blocks = sum(m * n for m, n in zip(m_blocks, n_blocks))
    scheduler_heads = 1 if pack_gqa else heads
    blocks_per_sm = math.ceil(total_blocks * 1.1 * scheduler_heads / num_sms)
    blocks_per_sm = max(blocks_per_sm, 1)
    return tuple(
        max(min(_ceil_div(n, blocks_per_sm), split_upper_bound), 1)
        for n in n_blocks
    )


def _split_n_block_count(
    num_n_blocks: int,
    split_idx: int,
    num_splits: int,
) -> int:
    blocks_per_split = _ceil_div(num_n_blocks, num_splits)
    split_begin = split_idx * blocks_per_split
    return max(min(blocks_per_split, num_n_blocks - split_begin), 0)


def _fifo_attention_profile(
    q_lengths: Sequence[int],
    history_n_blocks: Sequence[int],
    *,
    hq_local: int,
    dcp_size: int,
    block_n: int,
    num_compute_ctas: int,
    split_sequence_idx: int | None = None,
    split_sequence_splits: int = 1,
    history_sequence_splits: Sequence[int] | None = None,
) -> _AttentionScheduleProfile:
    """Estimate wave quantization with the kernel's FIFO descriptor order."""
    if history_sequence_splits is not None:
        if split_sequence_idx is not None or split_sequence_splits != 1:
            raise ValueError(
                "history_sequence_splits cannot be combined with a single "
                "split override"
            )
        if len(history_sequence_splits) != len(q_lengths):
            raise ValueError("history_sequence_splits must match batch size")
        sequence_splits = tuple(int(value) for value in history_sequence_splits)
        if any(value < 1 or value > 128 for value in sequence_splits):
            raise ValueError("history sequence splits must be in [1, 128]")
    else:
        sequence_splits = tuple(
            split_sequence_splits if index == split_sequence_idx else 1
            for index in range(len(q_lengths))
        )
    task_costs: list[tuple[int, int | None]] = []
    for q_len in q_lengths:
        for m_block in range(_ceil_div(q_len * hq_local, 128)):
            causal_tokens = min(q_len, _ceil_div((m_block + 1) * 128, hq_local))
            task_costs.append(
                (
                    _ceil_div(causal_tokens, block_n) + HEURISTIC_TASK_OVERHEAD,
                    None,
                )
            )
    for sequence_idx, (q_len, num_n_blocks, splits) in enumerate(
        zip(q_lengths, history_n_blocks, sequence_splits)
    ):
        for _ in range(_ceil_div(q_len * dcp_size * hq_local, 128)):
            for split_idx in range(splits):
                task_costs.append(
                    (
                        _split_n_block_count(num_n_blocks, split_idx, splits)
                        + HEURISTIC_TASK_OVERHEAD,
                        sequence_idx,
                    )
                )

    return _schedule_attention_tasks(task_costs, num_compute_ctas)


def _schedule_attention_tasks(
    task_costs: Sequence[tuple[int, int | None] | tuple[int, int | None, int]],
    num_compute_ctas: int,
) -> _AttentionScheduleProfile:
    """List-schedule attention descriptors and retain completion times."""
    if num_compute_ctas <= 0:
        raise ValueError("num_compute_ctas must be positive")
    worker_loads = [(0, worker) for worker in range(num_compute_ctas)]
    heapq.heapify(worker_loads)
    final_sequence_by_worker: list[int | None] = [None] * num_compute_ctas
    completion_finish_times = [0] * len(task_costs)
    for ordinal, task in enumerate(task_costs):
        task_cost, sequence_idx = task[0], task[1]
        completion_id = task[2] if len(task) == 3 else ordinal
        load, worker = heapq.heappop(worker_loads)
        load += task_cost
        completion_finish_times[completion_id] = load
        final_sequence_by_worker[worker] = sequence_idx
        heapq.heappush(worker_loads, (load, worker))
    makespan = max(load for load, _ in worker_loads)
    cta_finish_times = [0] * num_compute_ctas
    for load, worker in worker_loads:
        cta_finish_times[worker] = load
    critical_history_sequences = tuple(
        sorted(
            {
                final_sequence_by_worker[worker]
                for load, worker in worker_loads
                if load == makespan
                and final_sequence_by_worker[worker] is not None
            }
        )
    )
    return _AttentionScheduleProfile(
        makespan=makespan,
        task_count=len(task_costs),
        critical_history_sequences=critical_history_sequences,
        cta_finish_times=tuple(cta_finish_times),
        completion_finish_times=tuple(completion_finish_times),
    )


def _fifo_attention_makespan(
    q_lengths: Sequence[int],
    history_n_blocks: Sequence[int],
    *,
    hq_local: int,
    dcp_size: int,
    block_n: int,
    num_compute_ctas: int,
    split_sequence_idx: int | None = None,
    split_sequence_splits: int = 1,
    history_sequence_splits: Sequence[int] | None = None,
) -> tuple[int, int]:
    profile = _fifo_attention_profile(
        q_lengths,
        history_n_blocks,
        hq_local=hq_local,
        dcp_size=dcp_size,
        block_n=block_n,
        num_compute_ctas=num_compute_ctas,
        split_sequence_idx=split_sequence_idx,
        split_sequence_splits=split_sequence_splits,
        history_sequence_splits=history_sequence_splits,
    )
    return profile.makespan, profile.task_count


def _choose_critical_wave_split(
    q_lengths: Sequence[int],
    history_lengths: Sequence[int],
    *,
    hq_local: int,
    dcp_size: int,
    block_n: int,
    num_compute_ctas: int,
) -> tuple[int | None, int | None, int, int, float]:
    """Choose at most one decode-history split using a dimensionless proxy."""
    history_n_blocks = tuple(
        _ceil_div(length, block_n) for length in history_lengths
    )
    baseline, _ = _fifo_attention_makespan(
        q_lengths,
        history_n_blocks,
        hq_local=hq_local,
        dcp_size=dcp_size,
        block_n=block_n,
        num_compute_ctas=num_compute_ctas,
    )
    candidates: list[tuple[int, int, int, float]] = []
    for batch_idx, (q_len, num_n_blocks) in enumerate(
        zip(q_lengths, history_n_blocks)
    ):
        if q_len > 16 or num_n_blocks < num_compute_ctas:
            continue
        split2, _ = _fifo_attention_makespan(
            q_lengths,
            history_n_blocks,
            hq_local=hq_local,
            dcp_size=dcp_size,
            block_n=block_n,
            num_compute_ctas=num_compute_ctas,
            split_sequence_idx=batch_idx,
            split_sequence_splits=2,
        )
        split4, _ = _fifo_attention_makespan(
            q_lengths,
            history_n_blocks,
            hq_local=hq_local,
            dcp_size=dcp_size,
            block_n=block_n,
            num_compute_ctas=num_compute_ctas,
            split_sequence_idx=batch_idx,
            split_sequence_splits=4,
        )
        if (split2 - split4) / baseline >= HEURISTIC_SPLIT4_EXTRA_GAIN:
            selected_splits, selected = 4, split4
        else:
            selected_splits, selected = 2, split2
        gain = (baseline - selected) / baseline
        if gain >= HEURISTIC_MIN_MAKESPAN_GAIN:
            candidates.append((selected, selected_splits, batch_idx, gain))

    if not candidates:
        return None, None, baseline, baseline, 0.0
    selected, selected_splits, batch_idx, gain = min(candidates)
    return batch_idx, selected_splits, baseline, selected, gain


def _critical_wave_plan(
    q_lengths: Sequence[int],
    history_lengths: Sequence[int],
    *,
    hq_local: int,
    dcp_size: int,
    block_n: int,
    num_sms: int,
    num_comm_sm: int,
    chunk_sequence_splits: Sequence[int],
    include_legacy_candidate: bool,
    max_num_splits: int,
    native_chunk_sequence_splits: Sequence[int] | None,
    legacy_is_native: bool,
    reorder_no_split: bool,
    reorder_split: bool,
) -> tuple[_CriticalWavePlan, _CriticalWavePlan]:
    """Compare iterative and legacy multi-sequence split candidates."""
    num_compute_ctas = num_sms - num_comm_sm
    history_n_blocks = tuple(
        _ceil_div(length, block_n) for length in history_lengths
    )
    cu_q_values = [0]
    for q_len in q_lengths:
        cu_q_values.append(cu_q_values[-1] + q_len)
    cu_q = tuple(cu_q_values)

    def profile(sequence_splits: Sequence[int], source: str) -> _CriticalWavePlan:
        splits = tuple(int(value) for value in sequence_splits)
        effective_chunk_splits = (
            tuple(native_chunk_sequence_splits)
            if legacy_is_native
            and source == "legacy_dynamic"
            and native_chunk_sequence_splits is not None
            else tuple(chunk_sequence_splits)
        )
        attention_profile, _, completion_for_vector = _attention_schedule_profile(
            q_lengths,
            history_n_blocks,
            hq_local=hq_local,
            dcp_size=dcp_size,
            block_n=block_n,
            num_compute_ctas=num_compute_ctas,
            num_comm_sm=num_comm_sm,
            chunk_sequence_splits=effective_chunk_splits,
            history_sequence_splits=splits,
            reorder_history=(
                False
                if legacy_is_native and source == "legacy_dynamic"
                else reorder_split
                if any(value > 1 for value in splits)
                else reorder_no_split
            ),
        )
        copy_vectors = _choose_history_copy_vectors_per_task(
            cu_q,
            splits,
            hq_local=hq_local,
            dcp_size=dcp_size,
            num_sms=num_sms,
            num_comm_sm=num_comm_sm,
        )
        combine_profile = _history_combine_profile(
            cu_q,
            splits,
            hq_local=hq_local,
            dcp_size=dcp_size,
            copy_vectors_per_task=copy_vectors,
        )
        combine_tasks = _history_combine_schedule_tasks(
            cu_q,
            splits,
            completion_for_vector,
            hq_local=hq_local,
            dcp_size=dcp_size,
            copy_vectors_per_task=copy_vectors,
        )
        if (
            len(combine_tasks) != combine_profile.task_count
            or sum(task.work for task in combine_tasks) != combine_profile.work
        ):
            raise AssertionError("combine schedule/profile accounting mismatch")
        score = _overlapped_attention_combine_makespan(
            attention_profile, combine_tasks
        )
        combine_penalty = score - attention_profile.makespan
        return _CriticalWavePlan(
            sequence_splits=splits,
            source=source,
            attention_makespan=attention_profile.makespan,
            attention_tasks=attention_profile.task_count,
            combine_tasks=combine_profile.task_count,
            combine_partial_vectors=combine_profile.partial_vector_count,
            combine_work=combine_profile.work,
            combine_penalty=combine_penalty,
            critical_history_sequences=(
                attention_profile.critical_history_sequences
            ),
            score=score,
        )

    no_split = profile((1,) * len(q_lengths), "nosplit")
    if include_legacy_candidate:
        legacy_upper_bound = choose_split_upper_bound(
            max_seqlen_q=max(q_lengths),
            max_seqlen_k=max(history_lengths),
            q_heads=dcp_size * hq_local,
            num_sms=num_sms,
            block_n=block_n,
            is_causal=False,
            requested_num_splits=0,
            max_num_splits=max_num_splits,
        )
        legacy_splits = _dynamic_sequence_splits(
            q_lengths,
            history_lengths,
            heads=dcp_size * hq_local,
            pack_gqa=True,
            split_upper_bound=legacy_upper_bound,
            num_sms=num_sms,
            block_n=block_n,
        )
    else:
        legacy_splits = no_split.sequence_splits

    split_caps = list(legacy_splits)
    for index, (q_len, num_n_blocks) in enumerate(
        zip(q_lengths, history_n_blocks)
    ):
        if q_len <= 16 and num_n_blocks >= num_compute_ctas:
            split_caps[index] = min(max_num_splits, max(split_caps[index], 4))

    iterative = no_split
    while True:
        candidate_split_vectors: set[tuple[int, ...]] = set()
        for index, cap in enumerate(split_caps):
            if iterative.sequence_splits[index] >= cap:
                continue
            candidate_splits = list(iterative.sequence_splits)
            candidate_splits[index] += 1
            candidate_split_vectors.add(tuple(candidate_splits))
        critical_sequences = iterative.critical_history_sequences
        if len(critical_sequences) > 1 and all(
            iterative.sequence_splits[index] < split_caps[index]
            for index in critical_sequences
        ):
            candidate_splits = list(iterative.sequence_splits)
            for index in critical_sequences:
                candidate_splits[index] += 1
            candidate_split_vectors.add(tuple(candidate_splits))
        candidates = [
            profile(sequence_splits, "iterative")
            for sequence_splits in candidate_split_vectors
        ]
        if not candidates:
            break
        candidate = min(
            candidates,
            key=lambda plan: (
                plan.score,
                plan.attention_makespan,
                plan.attention_tasks,
                sum(plan.sequence_splits),
                plan.sequence_splits,
            ),
        )
        if candidate.score < iterative.score:
            iterative = candidate
        else:
            break

    plans = [no_split, iterative]
    if legacy_splits != no_split.sequence_splits:
        plans.append(profile(legacy_splits, "legacy_dynamic"))

    selected = min(
        plans,
        key=lambda plan: (
            plan.score,
            plan.attention_makespan,
            plan.attention_tasks,
            sum(plan.sequence_splits),
            plan.sequence_splits,
        ),
    )
    gain = (no_split.score - selected.score) / no_split.score
    if gain < HEURISTIC_MIN_MAKESPAN_GAIN or selected.score >= no_split.score:
        selected = no_split
    return no_split, selected


def _append_attention_domain(
    rows: list[tuple[int, ...]],
    q_dependencies: list[int],
    completion_for_vector: dict[tuple[int, int, int, int], tuple[int, ...]],
    *,
    kind: int,
    cu_q: tuple[int, ...],
    heads: int,
    sequence_splits: tuple[int, ...],
) -> None:
    for batch_idx, (q_begin, q_end) in enumerate(zip(cu_q, cu_q[1:])):
        q_len = q_end - q_begin
        splits = sequence_splits[batch_idx]
        tile_coordinates = (
            (m_block, 0) for m_block in range(_ceil_div(q_len * heads, 128))
        )
        for m_block, head_coord in tile_coordinates:
            dependencies: tuple[int, ...] = ()
            if kind == HISTORY:
                dependency_set: set[int] = set()
                packed_begin = m_block * 128
                packed_end = min(packed_begin + 128, q_len * heads)
                # History Q TMA may speculatively read the rest of this 128-row
                # tile, but only valid packed rows gate the existing q_ready data.
                # Invalid tail rows must never enter a cross-row reduction or a
                # predicated output store.
                for packed in range(packed_begin, packed_end):
                    token_rel, _ = divmod(packed, heads)
                    dependency_set.add((q_begin + token_rel) // 16)
                dependencies = tuple(sorted(dependency_set))
            dep_begin = len(q_dependencies)
            q_dependencies.extend(dependencies)
            completion_ids: list[int] = []
            for split_idx in range(splits):
                completion_id = len(rows)
                completion_ids.append(completion_id)
                rows.append(
                    (
                        kind,
                        batch_idx,
                        m_block,
                        head_coord,
                        split_idx,
                        dep_begin,
                        len(dependencies),
                        completion_id,
                    )
                )
            packed_begin = m_block * 128
            packed_end = min(packed_begin + 128, q_len * heads)
            physical_vectors = (
                (q_begin + packed // heads, packed % heads)
                for packed in range(packed_begin, packed_end)
            )
            for token, physical_head in physical_vectors:
                completion_for_vector[(kind, batch_idx, token, physical_head)] = tuple(
                    completion_ids
                )


def _release_lpt_attention_order(
    attention: Sequence[tuple[int, ...]],
    q_dependencies: Sequence[int],
    *,
    num_token_subtiles: int,
    history_n_blocks: Sequence[int],
    history_sequence_splits: Sequence[int],
    num_comm_sm: int,
    dcp_size: int,
) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
    """Return the Q release order and the matching attention descriptor order."""
    unlock_values = [0.0] * num_token_subtiles
    for row in attention:
        if row[0] != HISTORY:
            continue
        dependencies = q_dependencies[row[5] : row[5] + row[6]]
        tile_work = (
            _split_n_block_count(
                history_n_blocks[row[1]],
                row[4],
                history_sequence_splits[row[1]],
            )
            + HEURISTIC_TASK_OVERHEAD
        )
        for dependency in dependencies:
            unlock_values[dependency] += tile_work / len(dependencies)
    q_block_order = tuple(
        sorted(
            range(num_token_subtiles),
            key=lambda block: (-unlock_values[block], block),
        )
    )
    q_position = {
        block: position for position, block in enumerate(q_block_order)
    }
    q_counters_per_epoch = max(1, num_comm_sm // dcp_size)

    def history_order(row: tuple[int, ...]) -> tuple[int, ...]:
        dependencies = q_dependencies[row[5] : row[5] + row[6]]
        release_position = max(q_position[dependency] for dependency in dependencies)
        tile_work = (
            _split_n_block_count(
                history_n_blocks[row[1]],
                row[4],
                history_sequence_splits[row[1]],
            )
            + HEURISTIC_TASK_OVERHEAD
        )
        return (
            release_position // q_counters_per_epoch,
            -tile_work,
            row[1],
            row[2],
            row[4],
            row[7],
        )

    chunk_attention = [row for row in attention if row[0] == CHUNK]
    history_attention = [row for row in attention if row[0] == HISTORY]
    return q_block_order, chunk_attention + sorted(
        history_attention, key=history_order
    )


def _attention_schedule_profile(
    q_lengths: Sequence[int],
    history_n_blocks: Sequence[int],
    *,
    hq_local: int,
    dcp_size: int,
    block_n: int,
    num_compute_ctas: int,
    num_comm_sm: int,
    chunk_sequence_splits: Sequence[int],
    history_sequence_splits: Sequence[int],
    reorder_history: bool,
) -> tuple[
    _AttentionScheduleProfile,
    tuple[int, ...],
    dict[tuple[int, int, int, int], tuple[int, ...]],
]:
    """Model the exact attention queue order used by a candidate plan."""
    cu_q_values = [0]
    for q_len in q_lengths:
        cu_q_values.append(cu_q_values[-1] + q_len)
    cu_q = tuple(cu_q_values)
    attention: list[tuple[int, ...]] = []
    q_dependencies: list[int] = []
    completion_for_vector: dict[
        tuple[int, int, int, int], tuple[int, ...]
    ] = {}
    _append_attention_domain(
        attention,
        q_dependencies,
        completion_for_vector,
        kind=CHUNK,
        cu_q=cu_q,
        heads=hq_local,
        sequence_splits=tuple(chunk_sequence_splits),
    )
    _append_attention_domain(
        attention,
        q_dependencies,
        completion_for_vector,
        kind=HISTORY,
        cu_q=cu_q,
        heads=dcp_size * hq_local,
        sequence_splits=tuple(history_sequence_splits),
    )
    if reorder_history:
        _, attention = _release_lpt_attention_order(
            attention,
            q_dependencies,
            num_token_subtiles=_ceil_div(cu_q[-1], 16),
            history_n_blocks=history_n_blocks,
            history_sequence_splits=history_sequence_splits,
            num_comm_sm=num_comm_sm,
            dcp_size=dcp_size,
        )

    task_costs: list[tuple[int, int | None, int]] = []
    for row in attention:
        kind, batch_idx, m_block, _, split_idx, _, _, completion_id = row
        if kind == CHUNK:
            causal_tokens = min(
                q_lengths[batch_idx],
                _ceil_div((m_block + 1) * 128, hq_local),
            )
            n_blocks = _ceil_div(causal_tokens, block_n)
            splits = chunk_sequence_splits[batch_idx]
            sequence_idx = None
        else:
            n_blocks = history_n_blocks[batch_idx]
            splits = history_sequence_splits[batch_idx]
            sequence_idx = batch_idx
        task_costs.append(
            (
                _split_n_block_count(n_blocks, split_idx, splits)
                + HEURISTIC_TASK_OVERHEAD,
                sequence_idx,
                completion_id,
            )
        )
    return (
        _schedule_attention_tasks(task_costs, num_compute_ctas),
        cu_q,
        completion_for_vector,
    )


def _history_combine_schedule_tasks(
    cu_q: tuple[int, ...],
    history_sequence_splits: Sequence[int],
    completion_for_vector: dict[
        tuple[int, int, int, int], tuple[int, ...]
    ],
    *,
    hq_local: int,
    dcp_size: int,
    copy_vectors_per_task: int,
) -> tuple[_HistoryCombineScheduleTask, ...]:
    """Build combine work in the same final-tile/destination FIFO order."""
    total_q = cu_q[-1]
    batch_for_token = [0] * total_q
    for batch_idx, (begin, end) in enumerate(zip(cu_q, cu_q[1:])):
        batch_for_token[begin:end] = [batch_idx] * (end - begin)

    tasks: list[_HistoryCombineScheduleTask] = []
    tile_vectors = 16 * hq_local
    total_vectors = total_q * hq_local
    for vector_begin in range(0, total_vectors, tile_vectors):
        tile_end = min(vector_begin + tile_vectors, total_vectors)
        for dst_rank in range(dcp_size):
            region_begin = vector_begin
            while region_begin < tile_end:
                token = region_begin // hq_local
                batch_idx = batch_for_token[token]
                region_end = min(tile_end, cu_q[batch_idx + 1] * hq_local)
                actual_splits = history_sequence_splits[batch_idx]
                vectors_per_task = (
                    1 if actual_splits > 1 else copy_vectors_per_task
                )
                for task_vector_begin in range(
                    region_begin, region_end, vectors_per_task
                ):
                    valid_vectors = min(
                        vectors_per_task, region_end - task_vector_begin
                    )
                    dependency_set: set[int] = set()
                    for vector in range(
                        task_vector_begin,
                        task_vector_begin + valid_vectors,
                    ):
                        vector_token, local_head = divmod(vector, hq_local)
                        history_head = dst_rank * hq_local + local_head
                        dependency_set.update(
                            completion_for_vector[
                                (HISTORY, batch_idx, vector_token, history_head)
                            ]
                        )
                    tasks.append(
                        _HistoryCombineScheduleTask(
                            dependencies=tuple(sorted(dependency_set)),
                            work=(
                                HEURISTIC_COMBINE_TASK_OVERHEAD
                                + valid_vectors * actual_splits
                            ),
                        )
                    )
                region_begin = region_end
    return tuple(tasks)


def _overlapped_attention_combine_makespan(
    attention_profile: _AttentionScheduleProfile,
    combine_tasks: Sequence[_HistoryCombineScheduleTask],
) -> int:
    """Schedule dependent combine warps as each CTA leaves attention."""
    if not combine_tasks:
        return attention_profile.makespan
    worker_heap = [
        (available, cta, warp)
        for cta, available in enumerate(attention_profile.cta_finish_times)
        for warp in range(MEGA_COMPUTE_WARPS)
    ]
    heapq.heapify(worker_heap)
    for task in combine_tasks:
        available, cta, warp = heapq.heappop(worker_heap)
        dependency_ready = max(
            attention_profile.completion_finish_times[completion_id]
            for completion_id in task.dependencies
        )
        finish = max(available, dependency_ready) + task.work
        heapq.heappush(worker_heap, (finish, cta, warp))
    combine_finish = max(available for available, _, _ in worker_heap)
    return max(attention_profile.makespan, combine_finish)


def _history_combine_profile(
    cu_q: tuple[int, ...],
    history_sequence_splits: tuple[int, ...],
    *,
    hq_local: int,
    dcp_size: int,
    copy_vectors_per_task: int,
) -> _HistoryCombineProfile:
    tasks_per_destination = 0
    partial_vectors_per_destination = 0
    for tile_begin in range(0, cu_q[-1], 16):
        tile_end = min(tile_begin + 16, cu_q[-1])
        for batch_idx, (q_begin, q_end) in enumerate(zip(cu_q, cu_q[1:])):
            region_begin = max(tile_begin, q_begin)
            region_end = min(tile_end, q_end)
            if region_begin >= region_end:
                continue
            region_vectors = (region_end - region_begin) * hq_local
            actual_splits = history_sequence_splits[batch_idx]
            partial_vectors_per_destination += region_vectors * actual_splits
            if actual_splits > 1:
                tasks_per_destination += region_vectors
            else:
                tasks_per_destination += _ceil_div(
                    region_vectors, copy_vectors_per_task
                )
    task_count = dcp_size * tasks_per_destination
    partial_vector_count = dcp_size * partial_vectors_per_destination
    work = (
        HEURISTIC_COMBINE_TASK_OVERHEAD * task_count
        + HEURISTIC_COMBINE_PARTIAL_VECTOR_COST * partial_vector_count
    )
    return _HistoryCombineProfile(
        task_count=task_count,
        partial_vector_count=partial_vector_count,
        work=work,
    )


def _history_combine_task_count(
    cu_q: tuple[int, ...],
    history_sequence_splits: tuple[int, ...],
    *,
    hq_local: int,
    dcp_size: int,
    copy_vectors_per_task: int,
) -> int:
    return _history_combine_profile(
        cu_q,
        history_sequence_splits,
        hq_local=hq_local,
        dcp_size=dcp_size,
        copy_vectors_per_task=copy_vectors_per_task,
    ).task_count


def _choose_history_copy_vectors_per_task(
    cu_q: tuple[int, ...],
    history_sequence_splits: tuple[int, ...],
    *,
    hq_local: int,
    dcp_size: int,
    num_sms: int,
    num_comm_sm: int,
) -> int:
    candidates = sorted(
        {
            1,
            hq_local,
            2 * hq_local,
            4 * hq_local,
            min(8 * hq_local, HISTORY_MAX_COPY_VECTORS),
        }
    )
    worker_warps = (num_sms - num_comm_sm) * MEGA_COMPUTE_WARPS
    target_claims = math.ceil(HISTORY_TASK_WAVE_TARGET * worker_warps)
    for candidate in reversed(candidates):
        task_count = _history_combine_profile(
            cu_q,
            history_sequence_splits,
            hq_local=hq_local,
            dcp_size=dcp_size,
            copy_vectors_per_task=candidate,
        ).task_count
        if task_count >= target_claims:
            return candidate
    return 1


def build_dcp_mega_metadata(
    cu_seqlens_q: Sequence[int],
    cu_seqlens_history: Sequence[int],
    *,
    hq_local: int,
    dcp_size: int,
    num_sms: int,
    num_comm_sm: int,
    requested_num_splits: int = 0,
    block_n_override: int | None = None,
    max_num_splits: int = 128,
    scheduler_heuristic: bool | None = None,
    reorder_history_override: bool | None = None,
) -> DCPMegaMetadata:
    """Build all host queues for one packed-varlen chunk prefill call."""
    cu_q = _validate_cu_seqlens(cu_seqlens_q, "cu_seqlens_q")
    cu_history = _validate_cu_seqlens(
        cu_seqlens_history, "cu_seqlens_history"
    )
    if len(cu_q) != len(cu_history):
        raise ValueError("Q and history cu_seqlens must have the same batch size")
    if hq_local <= 0:
        raise ValueError("hq_local must be positive")
    if num_comm_sm <= 0 or num_comm_sm >= num_sms:
        raise ValueError("num_comm_sm must be positive and smaller than num_sms")
    q_lengths = tuple(end - begin for begin, end in zip(cu_q, cu_q[1:]))
    history_lengths = tuple(
        end - begin for begin, end in zip(cu_history, cu_history[1:])
    )
    if reorder_history_override is not None and not isinstance(
        reorder_history_override, bool
    ):
        raise ValueError("reorder_history_override must be a bool or None")
    if scheduler_heuristic is not None and not isinstance(
        scheduler_heuristic, bool
    ):
        raise ValueError("scheduler_heuristic must be a bool or None")
    if max_num_splits < 1 or max_num_splits > 128:
        raise ValueError("max_num_splits must be in [1, 128]")
    auto_mode = scheduler_heuristic is None and requested_num_splits == 0
    if scheduler_heuristic is None and requested_num_splits == 1:
        scheduler_heuristic = True
    elif scheduler_heuristic is None and requested_num_splits > 1:
        scheduler_heuristic = False
    if scheduler_heuristic and requested_num_splits not in (0, 1):
        raise ValueError(
            "scheduler_heuristic requires requested_num_splits in {0, 1}"
        )
    if auto_mode:
        scheduler_mode = "auto"
    elif scheduler_heuristic:
        scheduler_mode = "critical_wave"
    else:
        scheduler_mode = "fa3_native"
    if scheduler_mode in ("auto", "critical_wave"):
        split_policy = SPLIT_POLICY_CRITICAL_WAVE
    else:
        split_policy = SPLIT_POLICY_FA3_NATIVE
    decode_only = all(q_len <= 16 for q_len in q_lengths)
    if reorder_history_override is None:
        reorder_no_split = False
        reorder_split = (
            split_policy == SPLIT_POLICY_CRITICAL_WAVE and decode_only
        )
    else:
        reorder_no_split = reorder_history_override
        reorder_split = reorder_history_override
    auto_block_n = (
        block_n_override is None
        and scheduler_mode in ("auto", "critical_wave")
    )
    dispatch = choose_dispatch(
        max_seqlen_q=max(q_lengths),
        max_seqlen_history=max(history_lengths),
        hq_local=hq_local,
        dcp_size=dcp_size,
        num_sms=num_sms,
        requested_num_splits=(
            1 if auto_mode or scheduler_heuristic else requested_num_splits
        ),
        block_n_override=block_n_override,
        max_num_splits=max_num_splits,
    )
    critical_wave_mode = scheduler_mode in ("auto", "critical_wave")
    native_dispatch = dispatch
    if auto_mode:
        native_dispatch = choose_dispatch(
            max_seqlen_q=max(q_lengths),
            max_seqlen_history=max(history_lengths),
            hq_local=hq_local,
            dcp_size=dcp_size,
            num_sms=num_sms,
            requested_num_splits=0,
            block_n_override=block_n_override,
            max_num_splits=max_num_splits,
        )
    heuristic_model_block_n = (
        dispatch.block_n if critical_wave_mode else None
    )

    chunk_sequence_splits = _dynamic_sequence_splits(
        q_lengths,
        q_lengths,
        heads=hq_local,
        pack_gqa=dispatch.pack_gqa,
        split_upper_bound=dispatch.chunk_num_splits,
        num_sms=num_sms,
        block_n=dispatch.block_n,
    )
    history_sequence_splits = _dynamic_sequence_splits(
        q_lengths,
        history_lengths,
        heads=dcp_size * hq_local,
        pack_gqa=dispatch.pack_gqa,
        split_upper_bound=dispatch.history_num_splits,
        num_sms=num_sms,
        block_n=dispatch.block_n,
    )
    native_chunk_sequence_splits = _dynamic_sequence_splits(
        q_lengths,
        q_lengths,
        heads=hq_local,
        pack_gqa=native_dispatch.pack_gqa,
        split_upper_bound=native_dispatch.chunk_num_splits,
        num_sms=num_sms,
        block_n=native_dispatch.block_n,
    )
    heuristic_split_sequence_idx = None
    heuristic_split_sequence_splits = None
    heuristic_plan_source = None
    heuristic_history_sequence_splits = None
    heuristic_baseline_attention_tasks = None
    heuristic_selected_attention_tasks = None
    heuristic_baseline_combine_tasks = None
    heuristic_selected_combine_tasks = None
    heuristic_baseline_combine_partial_vectors = None
    heuristic_selected_combine_partial_vectors = None
    heuristic_baseline_combine_work = None
    heuristic_selected_combine_work = None
    heuristic_baseline_combine_penalty = None
    heuristic_selected_combine_penalty = None
    heuristic_baseline_makespan = None
    heuristic_selected_makespan = None
    heuristic_gain = None
    if scheduler_mode in ("auto", "critical_wave"):
        baseline_plan, selected_plan = _critical_wave_plan(
            q_lengths,
            history_lengths,
            hq_local=hq_local,
            dcp_size=dcp_size,
            block_n=dispatch.block_n,
            num_sms=num_sms,
            num_comm_sm=num_comm_sm,
            chunk_sequence_splits=chunk_sequence_splits,
            include_legacy_candidate=requested_num_splits == 0,
            max_num_splits=max_num_splits,
            native_chunk_sequence_splits=native_chunk_sequence_splits,
            legacy_is_native=auto_mode,
            reorder_no_split=reorder_no_split,
            reorder_split=reorder_split,
        )
        if auto_mode and selected_plan.source == "legacy_dynamic":
            split_policy = SPLIT_POLICY_FA3_NATIVE
            auto_block_n = False
            chunk_sequence_splits = native_chunk_sequence_splits
        heuristic_plan_source = selected_plan.source
        heuristic_baseline_attention_tasks = baseline_plan.attention_tasks
        heuristic_selected_attention_tasks = selected_plan.attention_tasks
        heuristic_baseline_combine_tasks = baseline_plan.combine_tasks
        heuristic_selected_combine_tasks = selected_plan.combine_tasks
        heuristic_baseline_combine_partial_vectors = (
            baseline_plan.combine_partial_vectors
        )
        heuristic_selected_combine_partial_vectors = (
            selected_plan.combine_partial_vectors
        )
        heuristic_baseline_combine_work = baseline_plan.combine_work
        heuristic_selected_combine_work = selected_plan.combine_work
        heuristic_baseline_combine_penalty = baseline_plan.combine_penalty
        heuristic_selected_combine_penalty = selected_plan.combine_penalty
        heuristic_baseline_makespan = baseline_plan.attention_makespan
        heuristic_selected_makespan = selected_plan.attention_makespan
        heuristic_gain = (
            baseline_plan.score - selected_plan.score
        ) / baseline_plan.score
        if selected_plan.sequence_splits != baseline_plan.sequence_splits:
            heuristic_history_sequence_splits = selected_plan.sequence_splits
            history_sequence_splits = selected_plan.sequence_splits
            split_indices = tuple(
                index
                for index, splits in enumerate(history_sequence_splits)
                if splits > 1
            )
            heuristic_split_sequence_idx = max(
                split_indices,
                key=lambda index: (
                    history_sequence_splits[index],
                    history_lengths[index],
                    -index,
                ),
            )
            heuristic_split_sequence_splits = history_sequence_splits[
                heuristic_split_sequence_idx
            ]
    chunk_num_splits = max(chunk_sequence_splits)
    history_num_splits = max(history_sequence_splits)
    effective_num_splits = max(chunk_num_splits, history_num_splits)
    dispatch = DCPMegaDispatch(
        effective_num_splits=effective_num_splits,
        chunk_num_splits=chunk_num_splits,
        history_num_splits=history_num_splits,
        pack_gqa=dispatch.pack_gqa,
        split=effective_num_splits > 1,
        block_n=dispatch.block_n,
        history_copy_vectors_per_task=1,
    )
    if auto_block_n and not dispatch.split:
        dispatch = DCPMegaDispatch(
            effective_num_splits=dispatch.effective_num_splits,
            chunk_num_splits=dispatch.chunk_num_splits,
            history_num_splits=dispatch.history_num_splits,
            pack_gqa=dispatch.pack_gqa,
            split=dispatch.split,
            block_n=176,
            history_copy_vectors_per_task=1,
        )

    copy_vectors_per_task = _choose_history_copy_vectors_per_task(
        cu_q,
        history_sequence_splits,
        hq_local=hq_local,
        dcp_size=dcp_size,
        num_sms=num_sms,
        num_comm_sm=num_comm_sm,
    )
    dispatch = DCPMegaDispatch(
        effective_num_splits=dispatch.effective_num_splits,
        chunk_num_splits=dispatch.chunk_num_splits,
        history_num_splits=dispatch.history_num_splits,
        pack_gqa=dispatch.pack_gqa,
        split=dispatch.split,
        block_n=dispatch.block_n,
        history_copy_vectors_per_task=copy_vectors_per_task,
    )

    total_q = cu_q[-1]
    num_token_subtiles = _ceil_div(total_q, 16)
    final_tokens_per_task = _choose_final_tokens_per_task(
        num_token_subtiles, num_sms - num_comm_sm
    )
    q_tasks: list[tuple[int, ...]] = []
    for token_subtile in range(num_token_subtiles):
        for src_rank in range(dcp_size):
            token_begin = token_subtile * 16
            valid_rows = min(16, total_q - token_begin)
            q_tasks.append((src_rank, 0, token_begin, valid_rows))

    attention: list[tuple[int, ...]] = []
    q_dependencies: list[int] = []
    completion_for_vector: dict[tuple[int, int, int, int], tuple[int, ...]] = {}
    _append_attention_domain(
        attention,
        q_dependencies,
        completion_for_vector,
        kind=CHUNK,
        cu_q=cu_q,
        heads=hq_local,
        sequence_splits=chunk_sequence_splits,
    )
    _append_attention_domain(
        attention,
        q_dependencies,
        completion_for_vector,
        kind=HISTORY,
        cu_q=cu_q,
        heads=dcp_size * hq_local,
        sequence_splits=history_sequence_splits,
    )

    heuristic_q_block_order = tuple(range(num_token_subtiles))
    if reorder_history_override is None:
        release_lpt_enabled = (
            split_policy == SPLIT_POLICY_CRITICAL_WAVE
            and heuristic_history_sequence_splits is not None
            and decode_only
        )
    else:
        release_lpt_enabled = reorder_history_override
    history_order_policy = (
        HISTORY_ORDER_POLICY_RELEASE_LPT
        if release_lpt_enabled
        else HISTORY_ORDER_POLICY_FIFO
    )
    if release_lpt_enabled:
        history_n_blocks = tuple(
            _ceil_div(length, dispatch.block_n) for length in history_lengths
        )
        heuristic_q_block_order, attention = _release_lpt_attention_order(
            attention,
            q_dependencies,
            num_token_subtiles=num_token_subtiles,
            history_n_blocks=history_n_blocks,
            history_sequence_splits=history_sequence_splits,
            num_comm_sm=num_comm_sm,
            dcp_size=dcp_size,
        )
        q_tasks = [
            (src_rank, 0, token_block * 16, min(16, total_q - token_block * 16))
            for token_block in heuristic_q_block_order
            for src_rank in range(dcp_size)
        ]

    publish: list[tuple[int, ...]] = []
    history_combine_by_publish: list[list[tuple[int, ...]]] = []
    publish_dependencies: list[int] = []
    publish_tile_size = 16 * hq_local
    batch_for_token = [0] * total_q
    for batch_idx, (begin, end) in enumerate(zip(cu_q, cu_q[1:])):
        batch_for_token[begin:end] = [batch_idx] * (end - begin)
    for dst_rank in range(dcp_size):
        for vector_begin in range(0, total_q * hq_local, publish_tile_size):
            valid_vectors = min(
                publish_tile_size, total_q * hq_local - vector_begin
            )
            publish_id = len(publish)
            publish_dependency_begin = len(publish_dependencies)
            combine_tasks: list[tuple[int, ...]] = []
            tile_end = vector_begin + valid_vectors
            region_begin = vector_begin
            while region_begin < tile_end:
                token = region_begin // hq_local
                batch_idx = batch_for_token[token]
                region_end = min(tile_end, cu_q[batch_idx + 1] * hq_local)
                actual_splits = history_sequence_splits[batch_idx]
                vectors_per_task = (
                    1 if actual_splits > 1 else copy_vectors_per_task
                )
                for task_vector_begin in range(
                    region_begin,
                    region_end,
                    vectors_per_task,
                ):
                    task_valid_vectors = min(
                        vectors_per_task, region_end - task_vector_begin
                    )
                    dependency_set: set[int] = set()
                    for vector in range(
                        task_vector_begin,
                        task_vector_begin + task_valid_vectors,
                    ):
                        vector_token, local_head = divmod(vector, hq_local)
                        history_head = dst_rank * hq_local + local_head
                        dependency_set.update(
                            completion_for_vector[
                                (HISTORY, batch_idx, vector_token, history_head)
                            ]
                        )
                    dependencies = tuple(sorted(dependency_set))
                    dependency_begin = len(publish_dependencies)
                    publish_dependencies.extend(dependencies)
                    combine_tasks.append(
                        (
                            publish_id,
                            task_vector_begin,
                            task_valid_vectors,
                            dependency_begin,
                            len(dependencies),
                            batch_idx,
                            actual_splits,
                            0,
                        )
                    )
                region_begin = region_end
            publish.append(
                (
                    dst_rank,
                    vector_begin,
                    valid_vectors,
                    publish_dependency_begin,
                    len(publish_dependencies) - publish_dependency_begin,
                    int(dispatch.pack_gqa),
                    len(combine_tasks),
                    0,
                )
            )
            history_combine_by_publish.append(combine_tasks)

    history_combine = tuple(
        task
        for parent_token_block in heuristic_q_block_order
        for dst_rank in range(dcp_size)
        for task in history_combine_by_publish[
            dst_rank * num_token_subtiles + parent_token_block
        ]
    )

    final: list[tuple[int, ...]] = []
    final_dependencies: list[int] = []
    total_vectors = total_q * hq_local
    for parent_token_block in heuristic_q_block_order:
        parent_token_begin = parent_token_block * 16
        parent_valid_tokens = min(16, total_q - parent_token_begin)
        for token_offset in range(0, parent_valid_tokens, final_tokens_per_task):
            token_begin = parent_token_begin + token_offset
            valid_tokens = min(
                final_tokens_per_task, parent_valid_tokens - token_offset
            )
            vector_begin = token_begin * hq_local
            valid_vectors = valid_tokens * hq_local
            dependency_set: set[int] = set()
            for vector in range(vector_begin, vector_begin + valid_vectors):
                token, local_head = divmod(vector, hq_local)
                batch_idx = batch_for_token[token]
                dependency_set.update(
                    completion_for_vector[(CHUNK, batch_idx, token, local_head)]
                )
            dependencies = tuple(sorted(dependency_set))
            dep_begin = len(final_dependencies)
            final_dependencies.extend(dependencies)
            final.append(
                (
                    vector_begin,
                    valid_vectors,
                    dep_begin,
                    len(dependencies),
                    parent_token_block,
                    0,
                    0,
                    0,
                )
            )

    result = DCPMegaMetadata(
        dispatch=dispatch,
        attention=tuple(attention),
        q_tasks=tuple(q_tasks),
        q_dependencies=tuple(q_dependencies),
        publish=tuple(publish),
        history_combine=history_combine,
        publish_dependencies=tuple(publish_dependencies),
        final=tuple(final),
        final_dependencies=tuple(final_dependencies),
        chunk_sequence_splits=chunk_sequence_splits,
        history_sequence_splits=history_sequence_splits,
        total_q=total_q,
        total_vectors=total_vectors,
        final_tokens_per_task=final_tokens_per_task,
        token_block_count=num_token_subtiles,
        q_ready_count=num_token_subtiles,
        receive_count=num_token_subtiles * (dcp_size - 1),
        tile_ready_count=num_token_subtiles * (dcp_size - 1),
        dcp_size=dcp_size,
        scheduler_mode=scheduler_mode,
        split_policy=split_policy,
        history_order_policy=history_order_policy,
        scheduler_policy=_combined_scheduler_policy(
            split_policy, history_order_policy
        ),
        heuristic_model_block_n=heuristic_model_block_n,
        heuristic_plan_source=heuristic_plan_source,
        heuristic_history_sequence_splits=heuristic_history_sequence_splits,
        heuristic_baseline_attention_tasks=heuristic_baseline_attention_tasks,
        heuristic_selected_attention_tasks=heuristic_selected_attention_tasks,
        heuristic_baseline_combine_tasks=heuristic_baseline_combine_tasks,
        heuristic_selected_combine_tasks=heuristic_selected_combine_tasks,
        heuristic_baseline_combine_partial_vectors=(
            heuristic_baseline_combine_partial_vectors
        ),
        heuristic_selected_combine_partial_vectors=(
            heuristic_selected_combine_partial_vectors
        ),
        heuristic_baseline_combine_work=heuristic_baseline_combine_work,
        heuristic_selected_combine_work=heuristic_selected_combine_work,
        heuristic_baseline_combine_penalty=heuristic_baseline_combine_penalty,
        heuristic_selected_combine_penalty=heuristic_selected_combine_penalty,
        heuristic_split_sequence_idx=heuristic_split_sequence_idx,
        heuristic_split_sequence_splits=heuristic_split_sequence_splits,
        heuristic_baseline_makespan=heuristic_baseline_makespan,
        heuristic_selected_makespan=heuristic_selected_makespan,
        heuristic_gain=heuristic_gain,
        heuristic_q_block_order=heuristic_q_block_order,
    )
    validate_dcp_mega_metadata(result, cu_q, hq_local=hq_local, dcp_size=dcp_size)
    return result


def validate_dcp_mega_metadata(
    metadata: DCPMegaMetadata,
    cu_seqlens_q: Sequence[int],
    *,
    hq_local: int,
    dcp_size: int,
) -> None:
    """Validate queue ranges, dependencies, and publication ordering."""
    cu_q = tuple(int(value) for value in cu_seqlens_q)
    if metadata.scheduler_mode not in ("auto", "critical_wave", "fa3_native"):
        raise AssertionError("unknown scheduler mode")
    if metadata.split_policy not in (
        SPLIT_POLICY_CRITICAL_WAVE,
        SPLIT_POLICY_FA3_NATIVE,
    ):
        raise AssertionError("unknown split policy")
    if metadata.history_order_policy not in (
        HISTORY_ORDER_POLICY_FIFO,
        HISTORY_ORDER_POLICY_RELEASE_LPT,
    ):
        raise AssertionError("unknown history order policy")
    if metadata.scheduler_policy != _combined_scheduler_policy(
        metadata.split_policy, metadata.history_order_policy
    ):
        raise AssertionError("combined scheduler policy is inconsistent")
    # Auto keeps the critical-wave candidate diagnostics even when the final
    # policy is native, so the selected policy can be audited without
    # reconstructing the host-side decision.
    heuristic_enabled = metadata.heuristic_plan_source is not None
    heuristic_split_enabled = (
        heuristic_enabled
        and metadata.heuristic_history_sequence_splits is not None
    )
    release_lpt_enabled = (
        metadata.history_order_policy == HISTORY_ORDER_POLICY_RELEASE_LPT
    )
    if heuristic_enabled:
        if metadata.heuristic_model_block_n not in (128, 176):
            raise AssertionError("heuristic model BlockN is invalid")
        if any(
            value is None
            for value in (
                metadata.heuristic_baseline_makespan,
                metadata.heuristic_selected_makespan,
                metadata.heuristic_gain,
                metadata.heuristic_plan_source,
                metadata.heuristic_baseline_attention_tasks,
                metadata.heuristic_selected_attention_tasks,
                metadata.heuristic_baseline_combine_tasks,
                metadata.heuristic_selected_combine_tasks,
                metadata.heuristic_baseline_combine_partial_vectors,
                metadata.heuristic_selected_combine_partial_vectors,
                metadata.heuristic_baseline_combine_work,
                metadata.heuristic_selected_combine_work,
                metadata.heuristic_baseline_combine_penalty,
                metadata.heuristic_selected_combine_penalty,
            )
        ):
            raise AssertionError("heuristic diagnostics are incomplete")
        if (
            metadata.heuristic_split_sequence_idx is None
        ) != (metadata.heuristic_split_sequence_splits is None):
            raise AssertionError("heuristic split selection is incomplete")
        if heuristic_split_enabled:
            if (
                metadata.heuristic_history_sequence_splits
                != metadata.history_sequence_splits
            ):
                raise AssertionError("heuristic split vector is inconsistent")
            if metadata.heuristic_split_sequence_idx is None:
                raise AssertionError("heuristic primary split is missing")
        elif metadata.heuristic_split_sequence_idx is not None:
            raise AssertionError("NoSplit heuristic has a primary split")
        if metadata.dispatch.block_n != metadata.heuristic_model_block_n and not (
            not metadata.dispatch.split
            and metadata.heuristic_model_block_n == 128
            and metadata.dispatch.block_n == 176
        ):
            raise AssertionError("heuristic model and dispatch BlockN are inconsistent")
    elif any(
        value is not None
        for value in (
            metadata.heuristic_split_sequence_idx,
            metadata.heuristic_split_sequence_splits,
            metadata.heuristic_baseline_makespan,
            metadata.heuristic_selected_makespan,
            metadata.heuristic_gain,
            metadata.heuristic_model_block_n,
            metadata.heuristic_plan_source,
            metadata.heuristic_history_sequence_splits,
            metadata.heuristic_baseline_attention_tasks,
            metadata.heuristic_selected_attention_tasks,
            metadata.heuristic_baseline_combine_tasks,
            metadata.heuristic_selected_combine_tasks,
            metadata.heuristic_baseline_combine_partial_vectors,
            metadata.heuristic_selected_combine_partial_vectors,
            metadata.heuristic_baseline_combine_work,
            metadata.heuristic_selected_combine_work,
            metadata.heuristic_baseline_combine_penalty,
            metadata.heuristic_selected_combine_penalty,
        )
    ):
        raise AssertionError("non-heuristic metadata contains heuristic diagnostics")
    if any(len(row) != ATTENTION_DESC_FIELDS for row in metadata.attention):
        raise AssertionError("invalid attention descriptor width")
    if any(len(row) != Q_TASK_FIELDS for row in metadata.q_tasks):
        raise AssertionError("invalid Q task width")
    if any(len(row) != PUBLISH_DESC_FIELDS for row in metadata.publish):
        raise AssertionError("invalid publish descriptor width")
    if any(
        len(row) != HISTORY_COMBINE_DESC_FIELDS
        for row in metadata.history_combine
    ):
        raise AssertionError("invalid history combine descriptor width")
    if any(len(row) != FINAL_DESC_FIELDS for row in metadata.final):
        raise AssertionError("invalid final descriptor width")

    completion_ids = [row[7] for row in metadata.attention]
    expected_completion_ids = list(range(len(metadata.attention)))
    if sorted(completion_ids) != expected_completion_ids:
        raise AssertionError("attention completion ids must be dense and unique")
    if not release_lpt_enabled and completion_ids != expected_completion_ids:
        raise AssertionError(
            "non-split FIFO attention completion ids must follow queue order"
        )
    chunk_count = sum(row[0] == CHUNK for row in metadata.attention)
    if any(row[0] != CHUNK for row in metadata.attention[:chunk_count]):
        raise AssertionError("chunk descriptors must precede history descriptors")
    if any(row[0] != HISTORY for row in metadata.attention[chunk_count:]):
        raise AssertionError("history descriptors must follow chunk descriptors")

    for row in metadata.attention:
        dep_begin, dep_count = row[5], row[6]
        if dep_begin < 0 or dep_begin + dep_count > len(metadata.q_dependencies):
            raise AssertionError("attention Q dependency range is out of bounds")
        if row[0] == CHUNK and dep_count != 0:
            raise AssertionError("chunk attention must not depend on gathered Q")
        for ready_id in metadata.q_dependencies[dep_begin : dep_begin + dep_count]:
            if ready_id < 0 or ready_id >= metadata.q_ready_count:
                raise AssertionError("Q dependency references an invalid ready counter")
    for descriptors, dependencies, dep_fields in (
        (metadata.publish, metadata.publish_dependencies, (3, 4)),
        (metadata.history_combine, metadata.publish_dependencies, (3, 4)),
        (metadata.final, metadata.final_dependencies, (2, 3)),
    ):
        for row in descriptors:
            dep_begin, dep_count = row[dep_fields[0]], row[dep_fields[1]]
            if dep_begin < 0 or dep_begin + dep_count > len(dependencies):
                raise AssertionError("partial dependency range is out of bounds")
            if any(
                dep < 0 or dep >= len(metadata.attention)
                for dep in dependencies[dep_begin : dep_begin + dep_count]
            ):
                raise AssertionError("partial dependency references an invalid completion")

    expected_q_tasks = dcp_size * _ceil_div(metadata.total_q, 16)
    if len(metadata.q_tasks) != expected_q_tasks:
        raise AssertionError("Q task queue does not cover its communication tiles")
    expected_publish = dcp_size * metadata.token_block_count
    if len(metadata.publish) != expected_publish:
        raise AssertionError("publish queue does not cover every destination vector")
    expected_history_combine = _history_combine_task_count(
        cu_q,
        metadata.history_sequence_splits,
        hq_local=hq_local,
        dcp_size=dcp_size,
        copy_vectors_per_task=(
            metadata.dispatch.history_copy_vectors_per_task
        ),
    )
    if len(metadata.history_combine) != expected_history_combine:
        raise AssertionError(
            "history combine queue does not cover every destination vector"
        )
    if metadata.final_tokens_per_task not in FINAL_TOKEN_GRANULARITIES:
        raise AssertionError("invalid final task granularity")
    expected_final = _ceil_div(metadata.total_q, metadata.final_tokens_per_task)
    if len(metadata.final) != expected_final:
        raise AssertionError("final queue does not cover every local output vector")

    expected_q_ready = metadata.token_block_count
    expected_receive = metadata.token_block_count * (dcp_size - 1)
    expected_tile_ready = expected_receive
    if metadata.q_ready_count != expected_q_ready:
        raise AssertionError("invalid Q-ready counter count")
    if metadata.receive_count != expected_receive:
        raise AssertionError("invalid receive queue count")
    if metadata.tile_ready_count != expected_tile_ready:
        raise AssertionError("invalid tile-ready counter count")
    if metadata.token_block_count != _ceil_div(metadata.total_q, 16):
        raise AssertionError("invalid token-block count")
    if metadata.dcp_size != dcp_size:
        raise AssertionError("metadata DCP size mismatch")
    valid_copy_vectors = {
        1,
        hq_local,
        2 * hq_local,
        4 * hq_local,
        min(8 * hq_local, HISTORY_MAX_COPY_VECTORS),
    }
    if (
        metadata.dispatch.history_copy_vectors_per_task
        not in valid_copy_vectors
    ):
        raise AssertionError("invalid history copy task granularity")

    q_coordinates = [(row[2] // 16, row[0]) for row in metadata.q_tasks]
    expected_q_coordinates = [
        (token_block, source)
        for token_block in metadata.heuristic_q_block_order
        for source in range(dcp_size)
    ]
    if q_coordinates != expected_q_coordinates:
        raise AssertionError("Q tasks do not follow their block order")
    if sorted(metadata.heuristic_q_block_order) != list(
        range(metadata.token_block_count)
    ):
        raise AssertionError("Q block order must be a permutation")
    if not release_lpt_enabled and metadata.heuristic_q_block_order != tuple(
        range(metadata.token_block_count)
    ):
        raise AssertionError("non-split FIFO Q blocks must remain token-major")
    if any(row[1] != 0 for row in metadata.q_tasks):
        raise AssertionError("Q tasks must not encode a local head")

    expected_final_layout = []
    for parent_token_block in metadata.heuristic_q_block_order:
        parent_token_begin = parent_token_block * 16
        parent_valid_tokens = min(16, metadata.total_q - parent_token_begin)
        for token_offset in range(
            0, parent_valid_tokens, metadata.final_tokens_per_task
        ):
            valid_tokens = min(
                metadata.final_tokens_per_task,
                parent_valid_tokens - token_offset,
            )
            expected_final_layout.append(
                (
                    (parent_token_begin + token_offset) * hq_local,
                    valid_tokens * hq_local,
                    parent_token_block,
                )
            )
    for row, expected in zip(metadata.final, expected_final_layout):
        vector_begin, valid_vectors, _, _, parent_token_block = row[:5]
        expected_vector_begin, expected_valid_vectors, expected_parent = expected
        if (
            vector_begin != expected_vector_begin
            or valid_vectors != expected_valid_vectors
            or parent_token_block != expected_parent
        ):
            raise AssertionError("final task granularity is invalid")
        if parent_token_block != vector_begin // (16 * hq_local):
            raise AssertionError("final task references an invalid parent tile")
        parent_vector_begin = parent_token_block * 16 * hq_local
        parent_vector_end = min(
            parent_vector_begin + 16 * hq_local,
            metadata.total_vectors,
        )
        if not (
            parent_vector_begin <= vector_begin
            and vector_begin + valid_vectors <= parent_vector_end
        ):
            raise AssertionError("final task crosses its parent tile")
        if any(value != 0 for value in row[5:8]):
            raise AssertionError("final reserved fields must be zero")

    publish_tiles_per_rank = metadata.token_block_count
    for publish_id, row in enumerate(metadata.publish):
        expected_dst, parent_token_block = divmod(
            publish_id, publish_tiles_per_rank
        )
        expected_vector_begin = parent_token_block * 16 * hq_local
        expected_valid_vectors = min(
            16 * hq_local,
            metadata.total_vectors - expected_vector_begin,
        )
        if (
            row[0] != expected_dst
            or row[1] != expected_vector_begin
            or row[2] != expected_valid_vectors
        ):
            raise AssertionError(
                "publish tasks must be destination-major communication tiles"
            )
        if row[5] != int(metadata.dispatch.pack_gqa):
            raise AssertionError("publish PackGQA marker mismatch")
        if row[6] <= 0 or row[7] != 0:
            raise AssertionError("publish combine completion target is invalid")

    combine_by_publish: list[list[tuple[int, ...]]] = [
        [] for _ in metadata.publish
    ]
    history_completion_ids: dict[tuple[int, int], tuple[int, ...]] = {}
    for row in metadata.attention:
        if row[0] != HISTORY:
            continue
        key = (row[1], row[2])
        history_completion_ids.setdefault(key, tuple())
        history_completion_ids[key] += (row[7],)
    for row in metadata.history_combine:
        publish_id, vector_begin, valid_vectors = row[0], row[1], row[2]
        dep_begin, dep_count, batch_idx, actual_splits = row[3:7]
        if publish_id < 0 or publish_id >= len(metadata.publish):
            raise AssertionError("history combine references an invalid publish task")
        publish_row = metadata.publish[publish_id]
        if (
            valid_vectors <= 0
            or vector_begin < publish_row[1]
            or vector_begin + valid_vectors
                > publish_row[1] + publish_row[2]
        ):
            raise AssertionError("history combine vectors are outside its publish tile")
        token = vector_begin // hq_local
        expected_batch = next(
            idx
            for idx, (begin, end) in enumerate(zip(cu_q, cu_q[1:]))
            if begin <= token < end
        )
        expected_dst = publish_row[0]
        if batch_idx != expected_batch:
            raise AssertionError("history combine batch mapping is invalid")
        if actual_splits != metadata.history_sequence_splits[batch_idx]:
            raise AssertionError("history combine split count is invalid")
        if actual_splits > 1:
            if valid_vectors != 1:
                raise AssertionError("split history combine task must contain one vector")
        else:
            copy_vectors = metadata.dispatch.history_copy_vectors_per_task
            region_end = min(
                publish_row[1] + publish_row[2],
                cu_q[batch_idx + 1] * hq_local,
            )
            expected_valid = min(copy_vectors, region_end - vector_begin)
            if valid_vectors != expected_valid:
                raise AssertionError("history copy task vector count is invalid")
        if row[7] != 0:
            raise AssertionError("history combine reserved field is invalid")
        expected_id_set: set[int] = set()
        for vector in range(vector_begin, vector_begin + valid_vectors):
            vector_token, local_head = divmod(vector, hq_local)
            if not cu_q[batch_idx] <= vector_token < cu_q[batch_idx + 1]:
                raise AssertionError("history combine task crosses a batch boundary")
            history_head = expected_dst * hq_local + local_head
            q_relative = vector_token - cu_q[batch_idx]
            packed = q_relative * (dcp_size * hq_local) + history_head
            expected_id_set.update(
                history_completion_ids[(batch_idx, packed // 128)]
            )
        expected_ids = tuple(sorted(expected_id_set))
        dependencies = metadata.publish_dependencies[
            dep_begin : dep_begin + dep_count
        ]
        if dependencies != expected_ids:
            raise AssertionError("history combine dependencies are not exact")
        combine_by_publish[publish_id].append(row)
    combine_coordinates = [
        (metadata.publish[row[0]][0], vector)
        for row in metadata.history_combine
        for vector in range(row[1], row[1] + row[2])
    ]
    expected_coordinates = [
        (dst_rank, vector)
        for parent_token_block in metadata.heuristic_q_block_order
        for dst_rank in range(dcp_size)
        for vector in range(
            parent_token_block * 16 * hq_local,
            min(
                (parent_token_block + 1) * 16 * hq_local,
                metadata.total_vectors,
            ),
        )
    ]
    if combine_coordinates != expected_coordinates:
        raise AssertionError("history combine descriptors are missing or duplicated")
    for publish_id, combine_rows in enumerate(combine_by_publish):
        publish_row = metadata.publish[publish_id]
        if len(combine_rows) != publish_row[6]:
            raise AssertionError("publish combine task count mismatch")
        if combine_rows:
            dependency_begin = combine_rows[0][3]
            dependency_end = combine_rows[-1][3] + combine_rows[-1][4]
            if (
                publish_row[3] != dependency_begin
                or publish_row[4] != dependency_end - dependency_begin
            ):
                raise AssertionError("publish dependency span is invalid")

    receive_sources = dcp_size - 1
    receive_ids = [
        parent_token_block * receive_sources + source
        for parent_token_block in range(metadata.token_block_count)
        for source in range(receive_sources)
    ]
    if receive_ids != list(range(metadata.token_block_count * receive_sources)):
        raise AssertionError("receive task ids must be dense parent-major/source-minor")
    if any(
        len(
            receive_ids[
                parent_token_block
                * receive_sources : (parent_token_block + 1)
                * receive_sources
            ]
        )
        != receive_sources
        for parent_token_block in range(metadata.token_block_count)
    ):
        raise AssertionError("each parent tile has an invalid receive fan-in")

    seen_final: list[int] = []
    for row in metadata.final:
        seen_final.extend(range(row[0], row[0] + row[1]))
    if sorted(seen_final) != list(range(metadata.total_vectors)):
        raise AssertionError("final descriptors have duplicate or missing vectors")


def pack_dcp_mega_metadata(
    metadata: DCPMegaMetadata,
    *,
    pre_phase: int,
    post_phase: int,
    capacity: int | None = None,
) -> array:
    """Serialize a validated metadata v7 image into native int32 values."""
    if pre_phase <= 0 or post_phase <= pre_phase:
        raise ValueError("metadata phases must be positive and strictly increasing")
    payload = array("i", [0] * METADATA_HEADER_INTS)

    def append_rows(rows: Sequence[Sequence[int]]) -> int:
        offset = len(payload)
        for row in rows:
            payload.extend(row)
        return offset

    attention_offset = append_rows(metadata.attention)
    q_tasks_offset = append_rows(metadata.q_tasks)
    q_dependencies_offset = len(payload)
    payload.extend(metadata.q_dependencies)
    publish_offset = append_rows(metadata.publish)
    history_combine_offset = append_rows(metadata.history_combine)
    publish_dependencies_offset = len(payload)
    payload.extend(metadata.publish_dependencies)
    final_offset = append_rows(metadata.final)
    final_dependencies_offset = len(payload)
    payload.extend(metadata.final_dependencies)
    chunk_splits_offset = len(payload)
    payload.extend(metadata.chunk_sequence_splits)
    history_splits_offset = len(payload)
    payload.extend(metadata.history_sequence_splits)
    used = len(payload)
    if capacity is not None and used > capacity:
        raise ValueError(
            f"DCP mega metadata exceeds capacity: used={used}, capacity={capacity}"
        )

    chunk_count = sum(row[0] == CHUNK for row in metadata.attention)
    header = (
        METADATA_VERSION,
        len(metadata.attention),
        len(metadata.q_tasks),
        len(metadata.q_dependencies),
        len(metadata.publish),
        len(metadata.publish_dependencies),
        len(metadata.final),
        len(metadata.final_dependencies),
        chunk_count,
        len(metadata.attention) - chunk_count,
        metadata.total_q,
        metadata.total_vectors,
        metadata.dispatch.effective_num_splits,
        metadata.dispatch.chunk_num_splits,
        metadata.dispatch.history_num_splits,
        int(metadata.dispatch.pack_gqa),
        int(metadata.dispatch.split),
        metadata.dispatch.block_n,
        len(metadata.chunk_sequence_splits),
        pre_phase,
        post_phase,
        attention_offset,
        q_tasks_offset,
        q_dependencies_offset,
        publish_offset,
        publish_dependencies_offset,
        final_offset,
        final_dependencies_offset,
        chunk_splits_offset,
        history_splits_offset,
        used,
        0,
        1,
        metadata.token_block_count,
        metadata.q_ready_count,
        metadata.receive_count,
        metadata.tile_ready_count,
        metadata.dcp_size,
        len(metadata.history_combine),
        history_combine_offset,
    )
    if len(header) != METADATA_HEADER_INTS:
        raise AssertionError("metadata v7 header width mismatch")
    payload[:METADATA_HEADER_INTS] = array("i", header)
    return payload


__all__ = [
    "ATTENTION_DESC_FIELDS",
    "CHUNK",
    "DCPMegaDispatch",
    "DCPMegaMetadata",
    "FINAL_DESC_FIELDS",
    "FINAL_TOKEN_GRANULARITIES",
    "FINAL_TOKENS_PER_TASK",
    "HISTORY",
    "HISTORY_COMBINE_DESC_FIELDS",
    "HISTORY_ORDER_POLICY_FIFO",
    "HISTORY_ORDER_POLICY_RELEASE_LPT",
    "HISTORY_MAX_COPY_VECTORS",
    "SCHEDULER_POLICY_CRITICAL_WAVE_FIFO",
    "SCHEDULER_POLICY_FIFO",
    "SCHEDULER_POLICY_HEURISTIC",
    "SCHEDULER_POLICY_NATIVE_RELEASE_LPT",
    "SPLIT_POLICY_CRITICAL_WAVE",
    "SPLIT_POLICY_FA3_NATIVE",
    "HISTORY_TASK_WAVE_TARGET",
    "MEGA_COMPUTE_WARPS",
    "METADATA_HEADER_INTS",
    "METADATA_VERSION",
    "PUBLISH_DESC_FIELDS",
    "Q_TASK_FIELDS",
    "_choose_final_tokens_per_task",
    "build_dcp_mega_metadata",
    "choose_dispatch",
    "choose_split_upper_bound",
    "num_splits_heuristic",
    "pack_dcp_mega_metadata",
    "validate_dcp_mega_metadata",
]
