"""Fixed-step, decode-first replay and eligible wall-clock sampling."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass

from .models import (
    SCHEMA_VERSION,
    SOURCE_NAME,
    ReplayConfig,
    TraceLoadResult,
    TraceRequest,
)
from .mooncake import PrefixBlockCache


class ReplayError(RuntimeError):
    """Raised when replay cannot produce the requested workload corpus."""


@dataclass
class _RequestState:
    request: TraceRequest
    cached_prefix_length: int
    computed_prompt_tokens: int
    history_len: int
    generated_tokens: int = 0

    @property
    def is_prefill(self) -> bool:
        return self.computed_prompt_tokens < self.request.input_length

    @property
    def complete(self) -> bool:
        return (
            not self.is_prefill
            and self.generated_tokens >= self.request.output_length
        )


@dataclass(frozen=True)
class _ScheduledQuery:
    state: _RequestState
    phase: str
    q_len: int
    physical_q_len: int
    history_len: int
    generated_tokens_before: int
    accepted_drafts: int | None


@dataclass(frozen=True)
class _Snapshot:
    sampled_time_us: int
    scheduler_step: int
    queries: tuple[_ScheduledQuery, ...]


@dataclass
class ReplayStats:
    trace_rows: int = 0
    loaded_requests: int = 0
    dropped_model_len: int = 0
    admitted_requests: int = 0
    completed_requests: int = 0
    scheduler_steps_processed: int = 0
    idle_steps: int = 0
    eligible_steps_in_window: int = 0
    scheduled_queries: int = 0
    mega_eligible_queries: int = 0
    filtered_zero_local_history: int = 0
    decode_queries: int = 0
    chunk_prefill_queries: int = 0
    cached_prefix_tokens: int = 0
    max_active_requests: int = 0


@dataclass(frozen=True)
class ReplayResult:
    cases: tuple[dict[str, object], ...]
    stats: ReplayStats


def _sample_accepted(probabilities: tuple[float, ...], rng: random.Random) -> int:
    draw = rng.random()
    cumulative = 0.0
    for accepted, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return accepted
    return len(probabilities) - 1


def _align_q_len(q_len: int, alignment: int) -> int:
    return (q_len + alignment - 1) // alignment * alignment


def _admit(
    waiting: deque[TraceRequest],
    active: list[_RequestState],
    cache: PrefixBlockCache,
    config: ReplayConfig,
    stats: ReplayStats,
) -> None:
    while waiting and len(active) < config.max_num_seqs:
        request = waiting.popleft()
        cached_prefix = cache.lookup_prefix(request)
        active.append(
            _RequestState(
                request=request,
                cached_prefix_length=cached_prefix,
                computed_prompt_tokens=cached_prefix,
                history_len=cached_prefix,
            )
        )
        stats.admitted_requests += 1
        stats.cached_prefix_tokens += cached_prefix
    stats.max_active_requests = max(stats.max_active_requests, len(active))


def _schedule(
    active: list[_RequestState],
    config: ReplayConfig,
    acceptance_rng: random.Random,
) -> list[_ScheduledQuery]:
    remaining_budget = config.max_num_batched_tokens
    scheduled: list[_ScheduledQuery] = []

    for state in active:
        if state.is_prefill or state.complete:
            continue
        q_len = config.mtp_query_len
        physical_q_len = _align_q_len(q_len, config.q_len_alignment)
        if physical_q_len > remaining_budget:
            break
        sampled_accepted = _sample_accepted(
            config.accepted_draft_pmf, acceptance_rng
        )
        remaining_outputs = state.request.output_length - state.generated_tokens
        accepted_drafts = min(
            sampled_accepted,
            max(remaining_outputs - 1, 0),
            q_len - 1,
        )
        scheduled.append(
            _ScheduledQuery(
                state=state,
                phase="decode",
                q_len=q_len,
                physical_q_len=physical_q_len,
                history_len=state.history_len,
                generated_tokens_before=state.generated_tokens,
                accepted_drafts=accepted_drafts,
            )
        )
        remaining_budget -= physical_q_len

    for state in active:
        if not state.is_prefill or remaining_budget == 0:
            continue
        aligned_budget = (
            remaining_budget // config.q_len_alignment
            * config.q_len_alignment
        )
        if aligned_budget == 0:
            break
        q_len = min(
            state.request.input_length - state.computed_prompt_tokens,
            config.prefill_chunk_size,
            aligned_budget,
        )
        physical_q_len = _align_q_len(q_len, config.q_len_alignment)
        scheduled.append(
            _ScheduledQuery(
                state=state,
                phase="chunk_prefill",
                q_len=q_len,
                physical_q_len=physical_q_len,
                history_len=state.history_len,
                generated_tokens_before=state.generated_tokens,
                accepted_drafts=None,
            )
        )
        remaining_budget -= physical_q_len
    return scheduled


def _apply_queries(
    scheduled: list[_ScheduledQuery],
    cache: PrefixBlockCache,
    stats: ReplayStats,
) -> None:
    for query in scheduled:
        state = query.state
        if query.phase == "chunk_prefill":
            old_computed = state.computed_prompt_tokens
            state.computed_prompt_tokens += query.q_len
            state.history_len += query.q_len
            cache.insert_completed(
                state.request, old_computed, state.computed_prompt_tokens
            )
            if (
                state.computed_prompt_tokens == state.request.input_length
                and state.request.output_length > 0
            ):
                state.generated_tokens = 1
            stats.chunk_prefill_queries += 1
        else:
            assert query.accepted_drafts is not None
            committed = 1 + query.accepted_drafts
            state.history_len += committed
            state.generated_tokens += committed
            stats.decode_queries += 1


def _snapshot_case(
    snapshot: _Snapshot,
    case_id: int,
    config: ReplayConfig,
    trace: TraceLoadResult,
) -> dict[str, object]:
    queries = snapshot.queries
    logical_q_lens = [query.q_len for query in queries]
    q_lens = [query.physical_q_len for query in queries]
    history_lens = [query.history_len for query in queries]
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": f"case_{case_id:06d}",
        "source": SOURCE_NAME,
        "trace_sha256": trace.trace_sha256,
        "config_sha256": config.config_sha256,
        "sampled_time_us": snapshot.sampled_time_us,
        "scheduler_step": snapshot.scheduler_step,
        "batch_size": len(queries),
        "q_lens": q_lens,
        "logical_q_lens": logical_q_lens,
        "history_lens": history_lens,
        "total_kv_lens": [
            history + query for history, query in zip(history_lens, q_lens)
        ],
        "request_ids": [query.state.request.request_id for query in queries],
        "phases": [query.phase for query in queries],
        "prompt_lengths": [query.state.request.input_length for query in queries],
        "output_lengths": [
            query.state.request.output_length for query in queries
        ],
        "cached_prefix_lengths": [
            query.state.cached_prefix_length for query in queries
        ],
        "generated_tokens_before": [
            query.generated_tokens_before for query in queries
        ],
        "num_speculative_tokens": config.num_speculative_tokens,
        "q_len_alignment": config.q_len_alignment,
        "accepted_drafts": [query.accepted_drafts for query in queries],
        "fixed_step_us": config.fixed_step_us,
        "timestamp_policy": config.timestamp_policy,
        "seed": config.seed,
    }


def replay_trace(
    trace: TraceLoadResult,
    config: ReplayConfig,
    acceptance_rng: random.Random,
    sampling_rng: random.Random,
) -> ReplayResult:
    """Replay all state from time zero and sample eligible steps in the window."""
    if not trace.requests:
        raise ReplayError("no requests remain after max_model_len filtering")
    stats = ReplayStats(
        trace_rows=trace.total_rows,
        loaded_requests=len(trace.requests),
        dropped_model_len=trace.dropped_model_len,
    )
    waiting: deque[TraceRequest] = deque()
    active: list[_RequestState] = []
    cache = PrefixBlockCache(config.prefix_cache_capacity_blocks)
    reservoir: list[_Snapshot] = []
    arrival_index = 0
    scheduler_step = 0
    sampling_start_us = config.sampling_start_ms * 1000
    sampling_end_us = config.sampling_end_ms * 1000

    while scheduler_step * config.fixed_step_us < sampling_end_us:
        current_time_us = scheduler_step * config.fixed_step_us
        if not active and not waiting:
            if arrival_index >= len(trace.requests):
                break
            next_arrival = trace.requests[arrival_index].arrival_us
            if next_arrival > current_time_us:
                next_step = (next_arrival + config.fixed_step_us - 1) // config.fixed_step_us
                if next_step > scheduler_step:
                    stats.idle_steps += next_step - scheduler_step
                    scheduler_step = next_step
                    continue

        while (
            arrival_index < len(trace.requests)
            and trace.requests[arrival_index].arrival_us <= current_time_us
        ):
            waiting.append(trace.requests[arrival_index])
            arrival_index += 1
        _admit(waiting, active, cache, config, stats)
        scheduled = _schedule(active, config, acceptance_rng)
        stats.scheduler_steps_processed += 1
        stats.scheduled_queries += len(scheduled)
        if not scheduled:
            stats.idle_steps += 1

        eligible = tuple(
            query for query in scheduled if query.history_len >= config.dcp_size
        )
        stats.mega_eligible_queries += len(eligible)
        stats.filtered_zero_local_history += len(scheduled) - len(eligible)
        if eligible and sampling_start_us <= current_time_us < sampling_end_us:
            stats.eligible_steps_in_window += 1
            snapshot = _Snapshot(current_time_us, scheduler_step, eligible)
            eligible_index = stats.eligible_steps_in_window
            if len(reservoir) < config.num_cases:
                reservoir.append(snapshot)
            else:
                replacement = sampling_rng.randrange(eligible_index)
                if replacement < config.num_cases:
                    reservoir[replacement] = snapshot

        _apply_queries(scheduled, cache, stats)
        retained: list[_RequestState] = []
        for state in active:
            if state.complete:
                stats.completed_requests += 1
            else:
                retained.append(state)
        active = retained
        scheduler_step += 1

    if len(reservoir) < config.num_cases:
        raise ReplayError(
            f"requested {config.num_cases} cases but sampling window contains "
            f"only {stats.eligible_steps_in_window} eligible steps"
        )
    reservoir.sort(key=lambda snapshot: (snapshot.sampled_time_us, snapshot.scheduler_step))
    cases = tuple(
        _snapshot_case(snapshot, index, config, trace)
        for index, snapshot in enumerate(reservoir)
    )
    return ReplayResult(cases=cases, stats=stats)


def stats_dict(stats: ReplayStats) -> dict[str, int]:
    return {key: int(value) for key, value in asdict(stats).items()}
