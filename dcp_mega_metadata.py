"""Host metadata builder for the batched varlen DCP mega forward path.

The split selection and dynamic per-sequence split calculation are copied and
trimmed from ``include/hopper_compat/heuristics.h`` and
``csrc/min_fa3_varlen_prepare_scheduler.cu``.  Keeping the builder pure Python
makes its queue coverage testable without a GPU and lets the runner fill a
preallocated pinned host buffer before each launch.
"""

from __future__ import annotations

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
METADATA_VERSION = 4

@dataclass(frozen=True)
class DCPMegaDispatch:
    effective_num_splits: int
    chunk_num_splits: int
    history_num_splits: int
    pack_gqa: bool
    split: bool
    block_n: int


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
    token_block_count: int
    q_ready_count: int
    receive_count: int
    tile_ready_count: int
    dcp_size: int

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
) -> int:
    if requested_num_splits < 0 or requested_num_splits > 128:
        raise ValueError("num_splits must be 0 (auto), 1, or in [2, 128]")
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
    )
    history_splits = choose_split_upper_bound(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_history,
        q_heads=dcp_size * hq_local,
        num_sms=num_sms,
        block_n=block_n,
        is_causal=False,
        requested_num_splits=requested_num_splits,
    )
    effective = max(chunk_splits, history_splits)
    return DCPMegaDispatch(
        effective_num_splits=effective,
        chunk_num_splits=chunk_splits,
        history_num_splits=history_splits,
        pack_gqa=True,
        split=effective > 1,
        block_n=block_n,
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


def build_dcp_mega_metadata(
    cu_seqlens_q: Sequence[int],
    cu_seqlens_history: Sequence[int],
    *,
    hq_local: int,
    dcp_size: int,
    num_sms: int,
    requested_num_splits: int = 0,
    block_n_override: int | None = None,
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
    q_lengths = tuple(end - begin for begin, end in zip(cu_q, cu_q[1:]))
    history_lengths = tuple(
        end - begin for begin, end in zip(cu_history, cu_history[1:])
    )
    dispatch = choose_dispatch(
        max_seqlen_q=max(q_lengths),
        max_seqlen_history=max(history_lengths),
        hq_local=hq_local,
        dcp_size=dcp_size,
        num_sms=num_sms,
        requested_num_splits=requested_num_splits,
        block_n_override=block_n_override,
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
    if all(split == 1 for split in chunk_sequence_splits) and all(
        split == 1 for split in history_sequence_splits
    ):
        dispatch = DCPMegaDispatch(
            effective_num_splits=1,
            chunk_num_splits=1,
            history_num_splits=1,
            pack_gqa=dispatch.pack_gqa,
            split=False,
            block_n=dispatch.block_n,
        )

    total_q = cu_q[-1]
    num_token_subtiles = _ceil_div(total_q, 16)
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
            vectors_per_task = 1 if dispatch.split else hq_local
            for task_vector_begin in range(
                vector_begin,
                vector_begin + valid_vectors,
                vectors_per_task,
            ):
                task_valid_vectors = min(
                    vectors_per_task,
                    vector_begin + valid_vectors - task_vector_begin,
                )
                token = task_vector_begin // hq_local
                batch_idx = batch_for_token[token]
                dependency_set: set[int] = set()
                for vector in range(
                    task_vector_begin,
                    task_vector_begin + task_valid_vectors,
                ):
                    vector_token, local_head = divmod(vector, hq_local)
                    if batch_for_token[vector_token] != batch_idx:
                        raise AssertionError(
                            "history combine task crosses a batch boundary"
                        )
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
                        history_sequence_splits[batch_idx],
                        0,
                    )
                )
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
        for final_id in range(num_token_subtiles)
        for dst_rank in range(dcp_size)
        for task in history_combine_by_publish[
            dst_rank * num_token_subtiles + final_id
        ]
    )

    final: list[tuple[int, ...]] = []
    final_dependencies: list[int] = []
    total_vectors = total_q * hq_local
    final_tile_size = 16 * hq_local
    for vector_begin in range(0, total_vectors, final_tile_size):
        valid_vectors = min(final_tile_size, total_vectors - vector_begin)
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
                0,
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
        token_block_count=num_token_subtiles,
        q_ready_count=num_token_subtiles,
        receive_count=len(final) * (dcp_size - 1),
        tile_ready_count=len(final) * (dcp_size - 1),
        dcp_size=dcp_size,
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
    if completion_ids != list(range(len(metadata.attention))):
        raise AssertionError("attention completion ids must be dense and unique")
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
    expected_history_combine = dcp_size * (
        metadata.total_vectors if metadata.dispatch.split else metadata.total_q
    )
    if len(metadata.history_combine) != expected_history_combine:
        raise AssertionError(
            "history combine queue does not cover every destination vector"
        )
    expected_final = metadata.token_block_count
    if len(metadata.final) != expected_final:
        raise AssertionError("final queue does not cover every local output vector")

    expected_q_ready = metadata.token_block_count
    expected_receive = len(metadata.final) * (dcp_size - 1)
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

    q_coordinates = [(row[2] // 16, row[0]) for row in metadata.q_tasks]
    expected_q_coordinates = [
        (token_block, source)
        for token_block in range(metadata.token_block_count)
        for source in range(dcp_size)
    ]
    if q_coordinates != expected_q_coordinates:
        raise AssertionError("Q tasks must be token-block-major/rank-minor")
    if any(row[1] != 0 for row in metadata.q_tasks):
        raise AssertionError("Q tasks must not encode a local head")

    final_count = len(metadata.final)
    for publish_id, row in enumerate(metadata.publish):
        expected_dst, final_id = divmod(publish_id, final_count)
        final_row = metadata.final[final_id]
        if row[0] != expected_dst or row[1:3] != final_row[0:2]:
            raise AssertionError(
                "publish tasks must be destination-major copies of final tiles"
            )
        if row[5] != int(metadata.dispatch.pack_gqa):
            raise AssertionError("publish PackGQA marker mismatch")
        expected_task_count = row[2] if metadata.dispatch.split else _ceil_div(
            row[2], hq_local
        )
        if row[6] != expected_task_count or row[7] != 0:
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
        expected_vectors_per_task = 1 if metadata.dispatch.split else hq_local
        if valid_vectors != expected_vectors_per_task:
            raise AssertionError("history combine task vector count is invalid")
        if not metadata.dispatch.split and vector_begin % hq_local != 0:
            raise AssertionError("non-split history combine task is not token-aligned")
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
        for final_row in metadata.final
        for dst_rank in range(dcp_size)
        for vector in range(final_row[0], final_row[0] + final_row[1])
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
        final_id * receive_sources + source
        for final_id in range(final_count)
        for source in range(receive_sources)
    ]
    if receive_ids != list(range(final_count * receive_sources)):
        raise AssertionError("receive task ids must be dense final-major/source-minor")
    if any(
        len(
            receive_ids[
                final_id * receive_sources : (final_id + 1) * receive_sources
            ]
        )
        != receive_sources
        for final_id in range(final_count)
    ):
        raise AssertionError("each final tile has an invalid receive fan-in")

    seen_final: list[int] = []
    for row in metadata.final:
        seen_final.extend(range(row[0], row[0] + row[1]))
    if seen_final != list(range(metadata.total_vectors)):
        raise AssertionError("final descriptors have duplicate or missing vectors")


def pack_dcp_mega_metadata(
    metadata: DCPMegaMetadata,
    *,
    pre_phase: int,
    post_phase: int,
    capacity: int | None = None,
) -> array:
    """Serialize a validated metadata v4 image into native int32 values."""
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
        raise AssertionError("metadata v4 header width mismatch")
    payload[:METADATA_HEADER_INTS] = array("i", header)
    return payload


__all__ = [
    "ATTENTION_DESC_FIELDS",
    "CHUNK",
    "DCPMegaDispatch",
    "DCPMegaMetadata",
    "FINAL_DESC_FIELDS",
    "HISTORY",
    "HISTORY_COMBINE_DESC_FIELDS",
    "METADATA_HEADER_INTS",
    "METADATA_VERSION",
    "PUBLISH_DESC_FIELDS",
    "Q_TASK_FIELDS",
    "build_dcp_mega_metadata",
    "choose_dispatch",
    "choose_split_upper_bound",
    "num_splits_heuristic",
    "pack_dcp_mega_metadata",
    "validate_dcp_mega_metadata",
]
