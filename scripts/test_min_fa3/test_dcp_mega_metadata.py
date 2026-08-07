#!/usr/bin/env python3
"""CPU-only queue coverage tests for the batched varlen DCP mega path."""

from __future__ import annotations

import unittest

from dcp_mega_metadata import (
    CHUNK,
    HISTORY,
    HISTORY_COMBINE_DESC_FIELDS,
    METADATA_HEADER_INTS,
    METADATA_VERSION,
    build_dcp_mega_metadata,
    choose_dispatch,
    choose_split_upper_bound,
    pack_dcp_mega_metadata,
)


_RECEIVE_CHUNKS = 6
_RECEIVE_SCAN_WINDOW = 8


def _receive_slot_tasks(total_tasks, num_comm_sm):
    """Return the current static strided receive ownership by pipeline slot."""
    stride = _RECEIVE_CHUNKS * num_comm_sm
    return [
        list(range(slot, total_tasks, stride))
        for slot in range(stride)
    ]


def _simulate_bounded_receive_slot(task_ids, ready_order):
    """Model one static slot when remote tasks become ready one at a time."""
    if sorted(task_ids) != sorted(ready_order):
        raise ValueError("ready_order must contain every slot task exactly once")
    if not task_ids:
        return [], []

    cursor = 0
    completed = set()
    ready = set()
    completion_order = []
    probes_per_window = []
    for newly_ready in ready_order:
        ready.add(newly_ready)
        miss_windows = 0
        while True:
            selected = None
            ordinal = cursor
            probes = 0
            for _ in range(min(_RECEIVE_SCAN_WINDOW, len(task_ids))):
                task_id = task_ids[ordinal]
                ordinal = (ordinal + 1) % len(task_ids)
                probes += 1
                if task_id not in completed and task_id in ready:
                    selected = task_id
                    cursor = ordinal
                    break
            probes_per_window.append(probes)
            if selected is not None:
                completed.add(selected)
                ready.remove(selected)
                completion_order.append(selected)
                break
            cursor = ordinal
            miss_windows += 1
            if miss_windows > (
                len(task_ids) + _RECEIVE_SCAN_WINDOW - 1
            ) // _RECEIVE_SCAN_WINDOW:
                raise AssertionError("bounded scan failed to cover its task ring")
    return completion_order, probes_per_window


def _simulate_unified_attention_queue(kinds, num_compute_ctas):
    """Model logical initial IDs followed by the shared monotonic counter."""
    if num_compute_ctas <= 0:
        raise ValueError("num_compute_ctas must be positive")
    assignments = [[] for _ in range(num_compute_ctas)]
    active = []
    for compute_cta_id in range(num_compute_ctas):
        if compute_cta_id < len(kinds):
            assignments[compute_cta_id].append(compute_cta_id)
            active.append(compute_cta_id)

    next_task_id = num_compute_ctas
    while active:
        next_active = []
        for compute_cta_id in active:
            task_id = next_task_id
            next_task_id += 1
            if task_id < len(kinds):
                assignments[compute_cta_id].append(task_id)
                next_active.append(compute_cta_id)
        active = next_active

    q_waits = [
        task_id
        for tasks in assignments
        for task_id in tasks
        if kinds[task_id] == HISTORY
    ]
    return assignments, q_waits


class DCPMegaMetadataTest(unittest.TestCase):
    def _assert_receive_slots(self, total_tasks, num_comm_sm):
        slots = _receive_slot_tasks(total_tasks, num_comm_sm)
        claimed = [task_id for tasks in slots for task_id in tasks]
        self.assertEqual(sorted(claimed), list(range(total_tasks)))
        self.assertEqual(len(claimed), len(set(claimed)))
        for task_ids in slots:
            orders = (
                task_ids,
                list(reversed(task_ids)),
                task_ids[::2] + task_ids[1::2],
            )
            for ready_order in orders:
                completed, probe_counts = _simulate_bounded_receive_slot(
                    task_ids, ready_order
                )
                self.assertEqual(completed, ready_order)
                self.assertEqual(len(completed), len(set(completed)))
                self.assertTrue(
                    all(
                        probes <= min(_RECEIVE_SCAN_WINDOW, len(task_ids))
                        for probes in probe_counts
                    )
                )
        return slots

    def _assert_unified_queue(self, kinds, num_compute_ctas):
        assignments, q_waits = _simulate_unified_attention_queue(
            kinds, num_compute_ctas
        )
        claimed = [task_id for tasks in assignments for task_id in tasks]
        self.assertEqual(sorted(claimed), list(range(len(kinds))))
        self.assertEqual(len(claimed), len(set(claimed)))
        self.assertEqual(
            sorted(q_waits),
            [task_id for task_id, kind in enumerate(kinds) if kind == HISTORY],
        )
        for tasks in assignments:
            task_kinds = [kinds[task_id] for task_id in tasks]
            transitions = sum(
                previous != current
                for previous, current in zip(task_kinds, task_kinds[1:])
            )
            self.assertLessEqual(transitions, 1)
            self.assertNotIn(
                (HISTORY, CHUNK), list(zip(task_kinds, task_kinds[1:]))
            )
        return assignments

    def _assert_attention_tiles_once(self, metadata, cu_q, hq_local, dcp_size):
        for kind, heads, sequence_splits in (
            (CHUNK, hq_local, metadata.chunk_sequence_splits),
            (HISTORY, dcp_size * hq_local, metadata.history_sequence_splits),
        ):
            actual = [row for row in metadata.attention if row[0] == kind]
            expected = set()
            for batch_idx, (begin, end) in enumerate(zip(cu_q, cu_q[1:])):
                q_len = end - begin
                splits = sequence_splits[batch_idx]
                for m_block in range((q_len * heads + 127) // 128):
                    for split in range(splits):
                        expected.add((batch_idx, m_block, 0, split))
            coordinates = [(row[1], row[2], row[3], row[4]) for row in actual]
            self.assertEqual(len(coordinates), len(set(coordinates)))
            self.assertEqual(set(coordinates), expected)

    def _assert_publish_receive_mapping(
        self, metadata, cu_q, hq_local, dcp_size
    ):
        final_count = len(metadata.final)
        self.assertEqual(len(metadata.publish), dcp_size * final_count)
        combine_counts = [0] * len(metadata.publish)
        for row in metadata.history_combine:
            combine_counts[row[0]] += 1
        for publish_id, publish in enumerate(metadata.publish):
            dst_rank, final_id = divmod(publish_id, final_count)
            self.assertEqual(publish[0], dst_rank)
            self.assertEqual(publish[1:3], metadata.final[final_id][0:2])
            self.assertEqual(publish[6], combine_counts[publish_id])
        self.assertTrue(
            all(
                len(row) == HISTORY_COMBINE_DESC_FIELDS
                for row in metadata.history_combine
            )
        )
        coordinates = []
        for row in metadata.history_combine:
            publish = metadata.publish[row[0]]
            coordinates.extend(
                (publish[0], vector)
                for vector in range(row[1], row[1] + row[2])
            )
            self.assertLessEqual(publish[1], row[1])
            self.assertLessEqual(row[1] + row[2], publish[1] + publish[2])
            if row[6] > 1:
                self.assertEqual(row[2], 1)
            else:
                self.assertLessEqual(
                    row[2], metadata.dispatch.history_copy_vectors_per_task
                )
            self.assertGreater(row[4], 0)
            self.assertEqual(
                row[6], metadata.history_sequence_splits[row[5]]
            )
            self.assertEqual(row[7], 0)
            expected_dependencies = set()
            for vector in range(row[1], row[1] + row[2]):
                token, local_head = divmod(vector, hq_local)
                token_in_batch = token - cu_q[row[5]]
                history_head = publish[0] * hq_local + local_head
                m_block = (
                    token_in_batch * (dcp_size * hq_local) + history_head
                ) // 128
                expected_dependencies.update(
                    attention[7]
                    for attention in metadata.attention
                    if attention[0] == HISTORY
                    and attention[1] == row[5]
                    and attention[2] == m_block
                )
            self.assertEqual(
                metadata.publish_dependencies[row[3] : row[3] + row[4]],
                tuple(sorted(expected_dependencies)),
            )
        self.assertEqual(
            coordinates,
            [
                (dst_rank, vector)
                for final in metadata.final
                for dst_rank in range(dcp_size)
                for vector in range(final[0], final[0] + final[1])
            ],
        )
        sources = dcp_size - 1
        receive_ids = [
            final_id * sources + source
            for final_id in range(final_count)
            for source in range(sources)
        ]
        self.assertEqual(receive_ids, list(range(final_count * sources)))
        for final_id in range(final_count):
            begin = final_id * sources
            self.assertEqual(
                receive_ids[begin : begin + sources],
                list(range(begin, begin + sources)),
            )

    def test_pack_ragged_split_tail_and_dependencies(self):
        cu_q = (0, 1, 9, 42)
        cu_history = (0, 129, 1153, 4284)
        metadata = build_dcp_mega_metadata(
            cu_q,
            cu_history,
            hq_local=4,
            dcp_size=8,
            num_sms=132,
            num_comm_sm=8,
            requested_num_splits=8,
        )
        self.assertTrue(metadata.dispatch.pack_gqa)
        self.assertTrue(metadata.dispatch.split)
        self.assertTrue(any(split > 1 for split in metadata.history_sequence_splits))
        self._assert_attention_tiles_once(metadata, cu_q, 4, 8)
        self._assert_publish_receive_mapping(metadata, cu_q, 4, 8)
        history = [row for row in metadata.attention if row[0] == HISTORY]
        self.assertTrue(all(row[6] > 0 for row in history))
        self.assertTrue(all(row[6] == 0 for row in metadata.attention if row[0] == CHUNK))
        self.assertEqual(
            metadata.final[-1][1], metadata.total_vectors % (16 * 4) or 16 * 4
        )
        self.assertEqual(metadata.q_ready_count, metadata.token_block_count)
        self.assertEqual(metadata.receive_count, 7 * metadata.token_block_count)
        self.assertEqual(metadata.tile_ready_count, metadata.receive_count)

    def test_publish_dependencies_follow_history_and_final_follow_chunk(self):
        cu_q = (0, 17, 148)
        metadata = build_dcp_mega_metadata(
            cu_q,
            (0, 257, 1281),
            hq_local=4,
            dcp_size=4,
            num_sms=132,
            num_comm_sm=8,
            requested_num_splits=1,
        )
        self.assertTrue(metadata.dispatch.pack_gqa)
        self.assertFalse(metadata.dispatch.split)
        self._assert_attention_tiles_once(metadata, cu_q, 4, 4)
        self._assert_publish_receive_mapping(metadata, cu_q, 4, 4)
        for row in metadata.publish:
            self.assertIn(row[0], range(4))
            deps = metadata.publish_dependencies[row[3] : row[3] + row[4]]
            self.assertTrue(deps)
            self.assertTrue(all(metadata.attention[dep][0] == HISTORY for dep in deps))
        for row in metadata.final:
            deps = metadata.final_dependencies[row[2] : row[2] + row[3]]
            self.assertTrue(deps)
            self.assertTrue(all(metadata.attention[dep][0] == CHUNK for dep in deps))

    def test_auto_dispatch_and_block_overrides(self):
        for dcp_size in (2, 4, 8):
            for block_n in (128, 176):
                metadata = build_dcp_mega_metadata(
                    (0, 8, 40),
                    (0, 1024, 4155),
                    hq_local=4,
                    dcp_size=dcp_size,
                    num_sms=132,
                    num_comm_sm=8,
                    block_n_override=block_n,
                )
                self.assertEqual(metadata.dispatch.block_n, block_n)
                self.assertGreaterEqual(metadata.dispatch.effective_num_splits, 1)

    def test_auto_dispatch_collapses_when_all_dynamic_splits_are_one(self):
        cu_q = tuple(batch_idx * 128 for batch_idx in range(17))
        cu_history = tuple(batch_idx * 8192 for batch_idx in range(17))
        upper_bound = choose_dispatch(
            max_seqlen_q=128,
            max_seqlen_history=8192,
            hq_local=4,
            dcp_size=8,
            num_sms=132,
            requested_num_splits=0,
            block_n_override=128,
        )
        self.assertEqual(upper_bound.history_num_splits, 4)
        self.assertTrue(upper_bound.split)

        metadata = build_dcp_mega_metadata(
            cu_q,
            cu_history,
            hq_local=4,
            dcp_size=8,
            num_sms=132,
            num_comm_sm=8,
            requested_num_splits=0,
            block_n_override=128,
        )
        self.assertEqual(metadata.chunk_sequence_splits, (1,) * 16)
        self.assertEqual(metadata.history_sequence_splits, (1,) * 16)
        self.assertEqual(metadata.dispatch.effective_num_splits, 1)
        self.assertEqual(metadata.dispatch.chunk_num_splits, 1)
        self.assertEqual(metadata.dispatch.history_num_splits, 1)
        self.assertFalse(metadata.dispatch.split)
        self.assertTrue(metadata.dispatch.pack_gqa)
        self.assertEqual(metadata.dispatch.block_n, 128)

    def test_block_override_controls_both_split_heuristics(self):
        for block_n in (128, 176):
            dispatch = choose_dispatch(
                max_seqlen_q=8,
                max_seqlen_history=180225,
                hq_local=4,
                dcp_size=8,
                num_sms=132,
                requested_num_splits=0,
                block_n_override=block_n,
            )
            self.assertEqual(
                dispatch.chunk_num_splits,
                choose_split_upper_bound(
                    max_seqlen_q=8,
                    max_seqlen_k=8,
                    q_heads=4,
                    num_sms=132,
                    block_n=block_n,
                    is_causal=True,
                    requested_num_splits=0,
                ),
            )
            self.assertEqual(
                dispatch.history_num_splits,
                choose_split_upper_bound(
                    max_seqlen_q=8,
                    max_seqlen_k=180225,
                    q_heads=32,
                    num_sms=132,
                    block_n=block_n,
                    is_causal=False,
                    requested_num_splits=0,
                ),
            )

    def test_unified_queue_once_coverage_around_compute_cta_count(self):
        for attention_count in (3, 4, 11):
            kinds = (CHUNK,) * min(2, attention_count) + (HISTORY,) * max(
                0, attention_count - 2
            )
            assignments = self._assert_unified_queue(kinds, 4)
            initial_ids = [tasks[0] for tasks in assignments if tasks]
            self.assertEqual(initial_ids, list(range(min(attention_count, 4))))
            dynamic_ids = [task for tasks in assignments for task in tasks[1:]]
            self.assertEqual(sorted(dynamic_ids), list(range(4, attention_count)))

    def test_unified_queue_kind_transition_and_dependency_cases(self):
        cases = {
            "history_dominates": ((CHUNK,) + (HISTORY,) * 19, 4),
            "initial_history_only": ((HISTORY,) * 3, 8),
            "chunk_to_history": ((CHUNK,) * 6 + (HISTORY,) * 10, 4),
            "no_dynamic_work": ((CHUNK, HISTORY, HISTORY), 3),
            "multiple_dynamic_rounds": ((CHUNK,) * 2 + (HISTORY,) * 30, 4),
        }
        results = {
            name: self._assert_unified_queue(kinds, num_compute_ctas)
            for name, (kinds, num_compute_ctas) in cases.items()
        }
        self.assertTrue(
            any(
                CHUNK in [cases["chunk_to_history"][0][task] for task in tasks]
                and HISTORY
                in [cases["chunk_to_history"][0][task] for task in tasks]
                for tasks in results["chunk_to_history"]
            )
        )
        self.assertTrue(
            any(len(tasks) >= 3 for tasks in results["multiple_dynamic_rounds"])
        )

    def test_receive_static_slots_and_bounded_scan(self):
        case007_slots = self._assert_receive_slots(21, 8)
        self.assertEqual([len(tasks) for tasks in case007_slots].count(1), 21)
        self.assertEqual([len(tasks) for tasks in case007_slots].count(0), 27)

        case004_slots = self._assert_receive_slots(1708, 8)
        self.assertEqual(set(map(len, case004_slots)), {35, 36})

        for dcp_size in (2, 4, 8):
            for final_count in (0, 1, 3, 17):
                for num_comm_sm in (1, 4, 8):
                    with self.subTest(
                        dcp_size=dcp_size,
                        final_count=final_count,
                        num_comm_sm=num_comm_sm,
                    ):
                        self._assert_receive_slots(
                            final_count * (dcp_size - 1), num_comm_sm
                        )

    def test_attention_order_and_completion_ids_are_dense(self):
        metadata = build_dcp_mega_metadata(
            (0, 5, 42),
            (0, 257, 3390),
            hq_local=4,
            dcp_size=4,
            num_sms=132,
            num_comm_sm=8,
            requested_num_splits=2,
        )
        kinds = [row[0] for row in metadata.attention]
        first_history = kinds.index(HISTORY)
        self.assertTrue(all(kind == CHUNK for kind in kinds[:first_history]))
        self.assertTrue(all(kind == HISTORY for kind in kinds[first_history:]))
        self.assertEqual(
            [row[7] for row in metadata.attention],
            list(range(len(metadata.attention))),
        )

    def test_rejects_nonpositive_sequences(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_dcp_mega_metadata(
                (0, 0),
                (0, 1),
                hq_local=4,
                dcp_size=2,
                num_sms=132,
                num_comm_sm=8,
            )

    def test_fixed_layout_queue_coverage_matrix(self):
        cases = (
            ((0, 1), (0, 129)),
            ((0, 16), (0, 257)),
            ((0, 3, 20, 53), (0, 131, 516, 1541)),
            ((0, 17, 49), (0, 176, 433)),
        )
        for dcp_size in (2, 4, 8):
            for hq_local in (4, 8):
                for cu_q, cu_history in cases:
                    with self.subTest(
                        dcp_size=dcp_size,
                        hq_local=hq_local,
                        total_q=cu_q[-1],
                    ):
                        metadata = build_dcp_mega_metadata(
                            cu_q,
                            cu_history,
                            hq_local=hq_local,
                            dcp_size=dcp_size,
                            num_sms=132,
                            num_comm_sm=8,
                            requested_num_splits=2,
                        )
                        token_blocks = (cu_q[-1] + 15) // 16
                        self.assertEqual(metadata.token_block_count, token_blocks)
                        self.assertEqual(len(metadata.q_tasks), token_blocks * dcp_size)
                        self.assertEqual(len(metadata.publish), token_blocks * dcp_size)
                        self.assertEqual(len(metadata.final), token_blocks)
                        self.assertEqual(
                            metadata.receive_count, token_blocks * (dcp_size - 1)
                        )
                        self.assertEqual(
                            [(row[2] // 16, row[0]) for row in metadata.q_tasks],
                            [
                                (token_block, rank)
                                for token_block in range(token_blocks)
                                for rank in range(dcp_size)
                            ],
                        )
                        self._assert_attention_tiles_once(
                            metadata, cu_q, hq_local, dcp_size
                        )
                        self._assert_publish_receive_mapping(
                            metadata, cu_q, hq_local, dcp_size
                        )
                        final_vectors = [
                            vector
                            for row in metadata.final
                            for vector in range(row[0], row[0] + row[1])
                        ]
                        self.assertEqual(
                            final_vectors, list(range(cu_q[-1] * hq_local))
                        )

    def test_small_non_split_history_combine_preserves_vector_parallelism(self):
        cu_q = (0, 7, 23)
        cu_history = (0, 257, 1281)
        for dcp_size in (2, 4, 8):
            for hq_local in (4, 8):
                for block_n in (128, 176):
                    with self.subTest(
                        dcp_size=dcp_size,
                        hq_local=hq_local,
                        block_n=block_n,
                    ):
                        metadata = build_dcp_mega_metadata(
                            cu_q,
                            cu_history,
                            hq_local=hq_local,
                            dcp_size=dcp_size,
                            num_sms=132,
                            num_comm_sm=8,
                            requested_num_splits=1,
                            block_n_override=block_n,
                        )
                        self.assertFalse(metadata.dispatch.split)
                        self.assertEqual(
                            metadata.dispatch.history_copy_vectors_per_task, 1
                        )
                        self.assertEqual(
                            len(metadata.history_combine),
                            dcp_size * metadata.total_vectors,
                        )
                        self.assertTrue(
                            all(
                                row[2] == 1
                                for row in metadata.history_combine
                            )
                        )
                        full_publish = next(
                            row
                            for row in metadata.publish
                            if row[2] == 16 * hq_local
                        )
                        self.assertEqual(full_publish[6], 16 * hq_local)
                        self._assert_publish_receive_mapping(
                            metadata, cu_q, hq_local, dcp_size
                        )

    def test_adaptive_copy_granularity_covers_each_task_wave(self):
        expected_granularity = {
            40: 1,
            200: 4,
            400: 8,
            800: 16,
            1200: 32,
        }
        for q_len, expected in expected_granularity.items():
            with self.subTest(q_len=q_len):
                metadata = build_dcp_mega_metadata(
                    (0, q_len),
                    (0, 257),
                    hq_local=4,
                    dcp_size=8,
                    num_sms=132,
                    num_comm_sm=8,
                    requested_num_splits=1,
                )
                self.assertEqual(
                    metadata.dispatch.history_copy_vectors_per_task,
                    expected,
                )
                self._assert_publish_receive_mapping(
                    metadata, (0, q_len), 4, 8
                )

    def test_chunk_split_does_not_force_fine_history_copy(self):
        metadata = build_dcp_mega_metadata(
            (0, 1200),
            (0, 64),
            hq_local=4,
            dcp_size=8,
            num_sms=132,
            num_comm_sm=8,
            requested_num_splits=2,
        )
        self.assertTrue(metadata.dispatch.split)
        self.assertGreater(metadata.chunk_sequence_splits[0], 1)
        self.assertEqual(metadata.history_sequence_splits, (1,))
        self.assertEqual(
            metadata.dispatch.history_copy_vectors_per_task, 32
        )
        self.assertTrue(
            all(row[2] == 32 for row in metadata.history_combine)
        )
        self.assertEqual(len(metadata.history_combine), 1200)

    def test_history_copy_tasks_stop_at_unaligned_batch_boundaries(self):
        cu_q = (0, 17, 40)
        metadata = build_dcp_mega_metadata(
            cu_q,
            (0, 257, 514),
            hq_local=4,
            dcp_size=8,
            num_sms=132,
            num_comm_sm=8,
            requested_num_splits=1,
        )
        self._assert_publish_receive_mapping(metadata, cu_q, 4, 8)
        for row in metadata.history_combine:
            begin_token = row[1] // 4
            end_token = (row[1] + row[2] - 1) // 4
            self.assertTrue(
                any(begin <= begin_token <= end_token < end
                    for begin, end in zip(cu_q, cu_q[1:]))
            )

    def test_split_history_combine_keeps_one_vector_per_task(self):
        cu_q = (0, 7, 23)
        cu_history = (0, 4097, 12290)
        for hq_local in (4, 8):
            for block_n in (128, 176):
                with self.subTest(hq_local=hq_local, block_n=block_n):
                    metadata = build_dcp_mega_metadata(
                        cu_q,
                        cu_history,
                        hq_local=hq_local,
                        dcp_size=8,
                        num_sms=132,
                        num_comm_sm=8,
                        requested_num_splits=2,
                        block_n_override=block_n,
                    )
                    self.assertTrue(metadata.dispatch.split)
                    self.assertEqual(
                        len(metadata.history_combine),
                        8 * metadata.total_vectors,
                    )
                    self.assertTrue(
                        all(row[2] == 1 for row in metadata.history_combine)
                    )
                    self._assert_publish_receive_mapping(
                        metadata, cu_q, hq_local, 8
                    )

    def test_q_dependencies_only_gate_history_token_blocks(self):
        metadata = build_dcp_mega_metadata(
            (0, 5, 22, 57),
            (0, 129, 642, 2191),
            hq_local=8,
            dcp_size=4,
            num_sms=132,
            num_comm_sm=8,
            requested_num_splits=2,
        )
        referenced = set()
        for row in metadata.attention:
            dependencies = metadata.q_dependencies[row[5] : row[5] + row[6]]
            if row[0] == CHUNK:
                self.assertEqual(dependencies, ())
            else:
                self.assertTrue(dependencies)
                self.assertTrue(
                    all(0 <= dependency < metadata.token_block_count for dependency in dependencies)
                )
                referenced.update(dependencies)
        self.assertEqual(referenced, set(range(metadata.token_block_count)))

    def test_history_q_tma_tail_dependencies_only_cover_valid_packed_rows(self):
        # A full 128-row TMA footprint reaches q_ready block 1 in every case,
        # but speculative tail rows belong to the next sequence and are invalid.
        for valid_packed_rows, q_begin in ((24, 10), (64, 5), (120, 1)):
            q_len = valid_packed_rows // 8
            cu_q = (0, q_begin, q_begin + q_len, q_begin + q_len + 20)
            metadata = build_dcp_mega_metadata(
                cu_q,
                (0, 129, 386, 899),
                hq_local=4,
                dcp_size=2,
                num_sms=132,
                num_comm_sm=8,
                requested_num_splits=1,
            )
            history_rows = [
                row
                for row in metadata.attention
                if row[0] == HISTORY and row[1] == 1 and row[2] == 0
            ]
            self.assertTrue(history_rows)
            for row in history_rows:
                dependencies = metadata.q_dependencies[row[5] : row[5] + row[6]]
                self.assertEqual(dependencies, (0,))
                self.assertNotIn(1, dependencies)
            self.assertTrue(
                all(
                    metadata.q_dependencies[row[5] : row[5] + row[6]] == ()
                    for row in metadata.attention
                    if row[0] == CHUNK
                )
            )

    def test_metadata_v5_header_offsets_counts_and_capacity(self):
        metadata = build_dcp_mega_metadata(
            (0, 3, 20, 53),
            (0, 129, 516, 1541),
            hq_local=4,
            dcp_size=8,
            num_sms=132,
            num_comm_sm=8,
            requested_num_splits=2,
        )
        image = pack_dcp_mega_metadata(metadata, pre_phase=11, post_phase=12)
        self.assertEqual(image[0], METADATA_VERSION)
        self.assertEqual(METADATA_VERSION, 5)
        self.assertEqual(image[30], len(image))
        self.assertEqual(image[32], 1)
        self.assertEqual(image[33], metadata.token_block_count)
        self.assertEqual(image[34], metadata.q_ready_count)
        self.assertEqual(image[35], metadata.receive_count)
        self.assertEqual(image[36], metadata.tile_ready_count)
        self.assertEqual(image[37], 8)
        self.assertEqual(image[38], len(metadata.history_combine))
        self.assertEqual(image[39], image[24] + len(metadata.publish) * 8)
        offsets = list(image[21:30])
        self.assertEqual(offsets[0], METADATA_HEADER_INTS)
        self.assertEqual(offsets, sorted(offsets))
        self.assertTrue(all(METADATA_HEADER_INTS <= offset <= len(image) for offset in offsets))
        expected_offsets = [METADATA_HEADER_INTS]
        expected_offsets.append(expected_offsets[-1] + len(metadata.attention) * 8)
        expected_offsets.append(expected_offsets[-1] + len(metadata.q_tasks) * 4)
        expected_offsets.append(expected_offsets[-1] + len(metadata.q_dependencies))
        expected_offsets.append(expected_offsets[-1] + len(metadata.publish) * 8)
        expected_offsets[-1] += len(metadata.history_combine) * 8
        expected_offsets.append(
            expected_offsets[-1] + len(metadata.publish_dependencies)
        )
        expected_offsets.append(expected_offsets[-1] + len(metadata.final) * 8)
        expected_offsets.append(
            expected_offsets[-1] + len(metadata.final_dependencies)
        )
        expected_offsets.append(
            expected_offsets[-1] + len(metadata.chunk_sequence_splits)
        )
        self.assertEqual(offsets, expected_offsets)
        self.assertEqual(
            len(image),
            offsets[-1] + len(metadata.history_sequence_splits),
        )
        with self.assertRaisesRegex(ValueError, "exceeds capacity"):
            pack_dcp_mega_metadata(
                metadata,
                pre_phase=11,
                post_phase=12,
                capacity=len(image) - 1,
            )

    def test_fixed_layout_rejects_unsupported_heads(self):
        for hq_local in (1, 2, 3, 16):
            with self.subTest(hq_local=hq_local):
                with self.assertRaisesRegex(ValueError, "hq_local in"):
                    build_dcp_mega_metadata(
                        (0, 16),
                        (0, 257),
                        hq_local=hq_local,
                        dcp_size=2,
                        num_sms=132,
                        num_comm_sm=8,
                    )


if __name__ == "__main__":
    unittest.main()
