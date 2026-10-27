"""CPU regressions for plan reuse and the compact, exact cost model."""

import heapq
import itertools
import random
import unittest
from unittest.mock import patch

import dcp_mega_metadata as metadata


def offsets(lengths):
    return (0, *itertools.accumulate(lengths))


def expanded_combine(cu_q, splits, completions, heads, world, copy_vectors):
    """Reference the physical vectors, independently of compact tile arithmetic."""
    batch_for_token = [batch for batch, (a, b) in enumerate(zip(cu_q, cu_q[1:]))
                       for _ in range(a, b)]
    tasks = []
    for tile in range(0, cu_q[-1] * heads, 16 * heads):
        end = min(tile + 16 * heads, cu_q[-1] * heads)
        for rank in range(world):
            begin = tile
            while begin < end:
                batch = batch_for_token[begin // heads]
                region_end = min(end, cu_q[batch + 1] * heads)
                step = 1 if splits[batch] > 1 else copy_vectors
                for first in range(begin, region_end, step):
                    last = min(first + step, region_end)
                    deps = set()
                    for vector in range(first, last):
                        token, head = divmod(vector, heads)
                        deps.update(completions[
                            (metadata.HISTORY, batch, token, rank * heads + head)
                        ])
                    tasks.append((tuple(sorted(deps)), 4 + (last - first) * splits[batch]))
                begin = region_end
    return tasks


def expanded_makespan(attention, tasks):
    workers = [(time, cta, warp)
               for cta, time in enumerate(attention.cta_finish_times)
               for warp in range(metadata.MEGA_COMPUTE_WARPS)]
    heapq.heapify(workers)
    for deps, work in tasks:
        time, cta, warp = heapq.heappop(workers)
        ready = max(attention.completion_finish_times[i] for i in deps)
        heapq.heappush(workers, (max(time, ready) + work, cta, warp))
    return max(attention.makespan, max(time for time, _, _ in workers))


class PlannerTests(unittest.TestCase):
    def setUp(self):
        metadata._critical_wave_plan_cached.cache_clear()
        self.addCleanup(metadata._critical_wave_plan_cached.cache_clear)

    def test_compact_tasks_and_schedule_match_vector_reference(self):
        rng = random.Random(481)
        for world, heads, reorder in itertools.product((2, 4, 8), (4, 8), (False, True)):
            q = (1, 3, 17, 33)
            history_blocks = (140, 37, 91, 13)
            splits = (1, 3, 2, 1)
            chunk_splits = (1, 1, 2, 1)
            cu_q = offsets(q)
            rows, q_deps, completions = [], [], {}
            for kind, domain_heads, domain_splits in (
                (metadata.CHUNK, heads, chunk_splits),
                (metadata.HISTORY, world * heads, splits),
            ):
                metadata._append_attention_domain(
                    rows, q_deps, completions, kind=kind, cu_q=cu_q,
                    heads=domain_heads, sequence_splits=domain_splits,
                )
            attention, _, bases = metadata._attention_schedule_profile(
                q, history_blocks, hq_local=heads, dcp_size=world, block_n=128,
                num_compute_ctas=7, num_comm_sm=8,
                chunk_sequence_splits=chunk_splits,
                history_sequence_splits=splits, reorder_history=reorder,
            )
            for copy in (1, heads, 2 * heads, 4 * heads):
                with self.subTest(world=world, heads=heads, reorder=reorder, copy=copy):
                    compact = metadata._history_combine_schedule_tasks(
                        cu_q, splits, bases, hq_local=heads, dcp_size=world,
                        copy_vectors_per_task=copy,
                    )
                    reference = expanded_combine(cu_q, splits, completions, heads, world, copy)
                    self.assertEqual(
                        [(task.dependencies, task.work) for task in compact
                         for _ in range(task.repeats)], reference,
                    )
                    self.assertEqual(
                        metadata._overlapped_attention_combine_makespan(attention, compact),
                        expanded_makespan(attention, reference),
                    )
            # Uneven repeated task runs must preserve FIFO blocking and tie behavior.
            compact = tuple(metadata._HistoryCombineScheduleTask(
                dependencies=(rng.randrange(len(rows)),),
                work=rng.randint(1, 20), repeats=rng.randint(1, 100),
            ) for _ in range(40))
            reference = [(task.dependencies, task.work) for task in compact
                         for _ in range(task.repeats)]
            self.assertEqual(
                metadata._overlapped_attention_combine_makespan(attention, compact),
                expanded_makespan(attention, reference),
            )

    def plan(self, q=(1, 1), history=(10001, 20001), **overrides):
        args = dict(hq_local=8, dcp_size=4, block_n=128, num_sms=78,
                    num_comm_sm=8, chunk_sequence_splits=(1,) * len(q),
                    include_legacy_candidate=True, max_num_splits=8,
                    native_chunk_sequence_splits=(1,) * len(q),
                    legacy_is_native=True, reorder_no_split=False, reorder_split=True)
        args.update(overrides)
        return metadata._critical_wave_plan(q, history, **args)

    def test_plan_reused_within_history_block_and_invalidated_at_boundary(self):
        first = self.plan()
        self.assertEqual(self.plan(history=(10002, 20002)), first)
        info = metadata._critical_wave_plan_cached.cache_info()
        self.assertEqual((info.hits, info.misses), (1, 1))
        self.plan(history=(10113, 20002))
        self.assertEqual(metadata._critical_wave_plan_cached.cache_info().misses, 2)

    def test_raw_length_native_threshold_is_resolved_before_cache_lookup(self):
        # These lengths share an N-block at BlockN=176 but cross FA's 50 MiB L2 bound.
        self.plan(q=(700, 1), history=(1, 102400), block_n=176)
        self.plan(q=(700, 1), history=(1, 102401), block_n=176)
        self.assertEqual(metadata._critical_wave_plan_cached.cache_info().misses, 2)

    def test_plan_cache_separates_queue_and_hardware_configurations(self):
        self.plan()
        variants = (
            dict(q=(2, 1)), dict(hq_local=4), dict(dcp_size=2), dict(block_n=176),
            dict(num_sms=80), dict(num_comm_sm=4), dict(max_num_splits=1),
            dict(chunk_sequence_splits=(2, 1)), dict(native_chunk_sequence_splits=(2, 1)),
            dict(legacy_is_native=False), dict(reorder_no_split=True),
            dict(reorder_split=False), dict(include_legacy_candidate=False),
        )
        for changes in variants:
            with self.subTest(changes=changes):
                misses = metadata._critical_wave_plan_cached.cache_info().misses
                self.plan(**changes)
                self.assertEqual(metadata._critical_wave_plan_cached.cache_info().misses,
                                 misses + 1)


@unittest.skipIf(metadata._native_critical_wave_plan is None, "C++ CPU planner not built")
class NativePlannerTests(unittest.TestCase):
    def test_native_search_and_pruning_match_python(self):
        rng = random.Random(719)
        pruned = 0
        cases = [(1,) * 64, (1,) * 4, (4097,), (1, 17, 129, 3)]
        cases += [tuple(rng.choice((1, 1, 3, 17, 65)) for _ in range(rng.randint(1, 12)))
                  for _ in range(80)]
        for index, q in enumerate(cases):
            args = dict(
                q_lengths=q, history_n_blocks=tuple(rng.randint(1, 1500) for _ in q),
                hq_local=rng.choice((4, 8)), dcp_size=rng.choice((2, 4, 8)),
                block_n=rng.choice((128, 176)), num_sms=rng.choice((16, 40, 78)),
                num_comm_sm=8, chunk_sequence_splits=tuple(rng.randint(1, 3) for _ in q),
                legacy_splits=tuple(rng.randint(1, 4) for _ in q), max_num_splits=8,
                native_chunk_sequence_splits=(None if index % 3 == 0 else (1,) * len(q)),
                legacy_is_native=bool(index % 2), reorder_no_split=bool(index % 3),
                reorder_split=bool(index % 4),
            )
            expected = metadata._critical_wave_plan_python(**args)
            for prune in (False, True):
                with self.subTest(index=index, prune=prune):
                    baseline, selected, stats = metadata._native_critical_wave_plan(**args, prune=prune)
                    actual = tuple(metadata._CriticalWavePlan(
                        tuple(p[0]), *p[1:8], tuple(p[8]), p[9]
                    ) for p in (baseline, selected))
                    self.assertEqual(actual, expected)
                    rejected = sum(stats[k] for k in ("bound_pruned", "attention_pruned", "combine_pruned"))
                    self.assertEqual(stats["candidates"], stats["scored"] + rejected)
                    if prune:
                        pruned += rejected
                    else:
                        self.assertEqual(rejected, 0)
        self.assertGreater(pruned, 0)

    def test_native_full_metadata_and_packed_queues_match_python(self):
        rng = random.Random(820)
        for index in range(24):
            q = tuple(rng.choice((1, 2, 17, 65)) for _ in range(rng.randint(1, 8)))
            h = tuple(rng.randint(1, 100000) for _ in q)
            config = dict(hq_local=rng.choice((4, 8)), dcp_size=rng.choice((2, 4, 8)),
                          num_sms=78, num_comm_sm=8, max_num_splits=8,
                          block_n_override=rng.choice((None, 128, 176)),
                          scheduler_heuristic=None if index % 2 else True,
                          reorder_history_override=(None, False, True)[index % 3])
            metadata._critical_wave_plan_cached.cache_clear()
            metadata._metadata_queue_cache.clear()
            expected = metadata.build_dcp_mega_metadata(offsets(q), offsets(h), **config)
            with patch.object(metadata, "_native_critical_wave_plan", None):
                metadata._critical_wave_plan_cached.cache_clear()
                metadata._metadata_queue_cache.clear()
                reference = metadata.build_dcp_mega_metadata(offsets(q), offsets(h), **config)
            with self.subTest(index=index):
                self.assertEqual(expected, reference)
                self.assertEqual(
                    metadata.pack_dcp_mega_metadata(expected, pre_phase=1, post_phase=2),
                    metadata.pack_dcp_mega_metadata(reference, pre_phase=1, post_phase=2),
                )
        metadata._critical_wave_plan_cached.cache_clear()
        metadata._metadata_queue_cache.clear()


if __name__ == '__main__':
    unittest.main()
