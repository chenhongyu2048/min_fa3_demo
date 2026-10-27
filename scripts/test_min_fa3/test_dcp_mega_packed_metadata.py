"""Byte-for-byte coverage of the C++ queue builder and serving payload cache."""
import itertools
import random
import unittest
from unittest.mock import patch

import dcp_mega_metadata as metadata


def offsets(lengths):
    return (0, *itertools.accumulate(lengths))


class PackedFallbackTests(unittest.TestCase):
    def test_python_fallback_keeps_payload_contract(self):
        q, h = (0, 1, 18), (0, 8192, 32768)
        config = dict(hq_local=8, dcp_size=4, num_sms=78, num_comm_sm=8,
                      scheduler_heuristic=False, max_num_splits=8)
        reference = metadata.build_dcp_mega_metadata(q, h, **config)
        expected = metadata.pack_dcp_mega_metadata(reference, pre_phase=7, post_phase=8)
        with patch.object(metadata, "_native_packed_queues", None):
            dispatch, payload = metadata.build_packed_dcp_mega_metadata(
                q, h, pre_phase=7, post_phase=8, capacity=len(expected), **config)
        self.assertEqual(dispatch, reference.dispatch)
        self.assertEqual(payload, expected)


@unittest.skipIf(metadata._native_packed_queues is None, "C++ packed queue builder not built")
class NativePackedTests(unittest.TestCase):
    def setUp(self):
        metadata._metadata_queue_cache.clear()
        self.addCleanup(metadata._metadata_queue_cache.clear)

    def test_all_packed_fields_match_validated_python_queues(self):
        rng = random.Random(5921)
        cases = [(1,) * 64, (1, 3, 17, 33), (1120,), (2240,), (4097,)]
        cases += [tuple(rng.choice((1, 1, 3, 17, 65, 257))
                        for _ in range(rng.randint(1, 12))) for _ in range(100)]
        for i, lengths in enumerate(cases):
            q = offsets(lengths)
            h = offsets(rng.randint(1, 150000) for _ in lengths)
            cap = rng.choice((1, 8, 128))
            config = dict(hq_local=rng.choice((4, 8)), dcp_size=rng.choice((2, 4, 8)),
                          num_sms=78, num_comm_sm=rng.choice((4, 8, 16)), max_num_splits=cap,
                          block_n_override=(None, 128, 176)[i % 3],
                          scheduler_heuristic=(False, None, True)[i % 3],
                          requested_num_splits=rng.choice((0, 1)) if i % 3 else rng.choice((0, cap)),
                          reorder_history_override=(None, False, True)[(i // 3) % 3])
            reference = metadata.build_dcp_mega_metadata(q, h, **config)
            expected = metadata.pack_dcp_mega_metadata(reference, pre_phase=11, post_phase=12)
            dispatch, payload = metadata.build_packed_dcp_mega_metadata(
                q, h, pre_phase=11, post_phase=12, capacity=len(expected), **config)
            with self.subTest(case=i, config=config):
                self.assertEqual(dispatch, reference.dispatch)
                self.assertEqual(payload, expected)

    def prepare(self, q=(0, 1, 2), h=(0, 8192, 16384), **options):
        config = dict(hq_local=8, dcp_size=4, num_sms=78, num_comm_sm=8,
                      scheduler_heuristic=False, max_num_splits=1,
                      pre_phase=1, post_phase=2)
        config.update(options)
        return metadata.build_packed_dcp_mega_metadata(q, h, **config)

    def test_fifo_cache_reuse_and_payload_ownership(self):
        with patch.object(metadata, "_native_packed_queues", wraps=metadata._native_packed_queues) as build:
            dispatch, first = self.prepare()
            original = first[:]
            second_dispatch, second = self.prepare(h=(0, 8321, 16642), pre_phase=3, post_phase=4)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(dispatch, second_dispatch)
            self.assertEqual(first, original)
            self.assertEqual(second[19:21].tolist(), [3, 4])
            self.assertEqual(second[:19], first[:19])
            self.assertEqual(second[21:], first[21:])
            first[40] = -123
            _, third = self.prepare()
            self.assertEqual(third, original)

    def test_layout_and_order_invalidate_cache(self):
        with patch.object(metadata, "_native_packed_queues", wraps=metadata._native_packed_queues) as build:
            self.prepare()
            self.prepare(q=(0, 2, 3))
            self.prepare(reorder_history_override=True)
            self.prepare(h=(0, 8321, 16642), reorder_history_override=True)
            self.assertEqual(build.call_count, 4)

    def test_cache_capacity_and_phase_checks(self):
        _, payload = self.prepare()
        with self.assertRaisesRegex(ValueError, "exceeds capacity"):
            self.prepare(capacity=len(payload) - 1)
        for pre, post in ((0, 1), (2, 2), (3, 2)):
            with self.assertRaisesRegex(ValueError, "phases"):
                self.prepare(pre_phase=pre, post_phase=post)
        for length in range(1, metadata._METADATA_QUEUE_CACHE_SIZE + 3):
            self.prepare(q=(0, length, length + 1))
        self.assertEqual(len(metadata._metadata_queue_cache), metadata._METADATA_QUEUE_CACHE_SIZE)


if __name__ == "__main__":
    unittest.main()
