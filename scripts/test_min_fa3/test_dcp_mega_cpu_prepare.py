"""CPU queue-cache coverage, invalidation, and diagnostics regressions."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import dcp_mega_metadata as metadata


class QueueCacheTests(unittest.TestCase):
    def setUp(self):
        metadata._metadata_queue_cache.clear()
        self.addCleanup(metadata._metadata_queue_cache.clear)

    def build(self, q=(0, 1, 2), history=(0, 8192, 16384), **options):
        args = dict(hq_local=8, dcp_size=4, num_sms=78, num_comm_sm=8,
                    max_num_splits=1, block_n_override=128,
                    scheduler_heuristic=False)
        args.update(options)
        return metadata.build_dcp_mega_metadata(q, history, **args)

    def test_fifo_queue_reused_across_history_blocks_and_validated_once(self):
        with patch.object(metadata, 'validate_dcp_mega_metadata',
                          wraps=metadata.validate_dcp_mega_metadata) as validate:
            first = self.build()
            second = self.build(history=(0, 8321, 16642))
            self.assertIs(first.attention, second.attention)
            self.assertEqual(first, second)
            self.assertEqual(validate.call_count, 1)
        metadata.validate_dcp_mega_metadata(second, (0, 1, 2), hq_local=8, dcp_size=4)

    def test_cached_queue_keeps_current_auto_diagnostics(self):
        first = self.build(scheduler_heuristic=None)
        second = self.build(history=(0, 16384, 32768), scheduler_heuristic=None)
        self.assertIs(first.attention, second.attention)
        self.assertNotEqual(first.heuristic_baseline_makespan,
                            second.heuristic_baseline_makespan)
        metadata._metadata_queue_cache.clear()
        uncached = self.build(history=(0, 16384, 32768), scheduler_heuristic=None)
        self.assertEqual(second, uncached)

    def test_queue_key_covers_layout_splits_order_and_hardware(self):
        self.build()
        variants = (
            dict(q=(0, 2, 3)), dict(hq_local=4), dict(dcp_size=2),
            dict(num_sms=80), dict(num_comm_sm=4), dict(block_n_override=176),
            dict(requested_num_splits=2, max_num_splits=2),
            dict(reorder_history_override=True),
        )
        for change in variants:
            with self.subTest(change=change):
                before = len(metadata._metadata_queue_cache)
                self.build(**change)
                self.assertEqual(len(metadata._metadata_queue_cache), before + 1)
        before = len(metadata._metadata_queue_cache)
        self.build(history=(0, 8321, 16642), reorder_history_override=True)
        self.assertEqual(len(metadata._metadata_queue_cache), before + 1)

    def test_external_validation_still_checks_cached_queue_contents(self):
        value = self.build()
        bad = replace(value, final=())
        with self.assertRaises(AssertionError):
            metadata.validate_dcp_mega_metadata(bad, (0, 1, 2), hq_local=8, dcp_size=4)

    def test_queue_cache_is_bounded(self):
        for length in range(1, metadata._METADATA_QUEUE_CACHE_SIZE + 3):
            self.build(q=(0, length, length + 1))
        self.assertEqual(len(metadata._metadata_queue_cache), metadata._METADATA_QUEUE_CACHE_SIZE)


if __name__ == '__main__':
    unittest.main()
