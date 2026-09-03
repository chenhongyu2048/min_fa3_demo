from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vllm_bench.client import _relative_offsets, prepare_requests, summarize
from vllm_bench.workload import (
    BLOCK_SIZE,
    WorkloadRequest,
    build_manifest,
    load_trace,
    prompt_token_ids,
)


class WorkloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _trace(self, rows: list[dict[str, object]]) -> Path:
        path = self.root / "trace.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    @staticmethod
    def _row(timestamp, length, hashes, output=3):
        return {
            "timestamp": timestamp,
            "input_length": length,
            "output_length": output,
            "hash_ids": hashes,
        }

    def test_lineage_first_turn_and_ambiguous_drop(self) -> None:
        path = self._trace(
            [
                self._row(0, 1100, [1, 2, 3]),
                self._row(1, 1300, [1, 2, 4]),
                self._row(2, 1300, [1, 2, 4]),
                self._row(3, 700, [1, 9]),
            ]
        )
        _, requests, ambiguous, model_length = load_trace(path)
        self.assertEqual(ambiguous, 1)
        self.assertEqual(model_length, 0)
        self.assertEqual([request.request_id for request in requests], [0, 1, 3])
        self.assertEqual(requests[0].history_tokens, 1099)
        self.assertEqual(requests[1].history_tokens, 2 * BLOCK_SIZE)
        self.assertEqual(requests[1].chunk_tokens, 276)
        self.assertIsNone(requests[2].lineage_request_id)

    def test_tokens_are_deterministic_legal_and_share_hash_blocks(self) -> None:
        first = WorkloadRequest(0, 0, 600, 1, 599, (7, 8), None)
        second = WorkloadRequest(1, 0, 700, 1, 512, (7, 9), 0)
        first_tokens = prompt_token_ids(first, 42)
        second_tokens = prompt_token_ids(second, 42)
        self.assertEqual(first_tokens, prompt_token_ids(first, 42))
        self.assertEqual(first_tokens[:512], second_tokens[:512])
        self.assertNotEqual(first_tokens[512:], second_tokens[512:600])
        self.assertEqual(len(first_tokens), 600)
        self.assertTrue(all(0 <= token < 128000 for token in first_tokens))

    def test_arrival_scaling_changes_only_time(self) -> None:
        requests = (
            WorkloadRequest(0, 1_000_000, 2, 1, 1, (1,), None),
            WorkloadRequest(1, 3_000_000, 2, 1, 1, (2,), None),
        )
        self.assertEqual(_relative_offsets(requests, 1), [0.0, 2.0])
        self.assertEqual(_relative_offsets(requests, 4), [0.0, 0.5])
        prepared = prepare_requests(
            requests,
            scale=4,
            model="dummy",
            token_seed=42,
            request_prefix="test",
        )
        body = json.loads(prepared[1].body)
        self.assertEqual(body["kv_transfer_params"]["history_tokens"], 1)
        self.assertEqual(body["max_tokens"], 1)
        self.assertEqual(body["min_tokens"], body["max_tokens"])
        self.assertTrue(body["return_token_ids"])
        self.assertTrue(body["ignore_eos"])
        self.assertFalse(body["add_special_tokens"])

    def test_manifest_partitions_are_fixed(self) -> None:
        rows = [self._row(index, 10, [index], output=1) for index in range(5)]
        manifest = build_manifest(
            self._trace(rows), warmup_requests=2, measured_requests=3, seed=42
        )
        selected = [*manifest.warmup, *manifest.measured]
        self.assertEqual(len(selected), 5)
        self.assertEqual(
            [request.request_id for request in selected], list(range(5))
        )


class TraceRegressionTest(unittest.TestCase):
    def test_repository_trace_lineage_counts(self) -> None:
        trace = (
            Path(__file__).resolve().parents[3]
            / "dcp_test/trace/conversation_trace.jsonl"
        )
        _, requests, ambiguous, model_length = load_trace(trace)
        self.assertEqual(ambiguous, 118)
        self.assertEqual(model_length, 0)
        self.assertEqual(len(requests), 12031 - 118)
        self.assertEqual(
            sum(request.lineage_request_id is not None for request in requests), 4540
        )


if __name__ == "__main__":
    unittest.main()
