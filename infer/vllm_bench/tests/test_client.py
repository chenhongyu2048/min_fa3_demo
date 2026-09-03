from __future__ import annotations

import asyncio
import time
import unittest

from vllm_bench.client import PreparedRequest, _send_one, summarize
from vllm_bench.workload import WorkloadRequest


class _Content:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def iter_any(self):
        for chunk in self.chunks:
            yield chunk


class _Response:
    status = 200

    def __init__(self, chunks: list[bytes]) -> None:
        self.content = _Content(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _Session:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    def post(self, *args, **kwargs):
        return _Response(self.chunks)


class ClientTest(unittest.TestCase):
    @staticmethod
    def _prepared(output_length: int) -> PreparedRequest:
        request = WorkloadRequest(1, 0, 4, output_length, 3, (1,), None)
        return PreparedRequest(request, 0.0, b"{}")

    def test_sse_token_timestamps_and_summary(self) -> None:
        chunks = [
            b'data: {"choices":[{"token_ids":[10]}]}\n\n',
            b'data: {"choices":[{"token_ids":[11]}]}\n\ndata: [DONE]\n\n',
        ]
        result = asyncio.run(
            _send_one(
                _Session(chunks),
                "http://test",
                self._prepared(2),
                time.monotonic(),
            )
        )
        self.assertTrue(result.success)
        self.assertTrue(result.tbt_valid)
        self.assertEqual(result.token_ids, [10, 11])
        self.assertEqual(len(result.itl_s), 1)
        summary = summarize([result], backend="mega", scale=1, elapsed_s=2)
        self.assertEqual(summary["successful_requests"], 1)
        self.assertEqual(summary["output_tokens"], 2)
        self.assertEqual(summary["output_token_throughput"], 1)

    def test_multi_token_delta_is_not_a_tbt_sample(self) -> None:
        chunks = [
            b'data: {"choices":[{"token_ids":[10,11]}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        result = asyncio.run(
            _send_one(
                _Session(chunks),
                "http://test",
                self._prepared(2),
                time.monotonic(),
            )
        )
        self.assertTrue(result.success)
        self.assertFalse(result.tbt_valid)
        summary = summarize([result], backend="mega", scale=1, elapsed_s=1)
        self.assertEqual(summary["tbt_valid_requests"], 0)
        self.assertEqual(summary["pooled_itl_s"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
