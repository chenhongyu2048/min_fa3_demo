import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
ROOT_DIR = THIS_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
RING_TEST_DIR = ROOT_DIR / "ring_test"
if str(RING_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(RING_TEST_DIR))

import allgather_attention as allgather_module
from allgather_attention import (
    Llama3AllGatherAttention,
    _llama3_block_metadata,
    llama3_rank_local_global_indices,
    llama3_rank_major_to_global_order,
    sequence_shards_to_global_order,
)


class _CompletedWork:
    def wait(self) -> None:
        pass


def _fake_all_gather_into_tensor(
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    group,
    async_op: bool,
) -> _CompletedWork:
    del group
    assert async_op
    output.copy_(input)
    return _CompletedWork()


def _fake_reduce_scatter_tensor(
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    group,
) -> None:
    del group
    output.copy_(input)


def _fake_local_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    cu_q_host: torch.Tensor,
    cu_k_host: torch.Tensor,
    max_q: int,
    max_k: int,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del v, cu_q, cu_k, cu_q_host, cu_k_host, max_q, max_k, causal
    assert q.is_contiguous() and k.is_contiguous()
    ratio = q.size(1) // k.size(1)
    k_bias = k.mean(dim=0, keepdim=True).repeat_interleave(ratio, dim=1)
    out = q + k_bias
    lse = out[..., 0].transpose(0, 1).contiguous()
    return out, lse


def _fake_local_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    causal: bool,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del q, out, lse, cu_q, cu_k, max_q, max_k, causal
    assert dout.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert dq.is_contiguous() and dk.is_contiguous() and dv.is_contiguous()
    dq.copy_(dout * 2)
    dk.copy_(k * 3)
    dv.copy_(v * 5)
    return dq, dk, dv


def rank_major_token_ids(order: list[int]) -> list[int]:
    rank_major = [-1] * len(order)
    for global_idx, source_idx in enumerate(order):
        rank_major[source_idx] = global_idx
    return rank_major


class Llama3AllGatherLayoutTest(unittest.TestCase):
    def test_per_sequence_causal_order_restores_global_tokens(self) -> None:
        order = sequence_shards_to_global_order([12, 8], world_size=2, causal=True)
        rank_major = rank_major_token_ids(order)
        self.assertEqual([rank_major[idx] for idx in order], list(range(20)))

    def test_per_sequence_noncausal_order_restores_global_tokens(self) -> None:
        order = sequence_shards_to_global_order([12, 8], world_size=2, causal=False)
        rank_major = rank_major_token_ids(order)
        self.assertEqual([rank_major[idx] for idx in order], list(range(20)))

    def test_llama3_rank_blocks_round_trip_global_tokens(self) -> None:
        rank_major = []
        for rank in range(2):
            rank_major.extend(llama3_rank_local_global_indices(20, 2, rank))
        order = llama3_rank_major_to_global_order(20, 2)
        self.assertEqual([rank_major[idx] for idx in order], list(range(20)))
        self.assertEqual(rank_major[:10], [0, 1, 2, 3, 4, 15, 16, 17, 18, 19])
        self.assertEqual(rank_major[10:], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    def test_causal_metadata_for_block_crossing_sequence_boundary(self) -> None:
        block = _llama3_block_metadata(
            [12, 8], 10, 15, local_begin=5, causal=True, device=torch.device("cpu")
        )
        self.assertEqual(block.local_slice, slice(5, 10))
        self.assertEqual(block.global_k_slice, slice(0, 15))
        self.assertEqual(block.cu_q_host.tolist(), [0, 2, 5])
        self.assertEqual(block.cu_k_host.tolist(), [0, 12, 15])
        self.assertEqual((block.max_q, block.max_k), (3, 12))

    def test_noncausal_metadata_for_block_crossing_sequence_boundary(self) -> None:
        block = _llama3_block_metadata(
            [12, 8], 10, 15, local_begin=5, causal=False, device=torch.device("cpu")
        )
        self.assertEqual(block.global_k_slice, slice(0, 20))
        self.assertEqual(block.cu_q_host.tolist(), [0, 2, 5])
        self.assertEqual(block.cu_k_host.tolist(), [0, 12, 20])
        self.assertEqual((block.max_q, block.max_k), (3, 12))

    def test_rejects_non_divisible_total(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            llama3_rank_local_global_indices(19, 2, 0)

    def test_kv_head_pipeline_preserves_forward_and_backward_values(self) -> None:
        q = torch.arange(8 * 4 * 2, dtype=torch.float32).view(8, 4, 2)
        k = torch.arange(8 * 2 * 2, dtype=torch.float32).view(8, 2, 2)
        v = (k + 100).clone()
        dout = torch.full_like(q, 2.0)

        with (
            mock.patch.object(allgather_module.dist, "get_rank", return_value=0),
            mock.patch.object(allgather_module.dist, "get_world_size", return_value=1),
            mock.patch.object(
                allgather_module.dist,
                "all_gather_into_tensor",
                side_effect=_fake_all_gather_into_tensor,
            ),
            mock.patch.object(
                allgather_module.dist,
                "reduce_scatter_tensor",
                side_effect=_fake_reduce_scatter_tensor,
            ),
            mock.patch.object(allgather_module, "_local_forward", side_effect=_fake_local_forward),
            mock.patch.object(allgather_module, "_local_backward", side_effect=_fake_local_backward),
        ):
            runner = Llama3AllGatherAttention(
                None,
                q,
                k,
                v,
                [8],
                True,
                "min_fa3",
                heads_k_stride=1,
                enable_backward=True,
            )
            out = runner.forward()
            dq, dk, dv = runner.backward(dout)

        expected_out = q.clone()
        for kv_head in range(k.size(1)):
            q_head_slice = slice(2 * kv_head, 2 * (kv_head + 1))
            expected_out[:4, q_head_slice] += k[:4, kv_head].mean(dim=0)
            expected_out[4:, q_head_slice] += k[:, kv_head].mean(dim=0)
        expected_dk = k * 3
        expected_dk[:4] += k[:4] * 3
        expected_dv = v * 5
        expected_dv[:4] += v[:4] * 5

        torch.testing.assert_close(out, expected_out)
        torch.testing.assert_close(dq, dout * 2)
        torch.testing.assert_close(dk, expected_dk)
        torch.testing.assert_close(dv, expected_dv)

    def test_kv_head_stride_must_divide_kv_heads(self) -> None:
        q = torch.empty((8, 4, 2))
        k = torch.empty((8, 2, 2))
        with self.assertRaisesRegex(ValueError, "heads_k_stride"):
            Llama3AllGatherAttention(
                None, q, k, k, [8], True, "min_fa3", heads_k_stride=3
            )


if __name__ == "__main__":
    unittest.main()
