import sys
import unittest
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from allgather_attention import (
    _llama3_block_metadata,
    llama3_rank_local_global_indices,
    llama3_rank_major_to_global_order,
    sequence_shards_to_global_order,
)


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


if __name__ == "__main__":
    unittest.main()
