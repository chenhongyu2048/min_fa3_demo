import sys
import unittest

import torch

import min_fa3_dcp
from dcp_test.benchmark_dcp import expanded_method_labels as dense_method_labels
from dcp_test.benchmark_dcp_varlen import (
    expanded_method_labels as varlen_method_labels,
)
from min_fa3_dcp import (
    VLLMA2ADCPAttentionRunner,
    _dcp_a2a_head_owner_ranges,
    _dcp_a2a_lse_weighted_combine_reference,
    _dcp_a2a_payload_bytes,
    make_topology,
    validate_group_ranks,
    validate_topology,
)


class DCPTopologyTest(unittest.TestCase):
    def test_six_supported_gqa_topologies(self) -> None:
        expected = {
            (32, 4, 2),
            (32, 2, 2),
            (32, 2, 4),
            (64, 4, 2),
            (64, 2, 2),
            (64, 2, 4),
        }
        valid = {
            (hq, hkv, dcp)
            for hq in (32, 64)
            for hkv in (2, 4)
            for dcp in (2, 4, 8)
            if not validate_topology(hq, hkv, 8, dcp)
        }
        self.assertEqual(valid, expected)

    def test_rank_mapping_stays_inside_kv_replica_group(self) -> None:
        topology = make_topology(32, 2, 8, 2)
        self.assertEqual(topology.kv_replica_ranks(0), (0, 1, 2, 3))
        self.assertEqual(topology.kv_replica_ranks(7), (4, 5, 6, 7))
        self.assertEqual(topology.dcp_group_ranks(0), (0, 1))
        self.assertEqual(topology.dcp_group_ranks(3), (2, 3))
        self.assertEqual(topology.dcp_group_ranks(4), (4, 5))
        self.assertEqual(topology.dcp_group_ranks(7), (6, 7))
        self.assertEqual(topology.kv_head_for_rank(3), 0)
        self.assertEqual(topology.kv_head_for_rank(4), 1)
        self.assertEqual(topology.q_head_range(3), (12, 16))

    def test_rejects_group_crossing_kv_replica_boundary(self) -> None:
        topology = make_topology(32, 2, 8, 2)
        codes = {issue.code for issue in validate_group_ranks(topology, (3, 4))}
        self.assertIn("dcp_group_crosses_kv_replica_boundary", codes)

    def test_rejects_dcp_larger_than_replica_count(self) -> None:
        codes = {issue.code for issue in validate_topology(32, 4, 8, 4)}
        self.assertIn("dcp_exceeds_kv_replicas", codes)

    def test_rejects_replica_count_not_divisible_by_dcp(self) -> None:
        codes = {issue.code for issue in validate_topology(24, 2, 12, 4)}
        self.assertIn("kv_replicas_not_divisible_by_dcp", codes)

    def test_rejects_q_per_kv_not_divisible_by_dcp(self) -> None:
        codes = {issue.code for issue in validate_topology(20, 4, 8, 2)}
        self.assertIn("q_per_kv_not_divisible_by_dcp", codes)

    def test_rejects_q_heads_not_divisible_by_tp(self) -> None:
        codes = {issue.code for issue in validate_topology(36, 4, 8, 2)}
        self.assertIn("q_heads_not_divisible_by_tp", codes)

    def test_a2a_rank_major_head_owners(self) -> None:
        self.assertEqual(
            _dcp_a2a_head_owner_ranges(16, 4),
            ((0, 4), (4, 8), (8, 12), (12, 16)),
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            _dcp_a2a_head_owner_ranges(10, 4)

    def test_a2a_payload_counts_buffer_and_remote_bytes(self) -> None:
        buffer_bytes, remote_bytes = _dcp_a2a_payload_bytes(6, 2, 128, 4)
        per_owner = 6 * 2 * (128 + 2) * 2
        self.assertEqual(buffer_bytes, 4 * per_owner)
        self.assertEqual(remote_bytes, 3 * per_owner)

    def test_vllm_category_expands_ag_rs_and_a2a(self) -> None:
        self.assertEqual(
            dense_method_labels(("vllm",)),
            ("vllm_ag_rs_min_fa3", "vllm_a2a_min_fa3", "full_kv_min_fa3"),
        )
        self.assertEqual(
            varlen_method_labels(("vllm",)),
            ("vllm_ag_rs_min_fa3_varlen", "vllm_a2a_min_fa3_varlen"),
        )

    def test_a2a_lse_weighted_equal_states(self) -> None:
        outputs = torch.tensor([[[[1.0]]], [[[3.0]]]])
        lses = torch.zeros((2, 1, 1), dtype=torch.float32)
        output, lse = _dcp_a2a_lse_weighted_combine_reference(outputs, lses)
        torch.testing.assert_close(output, torch.tensor([[[2.0]]]))
        torch.testing.assert_close(lse, torch.tensor([[torch.log(torch.tensor(2.0))]]))

    def test_a2a_lse_weighted_dominant_rank(self) -> None:
        outputs = torch.tensor([[[[1.0]]], [[[5.0]]]])
        lses = torch.tensor([[[0.0]], [[20.0]]], dtype=torch.float32)
        output, lse = _dcp_a2a_lse_weighted_combine_reference(outputs, lses)
        torch.testing.assert_close(output, torch.tensor([[[5.0]]]), atol=1.0e-6, rtol=0)
        torch.testing.assert_close(lse, torch.tensor([[20.0]]), atol=1.0e-6, rtol=0)

    def test_a2a_lse_weighted_all_invalid_states(self) -> None:
        outputs = torch.tensor([[[[1.0]]], [[[5.0]]], [[[9.0]]]])
        lses = torch.tensor(
            [[[float("nan")]], [[float("inf")]], [[-float("inf")]]],
            dtype=torch.float32,
        )
        output, lse = _dcp_a2a_lse_weighted_combine_reference(outputs, lses)
        torch.testing.assert_close(output, torch.zeros_like(output))
        self.assertTrue(torch.isneginf(lse).all())

    def test_a2a_runner_exports_without_vllm_runtime(self) -> None:
        self.assertIs(min_fa3_dcp.VLLMA2ADCPAttentionRunner, VLLMA2ADCPAttentionRunner)
        self.assertIn("VLLMA2ADCPAttentionRunner", min_fa3_dcp.__all__)
        self.assertFalse(any(name == "vllm" or name.startswith("vllm.") for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
