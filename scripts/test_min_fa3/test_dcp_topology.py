import unittest

from min_fa3_dcp import (
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


if __name__ == "__main__":
    unittest.main()
