"""CPU-only coverage for planner-to-Mega-Ring topology adapters."""

from __future__ import annotations

import unittest

import balancer
from baseline.megatron_hybrid_cp import build_hybrid_cp_plan_for_fa3_ring
from ring_test.load_balance_bench.topology import (
    PlannerControls,
    PlannerTopology,
    TopologySample,
    make_br_pbs_topology,
    make_megatron_cp_topology,
    make_planner_topologies,
    make_zepplin_topology,
    validate_fused_metadata,
)
from ring_test.zepplin import make_zepplin_plan


class PlannerTopologyTest(unittest.TestCase):
    lengths = (8192, 4096, 2048, 1024, 512)
    world_size = 2
    controls = PlannerControls(
        compute_balance_tolerance=0.05,
        token_balance_tolerance=0.10,
        beam_width=32,
        finalist_count=4,
        structure_threshold=0.5,
        max_repair_iterations=4,
    )

    def assert_same_layout(self, left: PlannerTopology, right: PlannerTopology) -> None:
        self.assertEqual(left.planner, right.planner)
        self.assertEqual(left.is_causal, right.is_causal)
        self.assertEqual(left.world_size, right.world_size)
        self.assertEqual(left.raw_lengths, right.raw_lengths)
        self.assertEqual(left.samples, right.samples)
        self.assertEqual(left.diagnostics, right.diagnostics)

    def test_adapters_are_deterministic_and_preserve_raw_identity(self) -> None:
        first = make_planner_topologies(
            self.lengths,
            self.world_size,
            True,
            controls=self.controls,
            zepplin_threshold=4096,
            megatron_max_seqlen_per_rank=4096,
        )
        second = make_planner_topologies(
            self.lengths,
            self.world_size,
            True,
            controls=self.controls,
            zepplin_threshold=4096,
            megatron_max_seqlen_per_rank=4096,
        )
        for left, right in zip(first, second):
            self.assert_same_layout(left, right)
            self.assertEqual(set(left.sample_ids), set(range(len(self.lengths))))
            self.assertEqual(
                sorted(sample.raw_length for sample in left.samples),
                sorted(self.lengths),
            )
            self.assertGreaterEqual(left.planner_build_ms, 0.0)
            validate_fused_metadata(left)

    def test_br_pbs_matches_existing_balancer_result(self) -> None:
        topology = make_br_pbs_topology(
            self.lengths, self.world_size, True, self.controls
        )
        workload = balancer.assign_hierarchical_rings(
            list(self.lengths),
            self.world_size,
            True,
            compute_balance_tolerance=self.controls.compute_balance_tolerance,
            token_balance_tolerance=self.controls.token_balance_tolerance,
            beam_width=self.controls.beam_width,
            finalist_count=self.controls.finalist_count,
            structure_threshold=self.controls.structure_threshold,
            max_repair_iterations=self.controls.max_repair_iterations,
        )
        self.assertEqual(topology.global_lengths, tuple(workload.global_lengths))
        self.assertEqual(topology.ring_sizes, tuple(workload.ring_sizes))
        self.assertEqual(topology.ring_starts, tuple(workload.ring_starts))
        self.assertEqual(topology.sample_ids, tuple(workload.sample_ids))

    def test_megatron_uses_final_padded_fa3_ring_assignments(self) -> None:
        raw_lengths = (16385, 8192, 4096)
        topology = make_megatron_cp_topology(
            raw_lengths,
            self.world_size,
            True,
            max_seqlen_per_rank=4096,
        )
        plan = build_hybrid_cp_plan_for_fa3_ring(
            raw_lengths, self.world_size, True, max_seqlen_per_rank=4096
        )
        by_sample = {sample.sample_id: sample for sample in topology.samples}
        self.assertGreater(topology.padding_tokens, 0)
        for assignment in plan.assignments:
            sample = by_sample[assignment.sample_id]
            self.assertEqual(sample.execution_length, assignment.global_length)
            self.assertEqual(sample.ring_size, assignment.cp_size)
            self.assertEqual(sample.ring_start, assignment.rank_start)
        validate_fused_metadata(topology)

    def test_zepplin_keeps_g1_owner_and_gworld_placement_after_ordering(self) -> None:
        topology = make_zepplin_topology(
            self.lengths, self.world_size, True, threshold=4096
        )
        plan = make_zepplin_plan(list(self.lengths), self.world_size, True, 4096)
        owners = dict(zip(plan.short_indices, plan.short_owners))
        for sample in topology.samples:
            if sample.sample_id in owners:
                self.assertEqual(sample.ring_size, 1)
                self.assertEqual(sample.ring_start, owners[sample.sample_id])
            else:
                self.assertEqual(sample.ring_size, self.world_size)
                self.assertEqual(sample.ring_start, 0)
        self.assertEqual(topology.ring_sizes, tuple(sorted(topology.ring_sizes, reverse=True)))
        validate_fused_metadata(topology)

    def test_causal_and_noncausal_topologies_are_separate(self) -> None:
        causal = make_planner_topologies(
            self.lengths,
            self.world_size,
            True,
            controls=self.controls,
            zepplin_threshold=4096,
            megatron_max_seqlen_per_rank=4096,
        )
        noncausal = make_planner_topologies(
            self.lengths,
            self.world_size,
            False,
            controls=self.controls,
            zepplin_threshold=4096,
            megatron_max_seqlen_per_rank=4096,
        )
        self.assertTrue(all(topology.is_causal for topology in causal))
        self.assertTrue(all(not topology.is_causal for topology in noncausal))
        for topology in (*causal, *noncausal):
            validate_fused_metadata(topology)

    def test_invalid_alignment_fails_without_silent_padding(self) -> None:
        with self.assertRaisesRegex(ValueError, r"zepplin.*128-row"):
            make_zepplin_topology((130,), self.world_size, False, threshold=4096)
        malformed = PlannerTopology(
            "br_pbs",
            True,
            self.world_size,
            (256,),
            (TopologySample(0, 256, 256, 2, 0),),
            0.0,
        )
        with self.assertRaisesRegex(ValueError, r"br_pbs.*local halves"):
            validate_fused_metadata(malformed)


if __name__ == "__main__":
    unittest.main()
