"""CPU-only tests for dataset sampling and hierarchical ring balancing."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import balancer
from balancer.load_balancer import (
    _build_jobs,
    _dominates,
    _evaluate_solution,
    _repair_neighbors,
)
from balancer.sampler import _align_sequence_length, _load_bucket_counts
from dataset.build_length_bucket_stats import (
    DATASET_NAMES,
    MAX_SEQUENCE_TOKENS,
    bucket_counts,
    build_statistics,
)
from ring_test import (
    benchmark_dataset_backward,
    benchmark_dataset_forward,
)


DEMO_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = DEMO_DIR / "dataset"
BUCKET_STATS_PATH = DATASET_DIR / "sequence_length_buckets.json"


class DatasetBucketStatisticsTest(unittest.TestCase):
    def test_bucket_boundaries_and_overflow_are_right_inclusive(self) -> None:
        counts = bucket_counts([1, 256, 257, 512, 131072, 131073])
        self.assertEqual(len(counts), MAX_SEQUENCE_TOKENS // 256)
        self.assertEqual(counts[0], 2)
        self.assertEqual(counts[1], 2)
        self.assertEqual(counts[-1], 2)
        self.assertEqual(sum(counts), 6)

    def test_checked_in_statistics_match_sampled_lengths(self) -> None:
        expected = build_statistics(DATASET_DIR)
        actual = json.loads(BUCKET_STATS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(actual, expected)
        self.assertEqual(tuple(actual["datasets"]), DATASET_NAMES)
        for statistics in actual["datasets"].values():
            self.assertEqual(statistics["sample_count"], 20_000)
            self.assertEqual(sum(statistics["bucket_counts"]), 20_000)

    def test_bucket_loader_rejects_inconsistent_sample_count(self) -> None:
        payload = json.loads(BUCKET_STATS_PATH.read_text(encoding="utf-8"))
        invalid = copy.deepcopy(payload)
        invalid["datasets"]["arxiv"]["sample_count"] += 1
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "sum to 20000"):
                _load_bucket_counts(path)


class DatasetSamplerTest(unittest.TestCase):
    def test_bucket_upper_bounds_cover_256_token_intervals(self) -> None:
        self.assertEqual(len(balancer.LENGTH_BUCKETS), 512)
        self.assertEqual(balancer.LENGTH_BUCKETS[:4], (256, 512, 768, 1024))
        self.assertEqual(balancer.LENGTH_BUCKETS[-1], 128 * 1024)
        self.assertEqual(tuple(balancer.DATASET_WEIGHTS), DATASET_NAMES)

    def test_alignment_boundaries_round_up(self) -> None:
        expected = {
            1: 256,
            1025: 1280,
            4095: 4096,
            4097: 5120,
            8191: 8192,
            8193: 10240,
            16383: 16384,
            16385: 18432,
            131071: 131072,
        }
        for length, aligned in expected.items():
            with self.subTest(length=length):
                self.assertEqual(_align_sequence_length(length), aligned)

    def test_alignment_enables_target_causal_ring(self) -> None:
        cases = (
            (1025, 1),
            (2049, 2),
            (4097, 4),
            (8193, 8),
            (16385, 8),
        )
        for length, ring_size in cases:
            with self.subTest(length=length, ring_size=ring_size):
                aligned = _align_sequence_length(length)
                self.assertEqual(aligned % (256 * ring_size), 0)

    def test_sampling_is_deterministic_and_stays_within_padding_bound(self) -> None:
        for dataset in balancer.DATASET_WEIGHTS:
            with self.subTest(dataset=dataset):
                first = balancer.generate_dataset_lengths(dataset, 131073, seed=7)
                second = balancer.generate_dataset_lengths(dataset, 131073, seed=7)
                self.assertEqual(first, second)
                self.assertTrue(all(0 < length <= 128 * 1024 for length in first))
                self.assertGreaterEqual(sum(first), 131073)
                self.assertLess(sum(first) - 131073, 2048)

    def test_multiple_cases_advance_one_rng_stream(self) -> None:
        cases = balancer.generate_dataset_length_cases(
            "pile", 16385, seed=7, num_cases=3
        )
        self.assertEqual(
            cases,
            balancer.generate_dataset_length_cases(
                "pile", 16385, seed=7, num_cases=3
            ),
        )
        self.assertEqual(
            cases[0], balancer.generate_dataset_lengths("pile", 16385, seed=7)
        )
        self.assertNotEqual(
            cases[1], balancer.generate_dataset_lengths("pile", 16385, seed=8)
        )
        self.assertEqual(len(cases), 3)

    def test_multiple_cases_reject_nonpositive_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_cases must be positive"):
            balancer.generate_dataset_length_cases(
                "arxiv", 4096, seed=0, num_cases=0
            )

    def test_sampling_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown dataset"):
            balancer.generate_dataset_lengths("missing", 1024, seed=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            balancer.generate_dataset_lengths("arxiv", 0, seed=0)


class HierarchicalLoadBalancerTest(unittest.TestCase):
    def assert_workload_consistent(
        self, workload: balancer.HybridWorkload, is_causal: bool
    ) -> None:
        expected_tokens = [0] * len(workload.rank_tokens)
        expected_compute = [0.0] * len(workload.rank_compute)
        expected_communication = [0.0] * len(workload.rank_communication)
        active_rings: set[tuple[int, int]] = set()
        communication_cost = 0.0
        for length, ring_size, ring_start in zip(
            workload.global_lengths,
            workload.ring_sizes,
            workload.ring_starts,
        ):
            self.assertIn(ring_size, balancer.RING_SIZES)
            self.assertEqual(ring_start % ring_size, 0)
            self.assertLessEqual(ring_start + ring_size, len(workload.rank_tokens))
            if is_causal and ring_size > 1:
                self.assertEqual(length % (256 * ring_size), 0)
            if not is_causal:
                self.assertEqual(length % ring_size, 0)

            token_increment = length // ring_size
            compute_increment = balancer.attention_compute(length, is_causal) / ring_size
            communication_increment = balancer.ring_communication_per_rank(
                length, ring_size
            )
            for rank in range(ring_start, ring_start + ring_size):
                expected_tokens[rank] += token_increment
                expected_compute[rank] += compute_increment
                expected_communication[rank] += communication_increment
            communication_cost += communication_increment
            active_rings.add((ring_size, ring_start))

        self.assertEqual(workload.rank_tokens, expected_tokens)
        self.assertEqual(workload.rank_compute, expected_compute)
        self.assertEqual(workload.rank_communication, expected_communication)
        self.assertEqual(workload.communication_cost, communication_cost)
        self.assertEqual(workload.active_ring_count, len(active_rings))
        expected_violation = max(
            0.0,
            workload.compute_deviation - workload.compute_balance_tolerance,
            workload.token_deviation - workload.token_balance_tolerance,
        )
        self.assertAlmostEqual(workload.load_violation, expected_violation)
        self.assertEqual(workload.feasible, expected_violation <= 1e-12)

    def test_causal_placements_and_rank_totals_are_consistent(self) -> None:
        lengths = [512, 1536, 5120, 10240, 18432, 24576]
        workload = balancer.assign_hierarchical_rings(
            lengths,
            world_size=8,
            is_causal=True,
            compute_balance_tolerance=0.05,
            max_repair_iterations=2,
        )
        self.assert_workload_consistent(workload, is_causal=True)

    def test_noncausal_placements_and_rank_totals_are_consistent(self) -> None:
        workload = balancer.assign_hierarchical_rings(
            [256, 768, 2048, 5120, 8192],
            world_size=4,
            is_causal=False,
            beam_width=32,
            finalist_count=4,
            max_repair_iterations=4,
        )
        self.assert_workload_consistent(workload, is_causal=False)

    def test_buddy_tree_has_all_fifteen_candidates_for_g8_sequence(self) -> None:
        length = 16 * 1024
        compute = balancer.attention_compute(length, True)
        jobs = _build_jobs(
            [length],
            world_size=8,
            is_causal=True,
            average_compute=compute / 8,
            average_tokens=length / 8,
            compute_tolerance=0.05,
            token_tolerance=0.10,
        )
        self.assertEqual(jobs[0].legal_sizes, (1, 2, 4, 8))
        self.assertEqual(len(jobs[0].candidates), 15)
        self.assertEqual(jobs[0].minimum_ring_size, 8)

    def test_pareto_dominance_requires_no_worse_metric(self) -> None:
        self.assertTrue(_dominates((0.0, 1.0, 2), (0.0, 1.5, 2)))
        self.assertFalse(_dominates((0.0, 2.0, 1), (0.0, 1.0, 2)))
        self.assertFalse(_dominates((1.0, 2.0), (1.0, 2.0)))

    def test_short_sequences_stay_local_when_a_feasible_plan_exists(self) -> None:
        workload = balancer.assign_hierarchical_rings(
            [512] * 8 + [4096, 4096],
            world_size=8,
            is_causal=True,
        )
        self.assertTrue(workload.feasible)
        placements = list(zip(workload.global_lengths, workload.ring_sizes))
        self.assertTrue(all(ring_size == 1 for length, ring_size in placements if length == 512))
        self.assertEqual(workload.split_counts[0], 0)

    def test_progressive_unlock_splits_longer_filler_before_shortest(self) -> None:
        workload = balancer.assign_hierarchical_rings(
            [512, 512, 2560, 20480, 18432, 12288, 28672, 26624],
            world_size=8,
            is_causal=True,
        )
        self.assertTrue(workload.feasible)
        self.assertEqual(workload.relaxation_label, "unlock 2K-4K G2")
        placements = list(zip(workload.global_lengths, workload.ring_sizes))
        self.assertTrue(all(ring_size == 1 for length, ring_size in placements if length == 512))
        self.assertIn((2560, 2), placements)
        self.assertEqual(workload.split_counts[0], 0)

    def test_infeasible_topology_returns_lowest_violation_plan(self) -> None:
        workload = balancer.assign_hierarchical_rings(
            [1280], world_size=8, is_causal=True
        )
        self.assertFalse(workload.feasible)
        self.assertGreater(workload.load_violation, 0.0)
        self.assertEqual(workload.ring_sizes, [1])
        self.assert_workload_consistent(workload, is_causal=True)

    def test_repair_neighborhood_contains_sibling_demotion(self) -> None:
        lengths = [4096, 4096]
        total_compute = sum(balancer.attention_compute(length, True) for length in lengths)
        jobs = _build_jobs(
            lengths,
            world_size=4,
            is_causal=True,
            average_compute=total_compute / 4,
            average_tokens=sum(lengths) / 4,
            compute_tolerance=0.05,
            token_tolerance=0.10,
        )
        placements = [
            next(
                candidate
                for candidate in job.candidates
                if candidate.ring_size == 4 and candidate.ring_start == 0
            )
            for job in jobs
        ]
        solution = _evaluate_solution(
            placements,
            jobs,
            world_size=4,
            average_compute=total_compute / 4,
            average_tokens=sum(lengths) / 4,
            compute_tolerance=0.05,
            token_tolerance=0.10,
        )
        allowed = {job.original_index: job.candidates for job in jobs}
        neighbors = list(_repair_neighbors(solution, jobs, allowed))
        self.assertTrue(
            any(
                len(changes) == 2
                and {placement.ring_size for _, placement in changes} == {2}
                and {placement.ring_start for _, placement in changes} == {0, 2}
                for changes in neighbors
            )
        )

    def test_repair_neighborhood_contains_singleton_and_ring_moves(self) -> None:
        def build_case(
            lengths: list[int],
            world_size: int,
            placement_keys: list[tuple[int, int]],
        ):
            total_compute = sum(
                balancer.attention_compute(length, True) for length in lengths
            )
            average_compute = total_compute / world_size
            average_tokens = sum(lengths) / world_size
            jobs = _build_jobs(
                lengths,
                world_size=world_size,
                is_causal=True,
                average_compute=average_compute,
                average_tokens=average_tokens,
                compute_tolerance=0.05,
                token_tolerance=0.10,
            )
            placements = [
                next(
                    candidate
                    for candidate in job.candidates
                    if (candidate.ring_size, candidate.ring_start) == key
                )
                for job, key in zip(jobs, placement_keys)
            ]
            solution = _evaluate_solution(
                placements,
                jobs,
                world_size,
                average_compute,
                average_tokens,
                compute_tolerance=0.05,
                token_tolerance=0.10,
            )
            allowed = {job.original_index: job.candidates for job in jobs}
            return placements, list(_repair_neighbors(solution, jobs, allowed))

        local_placements, local_neighbors = build_case(
            [4096, 512], 2, [(1, 0), (1, 1)]
        )
        self.assertTrue(
            any(
                len(changes) == 1
                and changes[0][1].ring_size == 1
                and changes[0][1].ring_start
                != local_placements[changes[0][0]].ring_start
                for changes in local_neighbors
            )
        )
        self.assertTrue(
            any(
                len(changes) == 2
                and all(placement.ring_size == 1 for _, placement in changes)
                for changes in local_neighbors
            )
        )

        _, ring_neighbors = build_case(
            [4096, 4096], 4, [(2, 0), (1, 0)]
        )
        first_job_targets = {
            (placement.ring_size, placement.ring_start)
            for changes in ring_neighbors
            for index, placement in changes
            if index == 0
        }
        self.assertIn((2, 2), first_job_targets)
        self.assertIn((4, 0), first_job_targets)

        _, demotion_neighbors = build_case(
            [4096, 4096], 4, [(4, 0), (1, 0)]
        )
        self.assertTrue(
            any(
                index == 0 and placement.ring_size == 2
                for changes in demotion_neighbors
                for index, placement in changes
            )
        )

    def test_workload_generation_targets_expected_ring_tiers(self) -> None:
        cases = (
            (1025, 1280, 1),
            (4097, 5120, 4),
            (8193, 10240, 8),
            (16385, 18432, 8),
        )
        for target_tokens, aligned_length, ring_size in cases:
            with self.subTest(target_tokens=target_tokens):
                workload = balancer.make_workload(
                    dataset="arxiv",
                    target_tokens=target_tokens,
                    seed=0,
                    world_size=8,
                    mode="causal",
                    compute_balance_tolerance=0.05,
                )
                self.assertEqual(workload.global_lengths, [aligned_length])
                self.assertEqual(workload.ring_sizes, [ring_size])

    def test_planner_is_deterministic(self) -> None:
        kwargs = dict(
            dataset="github",
            target_tokens=32769,
            seed=3,
            world_size=8,
            mode="causal",
            compute_balance_tolerance=0.05,
        )
        self.assertEqual(
            balancer.make_workload(**kwargs),
            balancer.make_workload(**kwargs),
        )

    def test_multi_workload_first_case_matches_single_workload(self) -> None:
        kwargs = dict(
            dataset="pile",
            target_tokens=16385,
            seed=7,
            world_size=8,
            mode="causal",
            compute_balance_tolerance=0.05,
        )
        workloads = balancer.make_workloads(**kwargs, num_cases=2)
        self.assertEqual(workloads[0], balancer.make_workload(**kwargs))
        self.assertNotEqual(workloads[0].global_lengths, workloads[1].global_lengths)

    def test_rejects_unsupported_world_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "2, 4, or 8"):
            balancer.assign_hierarchical_rings(
                [2048],
                world_size=3,
                is_causal=True,
                compute_balance_tolerance=0.05,
            )


class DatasetBenchmarkFrontendTest(unittest.TestCase):
    def test_pile_print_workload_uses_json_distribution(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            benchmark_dataset_forward.main(
                [
                    "--dataset",
                    "pile",
                    "--target-tokens",
                    "4097",
                    "--seed",
                    "0",
                    "--world-size",
                    "8",
                    "--print-workload",
                ]
            )
        rendered = output.getvalue()
        self.assertIn("dataset=pile", rendered)
        self.assertIn("actual_tokens=4352", rendered)
        self.assertIn("Planner status: feasible=", rendered)
        self.assertIn("Split protection", rendered)

    def test_print_workload_uses_balancer_without_cuda(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            benchmark_dataset_forward.main(
                [
                    "--dataset",
                    "arxiv",
                    "--target-tokens",
                    "4097",
                    "--seed",
                    "0",
                    "--world-size",
                    "8",
                    "--print-workload",
                ]
            )
        rendered = output.getvalue()
        self.assertIn("actual_tokens=5120", rendered)
        self.assertIn("global_seqlens=5120", rendered)
        self.assertIn("ring_sizes=4", rendered)

    def test_print_workload_samples_multiple_cases_from_one_seed(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            benchmark_dataset_forward.main(
                [
                    "--dataset",
                    "pile",
                    "--target-tokens",
                    "16385",
                    "--seed",
                    "7",
                    "--num-cases",
                    "2",
                    "--world-size",
                    "8",
                    "--print-workload",
                ]
            )
        rendered = output.getvalue()
        self.assertIn("Dataset case 1/2", rendered)
        self.assertIn("Dataset case 2/2", rendered)
        self.assertEqual(rendered.count("Planner workload: dataset=pile, seed=7"), 2)

    def test_forward_frontend_forwards_all_cases_in_one_call(self) -> None:
        calls: list[tuple[list[str], dict[str, object]]] = []

        def fake_main(argv, **kwargs) -> None:
            calls.append((list(argv), dict(kwargs)))

        fake_benchmark = types.SimpleNamespace(main=fake_main)
        with mock.patch.dict(
            os.environ,
            {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "8"},
        ), mock.patch.dict(
            sys.modules,
            {"benchmark_topology_forward": fake_benchmark},
        ), contextlib.redirect_stdout(io.StringIO()):
            benchmark_dataset_forward.main(
                [
                    "--dataset",
                    "pile",
                    "--target-tokens",
                    "16385",
                    "--seed",
                    "7",
                    "--num-cases",
                    "2",
                    "--no-check",
                ]
            )

        self.assertEqual(len(calls), 1)
        _forwarded, options = calls[0]
        workload_cases = options["workload_cases"]
        self.assertEqual(len(workload_cases), 2)
        self.assertEqual(workload_cases[0].case_index, 0)
        self.assertEqual(workload_cases[1].case_index, 1)
        self.assertNotEqual(
            workload_cases[0].global_lengths,
            workload_cases[1].global_lengths,
        )

    def test_backward_print_workload_uses_balancer_without_cuda(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            benchmark_dataset_backward.main(
                [
                    "--dataset",
                    "arxiv",
                    "--target-tokens",
                    "4097",
                    "--seed",
                    "0",
                    "--world-size",
                    "8",
                    "--print-workload",
                ]
            )
        rendered = output.getvalue()
        self.assertIn("mode=causal", rendered)
        self.assertIn("actual_tokens=5120", rendered)
        self.assertIn("global_seqlens=5120", rendered)
        self.assertIn("ring_sizes=4", rendered)

    def test_backward_frontend_forwards_explicit_topology(self) -> None:
        forwarded: list[str] = []
        options: dict[str, object] = {}

        def fake_main(argv, **kwargs) -> None:
            forwarded.extend(argv)
            options.update(kwargs)

        fake_benchmark = types.SimpleNamespace(
            main=fake_main
        )
        with mock.patch.dict(
            os.environ,
            {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "8"},
        ), mock.patch.dict(
            sys.modules,
            {"benchmark_topology_backward": fake_benchmark},
        ), contextlib.redirect_stdout(io.StringIO()):
            benchmark_dataset_backward.main(
                [
                    "--dataset",
                    "arxiv",
                    "--target-tokens",
                    "4097",
                    "--seed",
                    "0",
                    "--num-cases",
                    "2",
                    "--sm-configs",
                    "100:16",
                    "--no-check",
                ]
            )

        def value_after(flag: str) -> str:
            return forwarded[forwarded.index(flag) + 1]

        self.assertEqual(value_after("--global-seqlens"), "5120")
        self.assertEqual(value_after("--ring-sizes"), "4")
        self.assertEqual(value_after("--ring-starts"), "0")
        self.assertEqual(value_after("--sm-configs"), "100:16")
        self.assertEqual(value_after("--methods"), "all")
        self.assertIn("--no-check", forwarded)
        self.assertTrue(options["skip_incompatible_methods"])
        self.assertEqual(len(options["workload_cases"]), 2)


if __name__ == "__main__":
    unittest.main()
