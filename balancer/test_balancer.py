"""CPU-only tests for dataset sampling and hierarchical ring balancing."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import types
import unittest
from unittest import mock

import balancer
from balancer.sampler import _align_sequence_length
from ring_test import (
    benchmark_hybrid_dataset_backward,
    benchmark_hybrid_dataset_forward,
)


class DatasetSamplerTest(unittest.TestCase):
    def test_alignment_boundaries_round_up(self) -> None:
        expected = {
            1: 512,
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
            (1025, 2),
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

    def test_sampling_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown dataset"):
            balancer.generate_dataset_lengths("missing", 1024, seed=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            balancer.generate_dataset_lengths("arxiv", 0, seed=0)


class HierarchicalLoadBalancerTest(unittest.TestCase):
    def test_causal_placements_and_rank_totals_are_consistent(self) -> None:
        lengths = [512, 1536, 5120, 10240, 18432, 24576]
        workload = balancer.assign_hierarchical_rings(
            lengths,
            world_size=8,
            is_causal=True,
            balance_tolerance=0.05,
            local_search_passes=2,
        )

        expected_tokens = [0] * 8
        expected_compute = [0.0] * 8
        expected_communication = [0.0] * 8
        for length, ring_size, ring_start in zip(
            workload.global_lengths,
            workload.ring_sizes,
            workload.ring_starts,
        ):
            self.assertIn(ring_size, balancer.RING_SIZES)
            self.assertLessEqual(ring_size, 8)
            self.assertEqual(ring_start % ring_size, 0)
            self.assertLessEqual(ring_start + ring_size, 8)
            if ring_size > 1:
                self.assertEqual(length % (256 * ring_size), 0)

            token_increment = length // ring_size
            compute_increment = balancer.attention_compute(length, True) / ring_size
            communication_increment = balancer.ring_communication_per_rank(
                length, ring_size
            )
            for rank in range(ring_start, ring_start + ring_size):
                expected_tokens[rank] += token_increment
                expected_compute[rank] += compute_increment
                expected_communication[rank] += communication_increment

        self.assertEqual(workload.rank_tokens, expected_tokens)
        self.assertEqual(workload.rank_compute, expected_compute)
        self.assertEqual(workload.rank_communication, expected_communication)

    def test_workload_generation_targets_expected_ring_tiers(self) -> None:
        cases = (
            (1025, 1536, 2),
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
                    balance_tolerance=0.05,
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
            balance_tolerance=0.05,
        )
        self.assertEqual(
            balancer.make_workload(**kwargs),
            balancer.make_workload(**kwargs),
        )

    def test_rejects_unsupported_world_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "2, 4, or 8"):
            balancer.assign_hierarchical_rings(
                [2048],
                world_size=3,
                is_causal=True,
                balance_tolerance=0.05,
            )


class DatasetBenchmarkFrontendTest(unittest.TestCase):
    def test_print_workload_uses_balancer_without_cuda(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            benchmark_hybrid_dataset_forward.main(
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

    def test_backward_print_workload_uses_balancer_without_cuda(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            benchmark_hybrid_dataset_backward.main(
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
            {"benchmark_hybrid_backward": fake_benchmark},
        ), contextlib.redirect_stdout(io.StringIO()):
            benchmark_hybrid_dataset_backward.main(
                [
                    "--dataset",
                    "arxiv",
                    "--target-tokens",
                    "4097",
                    "--seed",
                    "0",
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


if __name__ == "__main__":
    unittest.main()
