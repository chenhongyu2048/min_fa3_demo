import unittest

import torch

from ring_test.forward_ablation import (
    PROFILES,
    build_hierarchy,
    canonicalize_lengths,
    local_lengths_for_rank,
    profile_dispatch_manifest,
    profiles_for_world_size,
)


class ForwardAblationProfileTest(unittest.TestCase):
    def test_six_profile_dispatch(self) -> None:
        manifest = profile_dispatch_manifest()
        self.assertEqual([entry["id"] for entry in manifest], list(range(1, 7)))
        self.assertEqual(manifest[0]["attention_launches"], 8)
        self.assertEqual(manifest[0]["reduction_launches"], 8)
        self.assertEqual(manifest[1]["attention_launches"], 8)
        self.assertEqual(manifest[1]["reduction_launches"], 0)
        self.assertEqual(manifest[2]["scheduler"], manifest[3]["scheduler"])
        self.assertFalse(manifest[2]["recycle_comm"])
        self.assertTrue(manifest[3]["recycle_comm"])
        self.assertEqual(manifest[4]["executor"], manifest[5]["executor"])
        self.assertEqual(manifest[4]["scheduler"], manifest[5]["scheduler"])
        self.assertEqual(len(PROFILES), 6)

    def test_canonical_lengths_are_aligned_once(self) -> None:
        self.assertEqual(
            canonicalize_lengths((1, 2048, 2049, 4096)),
            (2048, 2048, 4096, 4096),
        )

    def test_small_world_runs_all_profiles(self) -> None:
        for world_size in (2, 4):
            with self.subTest(world_size=world_size):
                profiles = profiles_for_world_size(world_size)
                self.assertEqual(
                    [profile.id for profile in profiles], list(range(1, 7)),
                )
                self.assertEqual(profiles[0].kernel_launches, 2 * world_size)
                self.assertEqual(profiles[1].kernel_launches, world_size)
                self.assertEqual(
                    [profile.kernel_launches for profile in profiles[2:]], [1] * 4,
                )
                manifest = profile_dispatch_manifest(world_size)
                self.assertEqual(manifest[0]["attention_launches"], world_size)
                self.assertEqual(manifest[0]["reduction_launches"], world_size)
                self.assertEqual(manifest[1]["attention_launches"], world_size)
                self.assertEqual(manifest[1]["reduction_launches"], 0)
                self.assertEqual(
                    [entry["scheduler"] for entry in manifest[:4]],
                    [f"causal_w{world_size}_step_linear"] * 2
                    + [f"causal_w{world_size}_linear_atomic"] * 2,
                )
                self.assertEqual(
                    [entry["topology"] for entry in manifest[:5]],
                    [f"all_cp_g{world_size}"] * 5,
                )
                self.assertEqual(
                    manifest[5]["topology"],
                    "br_pbs_g2_g1" if world_size == 2 else "br_pbs_g4_g2_g1",
                )
        self.assertEqual(profiles_for_world_size(8), PROFILES)
        self.assertEqual(profile_dispatch_manifest(8), profile_dispatch_manifest())
        with self.assertRaises(ValueError):
            profiles_for_world_size(1)

    def test_small_world_all_cp_alignment(self) -> None:
        for world_size in (2, 4):
            alignment = 256 * world_size
            with self.subTest(world_size=world_size):
                self.assertEqual(
                    canonicalize_lengths((1, alignment, alignment + 1), alignment),
                    (alignment, alignment, 2 * alignment),
                )

    def test_small_world_mixed_hierarchy(self) -> None:
        cases = (
            (2, (2, 1), ((8, 4, 2, 10), (4, 4, 4, 8))),
            (4, (4, 2, 1), ((12, 8, 4, 20), (8, 8, 8, 20),
                            (4, 4, 4, 14), (4, 4, 4, 16))),
        )
        for world_size, rings, expected_counts in cases:
            lengths = tuple(256 * ring for ring in rings)
            starts = (0,) * len(rings)
            for rank, expected in enumerate(expected_counts):
                with self.subTest(world_size=world_size, rank=rank):
                    local = local_lengths_for_rank(lengths, rings, starts, rank)
                    self.assertGreater(sum(local), 0)
                    cu_host = torch.tensor((0, *torch.tensor(local).cumsum(0).tolist()),
                                           dtype=torch.int32)
                    _, flat, summary = build_hierarchy(
                        cu_host, lengths, rings, starts, rank, q_heads=2,
                        world_size=world_size,
                    )
                    self.assertEqual(flat.numel(), 48)
                    self.assertEqual(
                        tuple(summary[key] for key in (
                            "base_work_tiles", "reduction_tiles", "remote_tiles",
                            "total_work_tiles",
                        )),
                        expected,
                    )

    def test_hierarchy_rejects_topology_outside_world(self) -> None:
        with self.assertRaisesRegex(ValueError, "ring size"):
            build_hierarchy(
                torch.tensor((0, 256), dtype=torch.int32),
                (1024,), (4,), (0,), rank=0, q_heads=2, world_size=2,
            )
        with self.assertRaisesRegex(ValueError, "ring start"):
            build_hierarchy(
                torch.tensor((0, 0), dtype=torch.int32),
                (256,), (1,), (2,), rank=0, q_heads=2, world_size=2,
            )

    def test_all_cp_hierarchy_bounds(self) -> None:
        local_lengths = (256, 512)
        cu_host = torch.tensor((0, 256, 768), dtype=torch.int32)
        half, flat, summary = build_hierarchy(
            cu_host,
            (2048, 4096),
            (8, 8),
            (0, 0),
            rank=3,
            q_heads=16,
        )
        self.assertEqual(tuple(half.tolist()), (0, 128, 384))
        self.assertEqual(flat.numel(), 48)
        self.assertEqual(summary["base_work_tiles"], 96)
        self.assertEqual(summary["reduction_tiles"], 96)
        self.assertEqual(summary["total_work_tiles"], 576)
        self.assertEqual(summary["remote_tiles"], 96)


if __name__ == "__main__":
    unittest.main()
