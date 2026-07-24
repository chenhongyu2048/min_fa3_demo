import unittest

import torch

from ring_test.forward_ablation import (
    PROFILES,
    build_hierarchy,
    canonicalize_lengths,
    profile_dispatch_manifest,
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
