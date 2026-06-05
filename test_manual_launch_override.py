import unittest

import torch

import _min_fa3_op
import min_fa3_op


class ManualLaunchOverrideTest(unittest.TestCase):
    def test_forward_rejects_invalid_manual_override_before_cuda_checks(self) -> None:
        q = torch.zeros((1, 128, 1, 128), dtype=torch.bfloat16)
        k = torch.zeros((1, 128, 1, 128), dtype=torch.bfloat16)
        v = torch.zeros((1, 128, 1, 128), dtype=torch.bfloat16)

        with self.assertRaises(Exception) as exc_info:
            min_fa3_op.forward(q, k, v, False, manual_block_count=0)
        self.assertIn("manual_block_count must be greater than 0", str(exc_info.exception))

    def test_forward_varlen_rejects_invalid_manual_override_before_cuda_checks(self) -> None:
        q = torch.zeros((128, 1, 128), dtype=torch.bfloat16)
        k = torch.zeros((128, 1, 128), dtype=torch.bfloat16)
        v = torch.zeros((128, 1, 128), dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, 128], dtype=torch.int32)

        with self.assertRaises(Exception) as exc_info:
            min_fa3_op.forward_varlen(q, k, v, cu_seqlens, cu_seqlens, 128, 128, False, manual_block_count=0)
        self.assertIn("manual_block_count must be greater than 0", str(exc_info.exception))

    def test_default_keeps_automatic_grid(self) -> None:
        self.assertEqual(_min_fa3_op._debug_resolve_launch_grid_shape(132), (132, 1, 1))

    def test_manual_override_replaces_grid_x(self) -> None:
        self.assertEqual(
            _min_fa3_op._debug_resolve_launch_grid_shape(132, manual_block_count=7),
            (7, 1, 1),
        )

    def test_invalid_manual_override_raises(self) -> None:
        for invalid_value in (0, -3):
            with self.subTest(manual_block_count=invalid_value):
                with self.assertRaises(Exception) as exc_info:
                    _min_fa3_op._debug_resolve_launch_grid_shape(132, manual_block_count=invalid_value)
                self.assertIn("manual_block_count must be greater than 0", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()
