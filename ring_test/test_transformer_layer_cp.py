"""CPU-only metadata and autograd tests for the single-layer CP benchmark."""

from __future__ import annotations

import unittest

import torch

from ring_test.benchmark_transformer_layer import select_cuda_critical_rank_timing
from ring_test.ring_common import get_half_index, selector_to_row_indices
from ring_test.transformer_layer_cp import (
    METHOD_ORDER,
    build_physical_layout,
    explicit_cp_attention,
    parse_methods,
    parse_sm_configs,
)


GLOBAL_LENGTHS = (20480, 12288, 18432, 26624, 28672, 18432, 3584, 2560)
RING_SIZES = (8, 8, 4, 4, 4, 4, 2, 1)
RING_STARTS = (0, 0, 0, 0, 4, 4, 0, 2)


class _FakeAdapter:
    note = "CPU fake"

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        self.q = q
        return q + k + v

    def backward(
        self, dout: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return 2 * dout, 3 * dout, 4 * dout


class TransformerLayerCPTest(unittest.TestCase):
    def test_cuda_timing_uses_one_critical_rank(self) -> None:
        # Independent FWD/BWD maxima are rank 0/rank 1, but rank 1 has the
        # maximum total CUDA path. Both reported components must come from it.
        samples = torch.tensor(
            [
                [12.0, 1.0, 15.0, 7.0, 0.5],
                [10.0, 8.0, 19.5, 6.0, 5.0],
                [9.0, 7.0, 22.0, 4.0, 3.0],
            ],
            dtype=torch.float64,
        )
        selected = select_cuda_critical_rank_timing(samples)
        self.assertEqual(selected.rank, 1)
        self.assertEqual(selected.forward_ms, 10.0)
        self.assertEqual(selected.backward_ms, 8.0)
        self.assertEqual(selected.forward_ms + selected.backward_ms, 18.0)
        self.assertEqual(selected.self_attn_forward_ms, 6.0)
        self.assertEqual(selected.self_attn_backward_ms, 5.0)
        self.assertEqual(selected.wall_max_ms, 22.0)

    def test_half_selectors_become_reusable_integer_rows(self) -> None:
        single_cu = torch.tensor([0, 8], dtype=torch.int32)
        single_back = selector_to_row_indices(
            get_half_index(single_cu, front=False), 8, torch.device("cpu")
        )
        torch.testing.assert_close(single_back, torch.tensor([4, 5, 6, 7]))

        packed_cu = torch.tensor([0, 4, 10], dtype=torch.int32)
        packed_front = selector_to_row_indices(
            get_half_index(packed_cu, front=True), 10, torch.device("cpu")
        )
        packed_back = selector_to_row_indices(
            get_half_index(packed_cu, front=False), 10, torch.device("cpu")
        )
        torch.testing.assert_close(
            packed_front, torch.tensor([0, 1, 4, 5, 6])
        )
        torch.testing.assert_close(
            packed_back, torch.tensor([2, 3, 7, 8, 9])
        )

    def test_method_parser_is_stable(self) -> None:
        self.assertEqual(parse_methods("all"), list(METHOD_ORDER))
        self.assertEqual(
            parse_methods("fa3_ring,fa3_ring,zeppelin"),
            ["fa3_ring", "zeppelin"],
        )

    def test_default_and_explicit_sm_configs(self) -> None:
        self.assertEqual(parse_sm_configs(None, 78)[0].label, "70:8")
        self.assertEqual(
            [config.label for config in parse_sm_configs("64:8,60:12", 78)],
            ["64:8", "60:12"],
        )

    def test_all_cpu_layouts(self) -> None:
        for method in METHOD_ORDER:
            with self.subTest(method=method):
                layout = build_physical_layout(
                    method,
                    GLOBAL_LENGTHS,
                    RING_SIZES,
                    RING_STARTS,
                    8,
                )
                self.assertEqual(layout.method, method)
                self.assertGreaterEqual(layout.execution_tokens, layout.original_tokens)
                if method != "magi_attention":
                    self.assertEqual(len(layout.rank_token_loads), 8)
                    self.assertTrue(all(tokens > 0 for tokens in layout.rank_token_loads))
        mega = build_physical_layout(
            "mega_ring_all_cp",
            GLOBAL_LENGTHS,
            RING_SIZES,
            RING_STARTS,
            8,
        )
        self.assertTrue(all(length % 2048 == 0 for length in mega.execution_lengths))

    def test_cp4_smoke_layout(self) -> None:
        layout = build_physical_layout(
            "mega_ring_hybrid",
            (8192, 4096, 2048, 1024),
            (4, 2, 1, 1),
            (0, 0, 2, 3),
            4,
        )
        self.assertEqual(layout.rank_token_loads, (4096, 4096, 4096, 3072))

    def test_explicit_autograd_bridge(self) -> None:
        q = torch.randn(3, 2, requires_grad=True)
        k = torch.randn(3, 2, requires_grad=True)
        v = torch.randn(3, 2, requires_grad=True)
        out = explicit_cp_attention(q, k, v, _FakeAdapter())
        out.backward(torch.ones_like(out))
        torch.testing.assert_close(q.grad, torch.full_like(q, 2.0))
        torch.testing.assert_close(k.grad, torch.full_like(k, 3.0))
        torch.testing.assert_close(v.grad, torch.full_like(v, 4.0))


if __name__ == "__main__":
    unittest.main()
