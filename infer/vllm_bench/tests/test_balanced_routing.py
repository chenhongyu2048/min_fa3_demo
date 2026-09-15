"""Verify expert and EP balance, real router dispatch, and graph replay."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and vLLM")
class BalancedRoutingTest(unittest.TestCase):
    def test_balance_and_distinct_experts(self):
        from min_fa3_vllm_plugin.balanced_routing import balanced_routes

        for ep in (4, 8):
            for batch in (1, 2, 7, 16, 17, 65, 4096):
                for dtype in (torch.int32, torch.int64):
                    with self.subTest(ep=ep, batch=batch, dtype=dtype):
                        weights, ids = balanced_routes(batch, 128, 8, ep, "cuda", dtype)
                        ids = ids.cpu().long()
                        self.assertEqual(ids.shape, (batch, 8))
                        self.assertTrue(bool(((ids >= 0) & (ids < 128)).all()))
                        ordered = ids.sort(-1).values
                        self.assertTrue(bool((ordered[:, 1:] != ordered[:, :-1]).all()))
                        # Check every active-token prefix, including when the
                        # graph has padded its batch to a larger capture size.
                        for active in sorted({1, batch // 2 or 1, batch}):
                            counts = ids[:active].flatten().bincount(minlength=128)
                            self.assertLessEqual(int(counts.max() - counts.min()), 1)
                            per_rank = counts.reshape(ep, 128 // ep).sum(-1)
                            self.assertTrue(bool((per_rank == active * 8 // ep).all()))
                        torch.testing.assert_close(weights, torch.full_like(weights, 1 / 8))

    def test_factory_dispatch_and_graph_replay(self):
        from min_fa3_vllm_plugin.balanced_routing import register_balanced_routing
        from vllm.model_executor.layers.fused_moe.router.router_factory import (
            create_fused_moe_router,
        )

        register_balanced_routing()
        with (
            patch.dict(os.environ, {"VLLM_MOE_ROUTING_SIMULATION_STRATEGY": "min_fa3_balanced"}),
            patch("min_fa3_vllm_plugin.balanced_routing.get_ep_group",
                  return_value=SimpleNamespace(world_size=8)),
        ):
            router = create_fused_moe_router(top_k=8, global_num_experts=128)
            hidden = torch.randn(17, 2048, device="cuda", dtype=torch.bfloat16)
            logits = torch.randn(17, 128, device="cuda")

            def route():
                return router.select_experts(hidden, logits, torch.int32)

            expected_weights, expected_ids = route()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                weights, ids = route()
            for _ in range(3):
                hidden.normal_()
                logits.normal_()
                ids.fill_(-1)
                weights.zero_()
                graph.replay()
                torch.testing.assert_close(ids, expected_ids)
                torch.testing.assert_close(weights, expected_weights)
