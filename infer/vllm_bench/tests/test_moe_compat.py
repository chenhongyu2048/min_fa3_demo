"""Focused CUDA checks for the pinned wheel's Qwen3 routing adapter."""

import unittest

import torch

from min_fa3_vllm_plugin.moe_compat import install_moe_compat


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and the vLLM wheel")
class MoECompatTest(unittest.TestCase):
    def test_topk_matches_reference_and_graph_replay(self):
        import vllm._custom_ops as ops

        install_moe_compat()
        for batch in (1, 16, 65):
            logits = torch.randn(batch, 128, device="cuda", dtype=torch.float32)
            weights = torch.empty(batch, 8, device="cuda", dtype=torch.float32)
            ids = torch.empty(batch, 8, device="cuda", dtype=torch.int32)
            token_experts = torch.empty_like(ids)

            def route():
                ops.topk_softmax(weights, ids, token_experts, logits, True)

            route()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                route()
            for _ in range(3):
                logits.normal_()
                graph.replay()
                expected_weights, expected_ids = logits.softmax(-1).topk(8)
                expected_weights /= expected_weights.sum(-1, keepdim=True)
                torch.testing.assert_close(ids.long(), expected_ids)
                torch.testing.assert_close(weights, expected_weights)

    def test_old_wheel_rejects_padding_mask(self):
        import vllm._custom_ops as ops

        if len(torch.ops._moe_C.topk_softmax.default._schema.arguments) != 6:
            self.skipTest("matching wheel supports padding masks")
        install_moe_compat()
        with self.assertRaisesRegex(ValueError, "VLLM_MOE_SKIP_PADDING=0"):
            ops.topk_softmax(None, None, None, None, is_padding=object())
