"""CPU-only metadata and autograd tests for the single-layer CP benchmark."""

from __future__ import annotations

import gc
import sys
import unittest
import weakref
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

from ring_test import benchmark_transformer_layer as benchmark
from ring_test.benchmark_transformer_layer import select_cuda_critical_rank_timing
from ring_test.ring_common import get_half_index, selector_to_row_indices
from ring_test.transformer_layer_cp import (
    METHOD_ORDER,
    SmConfig,
    _MegaRingAdapter,
    _RunnerAdapter,
    _empty_qkv,
    _uniform_topk_routing,
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


class _CachedOutputRunner(_FakeAdapter):
    def __init__(self) -> None:
        self.out = torch.zeros(3, 2)
        # Llama3 retains slices of its output in forward_out.
        self.forward_out = [self.out[:1], self.out[1:]]

    def bind_inputs(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        pass

    def forward(self) -> torch.Tensor:
        return self.out


class _CPUMegaRingAdapter(_FakeAdapter):
    """Exercise the real MegaRing forward wrapper with a CPU kernel stub."""

    forward = _MegaRingAdapter.forward

    def __init__(self) -> None:
        self.out = torch.zeros(3, 2)
        self.lse = torch.empty(0)
        self.remote_k = SimpleNamespace(data_=None)
        self.remote_v = SimpleNamespace(data_=None)
        self.cu = self.cu_host = None
        self.global_host = self.ring_sizes_host = self.ring_starts_host = None
        self.max_local_len = 3
        self.sm_config = SmConfig(1, 1)
        self.op = SimpleNamespace(
            forward_varlen_mega_ring=lambda *args, **kwargs: (
                kwargs["out"], kwargs["lse"]
            )
        )

    def _populate_kv(self, k: torch.Tensor, v: torch.Tensor) -> None:
        pass


class TransformerLayerCPTest(unittest.TestCase):
    def test_uniform_routing_balances_experts_and_ep_destinations(self) -> None:
        router = SimpleNamespace(config=SimpleNamespace(num_moe_experts=128), topk=8)
        for tokens in (1, 15, 16, 17, 2048):
            with self.subTest(tokens=tokens):
                logits = torch.randn(tokens, 1, 128)
                probs, route = _uniform_topk_routing(router, logits)
                self.assertTrue(torch.all(route.sum(dim=1) == 8))
                counts = route.sum(dim=0)
                self.assertLessEqual(int(counts.max() - counts.min()), 1)
                for ep in (4, 8):
                    destinations = route.reshape(tokens, ep, 128 // ep).sum(dim=2)
                    self.assertTrue(torch.all(destinations == 8 // ep))
                torch.testing.assert_close(probs.sum(dim=1), torch.ones(tokens))
                self.assertTrue(torch.all(probs[~route] == 0))
                biased_logits = logits.clone()
                biased_logits[..., :8] += 1000
                _, biased_route = _uniform_topk_routing(router, biased_logits)
                self.assertTrue(torch.equal(route, biased_route))

    def test_uniform_routing_preserves_selected_logits_gradients(self) -> None:
        router = SimpleNamespace(config=SimpleNamespace(num_moe_experts=128), topk=8)
        hidden = torch.randn(17, 12, requires_grad=True)
        gate = torch.randn(128, 12, requires_grad=True)
        logits = hidden @ gate.T
        logits.retain_grad()
        probs, route = _uniform_topk_routing(router, logits)
        expert_outputs = torch.randn_like(probs)
        (probs * expert_outputs).sum().backward()
        expected = probs * (expert_outputs - (probs * expert_outputs).sum(dim=1, keepdim=True))
        torch.testing.assert_close(logits.grad, expected)
        self.assertTrue(torch.all(logits.grad[~route] == 0))
        for tensor in (hidden, gate):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0)

    def test_core_timing_waits_for_all_qkv_gradients(self) -> None:
        events: list[str] = []

        class Event:
            def __init__(self, name: str) -> None:
                self.name = name

            def record(self) -> None:
                events.append(self.name)

            def elapsed_time(self, end: Any) -> float:
                return float(events.index(end.name) - events.index(self.name))

        class Core(torch.nn.Module):
            def forward(self, q, k, v):
                return q.square() + k.pow(3) + v.pow(4)

        core = Core()
        q, k, v = [torch.randn(3, 2, requires_grad=True) for _ in range(3)]
        for name, tensor in zip(("dq", "dk", "dv"), (q, k, v)):
            tensor.register_hook(lambda grad, name=name: events.append(name))
        probe = benchmark.CoreAttentionTimingProbe(core)
        try:
            # Reuse the same inputs to catch accumulated timing hooks.
            for _ in range(2):
                events.clear()
                for tensor in (q, k, v):
                    tensor.grad = None
                with patch.object(torch.cuda, "Event", side_effect=[
                    Event("forward_start"), Event("forward_end"),
                    Event("backward_start"), Event("backward_end"),
                ]):
                    probe.begin()
                    out = core(q, k, v)
                    out.sum().backward()
                    forward_ms, backward_ms = probe.finish()
                self.assertEqual(events[:3], [
                    "forward_start", "forward_end", "backward_start"
                ])
                self.assertCountEqual(events[3:-1], ["dq", "dk", "dv"])
                self.assertEqual(events[-1], "backward_end")
                self.assertGreater(forward_ms, 0)
                self.assertGreater(backward_ms, 0)
                torch.testing.assert_close(q.grad, 2 * q)
                torch.testing.assert_close(k.grad, 3 * k.square())
                torch.testing.assert_close(v.grad, 4 * v.pow(3))
        finally:
            probe.close()

    def test_qwen_inputs_keep_attention_width_independent_of_hidden_size(self) -> None:
        hidden, dout = benchmark._make_inputs(8, torch.device("cpu"), 0)
        q, k, v, attention_dout = _empty_qkv(8, torch.device("cpu"))
        self.assertEqual(hidden.shape, (8, 1, 2048))
        self.assertEqual(dout.shape, hidden.shape)
        self.assertEqual(q.shape, (8, 32, 128))
        self.assertEqual(k.shape, (8, 4, 128))
        self.assertEqual(v.shape, k.shape)
        self.assertEqual(attention_dout.shape, q.shape)

    def test_runner_with_cached_output_views_is_released(self) -> None:
        runner = _CachedOutputRunner()
        adapter = _RunnerAdapter(runner, True)
        runner_ref, adapter_ref = weakref.ref(runner), weakref.ref(adapter)
        for _ in range(2):
            q, k, v = [torch.randn(3, 2, requires_grad=True) for _ in range(3)]
            out = explicit_cp_attention(q, k, v, adapter)
            self.assertEqual(out.data_ptr(), runner.out.data_ptr())
            out.backward(torch.ones_like(out))
            torch.testing.assert_close(q.grad, torch.full_like(q, 2.0))
            torch.testing.assert_close(k.grad, torch.full_like(k, 3.0))
            torch.testing.assert_close(v.grad, torch.full_like(v, 4.0))
            del out, q, k, v
        del adapter, runner
        self.assertIsNone(adapter_ref())
        self.assertIsNone(runner_ref())

    def test_run_preserves_error_without_cleanup_collectives(self) -> None:
        args = benchmark.parse_args([
            "--dataset", "arxiv", "--output-jsonl", "unused.jsonl",
            "--methods", "llama3_allgather_attention",
        ])
        cp = SimpleNamespace(
            MEGA_RING_METHODS=frozenset(),
            build_megatron_layer=Mock(),
            build_physical_layout=Mock(),
            make_packed_seq_params=Mock(),
            parse_methods=lambda spec: spec.split(","),
            prepare_method=Mock(side_effect=ValueError("original failure")),
        )
        modules = {
            "transformer_layer_cp": cp,
            "baseline.megatron_hybrid_cp": SimpleNamespace(
                create_hybrid_cp_process_groups=Mock()
            ),
            "megatron.core": SimpleNamespace(parallel_state=Mock()),
        }
        setup_results = {
            "_init_distributed": (1, 8, torch.device("cpu")),
            "_collect_device_inventory": [],
            "_resolve_sm_configs_collectively": [],
            "_preflight": ("min_fa3", {}),
            "_initialize_megatron": None,
            "_all_rank_preflight": None,
            "_broadcast_cases": [benchmark.Case(0, (2048,), (8,), (0,))],
        }
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.dict("os.environ", {"LOCAL_RANK": "1"}))
            for name, result in setup_results.items():
                stack.enter_context(patch.object(benchmark, name, return_value=result))
            stack.enter_context(patch.object(benchmark.dist, "broadcast_object_list"))
            stack.enter_context(patch.object(benchmark.dist, "is_initialized", return_value=True))
            barrier = stack.enter_context(patch.object(
                benchmark.dist, "barrier", side_effect=RuntimeError("cleanup barrier")
            ))
            empty_cache = stack.enter_context(patch.object(torch.cuda, "empty_cache"))
            destroy = stack.enter_context(patch.object(benchmark, "_destroy_distributed_state"))
            with self.assertRaisesRegex(ValueError, "original failure"):
                benchmark._run(args)
            barrier.assert_not_called()
            empty_cache.assert_not_called()
            destroy.assert_not_called()

    def test_mega_ring_output_reuse_releases_adapter_without_gc(self) -> None:
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            adapter = _CPUMegaRingAdapter()
            adapter_ref = weakref.ref(adapter)
            for _ in range(2):
                q, k, v = [torch.randn(3, 2, requires_grad=True) for _ in range(3)]
                out = explicit_cp_attention(q, k, v, adapter)
                self.assertEqual(out.data_ptr(), adapter.out.data_ptr())
                out.backward(torch.ones_like(out))
                torch.testing.assert_close(q.grad, torch.full_like(q, 2.0))
                torch.testing.assert_close(k.grad, torch.full_like(k, 3.0))
                torch.testing.assert_close(v.grad, torch.full_like(v, 4.0))
                del out, q, k, v
            del adapter
            self.assertIsNone(adapter_ref())
        finally:
            if gc_enabled:
                gc.enable()

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
        self.assertEqual(selected.core_attn_forward_ms, 6.0)
        self.assertEqual(selected.core_attn_backward_ms, 5.0)
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
