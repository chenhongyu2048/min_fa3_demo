"""CPU numerical checks for the CP motivation controls."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ring_test import benchmark_cp_motivation as benchmark

# Numerical tests replace the FA3 block calls and need no compiled extension.
with ExitStack() as imports:
    if importlib.util.find_spec("_min_fa3_op") is None:
        imports.enter_context(patch.dict(sys.modules, {"min_fa3_op": SimpleNamespace()}))
    from baseline.megatron_hybrid_cp.motivation import (
        MegatronHybridCPComputeAttention, _ComputeOnlyRing, _OneStepSample,
    )
    from baseline.megatron_hybrid_cp.attention import MegatronHybridCPAttention
    from baseline.megatron_hybrid_cp.plan import (
        ExecutionGroup, HybridCPPlan, HybridCPProcessGroups, SampleAssignment,
    )
    from hybrid_backward_baselines import VarlenFa3RingBackward, _BlockBackend
    from ring_common import RingComm


def _scores(q, k, causal):
    expanded = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, expanded) * q.size(2) ** -0.5
    if causal:
        rows = torch.arange(q.size(0))[:, None] + k.size(0) - q.size(0)
        scores = scores.masked_fill((torch.arange(k.size(0))[None, :] > rows).unsqueeze(0), -float("inf"))
    return scores


def _dense(q, k, v):
    scores = _scores(q, k, True)
    expanded = v.repeat_interleave(q.size(1) // v.size(1), dim=1)
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), expanded), scores.logsumexp(-1)


def _forward(backend, q, k, v, cu_q, cu_k, host_q, host_k, max_q, max_k, causal):
    out, lse = [], []
    for qs, qe, ks, ke in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:]):
        scores = _scores(q[qs:qe], k[ks:ke], causal)
        values = v[ks:ke].repeat_interleave(q.size(1) // v.size(1), dim=1)
        out.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), values))
        lse.append(scores.logsumexp(-1))
    return torch.cat(out), torch.cat(lse, dim=1)


def _backward(backend, dout, q, k, v, out, lse, cu_q, cu_k, max_q, max_k, causal, dq, dk, dv):
    repeat = q.size(1) // k.size(1)
    for qs, qe, ks, ke in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:]):
        queries, keys, values, do = q[qs:qe], k[ks:ke], v[ks:ke], dout[qs:qe]
        probabilities = (_scores(queries, keys, causal) - lse[:, qs:qe, None]).exp()
        dp = torch.einsum("qhd,khd->hqk", do, values.repeat_interleave(repeat, dim=1))
        delta = (do * out[qs:qe]).sum(-1).T.unsqueeze(-1)
        ds = probabilities * (dp - delta)
        dq[qs:qe].copy_(torch.einsum("hqk,khd->qhd", ds, keys.repeat_interleave(repeat, dim=1)) * q.size(2) ** -0.5)
        expanded_dk = torch.einsum("hqk,qhd->khd", ds, queries) * q.size(2) ** -0.5
        expanded_dv = torch.einsum("hqk,qhd->khd", probabilities, do)
        dk[ks:ke].copy_(expanded_dk.reshape(-1, k.size(1), repeat, k.size(2)).sum(2))
        dv[ks:ke].copy_(expanded_dv.reshape(-1, v.size(1), repeat, v.size(2)).sum(2))
    return dq, dk, dv


def _zigzag_parts(tensor, cp_size):
    half = tensor.size(0) // (2 * cp_size)
    return [torch.cat((tensor[rank * half:(rank + 1) * half],
                       tensor[(2 * cp_size - 1 - rank) * half:(2 * cp_size - rank) * half]))
            for rank in range(cp_size)]


class CPMotivationTest(unittest.TestCase):
    def test_controls_match_dense_causal_attention_and_gradients(self):
        torch.manual_seed(73)
        for cp_size in (2, 4, 8):
            for kv_heads in (1, 2):
                with self.subTest(cp_size=cp_size, kv_heads=kv_heads):
                    total = cp_size * 8
                    q = torch.randn(total, 2, 4, requires_grad=True)
                    k = torch.randn(total, kv_heads, 4, requires_grad=True)
                    v = torch.randn_like(k, requires_grad=True)
                    dout = torch.randn_like(q)
                    expected_out, expected_lse = _dense(q, k, v)
                    expected_grads = torch.autograd.grad(expected_out, (q, k, v), dout)
                    parts_q, parts_k, parts_v, parts_do = [
                        _zigzag_parts(tensor.detach(), cp_size) for tensor in (q, k, v, dout)
                    ]
                    expected_q, expected_k, expected_v = [
                        _zigzag_parts(tensor, cp_size) for tensor in expected_grads
                    ]
                    for runner_type in (_ComputeOnlyRing, _OneStepSample):
                        grad_k, grad_v = [], []
                        for rank in range(cp_size):
                            with ExitStack() as stack:
                                stack.enter_context(patch.object(torch.distributed, "get_rank", return_value=rank))
                                stack.enter_context(patch.object(torch.distributed, "get_world_size", return_value=cp_size))
                                stack.enter_context(patch.object(RingComm, "send_recv_kv", side_effect=AssertionError("KV transfer")))
                                stack.enter_context(patch.object(torch.distributed, "all_reduce", side_effect=AssertionError("dKV transfer")))
                                forward = stack.enter_context(patch.object(_BlockBackend, "forward_block", autospec=True, side_effect=_forward))
                                backward = stack.enter_context(patch.object(_BlockBackend, "backward_block", autospec=True, side_effect=_backward))
                                source = VarlenFa3RingBackward(
                                    None, parts_q[rank], parts_k[rank], parts_v[rank], parts_do[rank], [8], "min_fa3",
                                )
                                runner = runner_type(source, list(zip(parts_k, parts_v)))
                                # Repeated iterations must not accumulate stale dKV.
                                for _ in range(2):
                                    runner.prepare_backward()
                                    out, lse = runner.forward()
                                    dq, _, _ = runner.backward()
                                    torch.testing.assert_close(out, _zigzag_parts(expected_out, cp_size)[rank], atol=2e-6, rtol=2e-5)
                                    logical_lse = _zigzag_parts(expected_lse.T, cp_size)[rank].T
                                    torch.testing.assert_close(lse, logical_lse, atol=2e-6, rtol=2e-5)
                                    torch.testing.assert_close(dq, expected_q[rank], atol=3e-6, rtol=3e-5)
                                calls = cp_size if runner_type is _ComputeOnlyRing else 1
                                self.assertEqual(forward.call_count, 2 * calls)
                                self.assertEqual(backward.call_count, 2 * calls)
                                if runner_type is _ComputeOnlyRing:
                                    grad_k.append(runner.dk_parts.clone())
                                    grad_v.append(runner.dv_parts.clone())
                                else:
                                    grad_k.append(torch.stack(_zigzag_parts(runner.dk_global, cp_size)))
                                    grad_v.append(torch.stack(_zigzag_parts(runner.dv_global, cp_size)))
                        torch.testing.assert_close(sum(grad_k), torch.stack(expected_k), atol=4e-6, rtol=3e-5)
                        torch.testing.assert_close(sum(grad_v), torch.stack(expected_v), atol=4e-6, rtol=3e-5)

    def test_one_step_preserves_sample_order_and_group_barriers(self):
        plan = HybridCPPlan(
            (32, 16, 8, 8), 2, 16,
            (ExecutionGroup(0, ((0,), (0,))), ExecutionGroup(1, ((1, 2), (3,)))),
            (SampleAssignment(0, 32, 2, 0, 0), SampleAssignment(1, 16, 1, 0, 1),
             SampleAssignment(2, 8, 1, 0, 1), SampleAssignment(3, 8, 1, 1, 1)),
        )
        groups = HybridCPProcessGroups(None, 2, {}, ())
        q, k, v, dout = [torch.randn(40, 2, 4) for _ in range(4)]

        def gather(parts, local, group):
            for part in parts:
                part.copy_(local)

        with ExitStack() as stack:
            stack.enter_context(patch.object(MegatronHybridCPAttention, "_validate_inputs"))
            stack.enter_context(patch.object(torch.distributed, "get_rank", return_value=0))
            stack.enter_context(patch.object(torch.distributed, "get_world_size", return_value=2))
            stack.enter_context(patch.object(torch.distributed, "all_gather", side_effect=gather))
            stack.enter_context(patch.object(_BlockBackend, "forward_block", _forward))
            stack.enter_context(patch.object(_BlockBackend, "backward_block", _backward))
            barriers = stack.enter_context(patch.object(torch.distributed, "barrier"))
            # A real process group is only needed by CP; controls gather via the stub.
            stack.enter_context(patch.object(HybridCPProcessGroups, "group_for", return_value=object()))
            runner = MegatronHybridCPComputeAttention(
                plan, groups, q, k, v, True, "min_fa3", dout=dout, one_step=True,
            )
            events = []
            for sample_id, sample in runner._runners.items():
                forward, backward = sample.forward, sample.backward
                sample.forward = lambda sid=sample_id, fn=forward: (events.append(("f", sid)), fn())[1]
                sample.backward = lambda sid=sample_id, fn=backward: (events.append(("b", sid)), fn())[1]
            barriers.side_effect = lambda **kwargs: events.append("barrier")
            runner.prepare_backward()
            runner.forward_all()
            runner.backward_all()
            self.assertEqual(events, [("f", 0), "barrier", ("f", 1), ("f", 2),
                                      ("b", 0), "barrier", ("b", 1), ("b", 2)])

    def test_five_datasets_use_identical_megatron_plans(self):
        args = benchmark.parse_args(["--num-cases", "2"])
        cases = benchmark.make_cases(args)
        self.assertEqual(len(cases), 10)
        self.assertEqual(cases, benchmark.make_cases(args))
        for case in cases:
            plan = case["plan"]
            self.assertEqual(case["original_tokens"], sum(case["global_lengths"]))
            self.assertGreaterEqual(case["execution_tokens"], case["original_tokens"])
            self.assertEqual(case["rank_samples"], [list(plan.sample_ids_for_rank(r)) for r in range(8)])

    def test_summary_uses_same_cases_and_phases(self):
        records = [
            {"dataset": "arxiv", "phase": phase, "mode": mode, "latency_ms": value + case}
            for phase in benchmark.PHASES for mode, value in zip(benchmark.MODES, (10, 5, 2))
            for case in (0, 2)
        ]
        for summary in benchmark.summarize(records):
            self.assertEqual(summary["mean_ms"], {"normal": 11, "compute-only": 6, "one-step": 3})
            self.assertAlmostEqual(summary["normal_over_compute_only"], 11 / 6)
            self.assertEqual(summary["compute_only_over_one_step"], 2)


if __name__ == "__main__":
    unittest.main()
