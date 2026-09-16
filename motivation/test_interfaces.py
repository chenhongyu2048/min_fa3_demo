"""Source ABI checks plus real CPU layout checks when torch is installed."""
import ast
import importlib.util
import re
import subprocess
import unittest

from .config import BASE_COMMIT, ROOT, STRATEGIES


def baseline(path):
    return subprocess.check_output(["git", "show", f"{BASE_COMMIT}:{path}"], cwd=ROOT, text=True)


def binding_args(text, name):
    block = text.split(f'"{name}",', 1)[1].split(".def(", 1)[0]
    return re.findall(r'py::arg\("([^\"]+)"\)', block)


def runner_method(text, name):
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)
               and n.name == "DCPMegaAttentionRunner")
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


class InterfaceTests(unittest.TestCase):
    def test_binding_parameters_extend_existing_positional_prefix(self):
        for path, name, extra in (
            ("csrc/dcp_mega_min_fa3_varlen_bindings.cu", "forward_chunk_prefill_varlen_dcp_mega", "cta_trace"),
            ("csrc/mega_ring_min_fa3_varlen_ring_bindings.cu", "forward_varlen_mega_ring_ablation", "compute_only"),
        ):
            before = binding_args(baseline(path), name)
            after = binding_args((ROOT / path).read_text(), name)
            self.assertEqual(after, before + [extra])
        self.assertIn('py::arg("compute_only") = false',
                      (ROOT / "csrc/mega_ring_min_fa3_varlen_ring_bindings.cu").read_text())
        self.assertIn('py::arg("cta_trace") = py::none()',
                      (ROOT / "csrc/dcp_mega_min_fa3_varlen_bindings.cu").read_text())

    def test_python_trace_default_and_public_argument_builder(self):
        old = baseline("min_fa3_dcp.py")
        new = (ROOT / "min_fa3_dcp.py").read_text()
        self.assertEqual(ast.dump(runner_method(old, "_backend_args")),
                         ast.dump(runner_method(new, "_backend_args")))
        before, after = (runner_method(text, "__init__").args for text in (old, new))
        self.assertEqual([a.arg for a in after.args], [a.arg for a in before.args])
        self.assertEqual([a.arg for a in after.kwonlyargs],
                         [a.arg for a in before.kwonlyargs] + ["record_cta_trace"])
        self.assertIs(ast.literal_eval(after.kw_defaults[-1]), False)


@unittest.skipUnless(importlib.util.find_spec("torch"), "real torch CPU environment unavailable")
class LayoutTests(unittest.TestCase):
    def test_static_and_execution_layout_identity(self):
        from .placements import build_placements, describe_placement
        from ring_test.transformer_layer_cp import build_physical_layout
        # Raw manifests have already passed the public sampler's alignment.
        # Keep non-2048 multiples to exercise additional placement padding.
        raw = (18432, 10240, 5120, 2048, 768)
        for world in (4, 8):
            plans = build_placements(raw, world)
            self.assertEqual(tuple(plans), STRATEGIES)
            for strategy, plan in plans.items():
                report = describe_placement(strategy, plan)
                layout = build_physical_layout("mega_ring_hybrid", plan.global_lengths,
                                               plan.ring_sizes, plan.ring_starts, world)
                self.assertEqual(report["execution_lengths"], layout.execution_lengths)
                self.assertEqual(report["raw_lengths"], raw)
                self.assertEqual(report["padding_tokens"], sum(layout.execution_lengths) - sum(raw))
                self.assertEqual(sum(layout.rank_token_loads), report["execution_tokens"])
                self.assertEqual(sorted(report["sample_ids"]), list(range(len(raw))))
                self.assertEqual(report["executor"], "mega_ring_hybrid")
                for sample in report["samples"]:
                    self.assertEqual(sample["raw_length"], raw[sample["sample_id"]])
                    alignment = 256 if sample["ring_size"] > 1 else 128
                    self.assertEqual(sample["execution_length"] % (alignment * sample["ring_size"]), 0)
                    self.assertEqual(sample["mapped_group_size"] & (sample["mapped_group_size"] - 1), 0)

    def test_four_and_eight_rank_t2_hierarchy(self):
        import torch
        from .forward_ablation import build_hierarchy, profile_dispatch_manifest
        for world in (4, 8):
            local = 131072 // world
            half, hierarchy, counts = build_hierarchy(torch.tensor([0, local], dtype=torch.int32),
                [131072], [world], [0], 0, 32, world)
            self.assertEqual(half.tolist(), [0, local // 2])
            self.assertEqual(hierarchy[0].item(), world)
            profiles = profile_dispatch_manifest(world)
            self.assertEqual(profiles[0]["attention_launches"], world)
            self.assertEqual(profiles[0]["reduction_launches"], world)
            self.assertEqual(profiles[1]["reduction_launches"], 0)


if __name__ == "__main__":
    unittest.main()
