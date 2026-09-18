"""CPU-only checks for the compact, standalone plot exporter."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .export_plot_data import COMPONENTS, DATASETS, STRATEGIES, export


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        env = {"gpu_name": "test", "sm_count": 100}
        config = {"world_size": 8, "input_case": {"batch": 2, "local_seqlen": 8192}}
        t1 = {f"{method}_{name}": {"p50_rank_max_ms": value}
              for method in ("ring", "allgather")
              for name, value in (("comm_only", 1), ("comp_only", 2), ("serial", 4), ("overlap", 2.5))}
        t2 = {name: {"timing": {"p50_rank_max_ms": value}} for name, value in (
            ("step_external_reduce", 3), ("step_fused_reduce", 2), ("linear_queue_recycle", 1))}
        for experiment, results in (("T1", t1), ("T2", t2)):
            self.write(f"training/{experiment}/case.json", {
                "environment": env, "config": config, "results": results,
            })
        for dataset in DATASETS:
            self.write(f"training/T3/{dataset}.jsonl", [
                {"case_id": case_id, "static": {"strategy": strategy},
                 "timing": {field: scale * (index + 1)
                            for index, (_, field) in enumerate(COMPONENTS)}}
                for strategy in STRATEGIES for case_id, scale in ((0, 1), (1, 3))
            ])
        self.write("d1_selected.json", {"cases": [{"case_id": "chosen"}]})
        self.write("d1_selected_trace/D1/cases.jsonl", [{
            "case_id": "chosen", "workload_kind": "mixed", "config": {},
            "selection_status": "manually_selected", "environment": env,
            "results": {method: {"timing": {"p50_rank_max_ms": 1}}
                        for method in ("critical_wave", "fa3_native", "vllm_a2a")},
            "diagnostics": {
                "vllm_a2a": {"critical_rank": 1, "stages_ms": {"attention_end_to_end_ms": 2}},
                **{method: {"trace_by_rank": {
                    "0": [[5000, 5300, 5, 7, 1], [5400, 5550, 5, 7, 4]],
                    "1": [[9000, 9900, 6, 8, 1]],
                }} for method in ("critical_wave", "fa3_native")},
            },
        }])

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = ("\n".join(json.dumps(row) for row in value)
                if path.suffix == ".jsonl" else json.dumps(value))
        path.write_text(text)

    def test_plot_values_and_same_rank_trace(self):
        output = io.StringIO()
        export(self.root, output)
        text = output.getvalue()
        self.assertIn("2,8192,ring,3,4,2.5\n", text)
        self.assertIn("2,8192,allgather,3,4,2.5\n", text)
        self.assertIn("2,8192,3,2,1\n", text)
        self.assertIn("arxiv,all_cp,2,2,4,6,8\n", text)
        self.assertEqual(sum(line.startswith(tuple(d + "," for d in DATASETS))
                             for line in text.splitlines()), 20)
        self.assertEqual(text.count("5,7,1,0,300\n5,7,4,400,150\n"), 2)
        self.assertNotIn("6,8,1,0,900", text)
        self.assertEqual(text.count("rank=0"), 2)
        self.assertTrue(text.endswith("# PLOT_DATA_V1 END\n"))

    def test_standalone_cli_outside_repository(self):
        script = Path(__file__).with_name("export_plot_data.py").resolve()
        result = subprocess.run([
            sys.executable, str(script), str(self.root),
        ], cwd=self.root, text=True, capture_output=True, check=True)
        self.assertIn("2,8192,allgather,3,4,2.5", result.stdout)
        self.assertEqual(result.stdout.count("rank=0"), 2)
        self.assertNotIn("TRACE method=critical_wave rank=1", result.stdout)
        self.assertIn("A2A critical_rank=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
