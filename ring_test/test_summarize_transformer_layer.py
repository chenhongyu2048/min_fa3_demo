import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ring_test.summarize_transformer_layer import main


def make_record(dataset, case_index, scale=1, method="fa3_ring", sm_config=None):
    return {
        "schema": "min_fa3.megatron_transformer_layer_cp.v2",
        "dataset": dataset,
        "case_index": case_index,
        "num_cases": 2,
        "seed": 0,
        "target_tokens": 128,
        "method": method,
        "sm_config": sm_config,
        "causal": True,
        "model": {"profile": "test", "q_heads": 2, "head_dim": 4},
        "parallelism": {"tp": 1, "cp": 8, "pp": 1, "dp": 1, "ep": 8},
        "iterations": {"measure": 10 * scale},
        "tokens": {"original_global": 16 * scale, "execution_global": 32 * scale},
        "workload": {"global_lengths": [16 * scale]},
        "timing": {
            "core_attn_forward_cuda_critical_rank_avg_ms": 1 * scale,
            "core_attn_backward_cuda_critical_rank_avg_ms": 4 * scale,
            "others_forward_cuda_critical_rank_avg_ms": 2 * scale,
            "others_backward_cuda_critical_rank_avg_ms": 5 * scale,
            "forward_cuda_critical_rank_avg_ms": 3 * scale,
            "backward_cuda_critical_rank_avg_ms": 9 * scale,
            "total_cuda_critical_rank_avg_ms": 12 * scale,
            "total_wall_max_avg_ms": 15 * scale,
        },
    }


class SummarizeTransformerLayerTest(unittest.TestCase):
    def run_report(self, files):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, records in enumerate(files):
                path = Path(directory) / f"part-{index}.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
                paths.append(str(path))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(paths)
            return output.getvalue()

    def test_equal_case_means_across_files_and_final_dataset_order(self):
        output = self.run_report([
            [make_record("github", 0, 2), make_record("arxiv", 0)],
            [make_record("arxiv", 1, 9)],
        ])
        summaries = output[output.index("SUMMARIES BY DATASET"):]
        self.assertLess(output.index("DATASET github |"), output.index("SUMMARIES BY DATASET"))
        self.assertLess(summaries.index("SUMMARY arxiv"), summaries.index("SUMMARY github"))
        arxiv = summaries.split("TIMING SUMMARY arxiv", 1)[1].split("SUMMARY github", 1)[0]
        rows = [line.split() for line in arxiv.splitlines() if line.startswith("fa3_ring")]
        self.assertEqual(rows, [
            ["fa3_ring", "-", "2/2", "GeoMean", "3.000", "12.000", "6.000", "15.000",
             "9.000", "27.000", "36.000", "45.000"],
            ["fa3_ring", "-", "2/2", "ArithMean", "5.000", "20.000", "10.000", "25.000",
             "15.000", "45.000", "60.000", "75.000"],
        ])
        self.assertIn("Missing cases: fa3_ring SM=-: 2", summaries)

    def test_single_file_keeps_methods_and_sm_configurations_separate(self):
        output = self.run_report([[
            make_record("arxiv", 0),
            make_record("arxiv", 0, method="mega_ring_hybrid",
                        sm_config={"num_comp_sm": 128, "num_comm_sm": 4}),
            make_record("arxiv", 0, method="mega_ring_hybrid",
                        sm_config={"num_comp_sm": 120, "num_comm_sm": 12}),
        ]])
        summary = output.split("TIMING SUMMARY arxiv", 1)[1]
        rows = [line.split() for line in summary.splitlines()
                if line.startswith(("fa3_ring", "mega_ring_hybrid"))]
        self.assertEqual(len(rows), 6)
        self.assertEqual({tuple(row[:3]) for row in rows}, {
            ("fa3_ring", "-", "1/2"),
            ("mega_ring_hybrid", "128:4", "1/2"),
            ("mega_ring_hybrid", "120:12", "1/2"),
        })
        for row in rows:
            self.assertEqual(row[4:], ["1.000", "4.000", "2.000", "5.000",
                                      "3.000", "9.000", "12.000", "15.000"])


if __name__ == "__main__":
    unittest.main()
