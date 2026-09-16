"""Synthetic measurement records test analysis rules, not GPU performance."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from .config import BASE_COMMIT, d1_config
from .plot import trace_intervals
from .select_d1_cases import select_cases
from .summarize import load_records, summary_rows


def measurements():
    rows = []
    for kind in ("decode_only_q16", "mixed"):
        for case_id, ratio in enumerate((1.125, 1.25, 1.375, 1.5)):
            for run in range(3):
                name = f"case_{case_id:06d}"
                rows.append({"schema": "motivation.v2.D1", "config": d1_config(8),
                    "base_commit": BASE_COMMIT, "commit": BASE_COMMIT,
                    "environment": {"gpu_name": "test GPU", "sm_count": 100},
                    "warmup": 20, "iters": 30, "execution_mode": "cuda_graph",
                    "case_id": name, "workload_kind": kind, "run_id": str(run),
                    "case": {"case_id": name, "workload_kind": kind, "q_lengths": [16]},
                    "results": {mode: {"trace_enabled": False, "timing": {"p50_rank_max_ms": t}}
                                for mode, t in (("critical_wave", 1), ("fa3_native", ratio))}})
    return rows


class SelectionTests(unittest.TestCase):
    def test_select_nearest_winner_median_with_numeric_tie_break(self):
        result = select_cases(list(reversed(measurements())))
        self.assertEqual([x["case_id"] for x in result["cases"]],
                         ["case_000001", "case_000002"] * 2)
        self.assertEqual(len(result["all_candidates"]), 8)

    def test_a_single_losing_run_excludes_candidate(self):
        rows = measurements()
        rows[3]["results"]["fa3_native"]["timing"]["p50_rank_max_ms"] = .9
        result = select_cases(rows)
        self.assertNotIn("case_000001", [c["case_id"] for c in result["cases"]
                                        if c["workload_kind"] == "decode_only_q16"])

    def test_reject_uncomparable_measurements(self):
        for change in ("smoke", "trace", "missing_run", "duplicate", "input", "hardware", "zero", "nan"):
            with self.subTest(change=change):
                rows = measurements()
                if change == "smoke": rows[0]["config"] = d1_config(4)
                elif change == "trace": rows[0]["results"]["critical_wave"]["trace_enabled"] = True
                elif change == "missing_run": rows.pop()
                elif change == "duplicate": rows.append(deepcopy(rows[0]))
                elif change == "input": rows[0]["case"]["q_lengths"] = [32]
                elif change == "hardware": rows[0]["environment"]["gpu_name"] = "different"
                else: rows[0]["results"]["critical_wave"]["timing"]["p50_rank_max_ms"] = 0 if change == "zero" else float("nan")
                with self.assertRaises(ValueError): select_cases(rows)

    def test_insufficient_winners_stay_unselected(self):
        rows = measurements()
        for row in rows:
            row["results"]["fa3_native"]["timing"]["p50_rank_max_ms"] = .9
        with self.assertRaisesRegex(ValueError, "pending"):
            select_cases(rows)


class ResultTests(unittest.TestCase):
    def test_summary_separates_timing_semantics(self):
        t1 = {"schema": "motivation.v2.T1", "config": {"case_id": "tokens128k_b1"},
              "results": {"ring_serial": {"p50_rank_max_ms": 4}}, "corun": {}}
        t2 = {"schema": "motivation.v2.T2", "config": {"case_id": "tokens128k_b1"},
              "results": {"step_fused_reduce": {"timing": {"p50_rank_max_ms": 3}}}}
        t3 = {"schema": "motivation.v2.T3", "case_id": 0, "dataset": "arxiv",
              "static": dict(strategy="br_pbs", token_imbalance=1, attention_imbalance=1.1,
                             communication_tx_bytes=1024, tile_work=32, padding_tokens=256),
              "timing": {"total_cuda_critical_rank_avg_ms": 8, "total_wall_max_avg_ms": 9}}
        rows = summary_rows([t1, t2, t3, measurements()[0]])
        self.assertEqual({r["statistic"] for r in rows},
                         {"p50_rank_max", "analytical", "mean_cuda_critical_rank", "p50_rank_max_uninstrumented"})
        self.assertNotIn("total_wall_max_avg_ms", [r["metric"] for r in rows])
        bad = measurements()[0]
        bad["results"]["critical_wave"]["trace_enabled"] = True
        with self.assertRaises(ValueError): summary_rows([bad])

    def test_old_results_are_not_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "T1"
            path.mkdir()
            (path / "old.json").write_text(json.dumps({"schema_version": 1}))
            with self.assertRaisesRegex(ValueError, "v2"): load_records(directory)

    def test_trace_plot_uses_phase_intervals_not_task_counts(self):
        self.assertEqual(trace_intervals([[2000, 5000, 0, 8, 1], [4000, 6000, 3, 9, 2]]),
                         [(8, 0, 3, 1, 0), (9, 2, 2, 2, 3)])
        self.assertEqual(trace_intervals([]), [])
        with self.assertRaises(ValueError): trace_intervals([[4, 2, 0, 8, 1]])


if __name__ == "__main__":
    unittest.main()
