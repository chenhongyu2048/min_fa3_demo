"""CPU-only checks for compact plotting data and representative trace selection."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .config import PHASES, TRACE_FIELDS
from .export_plot_data import compact_timing, compact_trace, export_plot_data, select_trace
from .test_experiments import example_trace


class PlotDataTest(unittest.TestCase):
    def test_module_and_direct_script_cli(self):
        script = Path(__file__).with_name("export_plot_data.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "t1.json").write_text(json.dumps({"records": [
                {"batch_size": 1, "method": "ring", "mode": "serial",
                 "timing": {"per_rank_ms": [[1, 9, 2], [8, 1, 10]]}}]}))
            commands = (
                (["-m", "motivation.export_plot_data"], script.parent.parent, "--output"),
                ([script.name], script.parent, "--output-dir"),
            )
            for command, cwd, option in commands:
                with self.subTest(command=command, option=option):
                    output = root / f"{option.lstrip('-')}.log"
                    result = subprocess.run(
                        [sys.executable, *command, "--input-dir", str(root), option, str(output)],
                        cwd=cwd, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    rows = [json.loads(line) for line in output.read_text().splitlines()]
                    timing = next(row["timing"] for row in rows if row["type"] == "t1_timing")
                    self.assertEqual(timing["rank_max_ms"], [8, 9, 10])

    def test_rank_reduction_precedes_percentiles(self):
        timing = {"per_rank_ms": [[1, 9, 2], [8, 1, 10]]}
        self.assertEqual(compact_timing(timing, keep_samples=True),
                         {"rank_max_ms": [8, 9, 10], "p50_ms": 9, "p90_ms": 9.8})
        self.assertEqual(select_trace(timing), (1, 0, 9))

    def test_selection_uses_one_replay_and_its_slowest_rank(self):
        timing = {"per_rank_ms": [[2, 12, 3], [9, 4, 10]]}
        self.assertEqual(select_trace(timing), (2, 1, 10))
        # Even sample count can make two samples equally near p50.
        self.assertEqual(select_trace({"per_rank_ms": [[1, 3], [1, 3]]}), (0, 0, 1))

    def test_integer_clock_subtraction_preserves_all_boundaries(self):
        trace = example_trace()
        offset = 1_789_701_938_208_398_176
        for phase in trace:
            for row in phase:
                for field in range(1, 7):
                    if row[field]:
                        row[field] += offset
        compact = compact_trace(trace, 2)
        self.assertEqual(compact["sm_ids_by_cta"], [7, 2])
        self.assertEqual(compact["phase_times_ns"][0], [[0, 20, 50, 90], [1, 22, 80, 91]])
        self.assertEqual(compact["rank_sync_ns"], [[0, 5, 15], [4, 405, 415]])
        for source_phase, phase in zip(trace, compact["phase_times_ns"]):
            for source, row in zip(source_phase, phase):
                self.assertEqual(row, [value - (offset + 100) for value in source[1:5]])

    def test_export_reads_only_selected_rank_and_omits_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timing = {"per_rank_ms": [[2, 12, 3], [9, 4, 10]]}
            t1 = {"environment": {"world_size": 2}, "records": [
                {"batch_size": 1, "method": "ring", "mode": "serial", "timing": timing}]}
            (root / "t1.json").write_text(json.dumps(t1))
            d1 = {"environment": {"world_size": 2, "sm_count": 2}, "records": [
                {"case_id": "case_000003", "case": {"q_lens": [16, 16], "history_lens": [8, 9]},
                 "timings": {"phased_trace_on": timing, "phased_trace_off": timing}}]}
            (root / "d1.json").write_text(json.dumps(d1))
            # Other rank files and unselected trace samples are deliberately absent.
            rank = {"rank": 1, "dcp_ranks": [0, 1], "trace_fields": TRACE_FIELDS,
                    "phases": PHASES, "sm_trace_samples": [None, None, example_trace()],
                    "metadata_image": [123], "vllm_phase_samples_ms": [{"unused": 42}]}
            (root / "case_000003.rank1.json").write_text(json.dumps(rank))
            output = root / "plots.log"
            rows = export_plot_data(root, output)
            self.assertEqual([json.loads(line) for line in output.read_text().splitlines()],
                             json.loads(json.dumps(rows)))
            timeline = next(row for row in rows if row["type"] == "d1_timeline")
            self.assertEqual((timeline["sample_index"], timeline["rank"], timeline["selected_graph_ms"]),
                             (2, 1, 10))
            self.assertEqual(timeline["total_q"], 32)
            self.assertEqual(timeline["phase_times_ns"], compact_trace(example_trace(), 2)["phase_times_ns"])
            for omitted in ("per_rank_ms", "metadata_image", "vllm_phase_samples_ms", "sm_trace_samples"):
                self.assertNotIn(omitted, output.read_text())
            # An experiment directory may contain only D1 (e.g. a replay check).
            (root / "t1.json").unlink()
            self.assertFalse(any(row["type"].startswith("t1_") for row in export_plot_data(root, output)))


if __name__ == "__main__":
    unittest.main()
