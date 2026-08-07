from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from dcp_test.summarize_dcp_mega_matrix import (
    BASELINE_METHODS,
    MEGA_METHOD,
    _backfill_bandwidth_metrics,
    parse_args,
    summarize,
)


def _method_summary(case_count: int) -> dict[str, object]:
    latency = {"min": 1.0, "mean": 2.0, "p50": 2.0, "max": 3.0}
    return {
        "case_count": case_count,
        "p50_latency_ms": latency,
        "p90_latency_ms": latency,
        "mean_effective_tflops": 10.0,
        "workload_weighted_effective_tflops": 9.0,
        "workload_weighted_effective_tflops_per_gpu": 1.125,
        "mean_effective_kv_bandwidth_gbps_per_gpu": 120.0,
        "workload_weighted_effective_kv_bandwidth_gbps_per_gpu": 100.0,
        "total_effective_flops": 18_000_000_000,
        "total_logical_kv_bytes_per_gpu": 200_000_000.0,
        "total_p50_latency_ms": 2.0,
    }


def _weighted(methods: set[str], case_count: int) -> dict[str, object]:
    return {
        "topologies": {
            "dcp2_hkv4": {
                "dcp_size": 2,
                "kv_heads": 4,
                "methods": {
                    method: _method_summary(case_count) for method in methods
                },
            }
        }
    }


class DCPMegaMatrixSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.result_dir = self.root / "results"
        self.combo = self.result_dir / "arrival_1" / "dcp_2"
        self.trace = {
            "arrival_time_scale": "1.0",
            "dcp_size": 2,
            "num_cases": 2,
            "config_sha256": "a" * 64,
            "trace_sha256": "b" * 64,
            "cases_jsonl": "trace/cases.jsonl",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, relative: str, payload: dict[str, object]) -> Path:
        path = self.combo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _args(self):
        return parse_args(
            [
                "--result-dir",
                str(self.result_dir),
                "--arrival-time-scales",
                "1",
                "--dcp-sizes",
                "2",
                "--mega-num-comm-sms",
                "4,8",
                "--num-cases",
                "2",
                "--output-json",
                str(self.root / "matrix.json"),
                "--output-csv",
                str(self.root / "matrix.csv"),
            ]
        )

    def _write_complete_matrix(self) -> None:
        variants = []
        for comm_sm in (4, 8):
            variants.append(
                {
                    "variant_id": f"comm_sm_{comm_sm}",
                    "mega_num_comm_sm": comm_sm,
                    "status": "complete",
                    "completed_case_count": 2,
                    "weighted_summary": _weighted({MEGA_METHOD}, 2),
                }
            )
        self._write(
            "mega/manifest.json",
            {
                "schema_version": 1,
                "manifest_kind": "mega_comm_sm_sweep",
                "status": "complete",
                "execution_mode": "eager",
                "trace": self.trace,
                "variant_total": 2,
                "completed_variant_count": 2,
                "case_total": 4,
                "completed_case_count": 4,
                "variants": variants,
            },
        )
        for directory, mode in (
            ("baseline_eager", "eager"),
            ("baseline_graph", "cuda_graph"),
        ):
            self._write(
                f"{directory}/manifest.json",
                {
                    "schema_version": 1,
                    "status": "complete",
                    "execution_mode": mode,
                    "case_total": 2,
                    "completed_case_count": 2,
                    "trace": self.trace,
                    "weighted_summary": _weighted(BASELINE_METHODS, 2),
                },
            )

    def test_complete_matrix_writes_expected_json_and_csv_rows(self) -> None:
        self._write_complete_matrix()
        args = self._args()
        payload = summarize(args)

        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["expected_launch_count"], 3)
        self.assertEqual(payload["completed_launch_count"], 3)
        self.assertEqual(payload["expected_summary_row_count"], 14)
        self.assertEqual(payload["summary_row_count"], 14)
        self.assertEqual(len(payload["summary_rows"]), 14)
        with args.output_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 14)
        mega_rows = [row for row in rows if row["suite"] == "mega"]
        self.assertEqual(
            [row["mega_num_comm_sm"] for row in mega_rows], ["4", "8"]
        )
        self.assertEqual(
            [
                row["workload_weighted_effective_kv_bandwidth_gbps_per_gpu"]
                for row in mega_rows
            ],
            ["100.0", "100.0"],
        )

    def test_old_manifest_bandwidth_is_backfilled_from_case_outputs(self) -> None:
        entries = []
        for index, (average_bytes, bandwidth) in enumerate(
            ((10_000_000.0, 100.0), (30_000_000.0, 200.0))
        ):
            output = self._write(
                f"outputs/case_{index}.json",
                {
                    "methods": {
                        MEGA_METHOD: {
                            "logical_kv_read": {
                                "average_bytes_per_gpu": average_bytes,
                                "effective_bandwidth_gbps_per_gpu": bandwidth,
                            }
                        }
                    }
                },
            )
            entries.append({"output_json": str(output)})

        metrics = _backfill_bandwidth_metrics(
            entries,
            {MEGA_METHOD},
            self.combo / "mega" / "manifest.json",
        )[MEGA_METHOD]

        self.assertEqual(
            metrics["mean_effective_kv_bandwidth_gbps_per_gpu"], 150.0
        )
        self.assertEqual(
            metrics["total_logical_kv_bytes_per_gpu"], 40_000_000.0
        )

    def test_missing_run_is_indexed_without_discarding_completed_rows(self) -> None:
        self._write_complete_matrix()
        (self.combo / "baseline_graph" / "manifest.json").unlink()
        payload = summarize(self._args())

        self.assertEqual(payload["status"], "incomplete")
        self.assertEqual(payload["completed_launch_count"], 2)
        self.assertEqual(payload["missing_launch_count"], 1)
        self.assertEqual(payload["summary_row_count"], 8)

    def test_default_matrix_contract_is_27_launches_and_153_rows(self) -> None:
        args = parse_args(
            [
                "--result-dir",
                str(self.result_dir),
                "--output-json",
                str(self.root / "default_matrix.json"),
                "--output-csv",
                str(self.root / "default_matrix.csv"),
            ]
        )
        payload = summarize(args)

        self.assertEqual(payload["expected_launch_count"], 27)
        self.assertEqual(payload["missing_launch_count"], 27)
        self.assertEqual(payload["expected_summary_row_count"], 153)
        self.assertEqual(payload["summary_row_count"], 0)


if __name__ == "__main__":
    unittest.main()
