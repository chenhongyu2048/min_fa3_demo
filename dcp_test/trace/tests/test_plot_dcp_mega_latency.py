from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from decimal import Decimal
from pathlib import Path

from dcp_test.plot_dcp_mega_latency import (
    LATENCY_COLUMNS,
    LATENCY_LABELS,
    METHOD_MEGA,
    METHOD_SGLANG,
    METHOD_VLLM_A2A,
    METHOD_VLLM_AG_RS,
    MetricSpec,
    best_baseline_comparison,
    load_records,
    main,
    select_plot_points,
)
from dcp_test.summarize_dcp_mega_matrix import CSV_FIELDS


class DCPMegaLatencyPlotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.csv_path = self.root / "matrix_summary.csv"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _row(
        self,
        *,
        arrival: int,
        dcp_size: int,
        method: str,
        mode: str,
        latency: float,
        comm_sm: int | None = None,
    ) -> dict[str, object]:
        row: dict[str, object] = {field: "" for field in CSV_FIELDS}
        row.update(
            {
                "arrival_time_scale": arrival,
                "dcp_size": dcp_size,
                "suite": "mega" if method == METHOD_MEGA else "baseline",
                "execution_mode": mode,
                "method": method,
                "mega_num_comm_sm": comm_sm if comm_sm is not None else "",
                "case_count": 2,
                "p50_latency_ms_min": latency * 0.8,
                "p50_latency_ms_mean": latency,
                "p50_latency_ms_p50": latency * 0.95,
                "p50_latency_ms_max": latency * 1.2,
                "workload_weighted_effective_tflops_per_gpu": 100.0 / latency,
                "workload_weighted_effective_kv_bandwidth_gbps_per_gpu": (
                    200.0 / latency
                ),
            }
        )
        return row

    def _write_matrix(self, *, omit: tuple[int, int, str, str] | None = None) -> None:
        rows: list[dict[str, object]] = []
        methods = (METHOD_VLLM_AG_RS, METHOD_VLLM_A2A, METHOD_SGLANG)
        for dcp_size in (2, 4, 8):
            for arrival in (1, 2, 4):
                for method_index, method in enumerate(methods, start=1):
                    for mode_index, mode in enumerate(("eager", "cuda_graph")):
                        if omit == (dcp_size, arrival, method, mode):
                            continue
                        rows.append(
                            self._row(
                                arrival=arrival,
                                dcp_size=dcp_size,
                                method=method,
                                mode=mode,
                                latency=float(
                                    dcp_size + arrival + method_index + mode_index
                                ),
                            )
                        )
                rows.append(
                    self._row(
                        arrival=arrival,
                        dcp_size=dcp_size,
                        method=METHOD_MEGA,
                        mode="eager",
                        latency=float(dcp_size + arrival + 2),
                        comm_sm=4,
                    )
                )
                rows.append(
                    self._row(
                        arrival=arrival,
                        dcp_size=dcp_size,
                        method=METHOD_MEGA,
                        mode="eager",
                        latency=float(dcp_size + arrival + 1) * 0.8,
                        comm_sm=8,
                    )
                )
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_selects_modes_and_best_mega_comm_sm(self) -> None:
        self._write_matrix()
        records = load_records(self.csv_path, "mean")
        points = select_plot_points(
            records,
            arrivals=(Decimal("1"), Decimal("2"), Decimal("4")),
            dcp_sizes=(2, 4, 8),
            mega_num_comm_sm="best",
            allow_incomplete=False,
        )

        self.assertEqual(len(points), 3 * 3 * 7)
        self.assertEqual(points[(2, Decimal("1"), "vllm_ag_rs_eager")].latency_ms, 4.0)
        self.assertEqual(points[(2, Decimal("1"), "vllm_ag_rs_graph")].latency_ms, 5.0)
        self.assertEqual(points[(2, Decimal("1"), "mega_eager")].mega_num_comm_sm, 8)
        self.assertAlmostEqual(
            points[(2, Decimal("1"), "mega_eager")].latency_ms, 3.2
        )
        comparison = best_baseline_comparison(
            points,
            dcp_size=2,
            arrival=Decimal("1"),
            mega=points[(2, Decimal("1"), "mega_eager")],
            metric=MetricSpec("latency_ms", "Latency", 1000.0, False),
        )
        self.assertIsNotNone(comparison)
        assert comparison is not None
        self.assertEqual(comparison.series.key, "vllm_ag_rs_eager")
        self.assertEqual(comparison.value, 4.0)
        self.assertAlmostEqual(comparison.speedup, 1.25)
        tflops_comparison = best_baseline_comparison(
            points,
            dcp_size=2,
            arrival=Decimal("1"),
            mega=points[(2, Decimal("1"), "mega_eager")],
            metric=MetricSpec("tflops_per_gpu", "TFLOPS/GPU", 1.0, True),
        )
        self.assertIsNotNone(tflops_comparison)
        assert tflops_comparison is not None
        self.assertEqual(tflops_comparison.series.key, "vllm_ag_rs_eager")
        self.assertAlmostEqual(tflops_comparison.speedup, 1.25)
        self.assertEqual(LATENCY_LABELS["mean"], "Mean per-case p50 latency (us)")

    def test_fixed_mega_comm_sm_and_latency_stat(self) -> None:
        self._write_matrix()
        records = load_records(self.csv_path, "p50")
        self.assertEqual(LATENCY_COLUMNS["p50"], "p50_latency_ms_p50")
        points = select_plot_points(
            records,
            arrivals=(Decimal("1"), Decimal("2"), Decimal("4")),
            dcp_sizes=(2, 4, 8),
            mega_num_comm_sm=4,
            allow_incomplete=False,
        )
        point = points[(2, Decimal("1"), "mega_eager")]
        self.assertEqual(point.mega_num_comm_sm, 4)
        self.assertAlmostEqual(point.latency_ms, 5.0 * 0.95)

    def test_missing_required_baseline_is_rejected(self) -> None:
        self._write_matrix(omit=(2, 1, METHOD_SGLANG, "cuda_graph"))
        records = load_records(self.csv_path, "mean")
        with self.assertRaisesRegex(ValueError, "missing 1 required plot points"):
            select_plot_points(
                records,
                arrivals=(Decimal("1"), Decimal("2"), Decimal("4")),
                dcp_sizes=(2, 4, 8),
                mega_num_comm_sm="best",
                allow_incomplete=False,
            )

    def test_cli_writes_png(self) -> None:
        self._write_matrix()
        output = self.root / "latency.png"
        with redirect_stdout(io.StringIO()):
            status = main(
                [str(self.csv_path), "--output", str(output), "--dpi", "80"]
            )
        self.assertEqual(status, 0)
        self.assertGreater(output.stat().st_size, 1000)
        self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        for name in (
            "dcp_tflops_per_gpu_best_comm_sm.png",
            "dcp_kv_bandwidth_per_gpu_best_comm_sm.png",
        ):
            path = self.root / name
            self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
    best_baseline_comparison,
