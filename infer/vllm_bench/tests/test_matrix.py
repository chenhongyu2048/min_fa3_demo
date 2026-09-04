from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from vllm_bench.matrix import _parse_positive_int_list
from vllm_bench.summarize import aggregate


class MatrixConfigTest(unittest.TestCase):
    def test_parse_mega_comm_sm_sweep(self) -> None:
        self.assertEqual(_parse_positive_int_list("4,8,12 16"), (4, 8, 12, 16))

    def test_reject_invalid_mega_comm_sm_sweep(self) -> None:
        for value in ("", "0", "132", "4,4", "4,eight"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _parse_positive_int_list(value)

    def test_summary_compares_every_mega_sm_to_each_baseline(self) -> None:
        summaries = (
            ("vllm-ag-rs-scale1", "vllm-ag-rs", None),
            ("vllm-a2a-scale1", "vllm-a2a", None),
            ("mega-comm_sm4-scale1", "mega", 4),
            ("mega-comm_sm8-scale1", "mega", 8),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            result_dir = Path(temporary_dir)
            for directory, backend, comm_sm in summaries:
                run_dir = result_dir / directory
                run_dir.mkdir()
                (run_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "backend": backend,
                            "arrival_time_scale": 1,
                            "mega_num_comm_sm": comm_sm,
                        }
                    ),
                    encoding="utf-8",
                )
            result = aggregate(result_dir)

        self.assertEqual(len(result["runs"]), 4)
        self.assertEqual(len(result["comparisons"]), 4)
        self.assertEqual(
            [comparison["mega_num_comm_sm"] for comparison in result["comparisons"]],
            [4, 4, 8, 8],
        )


if __name__ == "__main__":
    unittest.main()
