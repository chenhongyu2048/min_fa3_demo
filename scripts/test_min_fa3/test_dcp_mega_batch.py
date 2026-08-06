import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dcp_test import benchmark_dcp_varlen
from dcp_test.benchmark_dcp_mega_batch import (
    DEFAULT_CONFIG,
    _case_argv,
    _manifest_case,
    _weighted_summary,
    expand_cases,
    load_batch_config,
    load_trace_workloads,
    parse_args,
)
from dcp_test.utils import BenchmarkPhaseRecorder


class DCPMegaBatchTest(unittest.TestCase):
    def test_six_load_config_expands_in_stable_order(self) -> None:
        config = load_batch_config(DEFAULT_CONFIG)
        cases = expand_cases(config)

        self.assertEqual(config.name, "dcp_mega_six_loads")
        self.assertEqual(len(cases), 18)
        self.assertEqual(cases[0].case_id, "small1_dcp2_hkv4")
        self.assertEqual(cases[1].case_id, "small1_dcp4_hkv2")
        self.assertEqual(cases[2].case_id, "small1_dcp8_hkv1")
        self.assertEqual(cases[-1].case_id, "large2_dcp8_hkv1")
        self.assertEqual(cases[-1].workload.q_lengths, (128,) * 16)
        self.assertEqual(cases[-1].workload.history_lengths, (65536,) * 16)

    def test_workload_and_dcp_filters_preserve_config_order(self) -> None:
        config = load_batch_config(DEFAULT_CONFIG)
        cases = expand_cases(
            config,
            workloads="medium1,small1",
            dcp_sizes="8,2",
        )
        self.assertEqual(
            [case.case_id for case in cases],
            [
                "small1_dcp2_hkv4",
                "small1_dcp8_hkv1",
                "medium1_dcp2_hkv4",
                "medium1_dcp8_hkv1",
            ],
        )

    def test_scalar_and_explicit_length_forms(self) -> None:
        payload = self._payload()
        payload["workloads"] = [
            {
                "name": "ragged",
                "b": 2,
                "sq": [1, 8],
                "seqlen": 129,
            }
        ]
        config = self._load(payload)
        workload = config.workloads[0]
        self.assertEqual(workload.q_lengths, (1, 8))
        self.assertEqual(workload.history_lengths, (129, 129))

    def test_invalid_configs_report_the_offending_field(self) -> None:
        mutations = {
            "unsafe name": lambda payload: payload["workloads"][0].update(
                name="../escape"
            ),
            "duplicate workload": lambda payload: payload["workloads"].append(
                copy.deepcopy(payload["workloads"][0])
            ),
            "wrong length count": lambda payload: payload["workloads"][0].update(
                b=2, sq=[1]
            ),
            "unknown field": lambda payload: payload["topologies"][0].update(
                typo=1
            ),
            "invalid topology": lambda payload: payload["topologies"][0].update(
                kvhead=3
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                payload = self._payload()
                mutate(payload)
                with self.assertRaises(ValueError):
                    self._load(payload)

    def test_invalid_filters_are_rejected(self) -> None:
        config = load_batch_config(DEFAULT_CONFIG)
        for workloads, dcp_sizes in (
            ("missing", "all"),
            ("all,small1", "all"),
            ("all", "3"),
            ("all", "two"),
        ):
            with self.subTest(workloads=workloads, dcp_sizes=dcp_sizes):
                with self.assertRaises(ValueError):
                    expand_cases(
                        config,
                        workloads=workloads,
                        dcp_sizes=dcp_sizes,
                    )

    def test_colliding_combined_case_ids_are_rejected(self) -> None:
        payload = self._payload()
        payload["workloads"] = [
            {"name": "a_b", "b": 1, "sq": 8, "seqlen": 129},
            {"name": "a", "b": 1, "sq": 8, "seqlen": 129},
        ]
        payload["topologies"] = [
            {"name": "c", "dcp_size": 2, "kvhead": 4},
            {"name": "b_c", "dcp_size": 2, "kvhead": 4},
        ]
        config = self._load(payload)
        with self.assertRaisesRegex(ValueError, "duplicate case IDs"):
            expand_cases(config)

    def test_manifest_case_keeps_result_path_and_method_summary(self) -> None:
        config = load_batch_config(DEFAULT_CONFIG)
        case = expand_cases(config, workloads="small1", dcp_sizes="2")[0]
        result = {
            "global_effective_flops": 5_250_000_000,
            "methods": {
                "dcp_mega_varlen": {
                    "stages_ms": {
                        "attention_end_to_end_ms": {"p50": 0.125, "p90": 0.150}
                    },
                    "effective_tflops": 42.0,
                }
            }
        }
        entry = _manifest_case(case, Path("results/case.json"), result)
        self.assertEqual(entry["case_id"], "small1_dcp2_hkv4")
        self.assertEqual(entry["output_json"], "results/case.json")
        self.assertEqual(entry["global_effective_flops"], 5_250_000_000)
        self.assertEqual(
            entry["methods"]["dcp_mega_varlen"],
            {"p50_ms": 0.125, "p90_ms": 0.150, "effective_tflops": 42.0},
        )

    def test_trace_cases_are_validated_and_keep_provenance(self) -> None:
        trace_config = mock.Mock(
            num_cases=2,
            dcp_size=4,
            trace_sha256="trace-sha",
            config_sha256="config-sha",
        )
        cases = [
            {
                "schema_version": "mega_dcp_workload/v1",
                "case_id": f"case_{index:06d}",
                "source": "trace-test",
                "trace_sha256": "trace-sha",
                "config_sha256": "config-sha",
                "batch_size": 2,
                "q_lens": [1, 8],
                "history_lens": [4, 17],
                "total_kv_lens": [5, 25],
                "sampled_time_us": index * 20_000,
            }
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_cases.jsonl"
            path.write_text(
                "".join(json.dumps(case) + "\n" for case in cases),
                encoding="utf-8",
            )
            workloads = load_trace_workloads(path, trace_config)

        self.assertEqual([workload.name for workload in workloads], [
            "case_000000",
            "case_000001",
        ])
        self.assertEqual(workloads[0].q_lengths, (1, 8))
        self.assertEqual(workloads[0].history_lengths, (4, 17))
        self.assertEqual(workloads[1].trace_metadata["sampled_time_us"], 20_000)

    def test_weighted_summary_uses_total_work_over_total_time(self) -> None:
        entries = [
            {
                "topology": "dcp4_hkv2",
                "dcp_size": 4,
                "kv_heads": 2,
                "global_effective_flops": 1_000_000_000_000,
                "methods": {
                    "dcp_mega_varlen": {
                        "p50_ms": 1.0,
                        "p90_ms": 1.5,
                        "effective_tflops": 1000.0,
                    }
                },
            },
            {
                "topology": "dcp4_hkv2",
                "dcp_size": 4,
                "kv_heads": 2,
                "global_effective_flops": 3_000_000_000_000,
                "methods": {
                    "dcp_mega_varlen": {
                        "p50_ms": 6.0,
                        "p90_ms": 9.0,
                        "effective_tflops": 500.0,
                    }
                },
            },
        ]
        summary = _weighted_summary(entries, tp_size=8)
        method = summary["topologies"]["dcp4_hkv2"]["methods"][
            "dcp_mega_varlen"
        ]

        self.assertEqual(method["case_count"], 2)
        self.assertEqual(method["p50_latency_ms"]["mean"], 3.5)
        self.assertEqual(method["mean_effective_tflops"], 750.0)
        self.assertAlmostEqual(
            method["workload_weighted_effective_tflops"],
            4000.0 / 7.0,
        )
        self.assertAlmostEqual(
            method["workload_weighted_effective_tflops_per_gpu"],
            500.0 / 7.0,
        )

    def test_baseline_phase_timing_flag_is_forwarded_to_every_case(self) -> None:
        args = parse_args(
            [
                "--no-baseline-phase-timing",
                "--output-dir",
                "results",
                "--manifest",
                "results/manifest.json",
            ]
        )
        config = load_batch_config(DEFAULT_CONFIG)
        case = expand_cases(config, workloads="small1", dcp_sizes="2")[0]
        case_argv = _case_argv(args, config, case, Path("results/case.json"))

        self.assertFalse(args.baseline_phase_timing)
        self.assertIn("--no-baseline-phase-timing", case_argv)
        direct_args = benchmark_dcp_varlen.parse_args(case_argv)
        self.assertFalse(direct_args.baseline_phase_timing)

    @mock.patch("dcp_test.utils.torch.cuda.Event")
    def test_disabled_phase_timing_only_constructs_end_to_end_events(
        self, event_factory: mock.Mock
    ) -> None:
        event_factory.side_effect = (mock.Mock(), mock.Mock())
        recorder = BenchmarkPhaseRecorder(
            world_size=8,
            output_collective_kind="reduce_scatter",
            phase_timing=False,
        )
        stream = mock.Mock()

        self.assertEqual(event_factory.call_count, 2)
        self.assertEqual(
            set(recorder.events), {"attention_start", "attention_end"}
        )
        recorder.begin("chunk", stream)
        recorder.record("q_ag_start", stream)
        recorder.record("attention_end", stream)
        recorder.events["attention_start"].record.assert_called_once_with(stream)
        recorder.events["attention_end"].record.assert_called_once_with(stream)

    @staticmethod
    def _payload() -> dict[str, object]:
        return {
            "schema_version": 1,
            "name": "test_batch",
            "tp_size": 8,
            "qhead": 32,
            "headdim": 128,
            "workloads": [
                {"name": "case", "b": 1, "sq": 8, "seqlen": 129}
            ],
            "topologies": [
                {"name": "dcp2_hkv4", "dcp_size": 2, "kvhead": 4}
            ],
        }

    def _load(self, payload: dict[str, object]):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_batch_config(path)


if __name__ == "__main__":
    unittest.main()
