import inspect
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import torch

import min_fa3_dcp
from dcp_test.baselines import (
    SGLangDCPAttentionRunner,
    VLLMA2ADCPAttentionRunner,
    VLLMDCPAttentionRunner,
    _dcp_a2a_head_owner_ranges,
    _dcp_a2a_lse_weighted_combine_reference,
    _dcp_a2a_payload_bytes,
)
from dcp_test.benchmark_output import (
    DCPBenchmarkRow,
    effective_kv_bandwidth_gbps_per_gpu,
    print_benchmark_results,
    print_timing_breakdowns,
)
from dcp_test.benchmark_dcp import expanded_method_labels as dense_method_labels
from dcp_test.benchmark_dcp_varlen import (
    expanded_method_labels as varlen_method_labels,
)
from dcp_test.utils import (
    BenchmarkPhaseRecorder,
    PHASE_EVENT_NAMES,
    make_cu_seqlens,
    parse_lengths,
)
from min_fa3_dcp import (
    DCPAttentionRunner,
    make_topology,
    validate_group_ranks,
    validate_topology,
)


class DCPTopologyTest(unittest.TestCase):
    def test_runtime_runner_has_no_benchmark_timing_api(self) -> None:
        for name in (
            "forward_decode",
            "forward_chunk_prefill",
            "forward_decode_varlen",
            "forward_chunk_prefill_varlen",
        ):
            self.assertNotIn(
                "_record_timing",
                inspect.signature(getattr(DCPAttentionRunner, name)).parameters,
            )
        for name in (
            "capture_decode",
            "capture_chunk_prefill",
            "capture_decode_varlen",
            "capture_chunk_prefill_varlen",
        ):
            self.assertNotIn(
                "record_timing",
                inspect.signature(getattr(DCPAttentionRunner, name)).parameters,
            )
        self.assertFalse(hasattr(DCPAttentionRunner, "last_timing_ms"))

    def test_runtime_runner_allocates_dependency_events_only(self) -> None:
        event_kwargs: list[dict[str, object]] = []

        def event(*args, **kwargs):
            del args
            event_kwargs.append(kwargs)
            return object()

        with (
            mock.patch.object(torch.distributed, "is_available", return_value=True),
            mock.patch.object(torch.distributed, "is_initialized", return_value=True),
            mock.patch.object(torch.distributed, "get_world_size", return_value=2),
            mock.patch.object(torch.distributed, "get_rank", return_value=0),
            mock.patch.object(torch.distributed, "get_backend", return_value="nccl"),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "Stream", return_value=object()),
            mock.patch.object(torch.cuda, "Event", side_effect=event),
        ):
            for runner_type in (
                DCPAttentionRunner,
                VLLMDCPAttentionRunner,
                VLLMA2ADCPAttentionRunner,
                SGLangDCPAttentionRunner,
            ):
                with self.subTest(runner_type=runner_type.__name__):
                    event_kwargs.clear()
                    runner = runner_type(None)
                    self.assertIsNone(runner._phase_recorder)
                    self.assertFalse(hasattr(runner, "_timing_events"))
                    self.assertEqual(len(event_kwargs), 4)
                    self.assertTrue(
                        all(
                            not kwargs.get("enable_timing", False)
                            for kwargs in event_kwargs
                        )
                    )

    def test_benchmark_phase_recorder_collective_field_mapping(self) -> None:
        class FakeEvent:
            def __init__(self, timestamp: float) -> None:
                self.timestamp = timestamp

            def elapsed_time(self, other: "FakeEvent") -> float:
                return other.timestamp - self.timestamp

        timestamps = {
            "attention_start": 0.0,
            "attention_end": 30.0,
            "q_ag_start": 1.0,
            "q_ag_end": 5.0,
            "chunk_start": 1.0,
            "chunk_end": 7.0,
            "ag_chunk_end": 8.0,
            "history_start": 8.0,
            "history_end": 14.0,
            "lse_correct_start": 14.0,
            "lse_correct_end": 17.0,
            "reduce_scatter_start": 17.0,
            "reduce_scatter_end": 23.0,
            "a2a_pack_start": 14.0,
            "a2a_pack_end": 16.0,
            "a2a_all_to_all_start": 16.0,
            "a2a_all_to_all_end": 22.0,
            "a2a_unpack_combine_start": 22.0,
            "a2a_unpack_combine_end": 27.0,
            "merge_start": 23.0,
            "merge_end": 28.0,
        }

        def elapsed(output_collective_kind: str) -> dict[str, float]:
            recorder = BenchmarkPhaseRecorder.__new__(BenchmarkPhaseRecorder)
            recorder.world_size = 2
            recorder.output_collective_kind = output_collective_kind
            recorder.events = {
                name: FakeEvent(timestamps[name]) for name in PHASE_EVENT_NAMES
            }
            recorder.last_kind = "chunk"
            return recorder.elapsed_ms(synchronize=False)

        reduce_scatter = elapsed("bf16_reduce_scatter")
        self.assertEqual(reduce_scatter["lse_allgather_correct_ms"], 3.0)
        self.assertEqual(reduce_scatter["output_reduce_scatter_ms"], 6.0)
        self.assertEqual(reduce_scatter["output_collective_ms"], 6.0)

        all_to_all = elapsed("bf16_packed_all_to_all")
        self.assertEqual(all_to_all["lse_allgather_correct_ms"], 0.0)
        self.assertEqual(all_to_all["output_reduce_scatter_ms"], 0.0)
        self.assertEqual(all_to_all["a2a_pack_ms"], 2.0)
        self.assertEqual(all_to_all["a2a_all_to_all_ms"], 6.0)
        self.assertEqual(all_to_all["a2a_unpack_combine_ms"], 5.0)
        self.assertEqual(all_to_all["output_collective_ms"], 6.0)

        all_reduce = elapsed("fp32_all_reduce")
        self.assertEqual(all_reduce["output_allreduce_ms"], 6.0)
        self.assertEqual(all_reduce["output_collective_ms"], 6.0)
        self.assertEqual(all_reduce["output_reduce_scatter_ms"], 0.0)

    def test_shared_length_and_cu_seqlens_helpers(self) -> None:
        self.assertEqual(parse_lengths("3", 2, "--sq"), [3, 3])
        device_cu, host_cu = make_cu_seqlens([3, 5], torch.device("cpu"))
        self.assertEqual(host_cu.tolist(), [0, 3, 8])
        self.assertEqual(device_cu.tolist(), host_cu.tolist())

    def test_benchmark_console_format_matches_ring_table(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            print_benchmark_results(
                "B=1, QH=32, KVH=1, D=128, mode=causal",
                [
                    DCPBenchmarkRow(
                        method="dcp_mega_varlen",
                        p50_ms=0.1254,
                        p90_ms=0.1512,
                        aggregate_tflops=42.25,
                        avg_gpu_tflops=5.28125,
                        kv_bandwidth_gbps_per_gpu=1234.5,
                        check="ok",
                        note="eager",
                        rank_p50_ms=(0.12, 0.125),
                    )
                ],
            )
        rendered = output.getvalue()
        self.assertIn("Method", rendered)
        self.assertIn("Time ms", rendered)
        self.assertIn("Agg TFLOPS", rendered)
        self.assertIn("Avg/GPU", rendered)
        self.assertIn("KV GB/s/GPU", rendered)
        self.assertIn("1234.5", rendered)
        self.assertIn("Check", rendered)
        self.assertIn("Note", rendered)
        self.assertIn("t0=0.120, t1=0.125", rendered)
        self.assertIn("p50(max_across_ranks)=0.125", rendered)
        self.assertIn("p90(max_across_ranks)=0.151", rendered)

    def test_effective_kv_bandwidth_uses_average_rank_bytes(self) -> None:
        self.assertEqual(
            effective_kv_bandwidth_gbps_per_gpu(
                (128.0e6, 256.0e6), p50_ms=0.125
            ),
            1536.0,
        )

    def test_timing_breakdowns_print_nonzero_phases_only(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            print_timing_breakdowns(
                "DCP CUDA-event phase breakdown",
                {
                    "vllm_a2a_min_fa3": {
                        "attention_end_to_end_ms": {"p50": 4.0, "p90": 5.0},
                        "a2a_pack_ms": {"p50": 0.2, "p90": 0.3},
                        "a2a_all_to_all_ms": {"p50": 0.4, "p90": 0.5},
                        "overlapped_ag_chunk_window_ms": {
                            "p50": 3.0,
                            "p90": 3.1,
                        },
                        "output_collective_ms": {"p50": 0.4, "p90": 0.5},
                        "output_reduce_scatter_ms": {"p50": 0.0, "p90": 0.0},
                    },
                    "ours_overlap_varlen": {
                        "overlapped_ag_chunk_window_ms": {
                            "p50": 0.6,
                            "p90": 0.7,
                        },
                    },
                    "full_kv_min_fa3": {
                        "attention_end_to_end_ms": {"p50": 1.0, "p90": 1.1},
                        "local_history_attention_ms": {"p50": 0.0, "p90": 0.0},
                    },
                },
                unit="ms",
                aggregation="per-iteration maximum across benchmark ranks",
            )
        rendered = output.getvalue()
        self.assertIn("DCP CUDA-event phase breakdown", rendered)
        self.assertIn("a2a_pack_ms", rendered)
        self.assertIn("a2a_all_to_all_ms", rendered)
        self.assertNotIn("attention_end_to_end_ms", rendered)
        self.assertNotIn("output_collective_ms", rendered)
        self.assertEqual(rendered.count("overlapped_ag_chunk_window_ms"), 1)
        self.assertIn("ours_overlap_varlen", rendered)
        self.assertNotIn("full_kv_min_fa3", rendered)

    def test_six_supported_gqa_topologies(self) -> None:
        expected = {
            (32, 4, 2),
            (32, 2, 2),
            (32, 2, 4),
            (64, 4, 2),
            (64, 2, 2),
            (64, 2, 4),
        }
        valid = {
            (hq, hkv, dcp)
            for hq in (32, 64)
            for hkv in (2, 4)
            for dcp in (2, 4, 8)
            if not validate_topology(hq, hkv, 8, dcp)
        }
        self.assertEqual(valid, expected)

    def test_rank_mapping_stays_inside_kv_replica_group(self) -> None:
        topology = make_topology(32, 2, 8, 2)
        self.assertEqual(topology.kv_replica_ranks(0), (0, 1, 2, 3))
        self.assertEqual(topology.kv_replica_ranks(7), (4, 5, 6, 7))
        self.assertEqual(topology.dcp_group_ranks(0), (0, 1))
        self.assertEqual(topology.dcp_group_ranks(3), (2, 3))
        self.assertEqual(topology.dcp_group_ranks(4), (4, 5))
        self.assertEqual(topology.dcp_group_ranks(7), (6, 7))
        self.assertEqual(topology.kv_head_for_rank(3), 0)
        self.assertEqual(topology.kv_head_for_rank(4), 1)
        self.assertEqual(topology.q_head_range(3), (12, 16))

    def test_rejects_group_crossing_kv_replica_boundary(self) -> None:
        topology = make_topology(32, 2, 8, 2)
        codes = {issue.code for issue in validate_group_ranks(topology, (3, 4))}
        self.assertIn("dcp_group_crosses_kv_replica_boundary", codes)

    def test_rejects_dcp_larger_than_replica_count(self) -> None:
        codes = {issue.code for issue in validate_topology(32, 4, 8, 4)}
        self.assertIn("dcp_exceeds_kv_replicas", codes)

    def test_rejects_replica_count_not_divisible_by_dcp(self) -> None:
        codes = {issue.code for issue in validate_topology(24, 2, 12, 4)}
        self.assertIn("kv_replicas_not_divisible_by_dcp", codes)

    def test_rejects_q_per_kv_not_divisible_by_dcp(self) -> None:
        codes = {issue.code for issue in validate_topology(20, 4, 8, 2)}
        self.assertIn("q_per_kv_not_divisible_by_dcp", codes)

    def test_rejects_q_heads_not_divisible_by_tp(self) -> None:
        codes = {issue.code for issue in validate_topology(36, 4, 8, 2)}
        self.assertIn("q_heads_not_divisible_by_tp", codes)

    def test_a2a_rank_major_head_owners(self) -> None:
        self.assertEqual(
            _dcp_a2a_head_owner_ranges(16, 4),
            ((0, 4), (4, 8), (8, 12), (12, 16)),
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            _dcp_a2a_head_owner_ranges(10, 4)

    def test_a2a_payload_counts_buffer_and_remote_bytes(self) -> None:
        buffer_bytes, remote_bytes = _dcp_a2a_payload_bytes(6, 2, 128, 4)
        per_owner = 6 * 2 * (128 + 2) * 2
        self.assertEqual(buffer_bytes, 4 * per_owner)
        self.assertEqual(remote_bytes, 3 * per_owner)

    def test_vllm_category_expands_ag_rs_and_a2a(self) -> None:
        self.assertEqual(
            dense_method_labels(("vllm",)),
            ("vllm_ag_rs_min_fa3", "vllm_a2a_min_fa3", "full_kv_min_fa3"),
        )
        self.assertEqual(
            varlen_method_labels(("vllm",)),
            ("vllm_ag_rs_min_fa3_varlen", "vllm_a2a_min_fa3_varlen"),
        )

    def test_varlen_mega_category_precedes_full_reference(self) -> None:
        self.assertEqual(
            varlen_method_labels(("mega", "full")),
            ("dcp_mega_varlen", "full_kv_min_fa3_varlen"),
        )

    def test_a2a_lse_weighted_equal_states(self) -> None:
        outputs = torch.tensor([[[[1.0]]], [[[3.0]]]])
        lses = torch.zeros((2, 1, 1), dtype=torch.float32)
        output, lse = _dcp_a2a_lse_weighted_combine_reference(outputs, lses)
        torch.testing.assert_close(output, torch.tensor([[[2.0]]]))
        torch.testing.assert_close(lse, torch.tensor([[torch.log(torch.tensor(2.0))]]))

    def test_a2a_lse_weighted_dominant_rank(self) -> None:
        outputs = torch.tensor([[[[1.0]]], [[[5.0]]]])
        lses = torch.tensor([[[0.0]], [[20.0]]], dtype=torch.float32)
        output, lse = _dcp_a2a_lse_weighted_combine_reference(outputs, lses)
        torch.testing.assert_close(output, torch.tensor([[[5.0]]]), atol=1.0e-6, rtol=0)
        torch.testing.assert_close(lse, torch.tensor([[20.0]]), atol=1.0e-6, rtol=0)

    def test_a2a_lse_weighted_all_invalid_states(self) -> None:
        outputs = torch.tensor([[[[1.0]]], [[[5.0]]], [[[9.0]]]])
        lses = torch.tensor(
            [[[float("nan")]], [[float("inf")]], [[-float("inf")]]],
            dtype=torch.float32,
        )
        output, lse = _dcp_a2a_lse_weighted_combine_reference(outputs, lses)
        torch.testing.assert_close(output, torch.zeros_like(output))
        self.assertTrue(torch.isneginf(lse).all())

    def test_a2a_runner_exports_without_vllm_runtime(self) -> None:
        self.assertFalse(hasattr(min_fa3_dcp, "VLLMA2ADCPAttentionRunner"))
        self.assertNotIn("VLLMA2ADCPAttentionRunner", min_fa3_dcp.__all__)
        self.assertEqual(VLLMA2ADCPAttentionRunner.method_name, "vllm_a2a_min_fa3")
        self.assertFalse(any(name == "vllm" or name.startswith("vllm.") for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
