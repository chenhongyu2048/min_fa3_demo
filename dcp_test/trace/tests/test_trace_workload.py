from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from dcp_test.trace.generate import run
from dcp_test.trace.models import (
    ConfigError,
    TraceRequest,
    derived_rng,
    load_config,
)
from dcp_test.trace.mooncake import (
    PrefixBlockCache,
    TraceError,
    load_mooncake_trace,
)
from dcp_test.trace.replay import ReplayError, replay_trace
from dcp_test.trace.show_example import DISPLAY_FIELDS, show_examples


def _row(
    timestamp: int | float,
    input_length: int,
    output_length: int,
    hash_ids: list[int] | None = None,
) -> dict[str, object]:
    block_count = (input_length + 511) // 512
    return {
        "timestamp": timestamp,
        "input_length": input_length,
        "output_length": output_length,
        "hash_ids": list(range(block_count)) if hash_ids is None else hash_ids,
    }


class TraceWorkloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.file_index = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _config_path(
        self,
        rows: list[dict[str, object]],
        **overrides: object,
    ) -> Path:
        index = self.file_index
        self.file_index += 1
        trace_path = self.root / f"trace_{index}.jsonl"
        trace_text = "".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in rows
        )
        trace_path.write_text(trace_text, encoding="utf-8")
        raw: dict[str, object] = {
            "trace_path": trace_path.name,
            "trace_sha256": hashlib.sha256(trace_text.encode("utf-8")).hexdigest(),
            "timestamp_policy": "preserve",
            "arrival_time_scale": 1,
            "sampling_start_ms": 0,
            "sampling_end_ms": 100,
            "fixed_step_us": 1000,
            "num_cases": 1,
            "seed": 7,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 2048,
            "prefill_chunk_size": 512,
            "q_len_alignment": 1,
            "max_model_len": 10000,
            "dcp_size": 2,
            "prefix_cache_capacity_blocks": 16,
            "num_speculative_tokens": 0,
            "scheduled_q_rule": "target_plus_drafts",
            "accepted_draft_pmf": [1.0],
        }
        raw.update(overrides)
        if (
            "accepted_draft_pmf" not in overrides
            and (
                "num_speculative_tokens" in overrides
                or "scheduled_q_rule" in overrides
            )
        ):
            count = int(raw["num_speculative_tokens"])
            entries = count + 1
            if raw["scheduled_q_rule"] == "drafts_only":
                entries = count
            raw["accepted_draft_pmf"] = [1.0] + [0.0] * (entries - 1)
        config_path = self.root / f"config_{index}.json"
        config_path.write_text(json.dumps(raw), encoding="utf-8")
        return config_path

    def _load_and_replay(self, config_path: Path):
        config = load_config(config_path)
        trace = load_mooncake_trace(config, derived_rng(config.seed, "arrival"))
        result = replay_trace(
            trace,
            config,
            derived_rng(config.seed, "acceptance"),
            derived_rng(config.seed, "sampling"),
        )
        return config, trace, result

    def test_config_validation_and_query_rules(self) -> None:
        rows = [_row(0, 4, 2)]
        config_path = self._config_path(rows)
        config = load_config(config_path)
        self.assertEqual(config.mtp_query_len, 1)
        self.assertEqual(config.max_accepted_drafts, 0)

        overridden = load_config(config_path, num_cases=3)
        self.assertEqual(overridden.num_cases, 3)
        self.assertNotEqual(overridden.config_sha256, config.config_sha256)
        with self.assertRaisesRegex(ConfigError, "num_cases override must be >= 1"):
            load_config(config_path, num_cases=0)

        matrix_override = load_config(
            config_path,
            arrival_time_scale=4.0,
            dcp_size=8,
        )
        self.assertEqual(matrix_override.arrival_time_scale, Decimal("4.0"))
        self.assertEqual(matrix_override.dcp_size, 8)
        self.assertNotEqual(matrix_override.config_sha256, config.config_sha256)
        with self.assertRaisesRegex(ConfigError, "positive finite"):
            load_config(config_path, arrival_time_scale=0)
        with self.assertRaisesRegex(ConfigError, "must be one of"):
            load_config(config_path, dcp_size=3)

        drafts_only = load_config(
            self._config_path(
                rows,
                num_speculative_tokens=3,
                scheduled_q_rule="drafts_only",
                accepted_draft_pmf=[0.2, 0.3, 0.5],
            )
        )
        self.assertEqual(drafts_only.mtp_query_len, 3)
        self.assertEqual(drafts_only.max_accepted_drafts, 2)

        invalid_path = self._config_path(rows, accepted_draft_pmf=[0.5])
        raw = json.loads(invalid_path.read_text(encoding="utf-8"))
        raw["unexpected"] = 1
        invalid_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "unknown configuration fields"):
            load_config(invalid_path)

        bad_pmf = self._config_path(
            rows,
            num_speculative_tokens=1,
            accepted_draft_pmf=[0.3, 0.3],
        )
        with self.assertRaisesRegex(ConfigError, "must sum to 1"):
            load_config(bad_pmf)

    def test_trace_provenance_timestamps_schema_and_filtering(self) -> None:
        rows = [_row(1, 4, 1), _row(1, 513, 1), _row(2, 8, 1)]
        preserve_config = load_config(self._config_path(rows))
        preserve = load_mooncake_trace(
            preserve_config, derived_rng(preserve_config.seed, "arrival")
        )
        self.assertEqual([request.arrival_us for request in preserve.requests], [0, 0, 1000])

        jitter_config = load_config(
            self._config_path(rows, timestamp_policy="uniform_bucket_jitter")
        )
        jitter_a = load_mooncake_trace(
            jitter_config, derived_rng(jitter_config.seed, "arrival")
        )
        jitter_b = load_mooncake_trace(
            jitter_config, derived_rng(jitter_config.seed, "arrival")
        )
        self.assertEqual(jitter_a.requests, jitter_b.requests)
        self.assertEqual(min(request.arrival_us for request in jitter_a.requests), 0)

        filtered_config = load_config(
            self._config_path(
                [_row(0, 9, 2), _row(1, 2, 1)], max_model_len=10
            )
        )
        filtered = load_mooncake_trace(
            filtered_config, derived_rng(filtered_config.seed, "arrival")
        )
        self.assertEqual(filtered.total_rows, 2)
        self.assertEqual(filtered.dropped_model_len, 1)
        self.assertEqual(len(filtered.requests), 1)

        bad_hash_config_path = self._config_path(rows)
        bad_hash_raw = json.loads(bad_hash_config_path.read_text(encoding="utf-8"))
        bad_hash_raw["trace_sha256"] = "0" * 64
        bad_hash_config_path.write_text(json.dumps(bad_hash_raw), encoding="utf-8")
        bad_hash_config = load_config(bad_hash_config_path)
        with self.assertRaisesRegex(TraceError, "SHA-256 mismatch"):
            load_mooncake_trace(
                bad_hash_config, derived_rng(bad_hash_config.seed, "arrival")
            )

        bad_schema_config = load_config(
            self._config_path([_row(0, 513, 1, hash_ids=[])])
        )
        with self.assertRaisesRegex(TraceError, "expected 2 hash_ids"):
            load_mooncake_trace(
                bad_schema_config, derived_rng(bad_schema_config.seed, "arrival")
            )

    def test_prefix_cache_consecutive_hits_guard_and_lru(self) -> None:
        request = TraceRequest(0, 0, 0, 1025, 1, (10, 11, 12))
        cache = PrefixBlockCache(4)
        cache.insert_completed(request, 0, 1025)
        self.assertEqual(cache.lookup_prefix(request), 1024)

        exact_block_request = TraceRequest(1, 0, 0, 1024, 1, (10, 11))
        self.assertEqual(cache.lookup_prefix(exact_block_request), 512)

        broken_prefix = TraceRequest(2, 0, 0, 1025, 1, (10, 99, 12))
        self.assertEqual(cache.lookup_prefix(broken_prefix), 512)

        one_block_cache = PrefixBlockCache(1)
        one_block_cache.insert_completed(request, 0, 1025)
        self.assertEqual(len(one_block_cache), 1)
        self.assertEqual(one_block_cache.lookup_prefix(request), 0)

    def test_chunk_then_decode_state_and_case_invariants(self) -> None:
        config_path = self._config_path(
            [_row(0, 6, 3)],
            prefill_chunk_size=3,
            max_num_batched_tokens=3,
            num_cases=3,
            sampling_end_ms=4,
        )
        _, _, result = self._load_and_replay(config_path)
        self.assertEqual([case["sampled_time_us"] for case in result.cases], [1000, 2000, 3000])
        self.assertEqual(
            [case["phases"][0] for case in result.cases],
            ["chunk_prefill", "decode", "decode"],
        )
        self.assertEqual([case["q_lens"][0] for case in result.cases], [3, 1, 1])
        self.assertEqual([case["history_lens"][0] for case in result.cases], [3, 6, 7])
        self.assertEqual(
            [case["generated_tokens_before"][0] for case in result.cases],
            [0, 1, 2],
        )
        for case in result.cases:
            self.assertEqual(case["batch_size"], len(case["q_lens"]))
            self.assertEqual(
                case["total_kv_lens"],
                [
                    history + query
                    for history, query in zip(case["history_lens"], case["q_lens"])
                ],
            )
            self.assertTrue(all(history >= 2 for history in case["history_lens"]))

    def test_decode_priority_mtp_atomicity_and_mixed_step(self) -> None:
        config_path = self._config_path(
            [_row(0, 2, 4, [10]), _row(0, 6, 1, [20])],
            max_num_batched_tokens=4,
            prefill_chunk_size=4,
            num_speculative_tokens=2,
            accepted_draft_pmf=[1.0, 0.0, 0.0],
            sampling_start_ms=1,
            sampling_end_ms=2,
        )
        _, _, result = self._load_and_replay(config_path)
        case = result.cases[0]
        self.assertEqual(case["phases"], ["decode", "chunk_prefill"])
        self.assertEqual(case["q_lens"], [3, 1])
        self.assertEqual(case["history_lens"], [2, 2])
        self.assertEqual(case["accepted_drafts"], [0, None])
        self.assertEqual(sum(case["q_lens"]), 4)

    def test_aligned_physical_queries_preserve_logical_progress(self) -> None:
        config_path = self._config_path(
            [_row(0, 18, 3)],
            q_len_alignment=8,
            max_num_batched_tokens=16,
            prefill_chunk_size=10,
            num_cases=2,
            sampling_start_ms=1,
            sampling_end_ms=3,
        )
        config, _, result = self._load_and_replay(config_path)
        self.assertEqual(config.q_len_alignment, 8)
        self.assertEqual(
            [case["phases"][0] for case in result.cases],
            ["chunk_prefill", "decode"],
        )
        self.assertEqual(
            [case["logical_q_lens"][0] for case in result.cases],
            [8, 1],
        )
        self.assertEqual(
            [case["q_lens"][0] for case in result.cases],
            [8, 8],
        )
        self.assertEqual(
            [case["history_lens"][0] for case in result.cases],
            [10, 18],
        )
        for case in result.cases:
            self.assertEqual(case["q_len_alignment"], 8)
            self.assertEqual(case["q_lens"][0] % 8, 0)
            self.assertLess(
                case["q_lens"][0] - case["logical_q_lens"][0], 8
            )

    def test_query_alignment_config_validation(self) -> None:
        rows = [_row(0, 8, 1)]
        with self.assertRaisesRegex(ConfigError, "must be 1 or 8"):
            load_config(self._config_path(rows, q_len_alignment=4))
        with self.assertRaisesRegex(ConfigError, "fit one aligned query"):
            load_config(
                self._config_path(
                    rows,
                    q_len_alignment=8,
                    max_num_batched_tokens=4,
                )
            )

    def test_prefix_reuse_enters_later_chunk_prefill_case(self) -> None:
        shared = [101, 102, 103]
        config_path = self._config_path(
            [_row(0, 1025, 1, shared), _row(0, 1025, 1, shared)],
            max_num_seqs=1,
            max_num_batched_tokens=2048,
            prefill_chunk_size=2048,
            sampling_start_ms=1,
            sampling_end_ms=2,
            prefix_cache_capacity_blocks=8,
        )
        _, _, result = self._load_and_replay(config_path)
        case = result.cases[0]
        self.assertEqual(case["request_ids"], [1])
        self.assertEqual(case["phases"], ["chunk_prefill"])
        self.assertEqual(case["cached_prefix_lengths"], [1024])
        self.assertEqual(case["history_lens"], [1024])
        self.assertEqual(case["q_lens"], [1])

    def test_reservoir_determinism_seed_change_and_shortfall(self) -> None:
        rows = [_row(0, 2, 30)]
        config_path = self._config_path(
            rows,
            num_cases=5,
            sampling_end_ms=40,
            max_num_batched_tokens=2,
            prefill_chunk_size=2,
        )
        config = load_config(config_path)
        trace = load_mooncake_trace(config, derived_rng(config.seed, "arrival"))

        def replay_with_sampling_seed(seed: int):
            return replay_trace(
                trace,
                config,
                derived_rng(config.seed, "acceptance"),
                derived_rng(seed, "sampling"),
            )

        first = replay_with_sampling_seed(config.seed)
        second = replay_with_sampling_seed(config.seed)
        alternate = replay_with_sampling_seed(config.seed + 1)
        self.assertEqual(first.cases, second.cases)
        self.assertNotEqual(
            [case["sampled_time_us"] for case in first.cases],
            [case["sampled_time_us"] for case in alternate.cases],
        )

        short_config_path = self._config_path(
            [_row(0, 2, 2)], num_cases=3, sampling_end_ms=3
        )
        short_config = load_config(short_config_path)
        short_trace = load_mooncake_trace(
            short_config, derived_rng(short_config.seed, "arrival")
        )
        with self.assertRaisesRegex(ReplayError, "only 1 eligible steps"):
            replay_trace(
                short_trace,
                short_config,
                derived_rng(short_config.seed, "acceptance"),
                derived_rng(short_config.seed, "sampling"),
            )

    def test_cli_atomic_output_and_overwrite_protection(self) -> None:
        config_path = self._config_path(
            [_row(0, 6, 3)],
            prefill_chunk_size=3,
            max_num_batched_tokens=3,
            sampling_end_ms=4,
        )
        output = self.root / "nested" / "cases.jsonl"
        summary = run(config_path, output, force=False)
        first_bytes = output.read_bytes()
        self.assertEqual(summary["cases_written"], 1)
        self.assertEqual(len(first_bytes.splitlines()), 1)
        with self.assertRaisesRegex(ConfigError, "output already exists"):
            run(config_path, output, force=False)
        run(config_path, output, force=True)
        self.assertEqual(output.read_bytes(), first_bytes)

        override_output = self.root / "override_cases.jsonl"
        override_summary = run(
            config_path,
            override_output,
            force=False,
            num_cases=2,
        )
        self.assertEqual(override_summary["num_cases"], 2)
        self.assertEqual(override_summary["cases_written"], 2)
        override_cases = [
            json.loads(line) for line in override_output.read_text().splitlines()
        ]
        self.assertEqual(len(override_cases), 2)
        self.assertTrue(
            all(
                case["config_sha256"] == override_summary["config_sha256"]
                for case in override_cases
            )
        )

        matrix_output = self.root / "matrix_override_cases.jsonl"
        matrix_summary = run(
            config_path,
            matrix_output,
            force=False,
            num_cases=1,
            arrival_time_scale=2.0,
            dcp_size=4,
        )
        self.assertEqual(matrix_summary["arrival_time_scale"], "2.0")
        self.assertEqual(matrix_summary["dcp_size"], 4)
        matrix_case = json.loads(matrix_output.read_text(encoding="utf-8"))
        self.assertEqual(matrix_case["config_sha256"], matrix_summary["config_sha256"])

    def test_show_examples_is_focused_deterministic_and_capped(self) -> None:
        config_path = self._config_path(
            [_row(0, 2, 20)],
            num_cases=5,
            sampling_end_ms=30,
            max_num_batched_tokens=2,
            prefill_chunk_size=2,
        )
        _, _, result = self._load_and_replay(config_path)
        input_path = self.root / "viewer_cases.jsonl"
        input_path.write_text(
            "".join(json.dumps(case) + "\n" for case in result.cases),
            encoding="utf-8",
        )

        first_output = io.StringIO()
        first_summary = show_examples(
            input_path,
            num_examples=3,
            seed=99,
            full=False,
            output=first_output,
        )
        second_output = io.StringIO()
        show_examples(
            input_path,
            num_examples=3,
            seed=99,
            full=False,
            output=second_output,
        )
        self.assertEqual(first_output.getvalue(), second_output.getvalue())
        displayed = [
            json.loads(line) for line in first_output.getvalue().splitlines()
        ]
        self.assertEqual(first_summary["examples_displayed"], 3)
        self.assertEqual(len({case["case_id"] for case in displayed}), 3)
        self.assertTrue(
            all(tuple(case.keys()) == DISPLAY_FIELDS for case in displayed)
        )

        full_output = io.StringIO()
        capped_summary = show_examples(
            input_path,
            num_examples=100,
            seed=99,
            full=True,
            output=full_output,
        )
        self.assertEqual(capped_summary["examples_displayed"], 5)
        self.assertEqual(len(full_output.getvalue().splitlines()), 5)


if __name__ == "__main__":
    unittest.main()
