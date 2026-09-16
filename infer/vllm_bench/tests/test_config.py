from __future__ import annotations

import json
import os
import sys
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_bench.serve import BACKENDS, build_serve_command, dcp_size, main, model_dir


class ServiceConfigTest(unittest.TestCase):
    @staticmethod
    def _args(backend: str, kv_heads: int = 1) -> Namespace:
        return Namespace(
            backend=backend,
            kv_heads=kv_heads,
            tp_size=8,
            host="127.0.0.1",
            port=8000,
            served_model_name="dummy",
            gpu_memory_utilization=0.9,
            kv_cache_memory_bytes=None,
            num_hidden_layers=48,
            max_num_seqs=64,
            seed=42,
            fill_mean=0.015,
            history_kv_mode="synthetic",
            mega_max_total_q=4096,
            mega_max_num_splits=128,
            mega_num_comm_sm=8,
            mega_block_n="auto",
            dry_run=True,
        )

    def test_model_configs_keep_qwen3_moe_shape_and_change_kvh(self) -> None:
        for kv_heads in (1, 2, 4):
            config = json.loads((model_dir(kv_heads) / "config.json").read_text())
            self.assertEqual(config["hidden_size"], 2048)
            self.assertEqual(config["num_hidden_layers"], 48)
            self.assertEqual(config["num_attention_heads"], 32)
            self.assertEqual(config["num_key_value_heads"], kv_heads)
            self.assertEqual(config["max_position_embeddings"], 131072)
            self.assertEqual(config["head_dim"], 128)
            self.assertEqual(config["architectures"], ["Qwen3MoeForCausalLM"])
            self.assertEqual(config["num_experts"], 128)
            self.assertEqual(config["num_experts_per_tok"], 8)
            self.assertEqual(config["moe_intermediate_size"], 768)
            self.assertEqual(config["rope_scaling"], {
                "rope_type": "yarn", "factor": 4.0,
                "original_max_position_embeddings": 32768,
            })
            self.assertEqual(dcp_size(kv_heads), 8 // kv_heads)

    def test_backend_selector_changes_only_attention_path(self) -> None:
        commands = {}
        environments = {}
        for backend in BACKENDS:
            environments[backend], commands[backend] = build_serve_command(
                self._args(backend)
            )
        common_flags = [
            "--tensor-parallel-size",
            "--enable-expert-parallel",
            "--decode-context-parallel-size",
            "--max-num-batched-tokens",
            "--compilation-config",
            "--cudagraph-metrics",
            "--no-enable-flashinfer-autotune",
            "--no-enable-prefix-caching",
        ]
        for flag in common_flags:
            self.assertTrue(all(flag in command for command in commands.values()))
        for backend, command in commands.items():
            self.assertNotIn("--enforce-eager", command)
            compilation = json.loads(command[command.index("--compilation-config") + 1])
            self.assertEqual(compilation["cudagraph_mode"], "FULL")
            self.assertEqual(max(compilation["cudagraph_capture_sizes"]), 4096)
            self.assertIn("CUSTOM", command)
            self.assertNotIn("FLASH_ATTN", command)
            self.assertEqual(
                environments[backend]["MIN_FA3_DCP_BACKEND"], backend
            )
            self.assertEqual(
                environments[backend]["VLLM_USE_FLASHINFER_SAMPLER"], "0"
            )
            self.assertEqual(environments[backend]["VLLM_MOE_SKIP_PADDING"], "0")
            self.assertEqual(
                environments[backend]["MEGA_DCP_SCHEDULER_HEURISTIC"],
                "0" if backend == "mega-fa3-native" else "auto",
            )
            self.assertEqual(
                environments[backend]["MEGA_DCP_MAX_NUM_SPLITS"], "128"
            )
            self.assertIn("--hf-overrides", command)
            self.assertIn('{"num_hidden_layers":48}', command)
            self.assertIn(
                str(Path(__file__).resolve().parents[3] / "infer/vllm_plugin/src"),
                environments[backend]["PYTHONPATH"],
            )

    def test_four_gpu_smoke_topology(self) -> None:
        args = self._args("mega")
        args.tp_size = 4
        _, command = build_serve_command(args)
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "4")
        self.assertEqual(command[command.index("--decode-context-parallel-size") + 1], "4")
        args.kv_heads = 4
        with self.assertRaisesRegex(ValueError, "DCP"):
            build_serve_command(args)

    def test_batch_limit_matches_plugin_capacity(self) -> None:
        from vllm_bench.serve import parser

        for backend in BACKENDS:
            for limit in (64, 128):
                with self.subTest(backend=backend, limit=limit):
                    argv = ["--backend", backend]
                    if limit != 64:
                        argv.extend(["--max-num-seqs", str(limit)])
                    with patch.dict(os.environ, {"MEGA_DCP_MAX_BATCH": "32"}):
                        env, command = build_serve_command(parser().parse_args(argv))
                    self.assertEqual(command[command.index("--max-num-seqs") + 1], str(limit))
                    self.assertEqual(env["MEGA_DCP_MAX_BATCH"], str(limit))

    def test_batch_limit_fits_token_capacity(self) -> None:
        args = self._args("mega")
        for limit in (0, -1, 4097):
            args.max_num_seqs = limit
            with self.assertRaisesRegex(ValueError, "max_num_seqs"):
                build_serve_command(args)
        args.max_num_seqs = 128
        args.mega_max_total_q = 64
        with self.assertRaisesRegex(ValueError, "max_num_seqs"):
            build_serve_command(args)

    def test_history_kv_mode_reaches_connector_for_all_backends(self) -> None:
        from vllm_bench.serve import parser

        for backend in BACKENDS:
            for mode in ("synthetic", "paged"):
                argv = ["--backend", backend]
                if mode == "paged":
                    argv.extend(["--history-kv-mode", mode])
                _, command = build_serve_command(parser().parse_args(argv))
                transfer = json.loads(command[command.index("--kv-transfer-config") + 1])
                self.assertEqual(transfer["kv_connector_extra_config"]["history_kv_mode"], mode)

    def test_explicit_kv_cache_size_is_forwarded(self) -> None:
        args = self._args("mega")
        args.kv_cache_memory_bytes = 1 << 30
        _, command = build_serve_command(args)
        index = command.index("--kv-cache-memory-bytes")
        self.assertEqual(command[index + 1], str(1 << 30))

    def test_flashinfer_sampler_override_is_preserved(self) -> None:
        args = self._args("mega")
        with patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "1"}):
            environment, _ = build_serve_command(args)
        self.assertEqual(environment["VLLM_USE_FLASHINFER_SAMPLER"], "1")

    def test_custom_port_is_forwarded(self) -> None:
        args = self._args("mega")
        args.port = 18000
        _, command = build_serve_command(args)
        index = command.index("--port")
        self.assertEqual(command[index + 1], "18000")

    def test_balanced_routing_default_and_original_routing_override(self) -> None:
        key = "VLLM_MOE_ROUTING_SIMULATION_STRATEGY"
        for backend in BACKENDS:
            with patch.dict(os.environ, {}, clear=True):
                env, _ = build_serve_command(self._args(backend))
            self.assertEqual(env[key], "min_fa3_balanced")
            with patch.dict(os.environ, {key: ""}):
                env, _ = build_serve_command(self._args(backend))
            self.assertEqual(env[key], "")

    def test_dry_run_does_not_print_inherited_environment(self) -> None:
        output = StringIO()
        argv = ["serve.py", "--backend", "mega", "--dry-run"]
        with (
            patch.object(sys, "argv", argv),
            patch.dict(os.environ, {"BENCHMARK_TEST_SECRET": "do-not-print"}),
            redirect_stdout(output),
        ):
            main()
        rendered = output.getvalue()
        self.assertNotIn("BENCHMARK_TEST_SECRET", rendered)
        self.assertNotIn("do-not-print", rendered)
        self.assertIn("MEGA_DCP_MAX_TOTAL_Q", rendered)
        self.assertIn("MIN_FA3_DCP_BACKEND", rendered)


class PluginPureConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        plugin_src = Path(__file__).resolve().parents[3] / "infer/vllm_plugin/src"
        self.plugin_src = str(plugin_src)
        self.path_patch = patch.dict(os.environ, {}, clear=False)
        self.path_patch.start()
        sys.path.insert(0, self.plugin_src)

    def tearDown(self) -> None:
        sys.path.remove(self.plugin_src)
        self.path_patch.stop()

    def test_history_validation(self) -> None:
        from min_fa3_vllm_plugin.config import validate_history_tokens

        self.assertEqual(validate_history_tokens(3, 4), 3)
        for invalid in (True, -1, 4, 1.5):
            with self.assertRaises(ValueError):
                validate_history_tokens(invalid, 4)

    def test_short_history_warmup_detection_is_narrow(self) -> None:
        from min_fa3_vllm_plugin.config import is_vllm_short_history_warmup

        self.assertTrue(
            is_vllm_short_history_warmup(
                [1, 1], [2, 2], 8, saw_zero_history_warmup=True
            )
        )
        # It must not become a general escape hatch for short real requests.
        self.assertFalse(
            is_vllm_short_history_warmup(
                [1, 1], [2, 2], 8, saw_zero_history_warmup=False
            )
        )
        self.assertFalse(
            is_vllm_short_history_warmup(
                [1, 1], [1, 2], 8, saw_zero_history_warmup=True
            )
        )
        self.assertFalse(
            is_vllm_short_history_warmup(
                [2, 1], [2, 2], 8, saw_zero_history_warmup=True
            )
        )

    def test_runtime_env_defaults_and_validation(self) -> None:
        from min_fa3_vllm_plugin.config import MegaRuntimeConfig

        with patch.dict(os.environ, {}, clear=True):
            config = MegaRuntimeConfig.from_env()
        self.assertEqual(config.max_total_q, 4096)
        self.assertEqual(config.max_num_splits, 128)
        self.assertIsNone(config.block_n)
        self.assertIsNone(config.scheduler_heuristic)
        with patch.dict(
            os.environ, {"MEGA_DCP_SCHEDULER_HEURISTIC": "0"}, clear=True
        ):
            self.assertFalse(MegaRuntimeConfig.from_env().scheduler_heuristic)
        with patch.dict(
            os.environ, {"MEGA_DCP_SCHEDULER_HEURISTIC": "1"}, clear=True
        ):
            self.assertTrue(MegaRuntimeConfig.from_env().scheduler_heuristic)
        with patch.dict(
            os.environ, {"MEGA_DCP_SCHEDULER_HEURISTIC": "invalid"}, clear=True
        ):
            with self.assertRaises(ValueError):
                MegaRuntimeConfig.from_env()
        with patch.dict(os.environ, {"MEGA_DCP_BLOCK_N": "64"}, clear=True):
            with self.assertRaises(ValueError):
                MegaRuntimeConfig.from_env()

    def test_plugin_backend_selection_is_explicit(self) -> None:
        from min_fa3_vllm_plugin import selected_backend_class_path

        expected_suffixes = {
            "mega": "MegaDCPAttentionBackend",
            "mega-fa3-native": "MegaDCPAttentionBackend",
            "vllm-ag-rs": "VLLMAGRSDCPAttentionBackend",
            "vllm-a2a": "VLLMA2ADCPAttentionBackend",
        }
        for backend, suffix in expected_suffixes.items():
            with patch.dict(
                os.environ, {"MIN_FA3_DCP_BACKEND": backend}, clear=False
            ):
                self.assertTrue(selected_backend_class_path().endswith(suffix))
        with patch.dict(
            os.environ, {"MIN_FA3_DCP_BACKEND": "flash-attn"}, clear=False
        ):
            with self.assertRaises(ValueError):
                selected_backend_class_path()

    def test_plugin_backend_does_not_import_flash_attention_backend(self) -> None:
        backend = (
            Path(__file__).resolve().parents[3]
            / "infer/vllm_plugin/src/min_fa3_vllm_plugin/mega_backend.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn(
            "from vllm.v1.attention.backends.flash_attn import", backend
        )
        self.assertIn(
            "scheduler_heuristic=runtime.scheduler_heuristic", backend
        )
        self.assertNotIn("scheduler_heuristic=True", backend)

    def test_supported_service_config_uses_pinned_vllm_model_api(self) -> None:
        from min_fa3_vllm_plugin.config import validate_service_config

        parallel = SimpleNamespace(
            tensor_parallel_size=8,
            decode_context_parallel_size=8,
            cp_kv_cache_interleave_size=1,
            use_ubatching=False,
        )
        model = SimpleNamespace(
            get_num_attention_heads=lambda config: 4,
            get_total_num_kv_heads=lambda: 1,
            get_head_size=lambda: 128,
            get_num_kv_heads=lambda config: 1,
            dtype="torch.bfloat16",
        )
        config = SimpleNamespace(
            model_config=model,
            parallel_config=parallel,
            scheduler_config=SimpleNamespace(
                max_num_seqs=64,
                max_num_batched_tokens=4096,
                enable_chunked_prefill=True,
            ),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            compilation_config=SimpleNamespace(
                cudagraph_mode=SimpleNamespace(
                    has_full_cudagraphs=lambda: True
                )
            ),
            speculative_config=None,
        )
        validate_service_config(config)
        parallel.tensor_parallel_size = 4
        parallel.decode_context_parallel_size = 4
        model.get_num_attention_heads = lambda config: 8
        validate_service_config(config)
        config.scheduler_config.max_num_seqs = 128
        with patch.dict(os.environ, {"MEGA_DCP_MAX_BATCH": "128"}):
            validate_service_config(config)
        with patch.dict(os.environ, {"MEGA_DCP_MAX_BATCH": "64"}):
            with self.assertRaisesRegex(ValueError, "MEGA_DCP_MAX_BATCH"):
                validate_service_config(config)


if __name__ == "__main__":
    unittest.main()
