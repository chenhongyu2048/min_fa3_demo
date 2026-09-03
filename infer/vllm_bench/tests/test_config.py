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

from vllm_bench.serve import build_serve_command, dcp_size, main, model_dir


class ServiceConfigTest(unittest.TestCase):
    @staticmethod
    def _args(backend: str, kv_heads: int = 1) -> Namespace:
        return Namespace(
            backend=backend,
            kv_heads=kv_heads,
            host="127.0.0.1",
            port=8000,
            served_model_name="dummy",
            gpu_memory_utilization=0.9,
            kv_cache_memory_bytes=None,
            num_hidden_layers=32,
            seed=42,
            fill_mean=0.015,
            mega_max_total_q=4096,
            mega_max_num_splits=8,
            mega_num_comm_sm=8,
            mega_block_n="auto",
            dry_run=True,
        )

    def test_model_configs_keep_llama_shape_and_change_kvh(self) -> None:
        for kv_heads in (1, 2, 4):
            config = json.loads((model_dir(kv_heads) / "config.json").read_text())
            self.assertEqual(config["hidden_size"], 4096)
            self.assertEqual(config["num_hidden_layers"], 32)
            self.assertEqual(config["num_attention_heads"], 32)
            self.assertEqual(config["num_key_value_heads"], kv_heads)
            self.assertEqual(config["max_position_embeddings"], 131072)
            self.assertEqual(4096 // 32, 128)
            self.assertEqual(dcp_size(kv_heads), 8 // kv_heads)

    def test_backend_selector_changes_only_attention_path(self) -> None:
        commands = {}
        environments = {}
        for backend in ("vllm-ag-rs", "vllm-a2a", "mega"):
            environments[backend], commands[backend] = build_serve_command(
                self._args(backend)
            )
        common_flags = [
            "--tensor-parallel-size",
            "--decode-context-parallel-size",
            "--max-num-batched-tokens",
            "--enforce-eager",
            "--no-enable-flashinfer-autotune",
            "--no-enable-prefix-caching",
        ]
        for flag in common_flags:
            self.assertTrue(all(flag in command for command in commands.values()))
        for backend, command in commands.items():
            self.assertIn("CUSTOM", command)
            self.assertNotIn("FLASH_ATTN", command)
            self.assertEqual(
                environments[backend]["MIN_FA3_DCP_BACKEND"], backend
            )
            self.assertIn("--hf-overrides", command)
            self.assertIn('{"num_hidden_layers":32}', command)
            self.assertIn(
                str(Path(__file__).resolve().parents[3] / "infer/vllm_plugin/src"),
                environments[backend]["PYTHONPATH"],
            )

    def test_explicit_kv_cache_size_is_forwarded(self) -> None:
        args = self._args("mega")
        args.kv_cache_memory_bytes = 1 << 30
        _, command = build_serve_command(args)
        index = command.index("--kv-cache-memory-bytes")
        self.assertEqual(command[index + 1], str(1 << 30))

    def test_custom_port_is_forwarded(self) -> None:
        args = self._args("mega")
        args.port = 18000
        _, command = build_serve_command(args)
        index = command.index("--port")
        self.assertEqual(command[index + 1], "18000")

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
        self.assertEqual(config.max_num_splits, 8)
        self.assertIsNone(config.block_n)
        with patch.dict(os.environ, {"MEGA_DCP_BLOCK_N": "64"}, clear=True):
            with self.assertRaises(ValueError):
                MegaRuntimeConfig.from_env()

    def test_plugin_backend_selection_is_explicit(self) -> None:
        from min_fa3_vllm_plugin import selected_backend_class_path

        expected_suffixes = {
            "mega": "MegaDCPAttentionBackend",
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

    def test_supported_service_config_uses_pinned_vllm_model_api(self) -> None:
        import torch

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
            dtype=torch.bfloat16,
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
                    has_full_cudagraphs=lambda: False
                )
            ),
            speculative_config=None,
        )
        validate_service_config(config)


if __name__ == "__main__":
    unittest.main()
