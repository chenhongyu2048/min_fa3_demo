"""Configuration tests run on the actual CPU modules without CUDA mocks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .config import DATASETS, ROOT, TOKENS, d1_config, t3_manifest, uniform_manifest
from .prepare_d1_cases import candidate_manifest
from .run import build_commands, parse_args


class ConfigurationTests(unittest.TestCase):
    def test_uniform_budget_and_local_length(self):
        for world in (4, 8):
            cases = uniform_manifest(world)["cases"]
            self.assertEqual([c["batch"] for c in cases], [1, 2, 4, 8, 16])
            for case in cases:
                self.assertEqual(case["context"], TOKENS)
                self.assertEqual(case["batch"] * world * case["local_seqlen"], TOKENS)
                self.assertEqual(case["local_seqlen"] % 256, 0)

    def test_d1_head_sharding(self):
        for world in (4, 8):
            config = d1_config(world)
            self.assertEqual(config["tp_size"], world)
            self.assertEqual(world // config["kvhead"], config["dcp_size"])
            self.assertEqual(config["configuration_role"], "smoke" if world == 4 else "formal")

    def test_raw_cases_use_public_sampler(self):
        from balancer import generate_dataset_length_cases
        manifest = t3_manifest()
        self.assertEqual(tuple(manifest["datasets"]), DATASETS)
        for dataset, cases in manifest["datasets"].items():
            self.assertEqual(len(cases), 20)
            self.assertEqual([c["raw_lengths"] for c in cases],
                             generate_dataset_length_cases(dataset, TOKENS, 0, 20))
            for case in cases:
                self.assertEqual(case["raw_tokens"], sum(case["raw_lengths"]))
                self.assertGreaterEqual(case["raw_tokens"], TOKENS)

    def test_commands_preserve_defaults_and_explicit_smoke(self):
        for world in (4, 8):
            args = parse_args(["--gpus", str(world)])
            commands = build_commands(args, Path("results"))
            self.assertEqual(len(commands), 16)
            for command in commands:
                self.assertEqual(command[0], sys.executable)
                self.assertIn(f"--nproc_per_node={world}", command)
                module = command[command.index("--module") + 1]
                expected = (10, 40) if module.endswith("transformer_layer") else (
                    (20, 30) if module.endswith("decode_sm_trace") else (40, 60))
                self.assertEqual(tuple(int(command[command.index(flag) + 1]) for flag in ("--warmup", "--iters")), expected)
            smoke = build_commands(parse_args(["--gpus", str(world), "--case-limit", "1",
                                              "--warmup", "0", "--iters", "2"]), Path("results"))
            self.assertEqual(len(smoke), 8)
            self.assertEqual(uniform_manifest(world)["cases"][0]["context"], TOKENS)

    def test_dry_runs_do_not_import_torch_or_write_results(self):
        values = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            for world in (4, 8):
                code = ("import sys; from motivation.run import main; "
                        f"main(['--gpus', '{world}', '--dry-run', '--output-dir', {str(output)!r}]); "
                        "assert 'torch' not in sys.modules")
                result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True,
                                        capture_output=True, text=True)
                values.append(json.loads(result.stdout))
            self.assertFalse(output.exists())
        self.assertEqual(values[0]["t3_manifest"], values[1]["t3_manifest"])

    def test_d1_candidates_match_main_and_remain_pending(self):
        candidate = json.loads((ROOT / "motivation/d1_candidates.json").read_text())
        expected = candidate_manifest(ROOT / candidate["source"])
        self.assertEqual(candidate["cases"], expected["cases"])
        self.assertEqual(candidate["selection_status"], "pending_gpu_measurement")
        kinds = [c["workload_kind"] for c in candidate["cases"]]
        self.assertGreaterEqual(kinds.count("decode_only_q16"), 2)
        self.assertGreaterEqual(kinds.count("mixed"), 2)


if __name__ == "__main__":
    unittest.main()
