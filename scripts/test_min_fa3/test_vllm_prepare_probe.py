"""The vLLM reuse probe must fall back to installation without exiting Bash."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class VllmPrepareProbeTests(unittest.TestCase):
    def test_reuse_requires_all_checks_to_pass(self):
        source = (ROOT / "third_party/setup_vllm_dcp.sh").read_text()
        functions = "\n".join(
            re.search(r"^" + name + r"\(\) \{\n.*?^\}", source, re.M | re.S).group()
            for name in ("die", "verify_core_wheel", "verify_prepared_vllm")
        )
        condition = next(line for line in source.splitlines()
                         if line.strip().startswith("if ") and "verify_prepared_vllm" in line)
        cases = (
            ("missing_core", False, 0, 0, "INSTALL"),
            ("wrong_torch", True, 1, 0, "INSTALL"),
            ("bad_import", True, 0, 1, "INSTALL"),
            ("prepared", True, 0, 0, "REUSE"),
        )
        for force in ("0", "1"):
            for name, core_exists, torch_status, import_status, expected in cases:
                with self.subTest(case=name, force=force), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    (root / "vllm").mkdir()
                    if core_exists:
                        (root / "vllm/_C_stable_libtorch.abi3.so").touch()
                    (root / "bin").mkdir()
                    python = root / "bin/python"
                    python.write_text(f"#!/usr/bin/env bash\nexit {import_status}\n")
                    python.chmod(0o755)
                    script = "\n".join((
                        "set -euo pipefail", functions,
                        'ROOT_DIR="$1"; VLLM_DIR="$1"; VENV_DIR="$1"; FORCE_REBUILD="$2"',
                        f"verify_torch_cuda12_packages() {{ return {torch_status}; }}",
                        condition, "echo REUSE", "else", "echo INSTALL", "fi",
                        "echo CONTINUED",
                    ))
                    result = subprocess.run(
                        ["bash", "-c", script, "probe", tmp, force],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.splitlines(), [expected, "CONTINUED"])


if __name__ == "__main__":
    unittest.main()
