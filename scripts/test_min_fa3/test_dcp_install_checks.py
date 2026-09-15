"""An existing CUDA extension must not hide a missing DCP CPU installation."""
import contextlib
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
CHECKS = (
    ("setup_fresh_environment.sh", "verify_min_fa3"),
    ("third_party/setup_vllm_dcp.sh", "verify_min_fa3_extensions"),
)


def verification_code(path, function):
    source = (ROOT / path).read_text().split(f"{function}() {{", 1)[1]
    return source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


class InstallChecks(unittest.TestCase):
    def check(self, code, missing=None, outside=False):
        modules = {}
        for name in ("torch", "min_fa3_op", "_min_fa3_op", "_dcp_mega_planner"):
            module = types.ModuleType(name)
            module.__file__ = str(ROOT / (name + ".so"))
            modules[name] = module
        for name in ("forward_varlen", "backward_varlen", "forward_varlen_mega_ring", "backward_varlen_mega_ring"):
            setattr(modules["min_fa3_op"], name, lambda: None)
        for name in ("critical_wave_plan", "build_packed_queues"):
            if name != missing:
                setattr(modules["_dcp_mega_planner"], name, lambda: None)
        if missing == "module":
            modules["_dcp_mega_planner"] = None
        if outside:
            modules["_dcp_mega_planner"].__file__ = str(ROOT.parent / "_dcp_mega_planner.so")
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", ["-", str(ROOT)]), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "install-verifier", "exec"), {})

    def test_both_extensions_allow_reuse(self):
        for path, function in CHECKS:
            with self.subTest(script=path):
                self.check(verification_code(path, function))

    def test_old_or_missing_cpu_module_requires_build(self):
        for path, function in CHECKS:
            for missing in ("module", "critical_wave_plan", "build_packed_queues"):
                with self.subTest(script=path, missing=missing), self.assertRaises(ImportError):
                    self.check(verification_code(path, function), missing=missing)

    def test_cpu_extension_from_another_checkout_is_rejected(self):
        for path, function in CHECKS:
            with self.subTest(script=path), self.assertRaises(SystemExit):
                self.check(verification_code(path, function), outside=True)


if __name__ == "__main__":
    unittest.main()
