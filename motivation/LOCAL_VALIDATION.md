# Local validation — 2026-09-16

Branch: `codex/motivation-v2`.
Fetched main baseline: `5b0bf71240d4adf0168f9d3d991ff4ed952b08e1`.
Local interpreter: Python 3.13.3 on macOS. No project `.venv`, torch, numpy,
matplotlib or CUDA environment was available. No SSH connection was made.

| Check | Result |
|---|---|
| AST parse of 19 `motivation/*.py` files and `min_fa3_dcp.py` | 20 passed |
| `bash -n` on the seven README shell command blocks | 7 passed; no new shell files |
| `git diff --check` | Passed |
| `python3 -m unittest motivation.test_config motivation.test_results motivation.test_interfaces -v` | 15 passed, 2 skipped because real torch is unavailable |
| 4/8 GPU `--dry-run` | Passed, no torch import or result-directory creation |
| T3 raw manifests across GPU counts | Identical: five datasets × 20 cases, seed=0 |
| Existing DCP and T2 binding positional argument prefixes | Preserved; new optional argument appended |
| Existing DCP `_backend_args` builder AST | Unchanged |
| Production MegaRing forward binding body | Unchanged from main |
| CUDA compilation/execution/performance | Not executed |
| Figure rendering | Not executed: matplotlib absent; analysis/trace plotting inputs tested |

The two skipped tests are actual static/execution layout consistency and the
T2 CP4/CP8 hierarchy test. They import real torch and will run in the project
environment; they do not substitute fake tensor or CUDA implementations.
Synthetic timing records exercise the D1 selection and summary rules only.

The following existing CPU tests were attempted but failed during dependency
import, before executing tests:

```text
balancer.test_balancer                         missing numpy
ring_test.load_balance_bench.test_topology      missing torch
scripts.test_min_fa3.test_dcp_topology           missing torch
scripts.test_min_fa3.test_dcp_mega_batch         missing torch
```

Dry-run configuration verified:

| GPUs | T1/T2 local lengths for B=1,2,4,8,16 | D1 |
|---|---|---|
| 4 | 32768, 16384, 8192, 4096, 2048 | Q32/KV2/D128, TP4/DCP2, smoke |
| 8 | 16384, 8192, 4096, 2048, 1024 | Q32/KV4/D128, TP8/DCP2, formal |

T1/T2 global tokens are exactly 131072 in every case. The public T3 sampler
targets 131072 and produces raw totals of 131072–132864 for seed=0. These
original lengths are preserved identically across GPU counts; execution
padding is separately recorded.

CUDA review checked signature propagation, CP4/CP8 template dispatch,
hierarchy level sizes, compute-only readiness initialization, appended pybind
defaults, and trace-specialized launch selection. The trace buffer is created
once per runner; communication CTAs overwrite slots 0/3 and compute CTAs
overwrite 1/2/4 on every invocation. Unused slots remain zero. No claim of
CUDA compilation, runtime correctness, speedup, or reduced tracing overhead
follows from this source review.

Pending after SSH authorization: project environment setup/build, rerun the
dependency-blocked CPU tests, four-GPU full-length smoke including T3 backward,
eight-GPU validation, three independent uninstrumented D1 candidate runs,
trace replay/slot validation, and measurement of tracing overhead. D1
selection remains `pending_gpu_measurement` until those measurements exist.
