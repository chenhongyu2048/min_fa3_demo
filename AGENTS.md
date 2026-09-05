# Repository Guidelines

## Core Principles

1. **Think Before Coding** — State assumptions explicitly, surface ambiguities and tradeoffs, push back when a simpler path exists, and ask for clarification rather than guessing. When multiple interpretations are plausible, present them instead of picking silently.

2. **Simplicity First** — Write the minimum code that solves the stated problem. No speculative features, single-use abstractions, unrequested configurability, or error handling for impossible cases. If 200 lines could be 50, rewrite it. Test: would a senior engineer call this overcomplicated?

3. **Surgical Changes** — Touch only what the task requires. Don't "improve" adjacent code, refactor what isn't broken, or reformat to your preference — match existing style even if you'd do it differently. Clean up orphans (imports, variables, functions) that *your* changes made unused, but leave pre-existing dead code alone — mention it instead of deleting. Every changed line must trace directly to the user's request.

4. **Goal-Driven Execution** — Convert tasks into verifiable success criteria before coding:
   - "Fix the bug" → "Write a test that reproduces it, then make it pass"
   - "Add validation" → "Write tests for invalid inputs, then make them pass"
   - "Refactor X" → "Ensure tests pass before and after"
   
   For multi-step work, state a brief plan with a verification check per step, then loop until each check passes.

5. **Context Health Canary** — End every response with the marker " 喵~ (ฅ• . •ฅ)ﻌﻌﻌ♥ ". This serves as a liveness signal for instruction adherence: if the marker disappears, degrades (wrong format, skipped numbering, drift to plain "meow"), treat it as evidence that this CLAUDE.md is being crowded out of effective attention — context is likely saturated. Do not suppress the marker for "serious" outputs (code blocks, formal docs, tool calls) — append it on a new line after everything else. The marker's reliability is the signal; skipping it "just this once" defeats the purpose.

6. **先读再改，优先小范围编辑，避免整文件重写**; 不做与当前实验无关的重构或完整测试套件。

7. **严禁过度防御**: 我们不是在写安全攻防论文。除非任务明确要求，否则不主动增加 SHA256/hash/checksum；不为极低概率（大致 <0.1%）的 corner case 编写防御代码；能确定性解决的问题，不引入 rubric 或模糊判断。

8. **路径不硬编码**; 优先沿用项目现有环境和依赖管理方式，不确定时再询问。

## Project Structure

This repository is a minimal Hopper/SM90 FlashAttention and Mega-CP benchmark
stack. CUDA extension sources live in `csrc/`, public and copied CUDA headers in
`include/`, and Python bindings/wrappers at the repository root (`min_fa3_op.py`,
`min_fa3_dcp.py`). Correctness tests and multi-rank experiments are under
`scripts/test_min_fa3/` and `scripts/test_mega_ring/`; higher-level benchmark
frontends are in `ring_test/`, `dcp_test/`, `infer/`, and `balancer/`. Dataset
helpers are in `dataset/`, documentation and figures in `docs/` and `paper/`,
and pinned external projects are submodules under `third_party/`.

## Build and Development Commands

Install the locked Python environment with `uv sync --frozen --no-install-project
--group build --group transformer-layer`, then build the in-place extension:

```bash
make PYTHON=.venv/bin/python
# Use an external CUTLASS checkout when needed:
CUTLASS_DIR=/path/to/cutlass make PYTHON=.venv/bin/python
make clean
```

The build requires CUDA, a linkable driver library, CUTLASS, and an SM90-capable
GPU for execution. `./setup_fresh_environment.sh prepare|install|verify` is the
supported end-to-end setup path.

## Coding Style and Naming

Use four-space indentation in Python and follow existing type hints, docstrings,
and `snake_case` module/function names. Keep CUDA/C++20 code consistent with
neighboring files: descriptive `snake_case` filenames, conventional CUDA/C++
types, and small, surgical changes. Preserve provenance comments and the copied-
and-trimmed Hopper structure; do not edit generated `.so`, `build/`, or cache
files. No repository-wide formatter is configured, so review diffs carefully.

## Testing Guidelines

Name Python tests `test_*.py` and keep focused cases near the subsystem they
exercise. CPU-only checks can run without GPUs:

```bash
python -m unittest balancer.test_balancer ring_test.load_balance_bench.test_topology \
  scripts.test_min_fa3.test_dcp_topology scripts.test_min_fa3.test_dcp_mega_batch
```

After building, run the relevant kernel module (for example,
`python -m scripts.test_min_fa3.test_min_fa3 ...`). Distributed tests use
`torchrun`; document GPU count and `CUDA_VISIBLE_DEVICES` in results. Validate
both correctness and benchmark changes on Hopper hardware when possible.

## Commits and Pull Requests

Use short imperative commit subjects, optionally with the established prefixes
`feat:`, `fix:`, `perf:`, `bench:`, `build:`, `refactor:`, or `chore:` (for
example, `fix: guard varlen workspace reuse`). Pull requests should explain the
behavior or performance change, identify affected paths, list exact build/test
commands and GPU topology, and include benchmark tables or plots when results
change. Call out new dependencies, submodule updates, environment variables,
and any limitations or untested hardware explicitly.
