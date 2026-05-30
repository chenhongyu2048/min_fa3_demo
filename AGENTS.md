# AGENTS.md

This file provides local instructions for Codex when working inside `hopper/min_fa3_demo/`.

## Scope

This directory is a standalone minimal Hopper FlashAttention demo derived from the original FA3 Hopper forward path under `hopper/`.

It contains two sibling forward-only kernels:

- Fixed-layout BSHD forward
- Varlen forward

The goal of this directory is not to design a new attention implementation. The goal is to preserve a small, buildable, runnable, Python-callable demo that is clearly copied and trimmed from the original Hopper sources.

## Highest-priority rules

1. Preserve the "copied + trimmed" design.
   - Do not rewrite the main kernel path from scratch.
   - Do not replace copied Hopper structures with new custom abstractions just because they look simpler.
   - `params`, `launch`, `kernel`, `prologue`, `mainloop`, `epilogue`, and scheduler-related code should remain traceable to the original Hopper sources.

2. Preserve params provenance.
   - `include/min_fa3_params.h` and `include/min_fa3_varlen_params.h` are trimmed copies of the original Hopper forward params path.
   - If a field must be added or removed, keep original naming and layout style whenever possible.
   - Do not redesign params into a brand-new struct family.

3. Keep the demo self-contained.
   - Do not reintroduce direct `#include "hopper/..."` dependencies.
   - If a small Hopper helper is required, prefer copying it into `include/hopper_compat/` first, then trimming it there.
   - The extension should build from this directory with local headers plus CUTLASS, PyTorch, and CUDA.

4. Do not broaden feature scope unless explicitly requested.
   - This demo is intentionally narrow.
   - Avoid adding generality, extra template branches, or feature flags unless they are required by the task.

## Supported configurations

### BSHD path

- GPU: Hopper SM90 only
- Direction: forward only
- Dtype: `torch.bfloat16`
- Head dim: `128`
- Input layout:
  - `q: [B, S, QH, D]`
  - `k: [B, S, KVH, D]`
  - `v: [B, S, KVH, D]`
- Output layout:
  - `o: [B, S, QH, D]`
- Modes:
  - causal
  - noncausal
- GQA/MQA:
  - supported when `qhead % kvhead == 0`

### Varlen path

- GPU: Hopper SM90 only
- Direction: forward only
- Dtype: `torch.bfloat16`
- Head dim: `128`
- Input layout:
  - `q: [total_q, QH, D]`
  - `k: [total_k, KVH, D]`
  - `v: [total_k, KVH, D]`
  - `cu_seqlens_q: [B + 1]`
  - `cu_seqlens_k: [B + 1]`
- Modes:
  - causal
  - noncausal
- GQA/MQA:
  - supported when `qhead % kvhead == 0`

## Explicitly out of scope

Unless the user asks for it, do not add:

- backward
- fp16/fp8 or non-bf16 dtype support
- non-128 head dims
- non-SM90 architectures
- paged KV
- append KV
- rotary
- local attention
- softcap
- split-KV
- pack-gqa rearchitecture
- new public APIs unrelated to the minimal demo

## Important file ownership

Core copied-and-trimmed files:

- `include/min_fa3_params.h`
- `include/min_fa3_traits.h`
- `include/min_fa3_launch.h`
- `include/min_fa3_prologue.h`
- `include/min_fa3_epilogue.h`
- `include/min_fa3_mainloop.h`
- `include/min_fa3_kernel.h`
- `include/min_fa3_scheduler.h`
- `include/min_fa3_varlen_params.h`
- `include/min_fa3_varlen_traits.h`
- `include/min_fa3_varlen_launch.h`
- `include/min_fa3_varlen_scheduler.h`
- `csrc/min_fa3_kernel.cu`
- `csrc/min_fa3_launch.cu`
- `csrc/min_fa3_varlen_kernel.cu`
- `csrc/min_fa3_varlen_launch.cu`
- `csrc/min_fa3_varlen_prepare_scheduler.cu`
- `bindings.cpp`

Copied support headers live in:

- `include/hopper_compat/`

Do not edit generated artifacts unless the task explicitly requires it:

- `_min_fa3_op.so`
- `build/`
- `__pycache__/`
- `fa_min_demo_h200-*.out`

## Build

Default in-repo build:

```bash
cd hopper/min_fa3_demo
make
```

Standalone or out-of-tree style build with explicit CUTLASS path:

```bash
cd hopper/min_fa3_demo
CUTLASS_DIR=/path/to/cutlass make
```

`CUTLASS_DIR` may point either to the CUTLASS root or directly to its `include/` directory.

Clean:

```bash
cd hopper/min_fa3_demo
make clean
```

## Test

BSHD correctness:

```bash
cd hopper/min_fa3_demo
python test_min_fa3.py --b 1 --seqlen 128 --qhead 8 --kvhead 8 --headdim 128 --mode both
```

Varlen correctness:

```bash
cd hopper/min_fa3_demo
python test_min_fa3_varlen.py --b 1 --seqlen 128 --qhead 8 --kvhead 8 --headdim 128 --mode both
```

Benchmarks:

```bash
cd hopper/min_fa3_demo
python benchmark.py --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 32 --headdim 128 --mode both
python benchmark_varlen.py --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 --headdim 128 --mode both
```

Note:

- `benchmark.py` and `benchmark_varlen.py` try to compare against PyTorch, FA2, and FA3 when those imports are available.
- The demo itself does not require FA2 or FA3 to build.

## Unified CLI conventions

Keep the Python test and benchmark entry points aligned around the same arguments:

- `--b`
- `--seqlen`
- `--qhead`
- `--kvhead`
- `--headdim`
- `--mode`

Do not reintroduce older rectangular `SqxSk` CLI formats unless the user explicitly requests them.

## Modification guidelines

1. Prefer surgical edits.
   - Change the smallest number of files and lines needed.
   - Preserve existing naming, file roles, and directory layout.

2. Keep provenance comments at the top of copied files.
   - When adding a new copied helper file, include a short source comment.

3. Keep BSHD and varlen as sibling paths.
   - Do not collapse them into one confusing API layer.
   - Shared concepts are fine, but avoid premature abstraction.

4. Be careful with performance conclusions.
   - Current benchmarks measure end-to-end op time, not just kernel body time.
   - Host allocations and scheduler-prep overhead matter for short-sequence comparisons.

5. Validate on actual supported hardware when possible.
   - This demo targets SM90 only.
   - If Hopper hardware is unavailable, state that clearly rather than guessing.

## When adding dependencies

If a new helper is needed, prefer this order:

1. Reuse an existing local file in `include/` or `include/hopper_compat/`
2. Copy a small required Hopper helper into `include/hopper_compat/`
3. Use CUTLASS/CuTe includes directly

Avoid creating a new dependency on broader repository internals when a local copied helper is sufficient.

## Slurm

This directory contains `run.slurm` for cluster execution. If you change script arguments or benchmark entry points, keep `run.slurm` and `README.md` synchronized.
