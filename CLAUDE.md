# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is a standalone, minimal Hopper FlashAttention demo focused on a **copied-and-trimmed** FA3 forward path. Keep that provenance intact:

- Do **not** redesign the kernel stack from scratch.
- Keep `params`, `launch`, `kernel`, `prologue`, `mainloop`, `epilogue`, and scheduler code traceable to the original Hopper implementation.
- Keep the demo self-contained; prefer local headers under `include/` and `include/hopper_compat/` over reintroducing dependencies on a larger Hopper tree.

The repo intentionally supports a narrow configuration:

- Hopper / SM90 only
- forward only
- `torch.bfloat16` only
- head dim `128` only
- GQA/MQA only when `qhead % kvhead == 0`

Out of scope unless explicitly requested: backward, non-SM90 support, non-bf16 dtypes, non-128 head dims, paged KV, append KV, rotary, local attention, softcap, split-KV, and public API expansion.

## Common commands

## Build

```bash
make
```

If CUTLASS is not available via the vendored `third_party/cutlass/include`, point `CUTLASS_DIR` at either the CUTLASS root or its `include/` directory:

```bash
CUTLASS_DIR=/path/to/cutlass make
```

Clean build artifacts:

```bash
make clean
```

## Tests

Basic fixed-layout correctness test:

```bash
python test_min_fa3.py --b 1 --seqlen 128 --qhead 8 --kvhead 8 --headdim 128 --mode both
```

Basic varlen correctness test:

```bash
python test_min_fa3_varlen.py --b 1 --seqlen 128 --qhead 8 --kvhead 8 --headdim 128 --mode both
```

Single test case with a manual persistent-grid override:

```bash
python test_min_fa3.py --b 1 --seqlen 128 --qhead 8 --kvhead 8 --headdim 128 --mode causal --manual-block-count 132
```

Local ring-attention test:

```bash
python test_min_fa3_varlen_ring_local.py --b 2 --seqlen 128 --qhead 16 --kvhead 8 --num-comp-sm 1 --num-comm-sm 1 --mode both
```

Multi-rank single-node tests must be run with `torchrun`:

```bash
torchrun --nproc_per_node=2 test_parallel_remote_load.py --shape 256x384 --src-rank 0
torchrun --nproc_per_node=2 test_min_fa3_varlen_ring_multi_rank.py --b 2 --seqlen 128,256 --qhead 16 --kvhead 8 --src-rank 0 --num-comp-sm 1 --num-comm-sm 1 --mode both
```

## Benchmarks

Fixed-layout benchmark:

```bash
python benchmark.py --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 32 --headdim 128 --mode both
```

Varlen benchmark:

```bash
python benchmark_varlen.py --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 --headdim 128 --mode both
```

Remote-load benchmark:

```bash
torchrun --nproc_per_node=2 benchmark_parallel_remote_load.py --shape 4096x4096 --src-rank 0
```

## Cluster run

The repo includes `run.slurm` for cluster execution:

```bash
sbatch run.slurm
```

If you change benchmark or test entry points, keep `run.slurm` and `README.md` in sync.

## Linting

No dedicated lint target or lint config was found in this repo. Do not invent one in follow-up edits unless the user asks for it.

## Architecture

## Public API layer

`min_fa3_op.py` is the only Python-facing module. It is a thin wrapper over the compiled `_min_fa3_op` extension and exposes:

- `forward(...)` for fixed-layout BSHD attention
- `forward_varlen(...)` for flattened varlen attention
- `forward_varlen_ring(...)` for ring-attention experiments
- `create_parallel_tensor(...)`, `parallel_remote_load(...)`, and `parallel_remote_load_vec(...)` for ThunderKittens-based local-node IPC / remote-load helpers

Keep Python changes small and ergonomic; the actual algorithm and launch behavior live in the extension.

## Extension build and binding layer

`setup.py` builds a single CUDA extension named `_min_fa3_op` from:

- `bindings.cpp` for the fixed-layout and varlen PyTorch bindings
- `csrc/min_fa3_*.cu` for launch, kernel instantiation, and varlen scheduler prep
- `csrc/min_fa3_varlen_ring_*.cu` for the ring-attention path
- `csrc/parallel/remote_load*.cu` for the ThunderKittens remote-load helpers

`setup.py` is also where CUTLASS include resolution, `libcuda.so` discovery, and SM90a compile flags live. This repo is intentionally hard-wired to `sm_90a`.

## Three sibling execution paths

### 1. Fixed-layout BSHD forward

Main files:

- `bindings.cpp`
- `include/min_fa3_params.h`
- `include/min_fa3_launch.h`
- `csrc/min_fa3_launch.cu`
- `include/min_fa3_kernel.h`
- `include/min_fa3_mainloop.h`
- `include/min_fa3_epilogue.h`
- `include/min_fa3_scheduler.h`

Flow:

1. Python calls `min_fa3_op.forward(...)` with contiguous BSHD tensors.
2. `bindings.cpp` validates tensors and builds the host-side params struct.
3. `csrc/min_fa3_launch.cu` dispatches causal vs noncausal launch.
4. `include/min_fa3_launch.h` assembles the CUTLASS/CuTe launch arguments.
5. The actual SM90 kernel is composed from the copied prologue/mainloop/epilogue/scheduler stack in `include/min_fa3_*.h`.

### 2. Varlen forward

Main files:

- `bindings.cpp`
- `include/min_fa3_varlen_params.h`
- `include/min_fa3_varlen_launch.h`
- `include/min_fa3_varlen_scheduler.h`
- `csrc/min_fa3_varlen_prepare_scheduler.cu`
- `csrc/min_fa3_varlen_launch.cu`
- `csrc/min_fa3_varlen_kernel.cu`

This is not just “the same kernel with different shapes.” The varlen path has an extra scheduling-prep phase:

1. Python calls `forward_varlen(...)` with flattened `[total_tokens, H, D]` tensors and CUDA `int32` `cu_seqlens`.
2. `bindings.cpp` validates inputs and allocates scheduler metadata.
3. `csrc/min_fa3_varlen_prepare_scheduler.cu` prepares metadata for the dynamic persistent scheduler.
4. `csrc/min_fa3_varlen_launch.cu` dispatches the causal/noncausal varlen kernel.

Keep fixed-layout and varlen as sibling paths; do not collapse them into a single over-abstracted API or kernel path.

### 3. Ring attention + remote-load path

Main files:

- `csrc/min_fa3_varlen_ring_bindings.cu`
- `include/min_fa3_varlen_ring_launch.h`
- `csrc/min_fa3_varlen_ring_launch.cu`
- `include/parallel/remote_load.h`
- `csrc/parallel/remote_load.cu`
- `csrc/parallel/remote_load_bindings.cu`

This path layers communication onto the varlen path:

- compute CTAs run the local varlen attention work
- communication CTAs prefetch remote K/V through ThunderKittens parallel-tensor IPC helpers

`forward_varlen_ring(...)` should be understood as **varlen attention plus explicit communication CTAs**, not as a separate redesign of the attention stack.

The remote-load helpers are also exposed independently for focused testing and benchmarking.

## File-role guide

- `include/min_fa3_params.h` and `include/min_fa3_varlen_params.h`: trimmed copies of the original Hopper forward params path; preserve naming/layout style when editing.
- `include/hopper_compat/`: local copied helpers needed to keep the demo self-contained.
- `csrc/min_fa3_kernel.cu` and `csrc/min_fa3_varlen_kernel.cu`: explicit instantiation / compilation anchors for the heavy templated kernel stack.
- `test_min_fa3.py` and `test_min_fa3_varlen.py`: correctness checks against PyTorch SDPA.
- `test_min_fa3_varlen_ring_local.py`, `test_min_fa3_varlen_ring_multi_rank.py`, and `test_parallel_remote_load.py`: validation for the ring / remote-load functionality.
- `benchmark.py` and `benchmark_varlen.py`: end-to-end timing against PyTorch and, when installed, FA2 / FA3.

## Validation expectations

- These tests enforce the supported hardware/configuration at runtime; if you cannot run on Hopper SM90, say so explicitly.
- `benchmark.py` and `benchmark_varlen.py` compare against FA2 / FA3 only when those packages import successfully. The demo does not require them to build.
- End-to-end timings include host-side work such as allocations and scheduler prep, so avoid drawing kernel-only conclusions from benchmark output.

## Editing guardrails

- Prefer surgical edits in `include/`, `csrc/`, `bindings.cpp`, and `min_fa3_op.py`.
- Preserve provenance comments at the tops of copied files.
- Avoid editing generated artifacts such as `_min_fa3_op.so`, `build/`, and `__pycache__/`.
- Keep CLI conventions aligned across tests and benchmarks: `--b`, `--seqlen`, `--qhead`, `--kvhead`, `--headdim`, and `--mode`.
- Do not reintroduce older rectangular `SqxSk` CLI formats unless the task explicitly asks for them.
