# Third-party source and CUDA environment

The parent repository pins the source revisions used by the single-layer
Mega-CP benchmark as Git submodules:

| Source | Path | Pinned revision |
| --- | --- | --- |
| Megatron-LM | `third_party/Megatron-LM` | `5eb0744c700e3791f3992fdce08cc41d5a469326` |
| Transformer Engine | `third_party/TransformerEngine` | `4329ff84bfbdaa778a33cba02a15fb0807c64689` |
| MagiAttention | `third_party/MagiAttention` | `872717e1f88fa6938593e452a28a41597c849a00` |
| vLLM | `third_party/vllm` | `c6fe94b4d5b418fa213af0e5884eddd304333dcd` |

The gitlinks in the parent repository are the version source of truth. The
installation scripts validate those gitlinks and every recursive submodule;
they do not checkout an independently hard-coded revision.

Official FlashAttention is not a top-level dependency of this environment.
The single-layer benchmark falls back collectively to the in-repository
`min_fa3_op` when `flash_attn_interface` is absent, which is the configuration
used by the completed smoke test. MagiAttention retains its own recorded
FlashAttention submodule, but that nested checkout is not installed as the
top-level `flash-attn` package.

## Prepare the repository and Python environment

The root `setup_fresh_environment.sh` is the only fresh-clone environment
orchestrator. It bootstraps repository-local uv 0.12.4, prepares the base
environment, and invokes the component-specific installers in this directory
for native work. On the shared filesystem, run the two phases on their
corresponding nodes:

```bash
# Internet-connected node.
./setup_fresh_environment.sh prepare

# H20 CUDA node.
CUDA_VISIBLE_DEVICES=0 ./setup_fresh_environment.sh install
```

The scripted environment is pinned to Linux x86_64, Python 3.12, uv 0.12.4,
PyTorch 2.11.0+cu128, and CUDA toolkit 12.8. `prepare` needs Git and network
access. `install` needs GNU Make plus at least one visible SM90 Hopper GPU; it
does not require all eight benchmark GPUs to be available during compilation.

The optional vLLM service integration follows the same component-installer
layout:

```bash
# Internet-connected node, after the root prepare action.
./third_party/setup_vllm_dcp.sh prepare

# H20 CUDA node, after the root install action.
./third_party/setup_vllm_dcp.sh verify
```

It installs the pinned `third_party/vllm` source and local benchmark plugin
into the shared `.venv`; it is intentionally separate from the base
environment orchestrator. A vLLM-only environment may instead run
`CUDA_VISIBLE_DEVICES=0 ./third_party/setup_vllm_dcp.sh install`; that action
builds min-FA3 but intentionally omits TE, MagiAttention, and their AOT
artifacts.

`verify` repeats all source, package, extension, Megatron integration, and
three-kernel AOT checks without rebuilding. `all` is available only when one
machine has both network access and the supported CUDA environment. The script
writes phase logs below `.cache/mega_cp/logs/fresh_environment/`.

The equivalent manual steps follow.

On the Internet-connected node, from the repository root:

```bash
git submodule update --init --checkout --recursive

UV_CACHE_DIR=.cache/uv \
uv sync --frozen --no-install-project \
  --group build \
  --group transformer-layer
```

The lock file selects the official PyTorch CUDA 12.8 index and is restricted
to the supported Linux x86_64 environment. After syncing, verify that Python
reports exactly `torch==2.11.0+cu128` and `torch.version.cuda == "12.8"`.

An exact `uv sync` removes undeclared packages, including source-built TE and
MagiAttention. Run it before installing those packages. If the base lock must
be synchronized again without removing the completed native installations,
add `--inexact`.

`fetch` is available when only one dependency needs initialization:

```bash
third_party/install_transformer_engine.sh fetch
third_party/install_magi_attention.sh fetch
```

The normal fresh-clone command remains the single recursive `git submodule
update` above.

## CUDA-node installation

Build the in-repository extension first, then TE, MagiAttention, and the three
targeted Magi FFA training kernels:

```bash
make PYTHON=.venv/bin/python

third_party/install_transformer_engine.sh install
third_party/install_magi_attention.sh install
third_party/precompile_magi_ffa_training.sh

third_party/install_transformer_engine.sh verify
third_party/install_magi_attention.sh verify
```

The two installers use `--offline --no-build-isolation --no-deps`; all Python
runtime and build dependencies must therefore already be present from the
locked base environment. Both packages are installed non-editable from the
submodule paths.

The Magi installer disables its broad upstream FFA prebuild matrix. The next
step compiles only the three BF16/SM90/head-dim-128 forward/backward variants
used by this benchmark and installs them into Magi's package-local `lib/` tree.
The fresh-environment orchestrator removes only those three matching cache/AOT
directories before compilation, preventing reuse across an older Torch/CUDA
environment without deleting unrelated Magi caches.

The scripts automatically discover the NVIDIA wheel library directories.
TE 2.17.1 looks for the PyTorch CUDA runtime under the legacy package path
`nvidia/cuda_cudart`, while the PyTorch 2.11 cu128 wheel uses
`nvidia/cuda_runtime`. The TE script creates a repository-venv-local
compatibility link between those names; it never modifies the system CUDA
toolkit or a global Python environment.

Magi installation deliberately leaves `NVSHMEM_DIR` unset and uses the wheel's
versioned `libnvshmem_host.so.3`. No system or wheel-library symlink is created.

Magi FFA JIT/AOT output and the precompile log default to ignored paths below:

```text
.cache/mega_cp/magi_ffa_sm90_bf16_hd128
.cache/mega_cp/logs
```

Reinstalling MagiAttention may replace its package-local `lib/` tree. Run the
targeted precompile script again after every Magi reinstall, but do not repeat
it when the package and environment have not changed.

For the complete build history, failure analysis, and exact FFA specs, see
[`../docs/TRANSFORMER_ENGINE_MAGI_BUILD_GUIDE.md`](../docs/TRANSFORMER_ENGINE_MAGI_BUILD_GUIDE.md).
