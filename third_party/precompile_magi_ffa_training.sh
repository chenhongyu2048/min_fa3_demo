#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
PYTHON=${PYTHON:-$ROOT_DIR/.venv/bin/python}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
SITE_PACKAGES=$(
    "$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
) || { echo "Cannot start Python: $PYTHON" >&2; exit 1; }
CUDNN_ROOT=${CUDNN_ROOT:-$SITE_PACKAGES/nvidia/cudnn}
NVSHMEM_ROOT=${NVSHMEM_ROOT:-$SITE_PACKAGES/nvidia/nvshmem}
NVIDIA_LIBRARY_PATH=$(
    "$PYTHON" - <<'PY'
import sysconfig
from pathlib import Path

root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()))
PY
)
MAGI_WORKSPACE=${MAGI_WORKSPACE:-$ROOT_DIR/.cache/mega_cp/magi_ffa_sm90_bf16_hd128}
FORCE_REBUILD=${FORCE_REBUILD:-0}
LOG_DIR=${LOG_DIR:-$ROOT_DIR/.cache/mega_cp/logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/precompile_magi_ffa_training.log}

NVCC_OUTPUT=$("$CUDA_HOME/bin/nvcc" --version) || { echo "Cannot start nvcc: $CUDA_HOME/bin/nvcc" >&2; exit 1; }
printf '%s\n' "$NVCC_OUTPUT"
grep -Eq 'release 12\.[0-9]+([,.]|$)' <<<"$NVCC_OUTPUT" || {
    echo "Expected CUDA toolkit 12.x at $CUDA_HOME" >&2; exit 1;
}
test -f "$NVSHMEM_ROOT/lib/libnvshmem_host.so.3" || {
    echo "Missing NVSHMEM host library under: $NVSHMEM_ROOT/lib" >&2
    exit 1
}

mkdir -p "$MAGI_WORKSPACE" "$LOG_DIR"
unset NVSHMEM_DIR

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDNN_ROOT/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$NVIDIA_LIBRARY_PATH${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$NVIDIA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST=9.0
export MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=90
export MAGI_ATTENTION_WORKSPACE_BASE="$MAGI_WORKSPACE"
export MAGI_ATTENTION_KERNEL_BACKEND=ffa
export MAGI_ATTENTION_RANGE_MERGE=0
export MAGI_ATTENTION_CATGQA=0
export MAGI_ATTENTION_NO_BUILD_CACHE=0
export MAGI_ATTENTION_BUILD_VERBOSE=${MAGI_ATTENTION_BUILD_VERBOSE:-1}
export MAX_JOBS=${MAX_JOBS:-8}
export NVCC_THREADS=${NVCC_THREADS:-2}
export FORCE_REBUILD

echo "Precompiling targeted Magi FFA training kernels"
echo "  Python:    $PYTHON"
echo "  CUDA:      $CUDA_HOME"
echo "  workspace: $MAGI_WORKSPACE"
echo "  log:       $LOG_FILE"

set -o pipefail
"$PYTHON" - <<'PY' 2>&1 | tee "$LOG_FILE"
import os
import shutil
from pathlib import Path

import torch
import magi_attention
from magi_attention.common.jit import env as jit_env
from magi_attention.functional._flex_flash_attn_jit import get_ffa_jit_spec

if torch.__version__ != "2.11.0+cu128" or not (torch.version.cuda or "").startswith("12."):
    raise RuntimeError(
        f"expected PyTorch 2.11.0+cu128/CUDA 12.x, got "
        f"{torch.__version__}/{torch.version.cuda}"
    )

specs = {}


def add_spec(
    *,
    direction: str,
    output_dtype: torch.dtype | None,
    disable_atomic: bool,
    dq_dtype: torch.dtype | None,
    dkv_dtype: torch.dtype | None,
) -> None:
    spec, uri = get_ffa_jit_spec(
        arch=(9, 0),
        direction=direction,
        head_dim=128,
        compute_dtype=torch.bfloat16,
        output_dtype=output_dtype,
        softcap=False,
        disable_atomic_reduction=disable_atomic,
        disable_dq_atomic_reduction=False,
        deterministic=False,
        ref_block_size=None,
        range_merge=False,
        swap_ab=False,
        pack_gqa=False,
        cat_gqa=False,
        pack_gqa_factor=1,
        block_sparse=False,
        index_sparse=False,
        bwd_inner_loop_k=False,
        profile_mode=False,
        return_max_logits=False,
        dq_dtype=dq_dtype,
        dkv_dtype=dkv_dtype,
        sparse_k_block_size=1,
    )
    specs[uri] = spec


# Distributed FFA may either accumulate partial output or write fresh output.
for disable_atomic in (False, True):
    add_spec(
        direction="fwd",
        output_dtype=torch.float32,
        disable_atomic=disable_atomic,
        dq_dtype=None,
        dkv_dtype=None,
    )

# Training backward uses FP32 partial dQ/dK/dV accumulation.
add_spec(
    direction="bwd",
    output_dtype=None,
    disable_atomic=False,
    dq_dtype=torch.float32,
    dkv_dtype=torch.float32,
)

print(f"Targeted FFA kernel count: {len(specs)}")
for uri in sorted(specs):
    print(f"  {uri}")
if len(specs) != 3:
    raise RuntimeError(f"expected 3 targeted kernels, got {len(specs)}")

aot_root = Path(magi_attention.__file__).resolve().parent / "lib"
force_rebuild = os.environ["FORCE_REBUILD"] == "1"
if not force_rebuild:
    missing = [
        uri for uri in specs
        if not any((aot_root / uri).glob("*.so"))
    ]
    if not missing:
        print("All targeted Magi FFA AOT artifacts already exist; skipping precompile")
        raise SystemExit(0)
if force_rebuild:
    print("Removing the three targeted JIT/AOT directories before rebuilding")
    for uri in specs:
        shutil.rmtree(jit_env.MAGI_ATTENTION_JIT_DIR / uri, ignore_errors=True)
        shutil.rmtree(aot_root / uri, ignore_errors=True)
for index, (uri, spec) in enumerate(sorted(specs.items()), start=1):
    print(f"[precompile {index}/{len(specs)}] building {uri}", flush=True)
    spec.build()
    src_dir = (jit_env.MAGI_ATTENTION_JIT_DIR / uri).resolve()
    dst_dir = (aot_root / uri).resolve()
    if src_dir.exists():
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    shared_objects = sorted(dst_dir.glob("*.so"))
    if not shared_objects:
        raise RuntimeError(f"no AOT shared object produced for {uri}")
    for path in shared_objects:
        print(f"AOT: {path} ({path.stat().st_size} bytes)")
    print(f"[precompile {index}/{len(specs)}] OK: {uri}", flush=True)

print("Targeted Magi FFA training kernels: OK")
PY

echo "Magi FFA precompile: OK"
