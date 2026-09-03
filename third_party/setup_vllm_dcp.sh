#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VLLM_DIR="$ROOT_DIR/third_party/vllm"
VENV_DIR=${VENV_DIR:-$ROOT_DIR/.venv}
UV=${UV:-}
UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
MAX_JOBS=${MAX_JOBS:-8}
NVCC_THREADS=${NVCC_THREADS:-2}
VLLM_USE_PRECOMPILED=${VLLM_USE_PRECOMPILED:-1}
ACTION=${1:-}
VLLM_COMMIT=c6fe94b4d5b418fa213af0e5884eddd304333dcd
# The source revision has no x86_64 CUDA-12 wheel. This nearest published
# ancestor uses the same cache-op schemas and a stable-libtorch ABI compatible
# with the pinned torch 2.11 build.
VLLM_CORE_WHEEL_COMMIT=f25953cc59f9b4ba9b04b16228d2b86dcfbcbdb1
VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_PRECOMPILED_WHEEL_COMMIT:-$VLLM_CORE_WHEEL_COMMIT}
VLLM_PRECOMPILED_WHEEL_VARIANT=${VLLM_PRECOMPILED_WHEEL_VARIANT:-cu129}
PYTORCH_CUDA_INDEX=${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}

usage() {
    cat <<EOF
Usage: $0 prepare|install|verify|all

  prepare  On the networked node, install pinned vLLM with its precompiled
           extensions and install the local plugin into the shared .venv.
           This action does not require CUDA hardware or nvcc.
  install  On a CUDA 12.8 Hopper node, build min-FA3, reinstall the local
           plugin, and run complete runtime verification. Run prepare first.
  verify   Verify revisions/imports without rebuilding.
  all      Run prepare and install on one networked CUDA Hopper node.

Run './setup_fresh_environment.sh prepare' on the networked node first.
The benchmark sources are kept below 'infer/'.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

resolve_uv() {
    local candidate resolved
    if [[ -n "$UV" ]]; then
        if [[ "$UV" == */* ]]; then
            [[ -x "$UV" ]] || die "UV=$UV is unavailable or not executable"
            resolved=$UV
        else
            resolved=$(command -v "$UV" 2>/dev/null) ||
                die "UV=$UV is unavailable on PATH"
        fi
    else
        for candidate in \
            "$ROOT_DIR/.cache/tools/uv-0.12.4/uv" \
            "$ROOT_DIR/.cache/tools/uv-0.12.4/bin/uv"; do
            if [[ -x "$candidate" ]]; then
                resolved=$candidate
                break
            fi
        done
        if [[ -z "${resolved:-}" ]]; then
            resolved=$(command -v uv 2>/dev/null) ||
                die "uv 0.12.4 is unavailable; set UV=/path/to/uv"
        fi
    fi
    UV=$resolved
    [[ -x "$UV" ]] || die "uv is unavailable at $UV"
    local actual
    actual=$($UV --version 2>/dev/null | awk 'NR == 1 {print $2}')
    [[ "$actual" == "0.12.4" ]] ||
        die "expected uv 0.12.4, got ${actual:-<unknown>} at $UV"
    echo "uv: $UV ($actual)"
}

verify_sources() {
    [[ -d "$VLLM_DIR" ]] || die "missing initialized vLLM submodule"
    local actual
    actual=$(git -C "$VLLM_DIR" rev-parse HEAD)
    [[ "$actual" == "$VLLM_COMMIT" ]] ||
        die "vLLM commit mismatch: expected $VLLM_COMMIT, got $actual"
    [[ -z "$(git -C "$VLLM_DIR" status --porcelain --untracked-files=no)" ]] ||
        die "vLLM submodule has tracked modifications"
    [[ -x "$VENV_DIR/bin/python" ]] || die "missing $VENV_DIR/bin/python"
}

verify_core_wheel() {
    local core needed
    core="$VLLM_DIR/vllm/_C_stable_libtorch.abi3.so"
    [[ -f "$core" ]] || die "missing vLLM core extension at $core"
    command -v readelf >/dev/null || die "readelf is required to verify vLLM core"
    needed=$(readelf -d "$core")
    [[ "$needed" != *libcudart.so.13* ]] ||
        die "vLLM core wheel requires CUDA 13; rerun prepare with the pinned cu129 wheel"
    [[ "$needed" == *libcudart.so.12* ]] ||
        die "vLLM core wheel is not the pinned CUDA 12 build"
}

pin_torch_cuda12_packages() {
    # vLLM's CUDA requirements intentionally omit the local +cu128 build tag.
    # Without an explicit PyTorch index uv may satisfy torchvision/torchaudio
    # from a CUDA-13 wheel, which fails before any text model code is loaded.
    "$UV" pip install --python "$VENV_DIR/bin/python" \
        --index "$PYTORCH_CUDA_INDEX" \
        --index-strategy unsafe-best-match --no-deps \
        'torch==2.11.0+cu128' \
        'torchvision==0.26.0+cu128' \
        'torchaudio==2.11.0+cu128'
}

verify_torch_cuda12_packages() {
    "$VENV_DIR/bin/python" - <<'PY'
from importlib.metadata import version

expected = {
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "torchaudio": "2.11.0+cu128",
}
for name, wanted in expected.items():
    actual = version(name)
    if actual != wanted:
        raise SystemExit(f"expected {name} {wanted}, got {actual}")
PY
    local so needed
    for so in \
        "$VENV_DIR"/lib/python*/site-packages/torchvision/_C.so \
        "$VENV_DIR"/lib/python*/site-packages/torchaudio/lib/libtorchaudio.abi3.so; do
        [[ -f "$so" ]] || die "missing CUDA PyTorch extension $so"
        needed=$(readelf -d "$so")
        [[ "$needed" != *libcudart.so.13* ]] ||
            die "$so requires CUDA 13; rerun prepare to pin the cu128 PyTorch wheels"
        [[ "$needed" == *libcudart.so.12* ]] ||
            die "$so is not a CUDA 12 PyTorch extension"
    done
}

remove_incompatible_optional_cuda_packages() {
    # The vLLM CUDA requirements currently resolve torchcodec to a CUDA-13
    # wheel (its image library needs libcudart/libnvrtc.so.13).  vLLM imports
    # the multimedia module while loading text-model config, and that module
    # only treats a missing torchcodec as optional; an ABI OSError is fatal.
    # This benchmark is text-only, so leave the optional package absent and let
    # vLLM use its PlaceholderModule path.
    "$UV" pip uninstall --python "$VENV_DIR/bin/python" torchcodec >/dev/null 2>&1 || true
}

verify_runtime() {
    verify_sources
    verify_core_wheel
    verify_torch_cuda12_packages
    PYTHONPATH="$ROOT_DIR/infer/vllm_plugin/src:$ROOT_DIR/infer:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$VENV_DIR/bin/python" \
        - "$ROOT_DIR" "$VLLM_COMMIT" "$VLLM_CORE_WHEEL_COMMIT" <<'PY'
import sys
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import torch
import vllm
import _min_fa3_op
import min_fa3_vllm_plugin

root = Path(sys.argv[1]).resolve()
expected_vllm = sys.argv[2]
expected_core_wheel = sys.argv[3]
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("vLLM:", version("vllm"), Path(vllm.__file__).resolve())
print("plugin:", Path(min_fa3_vllm_plugin.__file__).resolve())
print("min-FA3:", Path(_min_fa3_op.__file__).resolve())
if torch.__version__ != "2.11.0+cu128" or torch.version.cuda != "12.8":
    raise SystemExit("expected torch 2.11.0+cu128")
if not Path(vllm.__file__).resolve().is_relative_to(root / "third_party" / "vllm"):
    raise SystemExit("vLLM is not imported from the pinned submodule")
if not Path(_min_fa3_op.__file__).resolve().is_relative_to(root):
    raise SystemExit("min-FA3 extension is not imported from this repository")
if not Path(min_fa3_vllm_plugin.__file__).resolve().is_relative_to(
        root / "infer" / "vllm_plugin"
):
    raise SystemExit("vLLM plugin is not imported from this repository")
min_fa3_vllm_plugin.register()
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

path = AttentionBackendEnum.CUSTOM.get_path()
assert path.startswith("min_fa3_vllm_plugin.")
assert "flash_attn" not in path.lower()
assert "HistoryDecodeBenchConnector" in KVConnectorFactory._registry
assert hasattr(torch.ops._C_cache_ops, "reshape_and_cache_flash")
assert hasattr(torch.ops._C_cache_ops, "cp_gather_cache")
for backend, class_path in min_fa3_vllm_plugin.BACKEND_CLASS_PATHS.items():
    module_name, class_name = class_path.rsplit(".", 1)
    backend_cls = getattr(import_module(module_name), class_name)
    assert backend_cls.get_name() == "CUSTOM"
    assert backend_cls.get_impl_cls().runner_kind == backend
assert not any(name.startswith("vllm.vllm_flash_attn") for name in sys.modules)
print("vLLM commit:", expected_vllm)
print("vLLM core wheel commit:", expected_core_wheel)
print("vLLM DCP environment: OK")
PY
}

prepare_runtime() {
    verify_sources
    resolve_uv
    mkdir -p "$UV_CACHE_DIR"
    export UV_CACHE_DIR
    [[ "$VLLM_USE_PRECOMPILED" == 1 ]] ||
        die "VLLM_USE_PRECOMPILED must remain 1; full vLLM builds fetch FlashAttention"
    if [[ -z "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" ]]; then
        [[ "$VLLM_PRECOMPILED_WHEEL_COMMIT" == "$VLLM_CORE_WHEEL_COMMIT" ]] ||
            die "CUDA 12 requires core wheel commit $VLLM_CORE_WHEEL_COMMIT; "\
                "unset VLLM_PRECOMPILED_WHEEL_COMMIT or use an exact compatible local wheel"
        [[ "$VLLM_PRECOMPILED_WHEEL_VARIANT" == "cu129" ]] ||
            die "CUDA 12 requires VLLM_PRECOMPILED_WHEEL_VARIANT=cu129"
    fi
    export VLLM_USE_PRECOMPILED
    export VLLM_PRECOMPILED_WHEEL_COMMIT
    export VLLM_PRECOMPILED_WHEEL_VARIANT
    echo "vLLM source commit:     $VLLM_COMMIT"
    echo "vLLM core wheel commit: $VLLM_PRECOMPILED_WHEEL_COMMIT"
    echo "vLLM core wheel variant: $VLLM_PRECOMPILED_WHEEL_VARIANT"
    "$UV" pip install --python "$VENV_DIR/bin/python" \
        -r "$VLLM_DIR/requirements/build/cuda.txt"
    "$UV" pip install --python "$VENV_DIR/bin/python" \
        --no-build-isolation --editable "$VLLM_DIR"
    pin_torch_cuda12_packages
    "$UV" pip install --python "$VENV_DIR/bin/python" \
        --no-build-isolation --no-deps --editable "$ROOT_DIR/infer/vllm_plugin"
    remove_incompatible_optional_cuda_packages
    verify_core_wheel
    verify_torch_cuda12_packages
    "$VENV_DIR/bin/python" - "$ROOT_DIR" <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path

import vllm

root = Path(sys.argv[1]).resolve()
print("prepared vLLM:", version("vllm"), Path(vllm.__file__).resolve())
if not Path(vllm.__file__).resolve().is_relative_to(root / "third_party" / "vllm"):
    raise SystemExit("vLLM is not imported from the pinned submodule")
PY
}

install_runtime() {
    verify_sources
    resolve_uv
    mkdir -p "$UV_CACHE_DIR"
    export UV_CACHE_DIR
    [[ -x "$CUDA_HOME/bin/nvcc" ]] || die "CUDA toolkit is unavailable at $CUDA_HOME"
    "$VENV_DIR/bin/python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    print("prepared vLLM:", version("vllm"))
except PackageNotFoundError as exc:
    raise SystemExit("vLLM is not installed; run setup_vllm_dcp.sh prepare first") from exc
PY
    verify_core_wheel
    verify_torch_cuda12_packages
    export CUDA_HOME MAX_JOBS NVCC_THREADS
    export PATH="$VENV_DIR/bin:$CUDA_HOME/bin:$PATH"
    make -C "$ROOT_DIR" clean
    make -C "$ROOT_DIR" PYTHON="$VENV_DIR/bin/python"
    "$UV" pip install --python "$VENV_DIR/bin/python" \
        --no-build-isolation --no-deps --editable "$ROOT_DIR/infer/vllm_plugin"
    verify_runtime
}

case "$ACTION" in
    prepare)
        prepare_runtime
        ;;
    install)
        install_runtime
        ;;
    verify)
        verify_runtime
        ;;
    all)
        prepare_runtime
        install_runtime
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
