#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
VLLM_DIR="$ROOT_DIR/third_party/vllm"
VENV_DIR=${VENV_DIR:-$ROOT_DIR/.venv}
UV=${UV:-}
UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
MAX_JOBS=${MAX_JOBS:-8}
NVCC_THREADS=${NVCC_THREADS:-2}
VLLM_USE_PRECOMPILED=${VLLM_USE_PRECOMPILED:-1}
FORCE_REBUILD=${FORCE_REBUILD:-0}
ACTION=${1:-}
VLLM_COMMIT=c6fe94b4d5b418fa213af0e5884eddd304333dcd

# Keep the original precompiled-wheel selection.
# CUDA compatibility is not verified during prepare.
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
           Only file presence and package versions are checked;
           CUDA shared-library dependencies are not checked.
  install  On a CUDA 12.x Hopper node, build min-FA3 when needed, install the
           local plugin, and run runtime verification. Run prepare first.
  verify   Verify revisions and runtime imports without rebuilding.
           Requires the built min-FA3 extension and its runtime dependencies.
  all      Run prepare and install on one networked CUDA Hopper node.

Set FORCE_REBUILD=1 to force a min-FA3 rebuild during install.

On a node without CUDA, run only:
  $0 prepare

The benchmark sources are kept below 'infer/'.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

resolve_uv() {
    local candidate resolved output actual
    local -a candidates=()

    if [[ -n "$UV" ]]; then
        if [[ "$UV" == */* ]]; then
            candidates+=("$UV")
        else
            resolved=$(command -v "$UV" 2>/dev/null) ||
                die "UV=$UV is unavailable on PATH"
            candidates+=("$resolved")
        fi
    else
        candidates+=(
            "$ROOT_DIR/.cache/tools/uv-0.12.4/uv"
            "$ROOT_DIR/.cache/tools/uv-0.12.4/bin/uv"
        )

        if resolved=$(command -v uv 2>/dev/null); then
            candidates+=("$resolved")
        fi
    fi

    for candidate in "${candidates[@]}"; do
        if output=$("$candidate" --version 2>&1); then
            actual=$(printf '%s\n' "$output" | awk 'NR == 1 {print $2}')

            if [[ "$actual" == "0.12.4" ]]; then
                UV=$candidate
                echo "uv: $UV ($actual)"
                return 0
            fi

            printf 'Rejected uv at %s: expected 0.12.4, got %s\n' \
                "$candidate" "${actual:-<unknown>}" >&2
        else
            printf 'Cannot start uv at %s:\n%s\n' \
                "$candidate" "$output" >&2
        fi
    done

    die "no working uv 0.12.4 found; set UV=/path/to/uv"
}

verify_sources() {
    [[ -d "$VLLM_DIR" ]] || die "missing initialized vLLM submodule"

    local actual
    actual=$(git -C "$VLLM_DIR" rev-parse HEAD)

    [[ "$actual" == "$VLLM_COMMIT" ]] ||
        die "vLLM commit mismatch: expected $VLLM_COMMIT, got $actual"

    [[ -z "$(git -C "$VLLM_DIR" status --porcelain --untracked-files=no)" ]] ||
        die "vLLM submodule has tracked modifications"

    "$VENV_DIR/bin/python" -c \
        'import sys; print("Python:", sys.executable, sys.version.split()[0])' ||
        die "Python environment cannot be started: $VENV_DIR/bin/python"
}

verify_core_wheel() {
    # Check file presence only; do not inspect or load the shared library.
    local core
    core="$VLLM_DIR/vllm/_C_stable_libtorch.abi3.so"

    [[ -f "$core" ]] ||
        die "missing vLLM core extension at $core"
}

verify_prepared_vllm() {
    verify_core_wheel
    verify_torch_cuda12_packages
    "$VENV_DIR/bin/python" - "$ROOT_DIR" <<'PY'
import sys
from pathlib import Path

import vllm

root = Path(sys.argv[1]).resolve()
if not Path(vllm.__file__).resolve().is_relative_to(root / "third_party" / "vllm"):
    raise SystemExit("vLLM is not imported from the pinned submodule")
PY
}

pin_torch_cuda12_packages() {
    # Explicitly select the pinned CUDA 12.8 PyTorch wheels.
    "$UV" pip install --python "$VENV_DIR/bin/python" \
        --index "$PYTORCH_CUDA_INDEX" \
        --index-strategy unsafe-best-match --no-deps \
        'torch==2.11.0+cu128' \
        'torchvision==0.26.0+cu128' \
        'torchaudio==2.11.0+cu128'
}

verify_torch_cuda12_packages() {
    # Check package metadata only.
    # Do not import torch/torchvision/torchaudio or inspect their binaries.
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
    print(f"{name}: {actual}")
PY
}

remove_incompatible_optional_cuda_packages() {
    # This benchmark is text-only. Keep optional torchcodec absent.
    "$UV" pip uninstall --python "$VENV_DIR/bin/python" \
        torchcodec >/dev/null 2>&1 || true
}

verify_min_fa3_extensions() {
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$VENV_DIR/bin/python" - "$ROOT_DIR" <<'PY'
import sys
from pathlib import Path

import torch
import _min_fa3_op
import _dcp_mega_planner
from _dcp_mega_planner import critical_wave_plan, build_packed_queues

root = Path(sys.argv[1]).resolve()
for module in (_min_fa3_op, _dcp_mega_planner):
    path = Path(module.__file__).resolve()
    if path.parent != root:
        raise SystemExit(f"{module.__name__} was not built in this repository")
    print(f"Native extension: {path}")
PY
}

verify_runtime() {
    verify_sources
    verify_core_wheel
    verify_torch_cuda12_packages
    verify_min_fa3_extensions

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

if torch.__version__ != "2.11.0+cu128" or not (torch.version.cuda or "").startswith("12."):
    raise SystemExit("expected torch 2.11.0+cu128")

if not Path(vllm.__file__).resolve().is_relative_to(
    root / "third_party" / "vllm"
):
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
    expected_runner = "mega" if backend == "mega-fa3-native" else backend
    assert backend_cls.get_impl_cls().runner_kind == expected_runner

assert not any(
    name.startswith("vllm.vllm_flash_attn")
    for name in sys.modules
)

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

    echo "vLLM source commit:      $VLLM_COMMIT"
    echo "vLLM core wheel commit:  $VLLM_PRECOMPILED_WHEEL_COMMIT"
    echo "vLLM core wheel variant: $VLLM_PRECOMPILED_WHEEL_VARIANT"

    if verify_prepared_vllm >/dev/null 2>&1; then
        echo "vLLM source and precompiled core are already installed; skipping source install"
    else
        "$UV" pip install --python "$VENV_DIR/bin/python" \
            -r "$VLLM_DIR/requirements/build/cuda.txt"

        "$UV" pip install --python "$VENV_DIR/bin/python" \
            --no-build-isolation --editable "$VLLM_DIR"

        pin_torch_cuda12_packages
    fi

    "$UV" pip install --python "$VENV_DIR/bin/python" \
        --no-build-isolation --no-deps \
        --editable "$ROOT_DIR/infer/vllm_plugin"

    remove_incompatible_optional_cuda_packages

    verify_core_wheel
    verify_torch_cuda12_packages

    "$VENV_DIR/bin/python" - "$ROOT_DIR" <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path

import vllm

root = Path(sys.argv[1]).resolve()
vllm_path = Path(vllm.__file__).resolve()

print("prepared vLLM:", version("vllm"), vllm_path)

if not vllm_path.is_relative_to(root / "third_party" / "vllm"):
    raise SystemExit("vLLM is not imported from the pinned submodule")
PY

    echo "Prepare completed."
    echo "CUDA shared-library dependency checks were skipped."
    echo "Run install later on the CUDA 12.x Hopper node."
}

install_runtime() {
    verify_sources
    resolve_uv

    mkdir -p "$UV_CACHE_DIR"
    export UV_CACHE_DIR

    local nvcc_output
    nvcc_output=$("$CUDA_HOME/bin/nvcc" --version) ||
        die "CUDA compiler cannot be started: $CUDA_HOME/bin/nvcc"
    printf '%s\n' "$nvcc_output"
    grep -Eq 'release 12\.[0-9]+([,.]|$)' <<<"$nvcc_output" ||
        die "expected CUDA toolkit 12.x under CUDA_HOME=$CUDA_HOME"

    "$VENV_DIR/bin/python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    print("prepared vLLM:", version("vllm"))
except PackageNotFoundError as exc:
    raise SystemExit(
        "vLLM is not installed; run setup_vllm_dcp.sh prepare first"
    ) from exc
PY

    verify_core_wheel
    verify_torch_cuda12_packages

    export CUDA_HOME MAX_JOBS NVCC_THREADS
    export PATH="$VENV_DIR/bin:$CUDA_HOME/bin:$PATH"

    if [[ "$FORCE_REBUILD" == 1 ]]; then
        make -C "$ROOT_DIR" clean
        make -C "$ROOT_DIR" PYTHON="$VENV_DIR/bin/python"
    elif verify_min_fa3_extensions >/dev/null 2>&1; then
        echo "In-repository CUDA and DCP CPU extensions pass verification; skipping build"
    else
        make -C "$ROOT_DIR" PYTHON="$VENV_DIR/bin/python"
    fi

    "$UV" pip install --python "$VENV_DIR/bin/python" \
        --no-build-isolation --no-deps \
        --editable "$ROOT_DIR/infer/vllm_plugin"

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
