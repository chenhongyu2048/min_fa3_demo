#!/usr/bin/env bash
set -euo pipefail

# Two-node Transformer Engine setup:
#   Internet node: third_party/install_transformer_engine.sh fetch
#   CUDA node:     third_party/install_transformer_engine.sh install

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
ACTION=${1:-}
PYTHON=${PYTHON:-$ROOT_DIR/.venv/bin/python}
UV=${UV:-uv}
UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
TE_SRC=${TE_SRC:-$ROOT_DIR/third_party/TransformerEngine}
MEGATRON_SRC=${MEGATRON_SRC:-$ROOT_DIR/third_party/Megatron-LM}
MAX_JOBS=${MAX_JOBS:-8}

usage() {
    cat <<EOF
Usage: $0 fetch|install|verify

  fetch    Internet node: initialize pinned TE and all recursive submodules.
  install  CUDA node: build/install from the prepared submodule without network.
  verify   Check source provenance, the installed extension, and Megatron TE spec.

Environment overrides:
  PYTHON=$PYTHON
  TE_SRC=$TE_SRC
  MEGATRON_SRC=$MEGATRON_SRC
  CUDA_HOME=$CUDA_HOME
  MAX_JOBS=$MAX_JOBS
EOF
}

verify_submodule_tree() {
    local source=$1
    local label=$2
    if [[ ! -e "$source/.git" ]]; then
        echo "$label source is missing or is not initialized: $source" >&2
        echo "Run 'git submodule update --init --checkout --recursive' first." >&2
        exit 1
    fi

    local relative_path expected_commit actual_commit submodule_status
    relative_path=${source#"$ROOT_DIR/"}
    expected_commit=$(git -C "$ROOT_DIR" ls-tree HEAD "$relative_path" | awk '{print $3}')
    if [[ -z "$expected_commit" ]]; then
        # Allow verification before the parent commit containing a new gitlink
        # is created by falling back to the index entry.
        expected_commit=$(git -C "$ROOT_DIR" ls-files --stage "$relative_path" | awk '$1 == 160000 {print $2}')
    fi
    actual_commit=$(git -C "$source" rev-parse HEAD)
    if [[ -z "$expected_commit" || "$actual_commit" != "$expected_commit" ]]; then
        echo "$label source mismatch: expected ${expected_commit:-<missing gitlink>}, got $actual_commit" >&2
        exit 1
    fi
    if [[ -n "$(git -C "$source" status --porcelain --untracked-files=no)" ]]; then
        echo "$label has tracked source modifications: $source" >&2
        git -C "$source" status --short --untracked-files=no >&2
        exit 1
    fi

    submodule_status=$(git -C "$source" submodule status --recursive)
    if printf '%s\n' "$submodule_status" | grep -Eq '^[-+U]'; then
        echo "$label recursive submodules are missing or not at recorded commits:" >&2
        printf '%s\n' "$submodule_status" >&2
        exit 1
    fi
}

verify_sources() {
    verify_submodule_tree "$TE_SRC" "Transformer Engine"
    verify_submodule_tree "$MEGATRON_SRC" "Megatron-LM"
}

verify_python() {
    if [[ ! -x "$PYTHON" ]]; then
        echo "Python environment does not exist or is not executable: $PYTHON" >&2
        exit 1
    fi

    "$PYTHON" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
if torch.__version__ != "2.11.0+cu128":
    raise SystemExit(f"expected PyTorch 2.11.0+cu128, got {torch.__version__}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"expected PyTorch CUDA 12.8, got {torch.version.cuda}")
PY
}

site_packages() {
    "$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
}

nvidia_library_path() {
    "$PYTHON" - <<'PY'
import sysconfig
from pathlib import Path

root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()))
PY
}

ensure_te_cuda_runtime_compat() {
    local site_packages=$1
    local runtime_dir=$site_packages/nvidia/cuda_runtime
    local compat_dir=$site_packages/nvidia/cuda_cudart
    if [[ ! -d "$runtime_dir" ]]; then
        echo "Missing PyTorch CUDA runtime wheel directory: $runtime_dir" >&2
        exit 1
    fi
    if [[ -L "$compat_dir" ]]; then
        if [[ "$(readlink "$compat_dir")" != "cuda_runtime" ]]; then
            echo "Unexpected Transformer Engine CUDA runtime compatibility link: $compat_dir" >&2
            exit 1
        fi
    elif [[ -e "$compat_dir" ]]; then
        echo "Refusing to replace existing path: $compat_dir" >&2
        exit 1
    else
        ln -s cuda_runtime "$compat_dir"
    fi
}

verify_installation() {
    local metadata_path direct_url editable
    metadata_path=$(
        PYTHONPATH="$MEGATRON_SRC:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON" - <<'PY'
import json
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import torch
import transformer_engine
import transformer_engine.pytorch
import transformer_engine_torch
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)

dist = distribution("transformer-engine")
installed_version = dist.version
direct_url_path = Path(dist._path) / "direct_url.json"
direct_url = json.loads(direct_url_path.read_text()) if direct_url_path.is_file() else {}
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("Transformer Engine Python:", transformer_engine.__file__)
print("Transformer Engine extension:", transformer_engine_torch.__file__)
for package in ("transformer-engine", "transformer-engine-torch", "transformer-engine-cu12"):
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        pass
print("Transformer Engine direct_url:", json.dumps(direct_url, sort_keys=True))
spec = get_gpt_layer_with_transformer_engine_submodules()
print("Megatron TE spec:", type(spec).__name__)
if torch.__version__ != "2.11.0+cu128" or torch.version.cuda != "12.8":
    raise SystemExit("TE installation changed the expected Torch/CUDA stack")
if installed_version != "2.17.1+4329ff84":
    raise SystemExit(f"unexpected Transformer Engine version: {installed_version}")
if direct_url.get("dir_info", {}).get("editable", False):
    raise SystemExit("Transformer Engine must not be installed editable")
print("Transformer Engine installation: OK")
print(f"__DIRECT_URL_PATH__={direct_url_path}")
PY
    )
    printf '%s\n' "$metadata_path"
    direct_url=$(printf '%s\n' "$metadata_path" | sed -n 's/^Transformer Engine direct_url: //p')
    editable=$(printf '%s\n' "$direct_url" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("dir_info", {}).get("editable", False))')
    if [[ "$editable" != "False" ]]; then
        echo "Transformer Engine is installed editable" >&2
        exit 1
    fi
    if [[ "$direct_url" != *"third_party/TransformerEngine"* ]]; then
        echo "Transformer Engine was not installed from $TE_SRC: $direct_url" >&2
        exit 1
    fi
}

case "$ACTION" in
    fetch)
        echo "Initializing pinned Transformer Engine and Megatron-LM submodules"
        git -C "$ROOT_DIR" submodule sync --recursive
        git -C "$ROOT_DIR" submodule update --init --checkout --recursive \
            third_party/TransformerEngine third_party/Megatron-LM
        verify_sources
        echo "Transformer Engine source preparation: OK"
        ;;
    install)
        echo "Installing Transformer Engine on the CUDA node"
        echo "  Python:   $PYTHON"
        echo "  source:   $TE_SRC"
        echo "  MAX_JOBS: $MAX_JOBS"
        verify_sources
        verify_python
        test -x "$CUDA_HOME/bin/nvcc" || { echo "Missing nvcc: $CUDA_HOME/bin/nvcc" >&2; exit 1; }
        if ! command -v "$UV" >/dev/null 2>&1; then
            echo "uv is unavailable: $UV" >&2
            exit 1
        fi
        mkdir -p "$UV_CACHE_DIR"
        export UV_CACHE_DIR

        SITE_PACKAGES=$(site_packages)
        ensure_te_cuda_runtime_compat "$SITE_PACKAGES"
        NVIDIA_LIBRARY_PATH=$(nvidia_library_path)
        CUDNN_ROOT=${CUDNN_ROOT:-$SITE_PACKAGES/nvidia/cudnn}
        test -d "$CUDNN_ROOT/include" || { echo "Missing cuDNN headers: $CUDNN_ROOT/include" >&2; exit 1; }
        test -d "$CUDNN_ROOT/lib" || { echo "Missing cuDNN libraries: $CUDNN_ROOT/lib" >&2; exit 1; }

        export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
        export CUDA_HOME
        export NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS:-90}
        export NVTE_FRAMEWORK=${NVTE_FRAMEWORK:-pytorch}
        export NVTE_WITH_NCCL_EP=${NVTE_WITH_NCCL_EP:-0}
        export MAX_JOBS

        CPATH="$CUDNN_ROOT/include${CPATH:+:$CPATH}" \
        LIBRARY_PATH="$NVIDIA_LIBRARY_PATH${LIBRARY_PATH:+:$LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$NVIDIA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$UV" pip install \
            --python "$PYTHON" \
            --offline \
            --no-build-isolation \
            --no-deps \
            --reinstall \
            "$TE_SRC"

        LD_LIBRARY_PATH="$NVIDIA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            verify_installation
        ;;
    verify)
        verify_sources
        verify_python
        SITE_PACKAGES=$(site_packages)
        ensure_te_cuda_runtime_compat "$SITE_PACKAGES"
        NVIDIA_LIBRARY_PATH=$(nvidia_library_path)
        CUDNN_ROOT=${CUDNN_ROOT:-$SITE_PACKAGES/nvidia/cudnn}
        LD_LIBRARY_PATH="$NVIDIA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            verify_installation
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
