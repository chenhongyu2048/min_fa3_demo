#!/usr/bin/env bash
set -euo pipefail

# Two-node MagiAttention setup:
#   Internet node: third_party/install_magi_attention.sh fetch
#   CUDA node:     third_party/install_magi_attention.sh install

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
ACTION=${1:-}
PYTHON=${PYTHON:-$ROOT_DIR/.venv/bin/python}
UV=${UV:-uv}
UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
MAGI_SRC=${MAGI_SRC:-$ROOT_DIR/third_party/MagiAttention}
MAGI_WORKSPACE=${MAGI_WORKSPACE:-$ROOT_DIR/.cache/mega_cp/magi_ffa_sm90_bf16_hd128}
MAX_JOBS=${MAX_JOBS:-8}

usage() {
    cat <<EOF
Usage: $0 fetch|install|verify

  fetch    Internet node: initialize pinned MagiAttention and all submodules.
  install  CUDA node: build/install from the prepared submodule without network.
  verify   Check source provenance, Python API, and both native extensions.

Environment overrides:
  PYTHON=$PYTHON
  MAGI_SRC=$MAGI_SRC
  MAGI_WORKSPACE=$MAGI_WORKSPACE
  CUDA_HOME=$CUDA_HOME
  MAX_JOBS=$MAX_JOBS
EOF
}

verify_source() {
    if [[ ! -e "$MAGI_SRC/.git" ]]; then
        echo "MagiAttention source is missing or is not initialized: $MAGI_SRC" >&2
        echo "Run 'git submodule update --init --checkout --recursive' first." >&2
        exit 1
    fi

    local relative_path expected_commit actual_commit submodule_status
    relative_path=${MAGI_SRC#"$ROOT_DIR/"}
    expected_commit=$(git -C "$ROOT_DIR" ls-tree HEAD "$relative_path" | awk '{print $3}')
    if [[ -z "$expected_commit" ]]; then
        expected_commit=$(git -C "$ROOT_DIR" ls-files --stage "$relative_path" | awk '$1 == 160000 {print $2}')
    fi
    actual_commit=$(git -C "$MAGI_SRC" rev-parse HEAD)
    if [[ -z "$expected_commit" || "$actual_commit" != "$expected_commit" ]]; then
        echo "Magi source mismatch: expected ${expected_commit:-<missing gitlink>}, got $actual_commit" >&2
        exit 1
    fi
    if [[ -n "$(git -C "$MAGI_SRC" status --porcelain --untracked-files=no)" ]]; then
        echo "MagiAttention has tracked source modifications: $MAGI_SRC" >&2
        git -C "$MAGI_SRC" status --short --untracked-files=no >&2
        exit 1
    fi

    submodule_status=$(git -C "$MAGI_SRC" submodule status --recursive)
    if printf '%s\n' "$submodule_status" | grep -Eq '^[-+U]'; then
        echo "Magi recursive submodules are missing or not at recorded commits:" >&2
        printf '%s\n' "$submodule_status" >&2
        exit 1
    fi
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

verify_installation() {
    local output direct_url
    output=$(
        PYTHONPATH="$ROOT_DIR:$ROOT_DIR/third_party/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON" - <<'PY'
import json
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import torch
import magi_attention.api
import magi_attention.magi_attn_comm
import magi_attention.magi_attn_ext
from baseline.magi_attention import probe_magi_attention

dist = distribution("magi-attention")
installed_version = dist.version
direct_url_path = Path(dist._path) / "direct_url.json"
direct_url = json.loads(direct_url_path.read_text()) if direct_url_path.is_file() else {}
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
for package in ("magi_attention", "magi-attention"):
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        pass
print("MagiAttention direct_url:", json.dumps(direct_url, sort_keys=True))
available, reason = probe_magi_attention()
print("MagiAttention probe:", available, reason)
if torch.__version__ != "2.11.0+cu128" or torch.version.cuda != "12.8":
    raise SystemExit("Magi installation changed the expected Torch/CUDA stack")
if installed_version != "1.1.1.post16+g872717e1":
    raise SystemExit(f"unexpected MagiAttention version: {installed_version}")
if direct_url.get("dir_info", {}).get("editable", False):
    raise SystemExit("MagiAttention must not be installed editable")
if not available:
    raise SystemExit(reason or "MagiAttention probe failed")
print("MagiAttention installation: OK")
PY
    )
    printf '%s\n' "$output"
    direct_url=$(printf '%s\n' "$output" | sed -n 's/^MagiAttention direct_url: //p')
    if [[ "$direct_url" != *"third_party/MagiAttention"* ]]; then
        echo "MagiAttention was not installed from $MAGI_SRC: $direct_url" >&2
        exit 1
    fi
}

case "$ACTION" in
    fetch)
        echo "Initializing pinned MagiAttention submodule"
        git -C "$ROOT_DIR" submodule sync --recursive
        git -C "$ROOT_DIR" submodule update --init --checkout --recursive \
            third_party/MagiAttention
        verify_source
        echo "MagiAttention source preparation: OK"
        ;;
    install)
        echo "Installing MagiAttention on the CUDA node"
        echo "  Python:   $PYTHON"
        echo "  source:   $MAGI_SRC"
        echo "  MAX_JOBS: $MAX_JOBS"
        verify_source
        verify_python
        test -x "$CUDA_HOME/bin/nvcc" || { echo "Missing nvcc: $CUDA_HOME/bin/nvcc" >&2; exit 1; }
        if ! command -v "$UV" >/dev/null 2>&1; then
            echo "uv is unavailable: $UV" >&2
            exit 1
        fi
        mkdir -p "$UV_CACHE_DIR"
        mkdir -p "$MAGI_WORKSPACE"
        export UV_CACHE_DIR
        export MAGI_ATTENTION_WORKSPACE_BASE="$MAGI_WORKSPACE"

        SITE_PACKAGES=$(site_packages)
        NVIDIA_LIBRARY_PATH=$(nvidia_library_path)
        CUDNN_ROOT=${CUDNN_ROOT:-$SITE_PACKAGES/nvidia/cudnn}
        NVSHMEM_ROOT=${NVSHMEM_ROOT:-$SITE_PACKAGES/nvidia/nvshmem}
        test -d "$CUDNN_ROOT/include" || { echo "Missing cuDNN headers: $CUDNN_ROOT/include" >&2; exit 1; }
        test -f "$NVSHMEM_ROOT/lib/libnvshmem_host.so.3" || {
            echo "Missing NVSHMEM host library: $NVSHMEM_ROOT/lib/libnvshmem_host.so.3" >&2
            exit 1
        }
        unset NVSHMEM_DIR

        export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
        export CUDA_HOME
        export MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12=1
        export MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=${MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY:-90}
        # The benchmark uses only three BF16/SM90/HD128 training variants.
        # Build those explicitly after package installation instead of paying
        # for Magi's broad default FFA prebuild matrix here.
        export MAGI_ATTENTION_PREBUILD_FFA=0
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
            "$MAGI_SRC"

        LD_LIBRARY_PATH="$NVIDIA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            verify_installation
        ;;
    verify)
        verify_source
        verify_python
        mkdir -p "$MAGI_WORKSPACE"
        export MAGI_ATTENTION_WORKSPACE_BASE="$MAGI_WORKSPACE"
        SITE_PACKAGES=$(site_packages)
        NVIDIA_LIBRARY_PATH=$(nvidia_library_path)
        CUDNN_ROOT=${CUDNN_ROOT:-$SITE_PACKAGES/nvidia/cudnn}
        NVSHMEM_ROOT=${NVSHMEM_ROOT:-$SITE_PACKAGES/nvidia/nvshmem}
        unset NVSHMEM_DIR
        LD_LIBRARY_PATH="$NVIDIA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            verify_installation
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
