#!/usr/bin/env bash
set -euo pipefail

# Reproduce the complete single-layer Mega-CP software environment.
#
# Shared-filesystem two-node workflow:
#   Internet node: third_party/setup_fresh_environment.sh prepare
#   H20 CUDA node: third_party/setup_fresh_environment.sh install
#
# A machine with both network access and the supported CUDA environment may use
# `all`. No action is selected by default because `prepare` performs an exact
# uv sync and intentionally removes undeclared packages such as TE and Magi.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
ACTION=${1:-}

UV=${UV:-uv}
PYTHON=${PYTHON:-$ROOT_DIR/.venv/bin/python}
UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
MAX_JOBS=${MAX_JOBS:-8}
NVCC_THREADS=${NVCC_THREADS:-2}
LOG_DIR=${LOG_DIR:-$ROOT_DIR/.cache/mega_cp/logs/fresh_environment}
EXPECTED_UV_VERSION=0.12.4

usage() {
    cat <<EOF
Usage: $0 prepare|install|verify|all

  prepare  Internet node: sync every submodule and exactly recreate the locked
           Python 3.12 / PyTorch 2.11.0+cu128 base environment.
  install  H20 CUDA node: build min-FA3, TE, MagiAttention, and the three
           targeted Magi FFA training kernels, then run native verification.
  verify   Check pinned sources, the locked Torch/CUDA stack, in-repo min-FA3,
           TE/Megatron integration, Magi native extensions, and three FFA AOT
           artifacts without rebuilding them.
  all      Run prepare followed by install on one networked CUDA machine.

Environment overrides:
  UV=$UV
  PYTHON=$PYTHON
  UV_CACHE_DIR=$UV_CACHE_DIR
  CUDA_HOME=$CUDA_HOME
  MAX_JOBS=$MAX_JOBS
  NVCC_THREADS=$NVCC_THREADS
  LOG_DIR=$LOG_DIR

Important: prepare uses exact 'uv sync'. It removes undeclared packages,
including previous TE/Magi/FlashAttention installations. Run prepare before
install. After native installation, use 'uv sync --inexact' for later base
dependency maintenance.
EOF
}

run_logged() {
    local name=$1
    shift
    mkdir -p "$LOG_DIR"
    echo "[$name] $*"
    "$@" 2>&1 | tee "$LOG_DIR/$name.log"
}

require_command() {
    local command=$1
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command is unavailable: $command" >&2
        exit 1
    fi
}

verify_uv() {
    local actual_version
    require_command "$UV"
    actual_version=$("$UV" --version | awk '{print $2}')
    if [[ "$actual_version" != "$EXPECTED_UV_VERSION" ]]; then
        echo "Expected uv $EXPECTED_UV_VERSION, got ${actual_version:-<unknown>}" >&2
        exit 1
    fi
    echo "uv: $actual_version"
}

verify_cuda_host() {
    local nvcc_output
    test -x "$CUDA_HOME/bin/nvcc" || {
        echo "Missing nvcc: $CUDA_HOME/bin/nvcc" >&2
        exit 1
    }
    nvcc_output=$("$CUDA_HOME/bin/nvcc" --version)
    printf '%s\n' "$nvcc_output"
    if ! grep -Eq 'release 12\.8([,.]|$)' <<<"$nvcc_output"; then
        echo "Expected CUDA toolkit 12.8 under CUDA_HOME=$CUDA_HOME" >&2
        exit 1
    fi

    "$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to PyTorch on the native-build node")
capability = torch.cuda.get_device_capability()
print("CUDA build device:", torch.cuda.get_device_name(), "SM", capability)
if capability != (9, 0):
    raise SystemExit(f"expected an SM90 Hopper build device, got SM{capability}")
PY
}

verify_python_stack() {
    if [[ ! -x "$PYTHON" ]]; then
        echo "Python environment does not exist or is not executable: $PYTHON" >&2
        exit 1
    fi
    "$PYTHON" - <<'PY'
import sys
from importlib.metadata import version

import torch

print("Python/Torch environment")
print("  Python:", sys.version.split()[0])
print("  NumPy:", version("numpy"))
print("  torch:", torch.__version__)
print("  torch CUDA:", torch.version.cuda)
print("  Triton:", version("triton"))
print("  cuDNN wheel:", version("nvidia-cudnn-cu12"))
print("  NVSHMEM wheel:", version("nvidia-nvshmem-cu12"))
print("  NCCL wheel:", version("nvidia-nccl-cu12"))

expected = {
    "numpy": "2.1.3",
    "torch": "2.11.0+cu128",
    "triton": "3.6.0",
    "nvidia-cudnn-cu12": "9.19.0.56",
    "nvidia-nvshmem-cu12": "3.4.5",
    "nvidia-nccl-cu12": "2.28.9",
}
actual = {
    "numpy": version("numpy"),
    "torch": torch.__version__,
    "triton": version("triton"),
    "nvidia-cudnn-cu12": version("nvidia-cudnn-cu12"),
    "nvidia-nvshmem-cu12": version("nvidia-nvshmem-cu12"),
    "nvidia-nccl-cu12": version("nvidia-nccl-cu12"),
}
if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"expected Python 3.12, got {sys.version_info.major}.{sys.version_info.minor}"
    )
for package, expected_version in expected.items():
    if actual[package] != expected_version:
        raise SystemExit(
            f"unexpected {package} version: expected {expected_version}, "
            f"got {actual[package]}"
        )
if torch.version.cuda != "12.8":
    raise SystemExit(f"expected PyTorch CUDA 12.8, got {torch.version.cuda}")
print("Locked Python/Torch environment: OK")
PY
}

verify_submodules() {
    local status path expected_commit actual_commit
    local top_level_submodules=(
        third_party/Megatron-LM
        third_party/TransformerEngine
        third_party/MagiAttention
    )

    status=$(git -C "$ROOT_DIR" submodule status --recursive)
    printf '%s\n' "$status"
    if printf '%s\n' "$status" | grep -Eq '^[-+U]'; then
        echo "One or more submodules are missing or not at recorded commits" >&2
        exit 1
    fi

    for path in "${top_level_submodules[@]}"; do
        expected_commit=$(
            git -C "$ROOT_DIR" ls-tree HEAD "$path" | awk '$1 == "160000" {print $3}'
        )
        if [[ -z "$expected_commit" ]]; then
            # This fallback also makes the check useful while a newly added
            # gitlink is staged but the parent commit has not yet been made.
            expected_commit=$(
                git -C "$ROOT_DIR" ls-files --stage "$path" |
                    awk '$1 == "160000" {print $2}'
            )
        fi
        if [[ -z "$expected_commit" || ! -e "$ROOT_DIR/$path/.git" ]]; then
            echo "Missing initialized gitlink: $path" >&2
            exit 1
        fi
        actual_commit=$(git -C "$ROOT_DIR/$path" rev-parse HEAD)
        printf '%s: %s\n' "$path" "$actual_commit"
        if [[ "$actual_commit" != "$expected_commit" ]]; then
            echo "$path mismatch: expected $expected_commit, got $actual_commit" >&2
            exit 1
        fi
        if [[ -n "$(git -C "$ROOT_DIR/$path" status --porcelain --untracked-files=no)" ]]; then
            echo "$path has tracked source modifications" >&2
            git -C "$ROOT_DIR/$path" status --short --untracked-files=no >&2
            exit 1
        fi
    done
    git -C "$ROOT_DIR" submodule foreach --recursive --quiet '
        if test -n "$(git status --porcelain --untracked-files=no)"; then
            echo "$displaypath has tracked source modifications" >&2
            git status --short --untracked-files=no >&2
            exit 1
        fi
    '
    echo "Pinned top-level submodules: OK"
}

verify_min_fa3() {
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - "$ROOT_DIR" <<'PY'
import sys
from pathlib import Path

import min_fa3_op
import _min_fa3_op

root = Path(sys.argv[1]).resolve()
print("min_fa3_op:", min_fa3_op.__file__)
print("min-FA3 extension:", _min_fa3_op.__file__)
if Path(min_fa3_op.__file__).resolve().parent != root:
    raise SystemExit("min_fa3_op was not imported from this repository")
if Path(_min_fa3_op.__file__).resolve().parent != root:
    raise SystemExit("_min_fa3_op was not built in this repository")
required = (
    "forward_varlen",
    "backward_varlen",
    "forward_varlen_mega_ring",
    "backward_varlen_mega_ring",
)
missing = [name for name in required if not hasattr(min_fa3_op, name)]
if missing:
    raise SystemExit(f"min_fa3_op is missing required APIs: {missing}")
print("In-repository min-FA3 extension: OK")
PY
}

verify_magi_aot() {
    "$PYTHON" - <<'PY'
from pathlib import Path

import magi_attention

root = Path(magi_attention.__file__).resolve().parent / "lib"
expected_names = (
    "flex_flash_attn_sm_90_bwd_128hd_compute_bfloat16_dq_float32_dkv_float32_atomic_mmunified_pr40_cr232",
    "flex_flash_attn_sm_90_fwd_128hd_compute_bfloat16_out_float32_atomic_m128n128_pr40_cr232",
    "flex_flash_attn_sm_90_fwd_128hd_compute_bfloat16_out_float32_m128n128_pr40_cr232",
)
expected = set(expected_names)
actual = {path.name for path in root.glob("flex_flash_attn_sm_90_*") if path.is_dir()}
if actual != expected:
    raise SystemExit(
        "unexpected Magi FFA AOT directory set: "
        f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
    )
for name in expected_names:
    path = root / name / f"{name}.so"
    if not path.is_file():
        raise SystemExit(f"missing targeted Magi FFA AOT artifact: {path}")
    if path.stat().st_size <= 0:
        raise SystemExit(f"empty Magi FFA AOT artifact: {path}")
    print(f"Magi FFA AOT: {path} ({path.stat().st_size} bytes)")
print("Targeted Magi FFA AOT artifacts: OK")
PY
}

prepare_environment() {
    require_command git
    verify_uv
    mkdir -p "$UV_CACHE_DIR" "$LOG_DIR"
    export UV_CACHE_DIR

    run_logged submodule_sync git -C "$ROOT_DIR" submodule sync --recursive
    run_logged submodule_update git -C "$ROOT_DIR" submodule update \
        --init --checkout --recursive
    run_logged uv_lock_check "$UV" lock --directory "$ROOT_DIR" --check
    run_logged uv_sync "$UV" sync \
        --directory "$ROOT_DIR" \
        --python 3.12 \
        --frozen \
        --no-install-project \
        --no-default-groups \
        --group build \
        --group transformer-layer

    verify_python_stack | tee "$LOG_DIR/verify_python_stack.log"
    verify_submodules | tee "$LOG_DIR/verify_submodules.log"
    echo "Fresh base environment preparation: OK"
}

install_native_environment() {
    require_command git
    require_command make
    verify_uv
    verify_python_stack
    verify_cuda_host
    verify_submodules
    mkdir -p "$UV_CACHE_DIR" "$LOG_DIR"
    export UV PYTHON UV_CACHE_DIR CUDA_HOME MAX_JOBS NVCC_THREADS
    export PATH="$(dirname -- "$PYTHON"):$CUDA_HOME/bin:$PATH"
    require_command cmake
    require_command ninja
    require_command gcc
    require_command g++

    run_logged clean_min_fa3 make -C "$ROOT_DIR" clean
    run_logged build_min_fa3 make -C "$ROOT_DIR" PYTHON="$PYTHON"
    run_logged install_transformer_engine \
        "$SCRIPT_DIR/install_transformer_engine.sh" install
    run_logged install_magi_attention \
        "$SCRIPT_DIR/install_magi_attention.sh" install

    echo "[precompile_magi_ffa_training] $SCRIPT_DIR/precompile_magi_ffa_training.sh"
    LOG_DIR="$LOG_DIR" \
    LOG_FILE="$LOG_DIR/precompile_magi_ffa_training.log" \
    FORCE_REBUILD=1 \
        "$SCRIPT_DIR/precompile_magi_ffa_training.sh"

    verify_environment
    echo "Fresh native CUDA environment installation: OK"
}

verify_environment() {
    require_command git
    verify_uv
    mkdir -p "$LOG_DIR"
    export UV PYTHON UV_CACHE_DIR CUDA_HOME MAX_JOBS NVCC_THREADS
    verify_python_stack | tee "$LOG_DIR/verify_python_stack.log"
    verify_cuda_host | tee "$LOG_DIR/verify_cuda_host.log"
    verify_submodules | tee "$LOG_DIR/verify_submodules.log"
    verify_min_fa3 | tee "$LOG_DIR/verify_min_fa3.log"
    run_logged verify_transformer_engine \
        "$SCRIPT_DIR/install_transformer_engine.sh" verify
    run_logged verify_magi_attention \
        "$SCRIPT_DIR/install_magi_attention.sh" verify

    MAGI_ATTENTION_WORKSPACE_BASE=${MAGI_ATTENTION_WORKSPACE_BASE:-$ROOT_DIR/.cache/mega_cp/magi_ffa_sm90_bf16_hd128} \
        verify_magi_aot | tee "$LOG_DIR/verify_magi_ffa_aot.log"
    echo "Complete fresh environment verification: OK"
}

case "$ACTION" in
    prepare)
        prepare_environment
        ;;
    install)
        install_native_environment
        ;;
    verify)
        verify_environment
        ;;
    all)
        prepare_environment
        install_native_environment
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
