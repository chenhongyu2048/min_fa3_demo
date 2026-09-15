#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a freshly cloned repository all the way to a verified Mega-CP
# single-layer benchmark environment. This is the single public environment
# entry point; component-specific installers remain under third_party/.

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
THIRD_PARTY_DIR="$ROOT_DIR/third_party"
ACTION=${1:-}

UV_VERSION=0.12.4
UV_TOOL_DIR=${UV_TOOL_DIR:-$ROOT_DIR/.cache/tools/uv-$UV_VERSION}
UV_INSTALL_URL=${UV_INSTALL_URL:-https://astral.sh/uv/$UV_VERSION/install.sh}
PYTHON=${PYTHON:-$ROOT_DIR/.venv/bin/python}
UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
MAX_JOBS=${MAX_JOBS:-8}
NVCC_THREADS=${NVCC_THREADS:-2}
LOG_DIR=${LOG_DIR:-$ROOT_DIR/.cache/mega_cp/logs/fresh_environment}
FORCE_REBUILD=${FORCE_REBUILD:-0}

usage() {
    cat <<EOF
Usage: $0 prepare|install|verify|all

Starting from a fresh clone:

  1. Internet-connected node, in this checkout:
       $0 prepare

  2. H20 CUDA node, using the same checkout/shared filesystem:
       CUDA_VISIBLE_DEVICES=0 $0 install

Actions:
  prepare  Bootstrap repository-local uv $UV_VERSION when necessary, initialize
           every recursive submodule, create .venv with Python 3.12, and
           synchronize the locked PyTorch 2.11.0+cu128 base environment.
  install  On one visible SM90 Hopper GPU with CUDA toolkit 12.x, build
           min-FA3 and its DCP CPU planner/queue module when needed,
           install pinned TE and MagiAttention offline
           when missing, compile the three targeted Magi FFA kernels when
           needed, and run complete verification.
  verify   Verify the completed environment without rebuilding.
  all      Run prepare and install on one machine that has both Internet access
           and a visible SM90 Hopper GPU.

For the optional vLLM benchmark, prepare vLLM while still on the networked
node, then verify it after this script's install action on the H20 node:
  ./third_party/setup_vllm_dcp.sh prepare
  CUDA_VISIBLE_DEVICES=0 ./third_party/setup_vllm_dcp.sh verify
See infer/vllm_bench/README.md for the 8-GPU smoke and formal matrix commands.

Fresh-clone host prerequisites:
  prepare: Linux x86_64, git, curl, a POSIX shell/CA certificates, and Internet
           access. uv downloads its managed CPython 3.12 when it is absent.
  install: the prepared .venv and submodules, GNU make/gcc/g++, CUDA toolkit
           12.x at CUDA_HOME (default /usr/local/cuda-12.8), an NVIDIA driver,
           and at least one visible SM90 Hopper GPU. Eight GPUs are not needed
           for building; they are only needed for the formal CP=8 benchmark.

Environment overrides:
  UV=/path/to/uv             Use an existing uv $UV_VERSION executable.
  UV_TOOL_DIR=$UV_TOOL_DIR
  UV_INSTALL_URL=$UV_INSTALL_URL
  PYTHON, UV_CACHE_DIR, CUDA_HOME, MAX_JOBS, NVCC_THREADS, and LOG_DIR
  customize the native build and its logs.
  FORCE_REBUILD=1  force rebuilding min-FA3 and targeted Magi FFA artifacts.

Important:
  prepare performs an exact uv sync for a fresh .venv. When .venv already
  exists it uses --inexact so source-built TE/Magi packages are preserved.
  Re-running install/prepare reuses verified native artifacts; set
  FORCE_REBUILD=1 when a rebuild is intentional. After install, use verify for
  non-mutating checks.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_command() {
    local command=$1
    command -v "$command" >/dev/null 2>&1 ||
        die "required command is unavailable: $command"
}

command_path() {
    local candidate=$1
    if [[ "$candidate" == */* ]]; then
        printf '%s\n' "$candidate"
    else
        command -v "$candidate"
    fi
}

uv_version() {
    local executable=$1
    "$executable" --version 2>/dev/null | awk 'NR == 1 {print $2}'
}

accept_uv() {
    local candidate=$1
    local resolved actual
    resolved=$(command_path "$candidate") || return 1
    actual=$(uv_version "$resolved") || return 1
    [[ "$actual" == "$UV_VERSION" ]] || return 1
    RESOLVED_UV=$resolved
}

bootstrap_uv() {
    require_command curl
    mkdir -p "$UV_TOOL_DIR"
    echo "Bootstrapping repository-local uv $UV_VERSION from:"
    echo "  $UV_INSTALL_URL"
    curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
        "$UV_INSTALL_URL" |
        env UV_UNMANAGED_INSTALL="$UV_TOOL_DIR" UV_NO_MODIFY_PATH=1 sh

    if accept_uv "$UV_TOOL_DIR/uv"; then
        return
    fi
    if accept_uv "$UV_TOOL_DIR/bin/uv"; then
        return
    fi
    die "uv installer completed but uv $UV_VERSION was not found under $UV_TOOL_DIR"
}

resolve_uv() {
    local may_bootstrap=$1
    local local_uv="$UV_TOOL_DIR/uv"
    local local_bin_uv="$UV_TOOL_DIR/bin/uv"

    if [[ -n "${UV:-}" ]]; then
        accept_uv "$UV" ||
            die "UV=$UV is unavailable or is not exactly uv $UV_VERSION"
        return
    fi
    if accept_uv "$local_uv" || accept_uv "$local_bin_uv" || accept_uv uv; then
        return
    fi
    if [[ "$may_bootstrap" == 1 ]]; then
        bootstrap_uv
        return
    fi
    die "uv $UV_VERSION is unavailable; run '$0 prepare' in this same checkout first"
}

verify_checkout() {
    require_command git
    local git_root

    git_root=$(git -C "$ROOT_DIR" rev-parse --show-toplevel 2>/dev/null) ||
        die "$ROOT_DIR is not a Git checkout"

    git_root=$(cd -- "$git_root" && pwd -P) ||
        die "cannot resolve Git checkout root: $git_root"

    if [[ "$git_root" != "$ROOT_DIR" ]]; then
        printf 'Script directory: <%s>\n' "$ROOT_DIR" >&2
        printf 'Git root:         <%s>\n' "$git_root" >&2
        die "script must reside at the root of its Git checkout"
    fi

    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] ||
        die "the locked environment supports only Linux x86_64"
}


run_logged() {
    local name=$1
    shift
    mkdir -p "$LOG_DIR"
    echo "[$name] $*"
    "$@" 2>&1 | tee "$LOG_DIR/$name.log"
}

verify_uv() {
    local actual_version
    actual_version=$(uv_version "$UV") ||
        die "unable to determine uv version from $UV"
    [[ "$actual_version" == "$UV_VERSION" ]] ||
        die "expected uv $UV_VERSION, got ${actual_version:-<unknown>}"
    echo "uv: $actual_version"
}

verify_cuda_host() {
    local nvcc_output
    nvcc_output=$("$CUDA_HOME/bin/nvcc" --version) ||
        die "CUDA compiler cannot be started: $CUDA_HOME/bin/nvcc"
    printf '%s\n' "$nvcc_output"
    if ! grep -Eq 'release 12\.[0-9]+([,.]|$)' <<<"$nvcc_output"; then
        die "expected CUDA toolkit 12.x under CUDA_HOME=$CUDA_HOME"
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
    "$PYTHON" -c 'import sys; print("Python executable:", sys.executable)' ||
        die "Python environment cannot be started: $PYTHON"
    "$PYTHON" - "${1:-runtime}" <<'PY'
import sys
from importlib.metadata import version

print("Python/Torch environment")
print("  Python:", sys.version.split()[0])
print("  NumPy:", version("numpy"))
print("  torch:", version("torch"))
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
    "torch": version("torch"),
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
if sys.argv[1] != "prepare":
    import torch

    print("  torch CUDA:", torch.version.cuda)
    if not (torch.version.cuda or "").startswith("12."):
        raise SystemExit(f"expected PyTorch CUDA 12.x, got {torch.version.cuda}")
print("Locked Python/Torch environment: OK")
PY
}

verify_submodules() {
    local status path expected_commit actual_commit
    local top_level_submodules=(
        third_party/Megatron-LM
        third_party/TransformerEngine
        third_party/MagiAttention
        third_party/vllm
    )

    status=$(git -C "$ROOT_DIR" submodule status --recursive)
    printf '%s\n' "$status"
    if printf '%s\n' "$status" | grep -Eq '^[-+U]'; then
        die "one or more submodules are missing or not at recorded commits"
    fi

    for path in "${top_level_submodules[@]}"; do
        expected_commit=$(
            git -C "$ROOT_DIR" ls-tree HEAD "$path" |
                awk '$1 == "160000" {print $3}'
        )
        if [[ -z "$expected_commit" ]]; then
            # Keep this useful while a newly added gitlink is staged but the
            # parent commit has not yet been made.
            expected_commit=$(
                git -C "$ROOT_DIR" ls-files --stage "$path" |
                    awk '$1 == "160000" {print $2}'
            )
        fi
        if [[ -z "$expected_commit" || ! -e "$ROOT_DIR/$path/.git" ]]; then
            die "missing initialized gitlink: $path"
        fi
        actual_commit=$(git -C "$ROOT_DIR/$path" rev-parse HEAD)
        printf '%s: %s\n' "$path" "$actual_commit"
        if [[ "$actual_commit" != "$expected_commit" ]]; then
            die "$path mismatch: expected $expected_commit, got $actual_commit"
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
import _dcp_mega_planner
from _dcp_mega_planner import critical_wave_plan, build_packed_queues

root = Path(sys.argv[1]).resolve()
print("min_fa3_op:", min_fa3_op.__file__)
print("min-FA3 extension:", _min_fa3_op.__file__)
print("DCP CPU planner/queue extension:", _dcp_mega_planner.__file__)
if Path(min_fa3_op.__file__).resolve().parent != root:
    raise SystemExit("min_fa3_op was not imported from this repository")
if Path(_min_fa3_op.__file__).resolve().parent != root:
    raise SystemExit("_min_fa3_op was not built in this repository")
if Path(_dcp_mega_planner.__file__).resolve().parent != root:
    raise SystemExit("_dcp_mega_planner was not built in this repository")
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
        raise SystemExit(f"empty targeted Magi FFA AOT artifact: {path}")
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
    local -a sync_args=(
        --directory "$ROOT_DIR"
        --python 3.12
        --frozen
        --no-install-project
        --no-default-groups
        --group build
        --group transformer-layer
    )
    # Keep source-built packages (TE/Magi) when prepare is rerun on an
    # already-completed checkout.  A fresh .venv still gets an exact sync.
    if [[ -x "$PYTHON" ]]; then
        sync_args+=(--inexact)
        echo "Existing Python environment detected; preserving undeclared native packages"
    fi
    run_logged uv_sync "$UV" sync "${sync_args[@]}"

    verify_python_stack prepare | tee "$LOG_DIR/verify_python_stack.log"
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
    export UV PYTHON UV_CACHE_DIR CUDA_HOME MAX_JOBS NVCC_THREADS FORCE_REBUILD
    export PATH="$(dirname -- "$PYTHON"):$CUDA_HOME/bin:$PATH"
    require_command cmake
    require_command ninja
    require_command gcc
    require_command g++

    if [[ "$FORCE_REBUILD" == 1 ]]; then
        run_logged clean_min_fa3 make -C "$ROOT_DIR" clean
        run_logged build_min_fa3 make -C "$ROOT_DIR" PYTHON="$PYTHON"
    elif verify_min_fa3 >/dev/null 2>&1; then
        echo "In-repository min-FA3 extension already passes verification; skipping build"
    else
        run_logged build_min_fa3 make -C "$ROOT_DIR" PYTHON="$PYTHON"
    fi
    run_logged install_transformer_engine \
        "$THIRD_PARTY_DIR/install_transformer_engine.sh" install
    run_logged install_magi_attention \
        "$THIRD_PARTY_DIR/install_magi_attention.sh" install

    if [[ "$FORCE_REBUILD" == 1 ]] || ! verify_magi_aot >/dev/null 2>&1; then
        echo "[precompile_magi_ffa_training] $THIRD_PARTY_DIR/precompile_magi_ffa_training.sh"
        LOG_DIR="$LOG_DIR" \
        LOG_FILE="$LOG_DIR/precompile_magi_ffa_training.log" \
        FORCE_REBUILD="$FORCE_REBUILD" \
            "$THIRD_PARTY_DIR/precompile_magi_ffa_training.sh"
    else
        echo "Targeted Magi FFA AOT artifacts already pass verification; skipping precompile"
    fi

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
        "$THIRD_PARTY_DIR/install_transformer_engine.sh" verify
    run_logged verify_magi_attention \
        "$THIRD_PARTY_DIR/install_magi_attention.sh" verify

    MAGI_ATTENTION_WORKSPACE_BASE=${MAGI_ATTENTION_WORKSPACE_BASE:-$ROOT_DIR/.cache/mega_cp/magi_ffa_sm90_bf16_hd128} \
        verify_magi_aot | tee "$LOG_DIR/verify_magi_ffa_aot.log"
    echo "Complete fresh environment verification: OK"
}

case "$ACTION" in
    prepare | all)
        verify_checkout
        resolve_uv 1
        ;;
    install | verify)
        verify_checkout
        resolve_uv 0
        ;;
    -h | --help | help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

echo "Repository: $ROOT_DIR"
echo "uv:         $RESOLVED_UV ($UV_VERSION)"
echo "Action:     $ACTION"

UV=$RESOLVED_UV

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
esac
