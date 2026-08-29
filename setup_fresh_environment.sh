#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a freshly cloned repository all the way to a verified Mega-CP
# single-layer benchmark environment. Native build details remain centralized
# in third_party/setup_fresh_environment.sh.

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ORCHESTRATOR="$ROOT_DIR/third_party/setup_fresh_environment.sh"
ACTION=${1:-}

UV_VERSION=0.12.4
UV_TOOL_DIR=${UV_TOOL_DIR:-$ROOT_DIR/.cache/tools/uv-$UV_VERSION}
UV_INSTALL_URL=${UV_INSTALL_URL:-https://astral.sh/uv/$UV_VERSION/install.sh}

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
           exactly install the locked PyTorch 2.11.0+cu128 base environment.
  install  On one visible SM90 Hopper GPU with CUDA toolkit 12.8, clean-build
           min-FA3, install pinned TE and MagiAttention offline, compile the
           three targeted Magi FFA kernels, and run complete verification.
  verify   Verify the completed environment without rebuilding.
  all      Run prepare and install on one machine that has both Internet access
           and a visible SM90 Hopper GPU.

Fresh-clone host prerequisites:
  prepare: Linux x86_64, git, curl, a POSIX shell/CA certificates, and Internet
           access. uv downloads its managed CPython 3.12 when it is absent.
  install: the prepared .venv and submodules, GNU make/gcc/g++, CUDA toolkit
           12.8 at CUDA_HOME (default /usr/local/cuda-12.8), an NVIDIA driver,
           and at least one visible SM90 Hopper GPU. Eight GPUs are not needed
           for building; they are only needed for the formal CP=8 benchmark.

Environment overrides:
  UV=/path/to/uv             Use an existing uv $UV_VERSION executable.
  UV_TOOL_DIR=$UV_TOOL_DIR
  UV_INSTALL_URL=$UV_INSTALL_URL
  PYTHON, UV_CACHE_DIR, CUDA_HOME, MAX_JOBS, NVCC_THREADS, and LOG_DIR are
  forwarded to the third-party environment orchestrator.

Important:
  prepare performs an exact uv sync and removes undeclared packages such as
  previous TE/Magi/FlashAttention installations. Always run prepare before
  install for a new checkout. After install, use verify for non-mutating checks.
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
        [[ -x "$candidate" ]] || return 1
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
    [[ -x "$ORCHESTRATOR" ]] ||
        die "missing executable orchestrator: $ORCHESTRATOR"
    local git_root
    git_root=$(git -C "$ROOT_DIR" rev-parse --show-toplevel 2>/dev/null) ||
        die "$ROOT_DIR is not a Git checkout"
    [[ "$(cd -- "$git_root" && pwd)" == "$ROOT_DIR" ]] ||
        die "script must reside at the root of its Git checkout"
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] ||
        die "the locked environment supports only Linux x86_64"
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

UV="$RESOLVED_UV" "$ORCHESTRATOR" "$ACTION"
