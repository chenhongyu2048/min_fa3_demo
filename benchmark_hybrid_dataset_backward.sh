#!/usr/bin/env bash

# Dataset-shaped hierarchical hybrid backward benchmark.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DIRECTION=backward exec "$SCRIPT_DIR/benchmark_hybrid_dataset.sh" "$@"
