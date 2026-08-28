#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
paper_dir="$(cd -- "${script_dir}/.." && pwd)"
source_dir="${paper_dir}/figures/src"
output_dir="${paper_dir}/figures"

if ! command -v dot >/dev/null 2>&1; then
    echo "error: Graphviz 'dot' is required to render paper figures" >&2
    exit 1
fi

shopt -s nullglob
sources=("${source_dir}"/*.dot)
if (( ${#sources[@]} == 0 )); then
    echo "error: no .dot figure sources found in ${source_dir}" >&2
    exit 1
fi

for source in "${sources[@]}"; do
    stem="$(basename -- "${source}" .dot)"
    dot -Tsvg "${source}" -o "${output_dir}/${stem}.svg"
    dot -Tpdf "${source}" -o "${output_dir}/${stem}.pdf"
    dot -Tpng -Gdpi=180 "${source}" -o "${output_dir}/${stem}.png"
    echo "rendered ${stem}"
done
