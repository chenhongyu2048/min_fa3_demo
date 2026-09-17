#!/usr/bin/env python3
"""Plot equal-case mean Transformer-layer latency from TIMING SUMMARY logs.

Use ArithMean so the four stacked components sum to mean layer latency.
Select each Mega-Ring variant's SM configuration by minimum LayerFB.
Hybrid speedup compares LayerFB against the fastest non-Mega-Ring baseline.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "transformer_layer_cp.log"
METHOD_LABELS = {
    "allgather_attention": "AllGather",
    "llama3_allgather_attention": "Llama3 AllGather",
    "fa3_ring": "FA3 Ring",
    "megatron_hybrid_cp": "Megatron Hybrid CP",
    "magi_attention": "MagiAttention",
    "zeppelin": "Zeppelin",
    "mega_ring_all_cp": "Mega-Ring All-CP",
    "mega_ring_hybrid": "Mega-Ring Hybrid",
}
# Match benchmark_logs/bench_cp/plot_weighted_flops.py.
METHOD_COLORS = {
    "allgather_attention": "#4C78A8",
    "llama3_allgather_attention": "#59A14F",
    "fa3_ring": "#9C755F",
    "megatron_hybrid_cp": "#F28E2B",
    "magi_attention": "#17A2B8",
    "zeppelin": "#ECA82C",
    "mega_ring_all_cp": "#E15759",
    "mega_ring_hybrid": "#B07AA1",
}
MEGA_METHODS = ("mega_ring_all_cp", "mega_ring_hybrid")
DATASET_LABELS = {
    "prolong": "ProLong",
    "arxiv": "ArXiv",
    "freelaw": "FreeLaw",
    "github": "GitHub",
    "pile": "Pile",
}
# Bottom-to-top stack order; color identifies the method, hatch the phase.
COMPONENTS = (
    ("OthersB", "OtherBwd", "//"),
    ("OthersF", "OtherFwd", "\\\\"),
    ("CoreB", "CoreBwd", "xx"),
    ("CoreF", "CoreFwd", ".."),
)


def load_summary(path: Path) -> dict[str, list[dict]]:
    grouped = {}
    dataset = None
    headers = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if line.startswith("TIMING SUMMARY "):
            dataset = fields[2]
            grouped[dataset] = []
        elif fields and fields[0] == "Method":
            headers = fields
        elif dataset is not None and len(fields) >= 4 and fields[3] == "ArithMean":
            row = dict(zip(headers, fields))
            for key in headers[4:]:
                row[key] = float(row[key])
            grouped[dataset].append(row)
    return grouped


def select_results(grouped: dict[str, list[dict]]) -> dict[str, list[dict]]:
    selected = {}
    for dataset in DATASET_LABELS:
        rows = grouped[dataset]
        selected[dataset] = []
        for method in METHOD_LABELS:
            candidates = [row for row in rows if row["Method"] == method]
            if method in MEGA_METHODS:
                candidates = [min(candidates, key=lambda row: row["LayerFB"])]
            selected[dataset].extend(candidates)
    return selected


def best_baseline(rows: list[dict]) -> dict:
    return min(
        (row for row in rows if row["Method"] not in MEGA_METHODS),
        key=lambda row: row["LayerFB"],
    )


def make_figure(selected: dict[str, list[dict]]) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": 0.2,
    })
    # Match plot_weighted_flops.py's width and height for one subplot row.
    figure_width = max(15.5, 5.0 + 2.4 * len(selected))
    figure, axis = plt.subplots(figsize=(figure_width, 4.8 + 1.8))
    bar_width = 0.84 / len(METHOD_LABELS)
    ymax = max(row["LayerFB"] for rows in selected.values() for row in rows) * 1.13
    for dataset_index, (dataset, rows) in enumerate(selected.items()):
        positions = [
            dataset_index + (i - (len(rows) - 1) / 2) * bar_width
            for i in range(len(rows))
        ]
        bottoms = [0.0] * len(rows)
        for key, label, hatch in COMPONENTS:
            values = [row[key] for row in rows]
            axis.bar(positions, values, bottom=bottoms, width=bar_width * 0.92,
                     color=[METHOD_COLORS[row["Method"]] for row in rows],
                     edgecolor="#333333", linewidth=0.25, hatch=hatch)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        baseline = best_baseline(rows)
        hybrid_index = next(i for i, row in enumerate(rows) if row["Method"] == "mega_ring_hybrid")
        speedup = baseline["LayerFB"] / rows[hybrid_index]["LayerFB"]
        axis.annotate(f"{speedup:.2f}x", (positions[hybrid_index], bottoms[hybrid_index]),
                      xytext=(-1.5, 3), textcoords="offset points", ha="left",
                      va="bottom", rotation=90, fontsize=20, fontweight="bold",
                      color="#704E6F",
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3})
    axis.set_xticks(range(len(selected)), [DATASET_LABELS[dataset] for dataset in selected])
    axis.set_xlabel("Dataset", fontsize=24, labelpad=6)
    axis.set_ylabel("Mean layer latency (ms)", fontsize=24, labelpad=6)
    axis.set_ylim(0, ymax)
    axis.set_xlim(-0.55, len(selected) - 0.45)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", length=0, labelsize=22, pad=5)
    axis.tick_params(axis="y", labelsize=22, length=3, pad=2)
    figure.legend(handles=[
        Patch(facecolor=METHOD_COLORS[method], label=label)
        for method, label in METHOD_LABELS.items()
    ], loc="upper center", bbox_to_anchor=(0.5, 1),
        ncol=4, frameon=False, fontsize=22, handlelength=1.1,
        handletextpad=0.4, columnspacing=0.9, labelspacing=0.35)
    figure.legend(handles=[
        Patch(facecolor="white", edgecolor="#333333", linewidth=0.25, hatch=hatch, label=label)
        for _, label, hatch in COMPONENTS
    ], loc="upper center", bbox_to_anchor=(0.5, 0.855),
        ncol=4, frameon=False, fontsize=22, handlelength=1.1,
        handletextpad=0.4, columnspacing=0.8)
    figure.tight_layout(rect=(0, 0, 1, 0.80))
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()
    selected = select_results(load_summary(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = make_figure(selected)
    for extension in ("png", "pdf"):
        output = args.output_dir / f"eval_megaring_transformer_layer.{extension}"
        figure.savefig(output, dpi=600, facecolor="white")
        print(output)
    plt.close(figure)
    output = args.output_dir / "transformer_layer_selected.csv"
    with output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["dataset", "method", "sm_config", "cases", "otherB_ms", "otherF_ms",
                         "coreB_ms", "coreF_ms", "layerFB_ms", "best_baseline", "hybrid_speedup"])
        for dataset, rows in selected.items():
            baseline = best_baseline(rows)
            for row in rows:
                hybrid = row["Method"] == "mega_ring_hybrid"
                writer.writerow([
                    dataset, row["Method"], row["SM"], row["Cases"],
                    *(row[key] for key, _, _ in COMPONENTS), row["LayerFB"],
                    baseline["Method"] if hybrid else "",
                    f"{baseline['LayerFB'] / row['LayerFB']:.6f}" if hybrid else "",
                ])
            allcp, hybrid = (next(row for row in rows if row["Method"] == method)
                             for method in MEGA_METHODS)
            print(f"{dataset}: All-CP SM={allcp['SM']}, Hybrid SM={hybrid['SM']}; "
                  f"{baseline['Method']} / Hybrid = "
                  f"{baseline['LayerFB'] / hybrid['LayerFB']:.4f}x")
    print(output)


if __name__ == "__main__":
    main()
