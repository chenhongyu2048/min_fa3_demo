#!/usr/bin/env python3
"""Plot p50 request-mean TBT and TTFT, minimizing each over SM configs."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter


ROOT = Path(__file__).resolve().parent
BACKENDS = (
    ("mega", "MegaDCP", "#0072B2"),
    ("mega-fa3-native", "MegaDCP (FA3 native)", "#D55E00"),
    ("vllm-a2a", "vLLM A2A", "#009E73"),
    ("vllm-ag-rs", "vLLM AG+RS", "#AA4499"),
)
METRICS = (
    ("per_request_mean_tbt_s.p50", "p50 request-mean TBT", 1000, "-", "o"),
    ("ttft_s.p50", "p50 TTFT", 1, "--", "s"),
)


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 26,
        "axes.titlesize": 26,
        "axes.labelsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.fontsize": 26,
        "pdf.fonttype": 42,
        "savefig.dpi": 220,
    })
    # Match megaring/plot_transformer_layer.py's five-dataset figure.
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6), sharey=True)
    selections = {}

    for ax, batch_size in zip(axes, (64, 128)):
        source = ROOT / f"vllm_dcp_qwen3_moe_kvh4_bs{batch_size}.csv"
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        scales = sorted({float(row["arrival_time_scale"]) for row in rows})
        ttft_ax = ax.twinx()
        for backend, label, color in BACKENDS:
            for metric_ax, (column, metric_label, factor, style, marker) in zip(
                (ax, ttft_ax), METRICS
            ):
                selected = [
                    min(
                        (row for row in rows if row["backend"] == backend
                         and float(row["arrival_time_scale"]) == scale),
                        key=lambda row: float(row[column]),
                    )
                    for scale in scales
                ]
                selections[batch_size, backend, column] = selected
                values = [float(row[column]) * factor for row in selected]
                metric_ax.plot(
                    scales, values, color=color, linestyle=style, marker=marker,
                    linewidth=2, markersize=6, markerfacecolor=color if style == "-" else "white",
                    markeredgewidth=1.4, label=f"{label}: {metric_label}",
                )
                if backend.startswith("mega"):
                    configs = [row["mega_num_comm_sm"] for row in selected]
                    print(f"BS={batch_size} {backend} {column}: SM={configs}, values={values}")

        ax.set_title(f"Batch size = {batch_size}", pad=12, fontweight="semibold")
        ax.set_xlabel("Arrival scale", labelpad=8)
        ax.set_ylabel("p50 per-request\nmean TBT (ms)", labelpad=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(scales, labels=[f"{scale:g}" for scale in scales])
        ax.set_xlim(min(scales) / 1.08, max(scales) * 1.08)
        ax.set_ylim(10, 22)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(axis="y", color="#D9DEE4", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ttft_ax.spines["top"].set_visible(False)
        ttft_ax.set_ylabel("p50 TTFT (s, log scale)", labelpad=8)
        ttft_ax.set_yscale("log")
        ttft_ax.set_ylim(0.1, 15)
        ttft_ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
        ttft_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ttft_ax.yaxis.set_minor_formatter(NullFormatter())

    fig.legend(
        handles=[Line2D([], [], color=color, linewidth=2.5, label=label)
                 for _, label, color in BACKENDS],
        loc="upper center", bbox_to_anchor=(0.5, 1), ncol=4, frameon=False,
        handlelength=1.1, handletextpad=0.3, columnspacing=0.6, labelspacing=0.25,
    )
    fig.legend(
        handles=[Line2D([], [], color="#444444", linestyle=style, marker=marker,
                        markerfacecolor="#444444" if style == "-" else "white",
                        label=f"{label} ({'left' if factor == 1000 else 'right'} axis)")
                 for _, label, factor, style, marker in METRICS],
        loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False,
        handlelength=1.1, handletextpad=0.3, columnspacing=0.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.865))
    for extension in ("png", "pdf"):
        destination = ROOT / f"eval_megadcp_end2end.{extension}"
        fig.savefig(destination, facecolor="white")
        print(destination)
    plt.close(fig)

    print("\nMegaDCP speedup = best vLLM latency / MegaDCP latency (higher is better)")
    print(f"{'BS':>3} {'Scale':>5} {'Metric':>8} {'Best vLLM':>12} {'SM':>3} "
          f"{'MegaDCP (ms)':>13} {'vLLM (ms)':>11} {'Speedup':>9}")
    for batch_size in (64, 128):
        for index, row in enumerate(selections[batch_size, "mega", METRICS[0][0]]):
            scale = float(row["arrival_time_scale"])
            for column, metric in ((METRICS[0][0], "p50 TBT"), (METRICS[1][0], "p50 TTFT")):
                mega = selections[batch_size, "mega", column][index]
                baseline = min(
                    (selections[batch_size, backend, column][index]
                     for backend in ("vllm-a2a", "vllm-ag-rs")),
                    key=lambda candidate: float(candidate[column]),
                )
                mega_ms = float(mega[column]) * 1000
                baseline_ms = float(baseline[column]) * 1000
                print(f"{batch_size:>3} {scale:>5g} {metric:>8} {baseline['backend']:>12} "
                      f"{mega['mega_num_comm_sm']:>3} {mega_ms:>13.4f} "
                      f"{baseline_ms:>11.4f} {baseline_ms / mega_ms:>8.4f}x")


if __name__ == "__main__":
    main()
