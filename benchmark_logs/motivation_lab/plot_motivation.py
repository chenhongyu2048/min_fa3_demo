"""Plot the compact motivation export: T1/T2, T3, and one CTA figure per D1 case."""
import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


COLORS = ("#6497BD", "#E2A16B", "#76AB92", "#AF92BA")
PHASES = ("Q all-gather", "Attention", "History combine", "Receive", "Final combine")
PHASE_COLORS = ("#D9A441", "#4B8FBB", "#8A79B5", "#DD8667", "#61A68B")
STRATEGIES = (
    ("all_cp", "All-CP"), ("br_pbs", "BR-PBS hybrid"),
    ("megatron_adapted", "Megatron-adapted"), ("zeppelin_adapted", "Zeppelin-adapted"),
)
COMPONENTS = (("OtherBwd", "///"), ("OtherFwd", "..."),
              ("CoreFwd", "\\\\"), ("CoreBwd", "xx"))
# Non-duplicated stage durations, in the local vLLM A2A execution order.
# Aggregate windows, aliases, and end-to-end timing are not additional stages.
A2A_STAGES = (
    ("q_allgather_and_reorder_ms", "Q all-gather / reorder", "#D9A441"),
    ("local_history_attention_ms", "History attention", "#4B8FBB"),
    ("a2a_pack_ms", "Pack", "#AF92BA"),
    ("a2a_all_to_all_ms", "All-to-all", "#DD8667"),
    ("a2a_unpack_combine_ms", "Unpack / combine", "#61A68B"),
    ("local_chunk_attention_ms", "Chunk attention", "#83BACE"),
    ("state_merge_ms", "State merge", "#C0AB8E"),
)


def read_log(path):
    lines = path.read_text().splitlines()
    if not lines or lines[-1].strip() != "# PLOT_DATA_V1 END":
        raise ValueError("run.log is missing the export completion marker")
    data = {"cases": []}
    case = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("# D1_CASE "):
            case = json.loads(line.removeprefix("# D1_CASE "))
            case["traces"] = {}
            data["cases"].append(case)
        elif line.startswith("# A2A critical_rank="):
            case["a2a_rank"] = int(re.search(r"critical_rank=(\d+)", line)[1])
        elif line in ("# T1", "# T2", "# T3", "# D1 trace-off", "# A2A stages") or line.startswith("# TRACE "):
            end = index + 1
            while end < len(lines) and not lines[end].startswith("#"):
                end += 1
            rows = list(csv.DictReader(item for item in lines[index + 1:end] if item.strip()))
            if line in ("# T1", "# T2", "# T3"):
                data[line[2:]] = rows
            elif line == "# D1 trace-off":
                case["p50"] = {row["method"]: float(row["p50_ms"]) for row in rows}
            elif line == "# A2A stages":
                case["a2a"] = {row["stage"]: float(row["ms"]) for row in rows}
            else:
                method, rank = re.search(r"method=(\w+) rank=(\d+)", line).groups()
                case["trace_rank"] = int(rank)
                case["traces"][method] = [{key: int(value) for key, value in row.items()} for row in rows]
            index = end
            continue
        index += 1
    return data


def style_axis(ax, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.6)
    ax.tick_params(length=3, color="#A0A6AD")
    if ylabel:
        ax.set_ylabel(ylabel)


def save(fig, output, name):
    for extension in ("png", "pdf"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
        print(path)
    plt.close(fig)


def plot_t12(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.27, top=0.72, wspace=0.23)
    batches = sorted({int(row["batch"]) for row in data["T1"]})
    labels = [f"B={b}\nL={128 // b}K" for b in batches]
    ax = axes[0]
    metrics = (("comm_plus_compute_ms", "Comm + compute"),
               ("serial_ms", "Serial"), ("overlap_ms", "Overlap"))
    width = 0.115
    for method_index, method in enumerate(("ring", "allgather")):
        by_batch = {int(row["batch"]): row for row in data["T1"] if row["method"] == method}
        for metric_index, (key, label) in enumerate(metrics):
            offset = (method_index * 3 + metric_index - 2.5) * width
            x = [i + offset + (method_index - 0.5) * 0.05 for i in range(len(batches))]
            ax.bar(x, [float(by_batch[b][key]) for b in batches], width=width,
                   color=COLORS[metric_index], edgecolor="#364152", linewidth=0.55,
                   hatch="//" if method == "allgather" else "",
                   label=f"{'Ring' if method == 'ring' else 'AllGather'} · {label}")
    ax.set_title("(a) T1 · Communication / compute overlap", loc="left", pad=72, fontweight="bold")
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(-0.01, 1.01), frameon=False, fontsize=9)
    ax = axes[1]
    profiles = (("step_external_reduce", "Step / external reduce"),
                ("step_fused_reduce", "Step / fused reduce"),
                ("linear_queue_recycle", "Linear queue recycle"))
    by_batch = {int(row["batch"]): row for row in data["T2"]}
    for i, (key, label) in enumerate(profiles):
        ax.bar([x + (i - 1) * 0.23 for x in range(len(batches))],
               [float(by_batch[b][key + "_ms"]) for b in batches], width=0.23,
               color=COLORS[i], edgecolor="#364152", linewidth=0.55, label=label)
    ax.set_title("(b) T2 · Execution organization", loc="left", pad=72, fontweight="bold")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.01), frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xticks(range(len(batches)), labels)
        ax.set_xlabel("Batch shape · global sequence length L")
        ax.set_ylim(bottom=0)
        style_axis(ax, "Latency (ms)")
    fig.text(0.06, 0.035, "8 × NVIDIA H800 · 128K total tokens · rank-max p50\n"
             "T1 Comm + compute = sum of isolated p50s.  T2 uses preloaded K/V (compute-only).",
             fontsize=9, color="#596574", linespacing=1.6)
    save(fig, output, "t1_t2")


def plot_t3(data, output):
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.72)
    datasets = list(dict.fromkeys(row["dataset"] for row in data["T3"]))
    lookup = {(row["dataset"], row["strategy"]): row for row in data["T3"]}
    for i, (strategy, _) in enumerate(STRATEGIES):
        x = [j + (i - 1.5) * 0.19 for j in range(len(datasets))]
        bottom = [0.0] * len(datasets)
        for component, hatch in COMPONENTS:
            heights = [float(lookup[(dataset, strategy)][component + "_ms"]) for dataset in datasets]
            ax.bar(x, heights, bottom=bottom, width=0.19, color=COLORS[i],
                   hatch=hatch, edgecolor="#35404D", linewidth=0.5)
            bottom = [a + b for a, b in zip(bottom, heights)]
        for pos, total in zip(x, bottom):
            ax.text(pos, total + 1, f"{total:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(datasets)), datasets)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.07)
    ax.set_xlabel("Dataset")
    style_axis(ax, "Mean layer FWD + BWD time (ms)")
    fig.suptitle("T3 · Transformer layer time by placement", x=0.075, ha="left", y=0.985,
                 fontsize=15, fontweight="bold")
    fig.legend([Patch(facecolor=COLORS[i], edgecolor="#35404D") for i in range(4)],
               [label for _, label in STRATEGIES], ncol=4, frameon=False,
               loc="upper left", bbox_to_anchor=(0.067, 0.915))
    fig.legend([Patch(facecolor="white", edgecolor="#35404D", hatch=hatch) for _, hatch in COMPONENTS],
               [label for label, _ in COMPONENTS], ncol=4, frameon=False,
               loc="upper left", bbox_to_anchor=(0.067, 0.84))
    counts = sorted({int(row["n"]) for row in data["T3"]})
    fig.text(0.075, 0.03, f"8 × NVIDIA H800 · equal-case means (n={','.join(map(str, counts))} per placement) · "
             "all placements use the same MegaRing executor\n"
             "Each iteration uses the rank with maximum full-layer CUDA FWD+BWD time; Others = full phase − core attention.",
             fontsize=9, color="#596574", linespacing=1.6)
    save(fig, output, "t3")


def plot_d1(case, output):
    fig = plt.figure(figsize=(13, 12.5))
    grid = fig.add_gridspec(3, 1, height_ratios=[4, 4, 0.85],
                           left=0.075, right=0.985, top=0.875, bottom=0.17, hspace=0.42)
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1]), fig.add_subplot(grid[2])]
    kind = "Decode-only Q16" if case["kind"] == "decode_only_q16" else "Mixed"
    fig.suptitle(f"D1 · {case['case_id']} · {kind}", x=0.075, y=0.985, ha="left",
                 fontsize=16, fontweight="bold")
    p50 = case["p50"]
    fig.text(0.075, 0.955, f"{case['gpu']} · TP8 / DCP2 · manually selected\n"
             f"Trace-off graph p50: CW {p50['critical_wave'] * 1000:.1f} µs   |   "
             f"FA3-native {p50['fa3_native'] * 1000:.1f} µs   |   vLLM A2A {p50['vllm_a2a'] * 1000:.1f} µs",
             fontsize=10, color="#596574", linespacing=1.65, va="top")
    fig.legend([Patch(facecolor=color) for color in PHASE_COLORS], PHASES,
               loc="upper left", bbox_to_anchor=(0.067, 0.915), ncol=5, frameon=False, fontsize=10)
    maximum = max(row["start_ns"] + row["duration_ns"]
                  for rows in case["traces"].values() for row in rows) / 1000
    for ax, method, title in zip(axes[:2], ("critical_wave", "fa3_native"), ("Critical-wave", "FA3-native / FIFO")):
        rows = case["traces"][method]
        ctas = sorted({row["cta"] for row in rows})
        for row in rows:
            ax.broken_barh([(row["start_ns"] / 1000, row["duration_ns"] / 1000)],
                           (row["cta"] - 0.44, 0.88),
                           facecolors=PHASE_COLORS[row["phase"]], linewidth=0)
        span = max(row["start_ns"] + row["duration_ns"] for row in rows) / 1000
        ax.set_title(f"{title} · rank {case['trace_rank']} · {len(ctas)} CTAs", loc="left",
                     fontsize=12, fontweight="bold", pad=8)
        ax.set_title(f"Trace span {span:.1f} µs", loc="right", fontsize=10, color="#596574", pad=8)
        ax.set_xlim(0, maximum * 1.025)
        ax.set_ylim(max(ctas) + 0.7, min(ctas) - 0.7)
        ax.set_yticks([cta for cta in ctas[::16] if cta <= ctas[-1] - 8] + [ctas[-1]])
        ax.set_ylabel("CTA ID")
        ax.set_xlabel("Time since first rank-local phase start (µs)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#DDE2E7", linewidth=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(7))

    ax = axes[2]
    left = 0
    handles, labels = [], []
    for key, label, color in A2A_STAGES:
        value = case["a2a"][key] * 1000
        ax.barh(0, value, left=left, height=0.5, color=color, edgecolor="white", linewidth=0.8)
        left += value
        handles.append(Patch(facecolor=color))
        labels.append(f"{label} · {value:.1f} µs")
    end_to_end = case["a2a"]["attention_end_to_end_ms"] * 1000
    ax.set_title(f"Local vLLM A2A · diagnostic rank {case['a2a_rank']}", loc="left",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_title(f"Stages {left:.1f} µs / measured end-to-end {end_to_end:.1f} µs",
                 loc="right", fontsize=9, color="#596574", pad=10)
    ax.set_xlim(0, left * 1.015)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("Cumulative measured stage duration (µs) · inter-stage gaps excluded")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.067, 0.122),
               ncol=3, frameon=False, fontsize=9, columnspacing=1.8)
    fig.text(0.075, 0.025,
             "CTA plots preserve recorded phase starts and gaps; each method has its own time origin. Phase intervals include waits/synchronization.\n"
             "A2A strip concatenates distinct stage durations; it is not a timestamped timeline. Diagnostic timings are separate from trace-off graph p50.",
             fontsize=8.5, color="#596574", linespacing=1.65)
    save(fig, output, f"d1_{case['case_id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("run.log"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": "#263445", "text.color": "#263445",
                         "axes.edgecolor": "#A0A6AD", "hatch.linewidth": 0.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    data = read_log(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_t12(data, args.output_dir)
    plot_t3(data, args.output_dir)
    for case in data["cases"]:
        plot_d1(case, args.output_dir)


if __name__ == "__main__":
    main()
