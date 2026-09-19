"""Render combined T1/T2 comparisons and stacked D1 cases as PNG + PDF figures.

Run: python plot_motivation.py
Requires matplotlib (the repository's plot dependency group); no CUDA or torch.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PHASES = ["q_allgather", "chunk_attention", "history_attention",
          "history_combine_publish", "a2a_pull", "final_combine"]
PHASE_LABELS = ["Q allgather", "Chunk attention", "History attention",
                "History publish", "A2A pull", "Final combine"]
# Reuse end2end/megaring/plot_transformer_layer.py's palette and typography.
METHOD_COLORS = {"allgather_attention": "#4C78A8", "fa3_ring": "#9C755F",
                 "megatron_hybrid_cp": "#F28E2B", "magi_attention": "#17A2B8",
                 "zeppelin": "#ECA82C"}
METHOD_LABELS = {"fa3_ring": "FA3 Ring", "megatron_hybrid_cp": "Megatron Hybrid CP",
                 "magi_attention": "MagiAttention", "zeppelin": "Zeppelin"}
DATASET_LABELS = {"prolong": "ProLong", "arxiv": "ArXiv", "freelaw": "FreeLaw",
                  "github": "GitHub", "pile": "Pile"}
COMPONENTS = (("otherB_ms", "OtherBwd", "//"), ("otherF_ms", "OtherFwd", "\\\\"),
              ("coreB_ms", "CoreBwd", "xx"), ("coreF_ms", "CoreFwd", ".."))
PHASE_COLORS = ["#4C78A8", "#59A14F", "#9C755F", "#B07AA1", "#F28E2B", "#E15759"]
MODES = ["comm_only", "comp_only", "serial", "overlap"]
MODE_LABELS = ["Comm-only", "Comp-only", "Serial (measured)", "Overlap"]
MODE_STYLES = [":", "--", "-.", "-"]
TAIL_COLOR = "#D8D8D8"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def check_data(rows):
    """Check the compact data that remain; discarded rank samples cannot be audited."""
    schema = next(row for row in rows if row["type"] == "schema")
    require(schema["schema"] == "motivation.v3.plot_data.v1", "unsupported log schema")
    t1_meta = next(row for row in rows if row["type"] == "t1_metadata")
    d1_meta = next(row for row in rows if row["type"] == "d1_metadata")
    t1 = [row for row in rows if row["type"] == "t1_timing"]
    d1 = [row for row in rows if row["type"] == "d1_timeline"]
    expected = set(itertools.product([1, 2, 4, 8, 16], ["ring", "allgather"], MODES))
    require(len(t1) == 40 and {(r["batch_size"], r["method"], r["mode"]) for r in t1} == expected,
            "T1 must contain all 40 unique configurations")
    for row in t1:
        require(row["batch_size"] * row["global_seqlen"] == row["total_tokens"] == 131072,
                "T1 global token mismatch")
        require(row["local_seqlen"] * t1_meta["environment"]["world_size"] == row["global_seqlen"],
                "T1 local sequence length mismatch")
        timing = row["timing"]
        samples = timing["rank_max_ms"]
        require(len(samples) == t1_meta["iters"] and all(math.isfinite(x) and x > 0 for x in samples),
                "T1 sample count or time invalid")
        for name, fraction in (("p50_ms", .5), ("p90_ms", .9)):
            require(math.isclose(timing[name], quantile(samples, fraction), rel_tol=1e-9),
                    f"T1 {name} inconsistent with samples")
    require(d1_meta["phases"] == PHASES, "unexpected D1 phases")
    require(d1_meta["time_fields"] == ["entry_ns", "work_start_ns", "work_end_ns", "exit_sync_done_ns"],
            "unexpected D1 timestamp layout")
    require(len(d1) == 2 and {r["case_id"] for r in d1} == {"case_000003", "case_000009"},
            "expected both D1 cases")
    for row in d1:
        ids, phases = row["sm_ids_by_cta"], row["phase_times_ns"]
        require(len(ids) == len(set(ids)) == d1_meta["environment"]["sm_count"], "D1 SM coverage mismatch")
        require(len(phases) == 6 and all(len(p) == len(ids) for p in phases), "D1 trace shape mismatch")
        require(0 <= row["sample_index"] < d1_meta["iters"], "D1 sample index invalid")
        require(row["rank"] in row["dcp_ranks"] and 0 <= row["rank"] < d1_meta["environment"]["world_size"],
                "D1 rank invalid")
        require(min(times[0] for times in phases[0]) == 0, "D1 clock origin mismatch")
        for phase, records in enumerate(phases):
            last_end = max(times[2] for times in records)
            for cta, times in enumerate(records):
                require(len(times) == 4 and all(isinstance(t, int) for t in times), "D1 timestamp format invalid")
                entry, start, end, exit_done = times
                require(0 <= entry <= start <= end <= exit_done and exit_done >= last_end,
                        "D1 phase order or grid completion invalid")
                if phase:
                    require(entry >= phases[phase - 1][cta][3], "D1 next phase precedes CTA exit")
        require([r[0] for r in row["rank_sync_ns"]] == [0, 4], "D1 communication boundaries missing")
        for phase, begin, end in row["rank_sync_ns"]:
            require(phases[phase][0][0] <= begin <= end <= phases[phase][0][1], "D1 rank barrier order invalid")
    return t1_meta, d1_meta, t1, d1


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=22, length=0, pad=5)
    ax.tick_params(axis="y", labelsize=22, length=3, pad=2)
    ax.set_axisbelow(True)


def save_figure(fig, output_dir, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{name}.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)


def load_t2(path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["method"] in METHOD_LABELS]
    expected = set(itertools.product(DATASET_LABELS, METHOD_LABELS))
    require(len(rows) == 20 and {(r["dataset"], r["method"]) for r in rows} == expected,
            "T2 requires five datasets and four unique methods each")
    for row in rows:
        for key in [c[0] for c in COMPONENTS] + ["layerFB_ms"]:
            row[key] = float(row[key])
            require(math.isfinite(row[key]) and row[key] >= 0, "T2 timing invalid")
        # Source summaries round each of the four components and total to 0.001 ms.
        require(abs(sum(row[key] for key, _, _ in COMPONENTS) - row["layerFB_ms"]) <= .00251,
                "T2 components do not sum to layer latency within source rounding")
        recorded, total = map(int, row["cases"].split("/"))
        require(recorded == total and total > 0, "T2 incomplete cases")
    return rows


def plot_t1_t2(records, layer_rows, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.2))
    batches = [1, 2, 4, 8, 16]
    lookup = {(r["batch_size"], r["method"], r["mode"]): r["timing"] for r in records}
    ax = axes[0]
    for method, color, marker in (("ring", METHOD_COLORS["fa3_ring"], "o"),
                                   ("allgather", METHOD_COLORS["allgather_attention"], "s")):
        for mode, linestyle in zip(MODES, MODE_STYLES):
            ratios = [lookup[b, method, mode]["p50_ms"] / lookup[b, method, "comp_only"]["p50_ms"]
                      for b in batches]
            ax.plot(range(5), ratios, color=color, marker=marker, linestyle=linestyle,
                    markersize=5.5, linewidth=2)
    ax.set_xticks(range(5), batches)
    ax.set_xlabel("Batch size", fontsize=24, labelpad=6)
    ax.set_ylabel("Latency / comp-only (×)", fontsize=24, labelpad=6)
    ax.yaxis.set_major_formatter("{x:.1f}×")
    ax.set_ylim(bottom=0)

    ax = axes[1]
    lookup = {(r["dataset"], r["method"]): r for r in layer_rows}
    bar_width = .84 / len(METHOD_LABELS)
    for dataset_index, dataset in enumerate(DATASET_LABELS):
        rows = [lookup[dataset, method] for method in METHOD_LABELS]
        positions = [dataset_index + (i - 1.5) * bar_width for i in range(4)]
        bottoms = [0.] * len(rows)
        for key, _, hatch in COMPONENTS:
            values = [row[key] for row in rows]
            ax.bar(positions, values, bottom=bottoms, width=bar_width * .92,
                   color=[METHOD_COLORS[row["method"]] for row in rows],
                   edgecolor="#333333", linewidth=.25, hatch=hatch)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_xticks(range(5), DATASET_LABELS.values())
    ax.set_xlabel("Dataset", fontsize=24, labelpad=6)
    ax.set_ylabel("Mean layer latency (ms)", fontsize=24, labelpad=6)
    ax.set_ylim(0, max(row["layerFB_ms"] for row in layer_rows) * 1.10)
    ax.set_xlim(-.55, 4.55)
    for ax in axes:
        ax.grid(axis="y", color="#D8D8D8", linewidth=.5, alpha=.8)
        style_axis(ax)
    fig.subplots_adjust(left=.09, right=.985, top=.73, bottom=.14, wspace=.18)
    centers = [(ax.get_position().x0 + ax.get_position().x1) / 2 for ax in axes]
    legend_args = dict(loc="upper center", ncol=2, frameon=False, fontsize=20,
                       handlelength=1.7, handletextpad=.4, columnspacing=1.0, labelspacing=.35)
    fig.legend(handles=[Line2D([0], [0], color=METHOD_COLORS[method], marker=marker, linewidth=2, label=label)
                        for method, marker, label in (("fa3_ring", "o", "Ring"),
                                                       ("allgather_attention", "s", "Allgather"))],
               bbox_to_anchor=(centers[0], .98), **{**legend_args, "fontsize": 22})
    fig.legend(handles=[Line2D([0], [0], color="#333333", linestyle=style, linewidth=2, label=label)
                        for style, label in zip(MODE_STYLES, MODE_LABELS)],
               bbox_to_anchor=(centers[0], .90), **{**legend_args, "fontsize": 22})
    fig.legend(handles=[Patch(facecolor=METHOD_COLORS[method], label=label)
                        for method, label in METHOD_LABELS.items()],
               bbox_to_anchor=(centers[1], .98), **legend_args)
    fig.legend(handles=[Patch(facecolor="white", edgecolor="#333333", linewidth=.25, hatch=hatch, label=label)
                        for _, label, hatch in COMPONENTS],
               bbox_to_anchor=(centers[1], .855), **{**legend_args, "ncol": 4, "fontsize": 18, "handlelength": 1.1})
    save_figure(fig, output_dir, "motivition_overlap_balance")


def compact_phase_times(row):
    """Concatenate work spans, preserving all within-stage SM time differences."""
    phases, offset = [], 0
    for times in row["phase_times_ns"]:
        first_start = min(t[1] for t in times)
        last_end = max(t[2] for t in times)
        phases.append([[offset + t[1] - first_start, offset + t[2] - first_start] for t in times])
        offset += last_end - first_start
    return phases


def draw_timeline(ax, row, phase_indices):
    sm_ids = sorted(row["sm_ids_by_cta"])
    sm_y = {sm: index for index, sm in enumerate(sm_ids)}
    vertices, colors = [], []
    phases = compact_phase_times(row)
    for phase in phase_indices:
        times = phases[phase]
        last_end = max(t[1] for t in times)
        for sm, (start, end) in zip(row["sm_ids_by_cta"], times):
            y = sm_y[sm]
            for begin, finish, color in ((start, end, PHASE_COLORS[phase]), (end, last_end, TAIL_COLOR)):
                vertices.append([(begin / 1000, y - .44), (finish / 1000, y - .44),
                                 (finish / 1000, y + .44), (begin / 1000, y + .44)])
                colors.append(color)
    ax.add_collection(PolyCollection(vertices, facecolors=colors, edgecolors="none"))
    start = min(t[0] for p in phase_indices for t in phases[p]) / 1000
    end = max(t[1] for p in phase_indices for t in phases[p]) / 1000
    ax.set_xlim(start, end + (end - start) * .012)
    ax.set_ylim(-1, len(sm_ids))
    ticks = list(range(0, len(sm_ids), 32))
    if ticks[-1] != len(sm_ids) - 1:
        ticks[-1] = len(sm_ids) - 1
    ax.set_yticks(ticks, [sm_ids[i] for i in ticks])
    ax.set_ylabel("Physical SM ID", fontsize=29, labelpad=6)
    style_axis(ax)
    ax.tick_params(axis="both", labelsize=27)
    if len(phase_indices) > 1:
        for phase in phase_indices:
            times = phases[phase]
            begin, finish = min(t[0] for t in times) / 1000, max(t[1] for t in times) / 1000
            ax.text((begin + finish) / 2, 1.025, str(phase + 1), transform=ax.get_xaxis_transform(),
                    ha="center", color=PHASE_COLORS[phase], fontsize=25, fontweight="bold")


def phase_stats(row):
    stats = []
    for name, times in zip(PHASES, row["phase_times_ns"]):
        last_end = max(t[2] for t in times)
        span = last_end - min(t[1] for t in times)
        mean_tail = sum(last_end - t[2] for t in times) / len(times)
        stats.append({"phase": name, "work_span_us": span / 1000,
                      "mean_work_us": sum(t[2] - t[1] for t in times) / len(times) / 1000,
                      "mean_tail_idle_us": mean_tail / 1000,
                      "tail_idle_fraction_of_work_span": mean_tail / span if span else 0,
                      "mean_entry_wait_us": sum(t[1] - t[0] for t in times) / len(times) / 1000,
                      "mean_post_last_work_wait_us": sum(t[3] - last_end for t in times) / len(times) / 1000})
    return stats


def plot_d1(rows, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(17, 11.4))
    for ax, row in zip(axes, rows):
        draw_timeline(ax, row, list(range(6)))
    handles = [Patch(facecolor=color, label=f"{i + 1}. {name}")
               for i, (name, color) in enumerate(zip(PHASE_LABELS, PHASE_COLORS))]
    handles += [Patch(facecolor=TAIL_COLOR, label="Tail idle")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1), ncol=4,
               frameon=False, fontsize=27, handlelength=1.1, handletextpad=.4, columnspacing=.9, labelspacing=.35)
    fig.supxlabel("Timeline across serial stages (µs)", fontsize=29, y=.025)
    fig.subplots_adjust(left=.085, right=.985, top=.84, bottom=.10, hspace=.20)
    save_figure(fig, output_dir, "motivation_wave_quant")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("motivation.log"))
    parser.add_argument("--layer-input", type=Path, default=Path(__file__).resolve().parent.parent /
                        "end2end" / "megaring" / "transformer_layer_selected.csv")
    parser.add_argument("--output-dir", type=Path, help="default: input log's directory")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input.parent
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    t1_meta, d1_meta, t1, d1 = check_data(rows)
    layer_rows = load_t2(args.layer_input)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": .2})
    plot_t1_t2(t1, layer_rows, output_dir)
    plot_d1(d1, output_dir)
    report = {"checks": "PASS: T1 coverage, token counts, samples and quantiles; T2 coverage and component sums; D1 SM coverage and timestamp/barrier order",
              "t2": {"source": str(args.layer_input), "aggregation": "equal-case arithmetic mean, layer forward + backward",
                     "records": layer_rows},
              "d1_axis": "Concatenated [earliest work_start, latest work_end] spans; entry/exit and rank barriers omitted. Within-stage offsets unchanged; not actual elapsed time.",
              "limits": "Compact log omits other ranks/replays: rank-max reduction and representative selection cannot be independently rechecked. Tail idle is not a direct wave-count measurement.",
              "t1_backend": t1_meta["backend"],
              "t1_normalization": "mode p50 / comp_only p50 for the same method and batch size; no percentile shading",
              "d1": []}
    for row in d1:
        off, on = (row["timings"][name]["p50_ms"] for name in ("phased_trace_off", "phased_trace_on"))
        report["d1"].append({"case_id": row["case_id"], "rank": row["rank"], "sample_index": row["sample_index"],
                             "compacted_span_us": max(t[1] for t in compact_phase_times(row)[-1]) / 1000,
                             "trace_overhead_percent": (on / off - 1) * 100,
                             "phased_minus_mixed_percent": (off / row["timings"]["mixed_trace_off"]["p50_ms"] - 1) * 100,
                             "rank_sync_us": {PHASES[p]: (end - begin) / 1000 for p, begin, end in row["rank_sync_ns"]},
                             "phases": phase_stats(row)})
    (output_dir / "plot_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["checks"])
    print(f"Wrote 2 figures (PNG + PDF) and plot_summary.json to {output_dir}")


if __name__ == "__main__":
    main()
