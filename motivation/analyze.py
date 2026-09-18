"""Export T1/D1 tables and per-case, all-rank SM timelines."""

import argparse
import csv
import json
from pathlib import Path

from .common import summarize, write_json
from .config import PHASES
from .trace import decode_trace


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_case(case_id, rank_records, sample, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = ("#4477AA", "#228833", "#66CCEE", "#AA3377", "#EE7733", "#CCBB44")
    subplot_rows = len(rank_records) // 2
    fig, axes = plt.subplots(subplot_rows, 2, figsize=(18, 3.5 * subplot_rows + 1), squeeze=False)
    for rank, (data, ax) in enumerate(zip(rank_records, axes.flat)):
        trace = data["sm_trace_samples"][sample]
        decoded = decode_trace(trace, len(trace[0]))
        origin = min(row[1] for row in trace[0])
        sm_ids = sorted(row[0] for row in trace[0])
        sm_position = {sm: position for position, sm in enumerate(sm_ids)}
        for phase, stats in enumerate(decoded):
            for row in stats["ctas"]:
                y = sm_position[row["sm"]]
                for begin, end, color in (
                    (row["entry_ns"], row["start_ns"], "#F3DBBE"),
                    (row["start_ns"], row["end_ns"], colors[phase]),
                    (row["end_ns"], row["exit_ns"], "#D5D5D5"),
                ):
                    ax.broken_barh([((begin - origin) / 1000, (end - begin) / 1000)],
                                   (y - 0.45, 0.9), facecolors=color, linewidth=0)
            if phase in (0, 4):
                begin, end = trace[phase][0][5:7]
                ax.broken_barh([((begin - origin) / 1000, (end - begin) / 1000)],
                               (len(sm_ids) + 1, 2), facecolors="#111111")
        positions = list(range(0, len(sm_ids), max(1, len(sm_ids) // 6)))
        ax.set_yticks(positions, [sm_ids[position] for position in positions])
        ax.set_ylabel("Physical SM ID")
        ax.set_xlabel("Rank-local time from first phase entry (µs)")
        ax.set_title(f'Rank {rank}; DCP ranks {data["dcp_ranks"]}')
    legend = [Patch(facecolor=color, label=phase) for color, phase in zip(colors, PHASES)]
    legend += [Patch(facecolor="#F3DBBE", label="entry / rank release wait"),
               Patch(facecolor="#D5D5D5", label="end-of-phase grid wait"),
               Patch(facecolor="#111111", label="CTA0 rank barrier (top strip)")]
    fig.legend(handles=legend, loc="lower center", ncol=3)
    fig.suptitle(f"{case_id}: phased native megakernel, trace sample {sample}\n"
                 "Separate GPU clocks; sample nearest trace-on rank-max p50")
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    fig.savefig(output_dir / f"{case_id}.png", dpi=180)
    fig.savefig(output_dir / f"{case_id}.pdf")
    plt.close(fig)


def analyze(input_dir, output_dir, plots=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    t1_path = input_dir / "t1.json"
    d1_path = input_dir / "d1.json"
    if not t1_path.exists() and not d1_path.exists():
        raise ValueError("input directory contains neither t1.json nor d1.json")
    if t1_path.exists():
        t1 = json.loads(t1_path.read_text())
        rows = [{**{key: row[key] for key in
                    ("batch_size", "global_seqlen", "local_seqlen", "method", "mode")},
                 "p50_ms": row["timing"]["p50_ms"], "p90_ms": row["timing"]["p90_ms"]}
                for row in t1["records"]]
        write_csv(output_dir / "t1.csv", rows)
    if d1_path.exists():
        d1 = json.loads(d1_path.read_text())
        rows, stage_rows, event_rows, overhead = [], [], [], []
        for record in d1["records"]:
            case_id, timings = record["case_id"], record["timings"]
            for method, timing in timings.items():
                rows.append({"case_id": case_id, "method": method,
                             "p50_ms": timing["p50_ms"], "p90_ms": timing["p90_ms"]})
            off, on = timings["phased_trace_off"]["p50_ms"], timings["phased_trace_on"]["p50_ms"]
            overhead.append({"case_id": case_id, "trace_delta_ms": on - off,
                             "trace_delta_percent": 100 * (on / off - 1),
                             "phased_minus_mixed_ms": off - timings["mixed_trace_off"]["p50_ms"]})
            ranks = [json.loads((input_dir / f"{case_id}.rank{rank}.json").read_text())
                     for rank in range(d1["environment"]["world_size"])]
            for rank, data in enumerate(ranks):
                for sample, trace in enumerate(data["sm_trace_samples"]):
                    for stage in decode_trace(trace, d1["environment"]["sm_count"]):
                        stage_rows.append({"case_id": case_id, "rank": rank, "sample": sample,
                                           "phase": stage["phase"], "work_span_ns": stage["work_span_ns"],
                                           "tail_idle_sm_ns": stage["tail_idle_sm_ns"],
                                           "rank_wait_ns": stage["rank_wait_ns"],
                                           "entry_wait_sm_ns": sum(row["entry_wait_ns"] for row in stage["ctas"]),
                                           "exit_wait_sm_ns": sum(row["exit_wait_ns"] for row in stage["ctas"])})
            for stage in ranks[0]["vllm_phase_samples_ms"][0]:
                stats = summarize([[sample[stage] for sample in data["vllm_phase_samples_ms"]]
                                   for data in ranks])
                event_rows.append({"case_id": case_id, "stage": stage,
                                   "p50_ms": stats["p50_ms"], "p90_ms": stats["p90_ms"]})
            traced = timings["phased_trace_on"]
            sample = min(range(len(traced["rank_max_ms"])),
                         key=lambda index: abs(traced["rank_max_ms"][index] - traced["p50_ms"]))
            if plots:
                plot_case(case_id, ranks, sample, output_dir)
        write_csv(output_dir / "d1.csv", rows)
        write_csv(output_dir / "d1_stages.csv", stage_rows)
        write_csv(output_dir / "d1_vllm_events.csv", event_rows)
        write_json(output_dir / "d1_overhead.json", overhead)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true", help="CPU-only tables without matplotlib")
    args = parser.parse_args(argv)
    analyze(args.input_dir, args.output_dir, plots=not args.no_plots)


if __name__ == "__main__":
    main()
