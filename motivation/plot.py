"""Plot v2 measurements; matplotlib is only needed when rendering figures."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .summarize import load_records, summary_rows

PHASES = ("Q all-gather", "attention", "history combine", "receive", "final combine")


def hardware_label(record):
    env = record["environment"]
    config = record.get("config", {})
    topology = (f"TP{config['tp_size']}/DCP{config['dcp_size']} Q{config['qhead']}/KV{config['kvhead']}/D{config['headdim']} "
                f"{config['configuration_role']}" if "tp_size" in config else
                f"CP{record.get('gpus', env['world_size'])}")
    return f"{env['gpu_name']} ({env['sm_count']} SM/GPU), {topology}"


def trace_intervals(rows):
    if not rows:
        return []
    origin = min(row[0] for row in rows)
    intervals = []
    for start, end, cta, sm, phase in rows:
        if end < start or phase not in range(5):
            raise ValueError("invalid CTA phase interval")
        intervals.append((sm, (start - origin) / 1000, (end - start) / 1000, phase, cta))
    return intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = load_records(args.run_dir)
    rows = summary_rows(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = defaultdict(list)
    for row in rows:
        panels[(row["experiment"], row["metric"], row["statistic"])].append(row)
    for (experiment, metric, statistic), values in panels.items():
        grouped = defaultdict(list)
        for row in values:
            category = row["dataset"] or str(row["case_id"])
            grouped[(category, row["method"])].append(row["value"])
        categories = sorted({key[0] for key in grouped})
        methods = sorted({key[1] for key in grouped})
        fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.4), 4.5))
        width = 0.8 / len(methods)
        for i, method in enumerate(methods):
            x = [n - 0.4 + width * (i + 0.5) for n in range(len(categories))]
            y = [mean(grouped[(category, method)]) if (category, method) in grouped else float("nan") for category in categories]
            ax.bar(x, y, width=width, label=method)
        ax.set_xticks(range(len(categories)), categories, rotation=20)
        ax.set_ylabel(metric)
        ax.set_title(f"{experiment}: {statistic}" + ("; equal-case arithmetic mean" if experiment == "T3" else ""))
        sources = [r for r in records if r["schema"] == f"motivation.v2.{experiment}"]
        labels = sorted({hardware_label(r) for r in sources})
        caption = "; ".join(labels)
        if experiment == "D1":
            caption += "\ntrace off; " + ", ".join(sorted({r["selection_status"] for r in sources}))
            caption += "\n" + sources[0]["selection_rule"]
        fig.suptitle(caption, fontsize=8)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{experiment}_{metric}.png", dpi=180)
        plt.close(fig)
    for record in records:
        if record["schema"] != "motivation.v2.D1":
            continue
        for mode, diagnostic in record["diagnostics"].items():
            if mode == "vllm_a2a":
                # Independent stage bars: no invented SM occupancy or additive
                # critical path made from overlapping/aggregate event intervals.
                phases = diagnostic["stages_ms"]
                names = [name for name in phases if name != "overlapped_ag_chunk_window_ms"]
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.barh(names, [phases[name] for name in names])
                ax.set_xlabel("CUDA event duration (ms)")
                ax.set_title(f"{hardware_label(record)}\n{record['case_id']}: A2A Graph rank {diagnostic['critical_rank']}, event-instrumented replay", fontsize=9)
                fig.tight_layout()
                fig.savefig(args.output_dir / f"D1_{record['case_id']}_a2a.png", dpi=180)
                plt.close(fig)
                continue
            for rank, trace in diagnostic["trace_by_rank"].items():
                fig, ax = plt.subplots(figsize=(10, 5))
                seen = set()
                for sm, start, duration, phase, cta in trace_intervals(trace):
                    ax.broken_barh([(start, duration)], (sm - 0.4, 0.8),
                                   facecolors=f"C{phase}", label=PHASES[phase] if phase not in seen else None)
                    seen.add(phase)
                ax.set_xlabel("rank-local time (µs); phase checkpoints include waits")
                ax.set_ylabel("SM ID")
                ax.set_title(f"{hardware_label(record)}\n{record['case_id']} {mode}, CTA trace on, rank {rank}", fontsize=9)
                ax.legend(fontsize=7)
                fig.tight_layout()
                fig.savefig(args.output_dir / f"D1_{record['case_id']}_{mode}_rank{rank}.png", dpi=180)
                plt.close(fig)


if __name__ == "__main__":
    main()
