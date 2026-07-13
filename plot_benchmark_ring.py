from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SECTION_RE = re.compile(r"^\[(forward_all_cp|forward_hybrid|backward)\]")
SM_RE = re.compile(r"^SM config: num_comp_sm=(\d+), num_comm_sm=(\d+)")
CASE_RE = re.compile(r"^Running B=(\d+), local_S=(\d+), causal=(True|False)")
TABLE_RESULT_RE = re.compile(
    r"^(\S+)\s+.*?max_across_ranks=(\d+(?:\.\d+)?)\s+"
    r"(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+"
)
HYBRID_RESULT_RE = re.compile(
    r"^method=(\S+)\s+mode=\S+\s+SM=(\d+):(\d+)\s+"
    r"max_ms=(\d+(?:\.\d+)?)\s+agg_TFLOPS=(\d+(?:\.\d+)?)\s+"
    r"avg_gpu_TFLOPS=(\d+(?:\.\d+)?)"
)

PANEL_METHODS = {
    "forward_all_cp": [
        "allgather_attention",
        "llama3_allgather_attention",
        "fa3",
        "min_varlen_mega_ring",
    ],
    "forward_hybrid": [
        "allgather_attention",
        "llama3_allgather_attention",
        "fa3_ring",
        "mega_ring_all_cp",
        "mega_ring_hybrid",
    ],
    "backward": [
        "allgather_attention",
        "llama3_allgather_attention",
        "min_varlen_python_ring",
        "min_varlen_mega_ring",
    ],
}

METHOD_LABELS = {
    "allgather_attention": "Per-seq AllGather",
    "llama3_allgather_attention": "Llama3 AllGather",
    "fa3": "FA3",
    "fa3_ring": "FA3",
    "min_varlen_python_ring": "FA3",
    "min_varlen_mega_ring": "Mega-ring",
    "mega_ring_all_cp": "Mega-ring All-CP",
    "mega_ring_hybrid": "Mega-ring Hybrid",
}

METHOD_COLORS = {
    "allgather_attention": "#4C78A8",
    "llama3_allgather_attention": "#59A14F",
    "fa3": "#9C755F",
    "fa3_ring": "#9C755F",
    "min_varlen_python_ring": "#9C755F",
    "min_varlen_mega_ring": "#E15759",
    "mega_ring_all_cp": "#F28E2B",
    "mega_ring_hybrid": "#B07AA1",
}

PANEL_TITLES = {
    "forward_all_cp": "(a) Forward - uniform",
    "forward_hybrid": "(b) Forward - hybrid varlen",
    "backward": "(c) Backward - uniform",
}


@dataclass(frozen=True)
class Record:
    section: str
    method: str
    workload: tuple[int, int] | None
    sm: tuple[int, int]
    max_ms: float
    aggregate_tflops: float
    avg_gpu_tflops: float


def parse_log(path: Path) -> list[Record]:
    records: list[Record] = []
    section = None
    sm = None
    workload = None

    for line in path.read_text(encoding="utf-8").splitlines():
        section_match = SECTION_RE.match(line)
        if section_match is not None:
            section = section_match.group(1)
            sm = None
            workload = None
            continue

        if section in ("forward_all_cp", "backward"):
            sm_match = SM_RE.match(line)
            if sm_match is not None:
                sm = (int(sm_match.group(1)), int(sm_match.group(2)))
                workload = None
                continue

            case_match = CASE_RE.match(line)
            if case_match is not None:
                workload = (int(case_match.group(1)), int(case_match.group(2)))
                continue

            result_match = TABLE_RESULT_RE.match(line)
            if result_match is None or sm is None or workload is None:
                continue
            records.append(
                Record(
                    section=section,
                    method=result_match.group(1),
                    workload=workload,
                    sm=sm,
                    max_ms=float(result_match.group(2)),
                    aggregate_tflops=float(result_match.group(3)),
                    avg_gpu_tflops=float(result_match.group(4)),
                )
            )
            continue

        if section == "forward_hybrid":
            result_match = HYBRID_RESULT_RE.match(line)
            if result_match is None:
                continue
            records.append(
                Record(
                    section=section,
                    method=result_match.group(1),
                    workload=None,
                    sm=(int(result_match.group(2)), int(result_match.group(3))),
                    max_ms=float(result_match.group(4)),
                    aggregate_tflops=float(result_match.group(5)),
                    avg_gpu_tflops=float(result_match.group(6)),
                )
            )

    missing = [section for section in PANEL_METHODS if not any(r.section == section for r in records)]
    if missing:
        raise ValueError(f"missing benchmark sections in {path}: {missing}")
    return records


def select_best_sm(records: list[Record]) -> dict[tuple[str, tuple[int, int] | None, str], Record]:
    best: dict[tuple[str, tuple[int, int] | None, str], Record] = {}
    for record in records:
        if record.method not in PANEL_METHODS[record.section]:
            continue
        key = (record.section, record.workload, record.method)
        current = best.get(key)
        if current is None or record.max_ms < current.max_ms:
            best[key] = record
    return best


def workload_label(workload: tuple[int, int]) -> str:
    batch, seqlen = workload
    seqlen_label = f"{seqlen // 1024}K" if seqlen % 1024 == 0 else str(seqlen)
    return f"B={batch}\nS={seqlen_label}"


def metric_value(record: Record, metric: str) -> float:
    return record.avg_gpu_tflops if metric == "avg_gpu_tflops" else record.max_ms


def plot_line_panel(ax, section: str, best: dict, metric: str) -> None:
    workloads = sorted(
        {
            workload
            for record_section, workload, _method in best
            if record_section == section and workload is not None
        },
        key=lambda item: item[1],
    )
    x = list(range(len(workloads)))
    for method in PANEL_METHODS[section]:
        values = []
        for workload in workloads:
            record = best.get((section, workload, method))
            if record is None:
                raise ValueError(f"missing {section} data for {method}, workload={workload}")
            values.append(metric_value(record, metric))
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_xticks(x, [workload_label(workload) for workload in workloads])
    ax.set_xlabel("Local workload")


def plot_hybrid_panel(ax, best: dict, metric: str) -> None:
    methods = PANEL_METHODS["forward_hybrid"]
    records = []
    for method in methods:
        record = best.get(("forward_hybrid", None, method))
        if record is None:
            raise ValueError(f"missing forward_hybrid data for {method}")
        records.append(record)

    x = list(range(len(methods)))
    values = [metric_value(record, metric) for record in records]
    ax.bar(
        x,
        values,
        width=0.72,
        color=[METHOD_COLORS[method] for method in methods],
        edgecolor="white",
        linewidth=0.8,
    )
    labels = [f"{METHOD_LABELS[method]}\nSM {record.sm[0]}:{record.sm[1]}" for method, record in zip(methods, records)]
    ax.set_xticks(x, labels, rotation=20, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Method and selected SM config")


def make_figure(records: list[Record], metric: str):
    best = select_best_sm(records)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20.0, 6.0),
        sharey=metric == "avg_gpu_tflops",
    )

    plot_line_panel(axes[0], "forward_all_cp", best, metric)
    plot_hybrid_panel(axes[1], best, metric)
    plot_line_panel(axes[2], "backward", best, metric)

    ylabel = "Average per-GPU TFLOPS" if metric == "avg_gpu_tflops" else "Max rank latency (ms)"
    axes[0].set_ylabel(ylabel)
    if metric == "max_ms":
        axes[1].set_ylabel(ylabel)
        axes[2].set_ylabel(ylabel)
    for ax, section in zip(axes, ("forward_all_cp", "forward_hybrid", "backward")):
        ax.set_title(PANEL_TITLES[section], fontsize=13)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.tick_params(axis="both", labelsize=9.5)

    handles_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        handles_by_label.update(zip(labels, handles))
    fig.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle("Ring Attention - Best SM Configuration per Method and Workload", y=1.06, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig, best


def print_selection(best: dict) -> None:
    for section in ("forward_all_cp", "forward_hybrid", "backward"):
        print(f"[{section}]")
        keys = sorted(
            (key for key in best if key[0] == section),
            key=lambda key: ((key[1] or (0, 0))[1], PANEL_METHODS[section].index(key[2])),
        )
        for _section, workload, method in keys:
            record = best[(_section, workload, method)]
            workload_text = "hybrid" if workload is None else f"B={workload[0]}, S={workload[1]}"
            print(
                f"  {workload_text:<18} {method:<27} SM={record.sm[0]}:{record.sm[1]:<2} "
                f"max_ms={record.max_ms:.4f} avg_gpu_TFLOPS={record.avg_gpu_tflops:.1f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot best-SM forward, hybrid, and backward results from a benchmark_ring log"
    )
    parser.add_argument("input", nargs="?", type=Path, default=Path("benchmark_ring_4.log"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--metric",
        choices=("avg_gpu_tflops", "max_ms"),
        default="avg_gpu_tflops",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_name(f"{args.input.stem}_best_sm.png")
    records = parse_log(args.input)
    fig, best = make_figure(records, args.metric)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print_selection(best)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
