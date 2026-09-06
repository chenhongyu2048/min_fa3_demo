#!/usr/bin/env python3
"""Plot the currently completed MegaCP motivation measurements.

The script intentionally only consumes the JSON artifacts produced by
``motivation_lab.sh``.  T1/T2/T3 form the training/full-prefill figure and D1
forms the decode figure.  D2 is not plotted: the current run has no reliable
NCU DRAM/L2 counters, so this script never turns a missing counter into zero.

Examples
--------
    .venv/bin/python motivation/plot_motivation.py \
        --run-dir benchmark_logs/motivation/20260906_042142 \
        --output-dir benchmark_logs/motivation/20260906_042142/figures

The plot dependency is an optional project group.  On a fresh environment use
``uv sync --frozen --group plot`` before running this entry point.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "benchmark_logs" / "motivation" / "motivation_4gpu_20260906_161823"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "figures"

COLORS = {
    "ring": "#4C78A8",
    "allgather": "#F58518",
    "llama3": "#54A24B",
    "mega": "#E45756",
    "megatron": "#4C78A8",
    "zeppelin": "#F58518",
    "all_cp": "#E45756",
    "q_allgather": "#4C78A8",
    "attention": "#59A14F",
    "history_combine": "#F28E2B",
    "receive": "#B279A2",
    "final_combine": "#E15759",
}

T1_METHODS = ("ring", "allgather")
T1_LABELS = {"ring": "FA3 Ring", "allgather": "AllGather"}
T3_METHODS = ("megatron_hybrid_cp", "zeppelin", "mega_ring_all_cp")
T3_LABELS = {
    "megatron_hybrid_cp": "Megatron",
    "zeppelin": "Zeppelin",
    "mega_ring_all_cp": "All-CP",
}
PHASES = ("q_allgather", "attention", "history_combine", "receive", "final_combine")
PHASE_LABELS = {
    "q_allgather": "Q all-gather",
    "attention": "Attention",
    "history_combine": "History combine",
    "receive": "Receive",
    "final_combine": "Final combine",
}
PHASE_IDS = {index: name for index, name in enumerate(PHASES)}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing motivation artifact: {path}")
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _finite_positive(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return result


def _finite_nonnegative(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative, got {value!r}")
    return result


def _p50_timing(result: dict[str, Any], *, expected_iters: int, label: str) -> float:
    values = result.get("rank_max_ms")
    if not isinstance(values, list) or len(values) != expected_iters:
        raise ValueError(f"{label}: expected {expected_iters} rank-max samples")
    values = [_finite_positive(value, f"{label} sample") for value in values]
    reported = _finite_positive(result.get("p50_rank_max_ms"), f"{label} p50")
    ordered = sorted(values)
    expected = ordered[round((len(ordered) - 1) * 0.5)]
    if not math.isclose(reported, expected, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{label}: reported p50 does not match rank-max samples")
    return reported


def _p50_component(result: dict[str, Any], *, expected_iters: int, label: str) -> float:
    values = result.get("rank_max_ms")
    if not isinstance(values, list) or len(values) != expected_iters:
        raise ValueError(f"{label}: expected {expected_iters} rank-max samples")
    values = [_finite_nonnegative(value, f"{label} sample") for value in values]
    reported = _finite_nonnegative(result.get("p50_rank_max_ms"), f"{label} p50")
    ordered = sorted(values)
    expected = ordered[round((len(ordered) - 1) * 0.5)]
    if not math.isclose(reported, expected, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{label}: reported p50 does not match rank-max samples")
    return reported


def _style_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color="#D8D8D8", linewidth=0.8, alpha=0.85)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _bar_labels(axis: plt.Axes, bars: Iterable[Any], fmt: str = "{:.2f}") -> None:
    for bar in bars:
        value = bar.get_height()
        axis.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )


def load_training(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    t1_dir = run_dir / "t1_training_overlap"
    t1_paths = sorted(t1_dir.glob("t1_*.json"))
    if not t1_paths:
        legacy = t1_dir / "t1_cp4.json"
        t1_paths = [legacy] if legacy.is_file() else []
    if not t1_paths:
        raise FileNotFoundError(f"missing T1 JSON files below {t1_dir}")
    t1_by_id = {path.stem.removeprefix("t1_"): path for path in t1_paths}
    t1_order: list[str] = []
    t1_manifest = t1_dir / "case_manifest.json"
    if t1_manifest.is_file():
        t1_order = [case["case_id"] for case in _read_json(t1_manifest).get("cases", [])]
    t1_order.extend(case_id for case_id in t1_by_id if case_id not in t1_order)
    t1 = {case_id: _read_json(t1_by_id[case_id]) for case_id in t1_order}
    t2_paths = sorted((run_dir / "t2_training_step").glob("t2_*.json"))
    if not t2_paths:
        legacy = run_dir / "t2_training_step" / "t2_cp4.json"
        t2_paths = [legacy] if legacy.is_file() else []
    t3_paths = sorted((run_dir / "t3_load_balance").glob("t3_cp*.json"))
    if not t2_paths or len(t3_paths) != 1:
        raise FileNotFoundError("missing T2 or T3 JSON artifacts")
    t2_by_id = {path.stem.removeprefix("t2_"): path for path in t2_paths}
    t2_order = [case["case_id"] for case in _read_json(run_dir / "t2_training_step" / "case_manifest.json").get("cases", [])] if (run_dir / "t2_training_step" / "case_manifest.json").is_file() else []
    t2_order.extend(case_id for case_id in t2_by_id if case_id not in t2_order)
    t2 = {case_id: _read_json(t2_by_id[case_id]) for case_id in t2_order}
    t3 = _read_json(t3_paths[0])
    for name, data in (("T1", next(iter(t1.values()))), ("T2", next(iter(t2.values())))):
        if data.get("environment", {}).get("gpu_name") != "NVIDIA H20":
            raise ValueError(f"{name}: expected the recorded H20 run artifact")
    reference_t1 = next(iter(t1.values()))
    if t3.get("config", {}).get("world_size") != reference_t1.get("config", {}).get("world_size"):
        raise ValueError("T3 world size does not match the T1 CP run")
    for case_id, data in t1.items():
        iters = int(data["config"]["iters"])
        for key, result in data["results"].items():
            _p50_timing(result, expected_iters=iters, label=f"T1 {case_id} {key}")
    for case_id, data in t2.items():
        t2_iters = int(data["config"]["iters"])
        for key, result in data["results"].items():
            _p50_timing(result["timing"], expected_iters=t2_iters, label=f"T2 {case_id} {key}")
    if "datasets" in t3 and isinstance(t3["cases"], dict):
        for dataset in t3["datasets"]:
            cases = t3["cases"].get(dataset)
            if not isinstance(cases, list) or len(cases) != int(t3["config"]["num_cases"]):
                raise ValueError(f"T3 {dataset} case count does not match its configuration")
            for case in cases:
                for method in T3_METHODS:
                    if method not in case:
                        raise ValueError(f"T3 {dataset} case {case.get('case_id')} is missing {method}")
                    for metric in ("token_imbalance", "attention_imbalance"):
                        _finite_positive(case[method][metric], f"T3 {dataset} {method} {metric}")
                    _finite_nonnegative(case[method]["communication_tx_bytes"], f"T3 {dataset} {method} communication_tx_bytes")
    else:
        cases = t3.get("cases")
        if not isinstance(cases, list) or len(cases) != int(t3["config"]["num_cases"]):
            raise ValueError("T3 case count does not match its configuration")
        for case in cases:
            for method in T3_METHODS:
                if method not in case:
                    raise ValueError(f"T3 case {case.get('case_id')} is missing {method}")
                for metric in ("token_imbalance", "attention_imbalance"):
                    _finite_positive(case[method][metric], f"T3 {method} {metric}")
                _finite_nonnegative(case[method]["communication_tx_bytes"], f"T3 {method} communication_tx_bytes")
    return t1, t2, t3


def plot_training(t1: dict[str, dict[str, Any]], t2: dict[str, Any], t3: dict[str, Any]) -> plt.Figure:
    figure, axes = plt.subplots(2, 4, figsize=(19.5, 8.8), squeeze=False)
    case_ids = list(t1)
    positions = list(range(len(case_ids)))
    component_methods = ("ring", "allgather")
    component_labels = {"ring": "FA3 Ring", "allgather": "AllGather"}
    axis = axes[0, 0]
    for method in component_methods:
        standalone = []
        corun = []
        for case_id in case_ids:
            data = t1[case_id]
            standalone.append(_p50_timing(data["results"][f"{method}_comm_only"], expected_iters=data["config"]["iters"], label=f"T1 {case_id} {method} comm-only"))
            corun.append(_p50_component(data["corun"][method]["communication_ms"], expected_iters=data["config"]["iters"], label=f"T1 {case_id} {method} co-run communication"))
        axis.plot(positions, standalone, color=COLORS[method], marker="o", linewidth=2.0, label=f"{component_labels[method]} standalone")
        axis.plot(positions, corun, color=COLORS[method], marker="s", linestyle="--", linewidth=2.0, label=f"{component_labels[method]} co-run")
    axis.set_xticks(positions)
    axis.set_xticklabels(case_ids, rotation=25, ha="right", fontsize=8)
    axis.set_ylabel("Rank-max p50 (ms)")
    axis.set_title("T1  Communication duration", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=7.3)
    _style_axis(axis)

    axis = axes[0, 1]
    compute_standalone = []
    compute_with_ring = []
    compute_with_allgather = []
    for case_id in case_ids:
        data = t1[case_id]
        compute_standalone.append(_p50_timing(data["results"]["ring_comp_only"], expected_iters=data["config"]["iters"], label=f"T1 {case_id} compute-only"))
        compute_with_ring.append(_p50_component(data["corun"]["ring"]["compute_ms"], expected_iters=data["config"]["iters"], label=f"T1 {case_id} compute with Ring"))
        compute_with_allgather.append(_p50_component(data["corun"]["allgather"]["compute_ms"], expected_iters=data["config"]["iters"], label=f"T1 {case_id} compute with AllGather"))
    axis.plot(positions, compute_standalone, color="#555555", marker="o", linewidth=2.0, label="COMP standalone")
    axis.plot(positions, compute_with_ring, color=COLORS["ring"], marker="s", linestyle="--", linewidth=2.0, label="COMP with Ring COMM")
    axis.plot(positions, compute_with_allgather, color=COLORS["allgather"], marker="D", linestyle="--", linewidth=2.0, label="COMP with AllGather COMM")
    axis.set_xticks(positions)
    axis.set_xticklabels(case_ids, rotation=25, ha="right", fontsize=8)
    axis.set_ylabel("Rank-max p50 (ms)")
    axis.set_title("T1  Attention duration", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=7.3)
    _style_axis(axis)

    axis = axes[0, 2]
    for method in T1_METHODS:
        serial = []
        overlap = []
        for case_id in case_ids:
            data = t1[case_id]
            serial.append(_p50_timing(data["results"][f"{method}_serial"], expected_iters=data["config"]["iters"], label=f"T1 {case_id} {method} serial"))
            overlap.append(_p50_timing(data["results"][f"{method}_overlap"], expected_iters=data["config"]["iters"], label=f"T1 {case_id} {method} overlap"))
        axis.plot(positions, serial, color=COLORS[method], marker="o", linewidth=2.0, label=f"{T1_LABELS[method].replace(chr(10), ' ')} serial")
        axis.plot(positions, overlap, color=COLORS[method], marker="s", linestyle="--", linewidth=2.0, label=f"{T1_LABELS[method].replace(chr(10), ' ')} overlap")
    axis.set_xticks(positions)
    axis.set_xticklabels(case_ids, rotation=25, ha="right", fontsize=8)
    axis.set_ylabel("Complete-call p50 (ms)")
    axis.set_title("T1  Serial vs overlap total time", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=6.8, ncol=2)
    _style_axis(axis)

    axis = axes[0, 3]
    t2_names = (
        "step_external_reduce",
        "step_fused_reduce",
        "linear_queue_recycle",
    )
    t2_labels = (
        "Step + external reduce",
        "Step + fused reduce",
        "Continuous segment",
    )
    t2_colors = ("#7F7F7F", "#F58518", COLORS["mega"])
    t2_case_ids = list(t2)
    t2_positions = list(range(len(t2_case_ids)))
    for offset, name in enumerate(t2_names):
        values = []
        for case_id in t2_case_ids:
            data = t2[case_id]
            values.append(_p50_timing(data["results"][name]["timing"], expected_iters=data["config"]["iters"], label=f"T2 {case_id} {name}"))
            step_stats = data["results"]["step_fused_reduce"]["stats"]
            segment_stats = data["results"]["linear_queue_recycle"]["stats"]
            if step_stats["qo_visits"] != segment_stats["qo_visits"] or step_stats["kv_tile_reads"] != segment_stats["kv_tile_reads"]:
                raise ValueError(f"T2 {case_id} profiles do not report the same mathematical work")
        axis.plot(t2_positions, values, color=t2_colors[offset], marker=("o", "s", "D")[offset], linewidth=2.0, label=t2_labels[offset])
    axis.set_xticks(t2_positions)
    axis.set_xticklabels(t2_case_ids, rotation=25, ha="right", fontsize=8)
    axis.set_ylabel("Compute-only p50 (ms)")
    axis.set_title("T2  Same work, fewer step boundaries", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=7.0)
    _style_axis(axis)

    datasets = list(t3.get("datasets", []))
    if not datasets:
        datasets = [str(t3["config"].get("dataset", "arxiv"))]
        summaries = {datasets[0]: t3["summary"]}
    else:
        summaries = t3["summary"]
    dataset_positions = list(range(len(datasets)))
    dataset_bar_width = 0.18
    for axis, metric, ylabel, title, scale in (
        (axes[1, 0], "token_imbalance", "max / mean tokens", "T3  Token balance across datasets", 1.0),
        (axes[1, 1], "attention_imbalance", "max / mean attention work", "T3  Attention-work balance across datasets", 1.0),
        (axes[1, 2], "communication_tx_bytes", "Total Tx (GiB, analytical)", "T3  Communication trade-off across datasets", 1.0 / (1024**3)),
    ):
        for offset, method in enumerate(T3_METHODS):
            values = [float(summaries[dataset][method][metric]) * scale for dataset in datasets]
            axis.bar([p + (offset - 1) * dataset_bar_width for p in dataset_positions], values, width=dataset_bar_width * 0.9, color=COLORS[{"megatron_hybrid_cp": "megatron", "zeppelin": "zeppelin", "mega_ring_all_cp": "all_cp"}[method]], label=T3_LABELS[method] if axis is axes[1, 2] else None)
        axis.set_xticks(dataset_positions)
        axis.set_xticklabels(datasets, rotation=25, ha="right", fontsize=8)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        _style_axis(axis)
    axes[1, 0].axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    axes[1, 1].axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    axes[1, 2].legend(frameon=False, fontsize=8)
    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.02,
        0.96,
        "T1 component panels\n"
        "solid: standalone COMM/COMP replay\n"
        "dashed: the same replay under controlled co-run\n\n"
        "T1 total panel\n"
        "solid: legal serial call\n"
        "dashed: legal overlap call\n\n"
        "Co-run components are diagnostics; they are not\n"
        "summed to reconstruct the complete critical path.",
        transform=axes[1, 3].transAxes,
        va="top",
        fontsize=10,
        color="#444444",
    )
    figure.suptitle("MegaCP motivation: execution granularity and multi-objective load organization", y=0.995, fontsize=16, fontweight="bold")
    figure.text(0.5, 0.012, "H20, CP4, BF16, D=128. T1/T2 are rank-max p50 CUDA-event timings; T3 is static accounting over shared cases per dataset.", ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0, 0.045, 1, 0.955))
    return figure


def load_decode(path: Path) -> dict[str, Any]:
    if path.is_dir():
        d1_paths = sorted(path.glob("d1_cp*.json"))
        if len(d1_paths) != 1:
            raise FileNotFoundError("select one D1 JSON artifact when a run has multiple topologies")
        path = d1_paths[0]
    data = _read_json(path)
    if data.get("experiment") != "D1_decode_sm_trace":
        raise ValueError("decode artifact is not a D1 trace")
    if isinstance(data.get("cases"), list):
        if len(data["cases"]) != 8:
            raise ValueError("D1 must contain four cases times two Mega policies")
        for record in data["cases"]:
            if record.get("workload_kind") not in ("decode_only_q16", "mixed"):
                raise ValueError("D1 case has an unknown workload kind")
            if not isinstance(record.get("mega", {}).get("trace_by_rank"), dict):
                raise ValueError("D1 case is missing per-rank Mega trace")
            for rank_rows in record["mega"]["trace_by_rank"].values():
                for row in rank_rows:
                    if not (row[0] <= row[1] <= row[2]):
                        raise ValueError("D1 contains an invalid start/useful-end/exit interval")
        return data
    timing = data["timing"]
    _p50_timing(timing, expected_iters=int(data["config"]["iters"]), label="D1")
    schema = data.get("trace_schema")
    required = ("start_ns", "useful_work_end_ns", "exit_ns", "sm_id", "phase_id", "valid_work_count")
    if not isinstance(schema, list) or any(name not in schema for name in required):
        raise ValueError("D1 trace schema is missing required columns")
    index = {name: schema.index(name) for name in required}
    rows = data.get("trace_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("D1 trace has no rows")
    for row in rows:
        if not (row[index["start_ns"]] <= row[index["useful_work_end_ns"]] <= row[index["exit_ns"]]):
            raise ValueError("D1 contains an invalid start/useful-end/exit interval")
    return data | {"_index": index}


def plot_decode(data: dict[str, Any]) -> plt.Figure:
    if isinstance(data.get("cases"), list):
        return _plot_decode_cases(data)
    index = data["_index"]
    rows = data["trace_rows"]
    trace_start = min(row[index["start_ns"]] for row in rows)
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 6.8), gridspec_kw={"width_ratios": (1.7, 1.0, 0.9)})

    axis = axes[0]
    legend_handles = []
    for phase_id, phase_name in PHASE_IDS.items():
        phase_rows = [row for row in rows if row[index["phase_id"]] == phase_id and row[index["valid_work_count"]] > 0]
        color = COLORS[phase_name]
        if phase_rows:
            axis.barh(
                [row[index["sm_id"]] for row in phase_rows],
                [(row[index["useful_work_end_ns"]] - row[index["start_ns"]]) / 1000.0 for row in phase_rows],
                left=[(row[index["start_ns"]] - trace_start) / 1000.0 for row in phase_rows],
                height=0.72,
                color=color,
                alpha=0.88,
                edgecolor="none",
            )
            for row in phase_rows:
                useful = (row[index["useful_work_end_ns"]] - trace_start) / 1000.0
                exit_time = (row[index["exit_ns"]] - trace_start) / 1000.0
                if exit_time > useful:
                    axis.plot([useful, exit_time], [row[index["sm_id"]]] * 2, color="#222222", linewidth=0.45, alpha=0.7)
            legend_handles.append(Patch(facecolor=color, label=PHASE_LABELS[phase_name]))
    axis.set_xlabel("Relative time (µs)")
    axis.set_ylabel("SM ID")
    sm_count = max(row[index["sm_id"]] for row in rows) + 1
    axis.set_yticks(range(0, sm_count, max(1, sm_count // 10)))
    axis.set_title("D1  Mega persistent-kernel trace", loc="left", fontweight="bold")
    axis.legend(handles=legend_handles, frameon=False, fontsize=7.5, loc="upper right")
    _style_axis(axis, grid_axis="x")

    axis = axes[1]
    phase_end_values: list[list[float]] = []
    for phase_id, phase_name in PHASE_IDS.items():
        values = sorted(
            (row[index["useful_work_end_ns"]] - trace_start) / 1000.0
            for row in rows
            if row[index["phase_id"]] == phase_id and row[index["valid_work_count"]] > 0
        )
        phase_end_values.append(values)
        if values:
            axis.scatter([phase_id] * len(values), values, s=11, color=COLORS[phase_name], alpha=0.65, zorder=2)
            p50 = values[round((len(values) - 1) * 0.5)]
            axis.scatter([phase_id], [p50], s=42, color="#111111", marker="D", zorder=3)
            axis.plot([phase_id, phase_id], [values[0], values[-1]], color=COLORS[phase_name], linewidth=2.2, zorder=1)
    axis.set_xticks(range(len(PHASES)))
    axis.set_xticklabels([PHASE_LABELS[name].replace(" ", "\n") for name in PHASES], fontsize=7.5)
    axis.set_ylabel("Useful-work end (relative µs)")
    axis.set_title("Stage tails within one replay", loc="left", fontweight="bold")
    _style_axis(axis)
    axis.text(0.03, 0.03, "dots: CTA ends\n◆: p50", transform=axis.transAxes, fontsize=8, color="#555555")

    axis = axes[2]
    tails = [float(data["phase_summary"][name]["tail_ns"]) / 1000.0 for name in PHASES]
    bars = axis.barh(range(len(PHASES)), tails, color=[COLORS[name] for name in PHASES], height=0.62)
    for bar, value in zip(bars, tails):
        axis.annotate(f"{value:.2f}", (value, bar.get_y() + bar.get_height() / 2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    axis.set_yticks(range(len(PHASES)))
    axis.set_yticklabels([PHASE_LABELS[name] for name in PHASES], fontsize=8)
    axis.set_xlabel("P100 − P50 useful-end (µs)")
    axis.set_title("Tail span", loc="left", fontweight="bold")
    _style_axis(axis, grid_axis="x")
    axis.text(
        0.03,
        0.03,
        f"DCP4, q=16, history=65,536\ncall p50 = {data['timing']['p50_rank_max_ms']:.2f} ms\nNCCL internal CTAs not traced",
        transform=axis.transAxes,
        fontsize=8,
        color="#555555",
    )
    figure.suptitle(
        "Decode motivation: fine-grained Mega stages expose useful-work tails",
        y=0.995,
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "H20 DCP4; bars show CTA start→useful-end, thin black segments show useful-end→exit. D2 NCU memory counters are pending.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.955))
    return figure


def _plot_decode_cases(data: dict[str, Any]) -> plt.Figure:
    """Plot two selected workload kinds, with optimized/unoptimized Mega pairs."""
    grouped: dict[str, list[dict[str, Any]]] = {"decode_only_q16": [], "mixed": []}
    for record in data["cases"]:
        grouped[record["workload_kind"]].append(record)
    figure, axes = plt.subplots(2, 4, figsize=(22.0, 9.4), squeeze=False)
    schema = ["start_ns", "useful_work_end_ns", "exit_ns", "iteration", "cta_id", "sm_id", "phase_id", "valid_work_count"]
    phase_names = {0: "q_allgather", 1: "attention", 2: "history_combine", 3: "receive", 4: "final_combine"}
    for row_index, (kind, records) in enumerate(grouped.items()):
        if len(records) != 4:
            raise ValueError(f"D1 {kind} must contain two cases times two policies")
        optimized = [record for record in records if record["optimized"]]
        unoptimized = [record for record in records if not record["optimized"]]
        case_ids = sorted({record["case_id"] for record in records})
        if len(optimized) != 2 or len(unoptimized) != 2:
            raise ValueError(f"D1 {kind} is missing an optimized/unoptimized pair")
        representative = optimized[0]
        control = next(record for record in unoptimized if record["case_id"] == representative["case_id"])
        trace_by_rank = representative["mega"]["trace_by_rank"]
        rank_name = sorted(trace_by_rank, key=int)[0]

        def plot_trace(axis: plt.Axes, record: dict[str, Any], title: str) -> None:
            trace_rows = record["mega"]["trace_by_rank"][rank_name]
            trace_start = min(entry[0] for entry in trace_rows)
            for phase_id, phase_name in phase_names.items():
                rows = [entry for entry in trace_rows if entry[6] == phase_id and entry[7] > 0]
                if not rows:
                    continue
                color = COLORS[phase_name]
                axis.barh(
                    [entry[5] for entry in rows],
                    [(entry[1] - entry[0]) / 1000.0 for entry in rows],
                    left=[(entry[0] - trace_start) / 1000.0 for entry in rows],
                    height=0.72,
                    color=color,
                    alpha=0.88,
                    edgecolor="none",
                )
            axis.set_title(title, loc="left", fontweight="bold")
            axis.set_xlabel("Relative time (µs)")
            axis.set_ylabel("SM ID")
            _style_axis(axis, grid_axis="x")
            axis.text(
                0.02,
                0.03,
                f"rank {rank_name}; NCCL internal CTAs not traced",
                transform=axis.transAxes,
                fontsize=8,
                color="#555555",
            )

        plot_trace(
            axes[row_index, 0],
            representative,
            f"{kind}: critical-wave trace ({representative['case_id']})",
        )
        plot_trace(
            axes[row_index, 1],
            control,
            f"{kind}: FA3-native + FIFO trace ({control['case_id']})",
        )

        axis = axes[row_index, 2]
        policy_values: list[float] = []
        policy_labels: list[str] = []
        for label, selected in (("critical-wave", optimized), ("FA3-native + FIFO", unoptimized)):
            for record in sorted(selected, key=lambda item: item["case_id"]):
                summary = record["mega"].get("phase_summary_by_rank", {}).get(rank_name, {})
                tails = [float(value["tail_ns"]) / 1000.0 for value in summary.values()]
                policy_values.append(max(tails) if tails else 0.0)
                policy_labels.append(f"{label}\n{record['case_id']}")
        bars = axis.bar(range(len(policy_values)), policy_values, color=[COLORS["mega"]] * 2 + ["#9C9C9C"] * 2, width=0.62)
        _bar_labels(axis, bars, fmt="{:.2f}")
        axis.set_xticks(range(len(policy_values)))
        axis.set_xticklabels(policy_labels, fontsize=7.5)
        axis.set_ylabel("Max stage tail (µs)")
        axis.set_title("Optimized vs unoptimized Mega", loc="left", fontweight="bold")
        _style_axis(axis)

        axis = axes[row_index, 3]
        stage_names = (
            "q_allgather_and_reorder_ms",
            "local_history_attention_ms",
            "a2a_pack_ms",
            "a2a_all_to_all_ms",
            "a2a_unpack_combine_ms",
            "local_chunk_attention_ms",
            "state_merge_ms",
        )
        stage_labels = ("Q AG", "History", "A2A pack", "A2A", "A2A unpack", "Chunk", "Merge")
        stage_colors = ("#4C78A8", "#54A24B", "#F28E2B", "#B279A2", "#E15759", "#59A14F", "#9C755F")
        baseline_records = sorted(optimized, key=lambda item: item["case_id"])
        critical_samples = []
        for record in baseline_records:
            critical_samples.extend(
                record["vllm_a2a_graph"].get("critical_rank_samples", [])
            )
        if critical_samples:
            critical_samples.sort(
                key=lambda sample: sample["stages_ms"]["attention_end_to_end_ms"]
            )
            timeline_sample = critical_samples[(len(critical_samples) - 1) // 2]
            stage_timing = timeline_sample["stages_ms"]
            timeline_note = (
                f"representative median replay, critical rank "
                f"{timeline_sample['critical_rank']}"
            )
        else:
            stage_timing = {
                name: sum(
                    record["vllm_a2a_graph"]["timing"][name]["p50"]
                    for record in baseline_records
                )
                / len(baseline_records)
                for name in (*stage_names, "attention_end_to_end_ms")
            }
            timeline_note = "legacy aggregate: mean of the two case p50 phases"
        cursor = 0.0
        for stage_name, stage_label, color in zip(stage_names, stage_labels, stage_colors):
            duration = _finite_nonnegative(stage_timing[stage_name], f"D1 vLLM {stage_name}")
            axis.barh(0, duration, left=cursor, height=0.42, color=color, edgecolor="white", linewidth=0.8, label=stage_label)
            cursor += duration
        total = _finite_positive(stage_timing["attention_end_to_end_ms"], "D1 vLLM total")
        residual = max(0.0, total - cursor)
        if residual > 0:
            axis.barh(0, residual, left=cursor, height=0.42, color="#D8D8D8", edgecolor="white", linewidth=0.8, label="Other / gaps")
        axis.axvline(total, color="#111111", linewidth=1.5, linestyle="--")
        axis.set_yticks([])
        axis.set_xlabel("Cumulative measured phase time (ms)")
        axis.set_title("vLLM A2A Graph cumulative phase timeline", loc="left", fontweight="bold")
        _style_axis(axis, grid_axis="x")
        axis.legend(frameon=False, fontsize=6.8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.35))
        axis.text(0.03, 0.91, f"{timeline_note}\ncomplete-call = {total:.3f} ms", transform=axis.transAxes, va="top", fontsize=8, color="#555555")
    figure.suptitle("D1 decode motivation: critical-wave vs FA3-native/FIFO SM traces", y=0.995, fontsize=16, fontweight="bold")
    config = data.get("config", {})
    figure.text(
        0.5,
        0.012,
        f"H20 TP/CP={config.get('tp_size', '?')}, DCP={config.get('dcp_size', '?')}, QH={config.get('qhead', '?')}, KVH={config.get('kvhead', '?')}; frozen cases selected from historical vLLM A2A Graph median; Mega is measured with and without scheduler optimization.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.955))
    return figure


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        if extension not in {"png", "pdf"}:
            raise ValueError(f"unsupported output format: {extension}")
        output = output_dir / f"{stem}.{extension}"
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {output}")
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", default="png,pdf", help="comma-separated output formats: png,pdf")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    formats = tuple(item.strip().lower() for item in args.formats.split(",") if item.strip())
    if not formats:
        raise SystemExit("--formats must not be empty")
    t1, t2, t3 = load_training(args.run_dir)
    d1_dir = args.run_dir / "d1_decode_sm_trace"
    d1_paths = sorted(d1_dir.glob("d1_cp*.json"))
    if not d1_paths:
        legacy_path = d1_dir / "d1_dcp4.json"
        d1_paths = [legacy_path] if legacy_path.is_file() else []
    if not d1_paths:
        raise FileNotFoundError(f"missing D1 JSON artifacts below {d1_dir}")
    _save_figure(plot_training(t1, t2, t3), args.output_dir, "fig1_training_load_balance", formats, args.dpi)
    for d1_path in d1_paths:
        d1 = load_decode(d1_path)
        suffix = d1_path.stem.removeprefix("d1_")
        stem = "fig2_decode_stage_trace" if len(d1_paths) == 1 else f"fig2_decode_stage_trace_{suffix}"
        _save_figure(plot_decode(d1), args.output_dir, stem, formats, args.dpi)


if __name__ == "__main__":
    main()
