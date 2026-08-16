#!/usr/bin/env python3
"""Plot Mega DCP metrics in decode-only and mixed-prefill rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

if __package__:
    from . import plot_dcp_mega_latency as base
else:
    import plot_dcp_mega_latency as base


BATCH_TYPE_DECODE = "decode_only"
BATCH_TYPE_MIXED = "mixed_prefill"
BATCH_TYPES = (BATCH_TYPE_DECODE, BATCH_TYPE_MIXED)
BATCH_TYPE_LABELS = {
    BATCH_TYPE_DECODE: "Decode-only batches",
    BATCH_TYPE_MIXED: "Mixed batches containing chunk prefill",
}
SUMMARY_NAME = "matrix_by_batch_type_summary.csv"


@dataclass(frozen=True)
class CaseContribution:
    case_id: str
    batch_type: str
    workload_signature: tuple[tuple[int, ...], tuple[int, ...], int]
    latency_ms: float
    global_effective_flops: int
    average_kv_bytes_per_gpu: float
    tp_size: int


def resolve_summary(path: Path | None) -> tuple[Path, Path]:
    selected = base._latest_run(base.DEFAULT_BENCHMARK_ROOT) if path is None else path
    selected = selected.expanduser().resolve()
    run_dir = selected if selected.is_dir() else selected.parent
    summary_path = run_dir / SUMMARY_NAME if selected.is_dir() else selected
    if not summary_path.is_file():
        raise FileNotFoundError(f"batch-type summary CSV does not exist: {summary_path}")
    return summary_path, run_dir


def load_batch_type_summary(
    path: Path, latency_stat: str
) -> tuple[dict[str, list[base.LatencyRecord]], dict[tuple[str, int, Decimal], int]]:
    latency_column = base.LATENCY_COLUMNS[latency_stat]
    required = {
        "batch_type", "arrival_time_scale", "dcp_size", "suite",
        "execution_mode", "method", "mega_num_comm_sm", "case_count",
        latency_column, base.TFLOPS_COLUMN, base.BANDWIDTH_COLUMN,
    }
    grouped = {batch_type: [] for batch_type in BATCH_TYPES}
    case_counts: dict[tuple[str, int, Decimal], int] = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            batch_type = row["batch_type"].strip()
            if batch_type not in BATCH_TYPES:
                raise ValueError(f"invalid batch_type at {path}:{line_number}")
            try:
                arrival = Decimal(row["arrival_time_scale"])
                dcp_size = int(row["dcp_size"])
                case_count = int(row["case_count"])
                comm_sm = int(row["mega_num_comm_sm"]) if row["mega_num_comm_sm"].strip() else None
                record = base.LatencyRecord(
                    arrival_time_scale=arrival,
                    dcp_size=dcp_size,
                    suite=row["suite"].strip(),
                    execution_mode=row["execution_mode"].strip(),
                    method=row["method"].strip(),
                    mega_num_comm_sm=comm_sm,
                    latency_ms=float(row[latency_column]),
                    tflops_per_gpu=float(row[base.TFLOPS_COLUMN]),
                    kv_bandwidth_gbps_per_gpu=float(row[base.BANDWIDTH_COLUMN]),
                )
            except (InvalidOperation, ValueError) as error:
                raise ValueError(f"invalid summary row at {path}:{line_number}") from error
            count_key = (batch_type, dcp_size, arrival)
            previous = case_counts.setdefault(count_key, case_count)
            if previous != case_count:
                raise ValueError(f"case count mismatch at {path}:{line_number}")
            grouped[batch_type].append(record)
    if not all(grouped.values()):
        raise ValueError(f"{path}: missing batch-type summary rows")
    return grouped, case_counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute Mega DCP metrics from per-case JSON and plot "
            "decode-only and mixed-prefill batches as a 2x3 figure"
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help=(
            "benchmark run directory or matrix_by_batch_type_summary.csv; defaults to the "
            "newest run under benchmark_logs/bench_dcp"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="latency figure output path"
    )
    parser.add_argument(
        "--flops-output",
        type=Path,
        default=None,
        help="per-GPU TFLOPS figure output path",
    )
    parser.add_argument(
        "--bandwidth-output",
        type=Path,
        default=None,
        help="effective per-GPU KV bandwidth figure output path",
    )
    parser.add_argument(
        "--latency-stat",
        choices=tuple(base.LATENCY_COLUMNS),
        default="mean",
        help="statistic over the per-case p50 latency distribution (default: mean)",
    )
    parser.add_argument(
        "--mega-num-comm-sm",
        type=base._mega_selector,
        default="best",
        metavar="{best,N}",
        help=(
            "plot the lowest-latency Mega comm-SM result per subgroup or a "
            "fixed comm-SM value (default: best)"
        ),
    )
    parser.add_argument(
        "--arrival-time-scales",
        default=None,
        help="comma-separated scales; defaults to all scales found in the CSV",
    )
    parser.add_argument(
        "--dcp-sizes",
        default=None,
        help="comma-separated sizes; defaults to all sizes found in the CSV",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="plot available summary bars and mark missing matrix points as N/A",
    )
    parser.add_argument(
        "--title", default="Mega DCP latency by arrival scale and batch type"
    )
    parser.add_argument(
        "--flops-title",
        default="Mega DCP per-GPU FLOPS by arrival scale and batch type",
    )
    parser.add_argument(
        "--bandwidth-title",
        default="Mega DCP effective KV bandwidth by arrival scale and batch type",
    )
    parser.add_argument("--dpi", type=base._positive_integer, default=220)
    args = parser.parse_args(argv)
    try:
        if args.arrival_time_scales is not None:
            args.arrival_time_scales = base._parse_arrivals(
                args.arrival_time_scales
            )
        if args.dcp_sizes is not None:
            args.dcp_sizes = base._parse_dcp_sizes(args.dcp_sizes)
    except ValueError as error:
        parser.error(str(error))
    if args.dcp_sizes is not None and len(args.dcp_sizes) != 3:
        parser.error("--dcp-sizes must contain exactly three values")
    return args


def _arrival_result_dir(run_dir: Path, arrival: Decimal) -> Path:
    result_root = run_dir / "results"
    matches = []
    for path in result_root.glob("arrival_*"):
        if not path.is_dir():
            continue
        try:
            value = Decimal(path.name.removeprefix("arrival_"))
        except InvalidOperation:
            continue
        if value == arrival:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(
            f"arrival={base._decimal_label(arrival)} does not identify one "
            f"result directory below {result_root}"
        )
    return matches[0]


def _finite_nonnegative(value: object, *, field: str, path: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field} in {path}: {value!r}") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and nonnegative in {path}")
    return parsed


def _load_case_contributions(
    path: Path,
    *,
    methods: Sequence[str],
    execution_mode: str,
) -> dict[str, CaseContribution]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        parameters = payload["parameters"]
        lengths = payload["lengths"]
        reports = payload["methods"]
        q_lengths = tuple(int(value) for value in lengths["q_global"])
        history_lengths = tuple(
            int(value) for value in lengths["history_or_cache_global"]
        )
        global_flops = int(payload["global_effective_flops"])
        tp_size = int(parameters["tp_size"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid benchmark case JSON {path}: {error}") from error
    case_id = path.name.split("_dcp", 1)[0]
    if not case_id.startswith("case_") or not q_lengths:
        raise ValueError(f"invalid case ID or empty Q batch in {path}")
    if len(q_lengths) != len(history_lengths):
        raise ValueError(f"Q/history batch lengths differ in {path}")
    if any(value <= 0 for value in q_lengths) or any(
        value < 0 for value in history_lengths
    ):
        raise ValueError(f"invalid Q/history lengths in {path}")
    if global_flops <= 0 or tp_size <= 0:
        raise ValueError(f"invalid FLOPs or TP size in {path}")
    batch_type = (
        BATCH_TYPE_DECODE if max(q_lengths) <= 16 else BATCH_TYPE_MIXED
    )
    signature = (q_lengths, history_lengths, global_flops)
    contributions = {}
    for method in methods:
        try:
            report = reports[method]
            actual_mode = str(report["execution"]["execution_mode"])
            latency_ms = float(
                report["stages_ms"]["attention_end_to_end_ms"]["p50"]
            )
            average_bytes = float(
                report["logical_kv_read"]["average_bytes_per_gpu"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid method {method} in {path}: {error}") from error
        if actual_mode != execution_mode:
            raise ValueError(
                f"execution mode for {method} in {path} is {actual_mode!r}, "
                f"expected {execution_mode!r}"
            )
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            raise ValueError(f"invalid p50 latency for {method} in {path}")
        _finite_nonnegative(
            average_bytes,
            field=f"{method} average logical KV bytes",
            path=path,
        )
        contributions[method] = CaseContribution(
            case_id=case_id,
            batch_type=batch_type,
            workload_signature=signature,
            latency_ms=latency_ms,
            global_effective_flops=global_flops,
            average_kv_bytes_per_gpu=average_bytes,
            tp_size=tp_size,
        )
    return contributions


def _aggregate_contributions(
    contributions: Sequence[CaseContribution],
    *,
    arrival: Decimal,
    dcp_size: int,
    suite: str,
    execution_mode: str,
    method: str,
    comm_sm: int | None,
    latency_stat: str,
) -> base.LatencyRecord:
    if not contributions:
        raise ValueError(
            f"no contributions for arrival={arrival}, DCP={dcp_size}, "
            f"suite={suite}, mode={execution_mode}, method={method}"
        )
    latencies = [record.latency_ms for record in contributions]
    latency_values = {
        "min": min(latencies),
        "mean": sum(latencies) / len(latencies),
        "p50": float(median(latencies)),
        "max": max(latencies),
    }
    tp_sizes = {record.tp_size for record in contributions}
    if len(tp_sizes) != 1:
        raise ValueError(f"TP size changes within {method} subgroup")
    total_latency_ms = sum(latencies)
    if total_latency_ms <= 0:
        raise ValueError(f"non-positive total latency for {method} subgroup")
    total_flops = sum(record.global_effective_flops for record in contributions)
    total_bytes = sum(
        record.average_kv_bytes_per_gpu for record in contributions
    )
    tp_size = next(iter(tp_sizes))
    return base.LatencyRecord(
        arrival_time_scale=arrival,
        dcp_size=dcp_size,
        suite=suite,
        execution_mode=execution_mode,
        method=method,
        mega_num_comm_sm=comm_sm,
        latency_ms=latency_values[latency_stat],
        tflops_per_gpu=total_flops / (total_latency_ms * 1.0e9) / tp_size,
        kv_bandwidth_gbps_per_gpu=total_bytes / (total_latency_ms * 1.0e6),
    )


def load_batch_type_records(
    run_dir: Path,
    *,
    arrivals: Sequence[Decimal],
    dcp_sizes: Sequence[int],
    latency_stat: str,
) -> tuple[
    dict[str, list[base.LatencyRecord]],
    dict[tuple[str, int, Decimal], int],
]:
    grouped_records = {batch_type: [] for batch_type in BATCH_TYPES}
    case_counts: dict[tuple[str, int, Decimal], int] = {}
    baseline_methods = tuple(
        dict.fromkeys(
            series.method
            for series in base.SERIES
            if series.method != base.METHOD_MEGA
        )
    )
    for arrival in arrivals:
        arrival_dir = _arrival_result_dir(run_dir, arrival)
        for dcp_size in dcp_sizes:
            dcp_dir = arrival_dir / f"dcp_{dcp_size}"
            mega_root = dcp_dir / "mega"
            comm_sm_dirs: list[tuple[int, Path]] = []
            for path in mega_root.glob("comm_sm_*"):
                try:
                    comm_sm = int(path.name.removeprefix("comm_sm_"))
                except ValueError:
                    continue
                if path.is_dir() and comm_sm > 0:
                    comm_sm_dirs.append((comm_sm, path))
            if not comm_sm_dirs:
                raise FileNotFoundError(f"no Mega comm-SM directories in {mega_root}")
            source_specs = [
                (
                    "baseline",
                    "eager",
                    None,
                    dcp_dir / "baseline_eager",
                    baseline_methods,
                ),
                (
                    "baseline",
                    "cuda_graph",
                    None,
                    dcp_dir / "baseline_graph",
                    baseline_methods,
                ),
                *[
                    ("mega", "eager", comm_sm, path, (base.METHOD_MEGA,))
                    for comm_sm, path in sorted(comm_sm_dirs)
                ],
            ]
            reference_signatures: dict[
                str, tuple[tuple[int, ...], tuple[int, ...], int]
            ] | None = None
            for suite, mode, comm_sm, source_dir, methods in source_specs:
                paths = sorted(source_dir.glob("case_*.json"))
                if not paths:
                    raise FileNotFoundError(f"no case JSON files in {source_dir}")
                contributions_by_method = {method: [] for method in methods}
                source_signatures = {}
                for path in paths:
                    contributions = _load_case_contributions(
                        path,
                        methods=methods,
                        execution_mode=mode,
                    )
                    first = next(iter(contributions.values()))
                    if first.case_id in source_signatures:
                        raise ValueError(
                            f"duplicate case {first.case_id} in {source_dir}"
                        )
                    source_signatures[first.case_id] = first.workload_signature
                    for method, contribution in contributions.items():
                        contributions_by_method[method].append(contribution)
                if reference_signatures is None:
                    reference_signatures = source_signatures
                elif source_signatures != reference_signatures:
                    raise ValueError(
                        f"workloads are not paired for arrival={arrival}, "
                        f"DCP={dcp_size}, source={source_dir}"
                    )
                for method, contributions in contributions_by_method.items():
                    for batch_type in BATCH_TYPES:
                        selected = [
                            contribution
                            for contribution in contributions
                            if contribution.batch_type == batch_type
                        ]
                        count_key = (batch_type, dcp_size, arrival)
                        previous_count = case_counts.setdefault(
                            count_key, len(selected)
                        )
                        if previous_count != len(selected):
                            raise ValueError(
                                f"case count changes for batch_type={batch_type}, "
                                f"arrival={arrival}, DCP={dcp_size}"
                            )
                        grouped_records[batch_type].append(
                            _aggregate_contributions(
                                selected,
                                arrival=arrival,
                                dcp_size=dcp_size,
                                suite=suite,
                                execution_mode=mode,
                                method=method,
                                comm_sm=comm_sm,
                                latency_stat=latency_stat,
                            )
                        )
    return grouped_records, case_counts


def plot_metric(
    points_by_batch_type: Mapping[
        str, Mapping[tuple[int, Decimal, str], base.PlotPoint]
    ],
    *,
    case_counts: Mapping[tuple[str, int, Decimal], int],
    arrivals: Sequence[Decimal],
    dcp_sizes: Sequence[int],
    metric: base.MetricSpec,
    mega_num_comm_sm: str | int,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18.0, 10.4),
        sharey="row",
        squeeze=False,
    )
    group_width = 0.88
    bar_width = group_width / len(base.SERIES)
    for row, batch_type in enumerate(BATCH_TYPES):
        points = points_by_batch_type[batch_type]
        available_values = [
            base._metric_value(point, metric) * metric.scale
            for point in points.values()
        ]
        y_max = max(available_values, default=1.0)
        annotation_offset = y_max * 0.018
        missing_y = y_max * 0.018
        for column, dcp_size in enumerate(dcp_sizes):
            axis = axes[row][column]
            mega_annotations: list[
                tuple[float, Decimal, base.PlotPoint]
            ] = []
            missing_positions: list[float] = []
            for series_index, series in enumerate(base.SERIES):
                offset = -group_width / 2 + (series_index + 0.5) * bar_width
                positions = [index + offset for index in range(len(arrivals))]
                values: list[float] = []
                alphas: list[float] = []
                selected_points: list[base.PlotPoint | None] = []
                for arrival in arrivals:
                    point = points.get((dcp_size, arrival, series.key))
                    selected_points.append(point)
                    values.append(
                        base._metric_value(point, metric) * metric.scale
                        if point is not None
                        else 0.0
                    )
                    alphas.append(1.0 if point is not None else 0.0)
                bars = axis.bar(
                    positions,
                    values,
                    width=bar_width * 0.9,
                    color=series.color,
                    edgecolor="#222222",
                    linewidth=0.6,
                    hatch=series.hatch,
                    zorder=3,
                )
                for position, arrival, bar, alpha, point in zip(
                    positions, arrivals, bars, alphas, selected_points
                ):
                    bar.set_alpha(alpha)
                    if point is None:
                        missing_positions.append(position)
                    elif series.method == base.METHOD_MEGA:
                        mega_annotations.append((position, arrival, point))

            if row == 0:
                axis.set_title(f"DCP size = {dcp_size}", fontsize=13, pad=10)
            axis.set_xticks(
                range(len(arrivals)),
                [
                    f"{base._decimal_label(arrival)}\n"
                    f"n={case_counts[(batch_type, dcp_size, arrival)]}"
                    for arrival in arrivals
                ],
            )
            axis.tick_params(axis="x", labelsize=9.5)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, zorder=0)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            for boundary in range(len(arrivals) - 1):
                axis.axvline(
                    boundary + 0.5,
                    color="#ECECEC",
                    linewidth=0.8,
                    zorder=1,
                )
            for position, arrival, point in mega_annotations:
                comparison = base.best_baseline_comparison(
                    points,
                    dcp_size=dcp_size,
                    arrival=arrival,
                    mega=point,
                    metric=metric,
                )
                speedup_label = (
                    f"{comparison.speedup:.2f}x"
                    if comparison is not None
                    else "N/A"
                )
                axis.text(
                    position,
                    base._metric_value(point, metric) * metric.scale
                    + annotation_offset,
                    f"{speedup_label} | SM {point.mega_num_comm_sm}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7.5,
                    color="#7A1F22",
                )
            for position in missing_positions:
                axis.text(
                    position,
                    missing_y,
                    "N/A",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                    color="#777777",
                )
            axis.set_ylim(0, y_max * 1.22)
        axes[row][0].set_ylabel(
            f"{BATCH_TYPE_LABELS[batch_type]}\n{metric.ylabel}",
            fontsize=11,
        )

    legend_handles = [
        Patch(
            facecolor=series.color,
            edgecolor="#222222",
            linewidth=0.6,
            hatch=series.hatch,
            label=base._legend_label(series, mega_num_comm_sm),
        )
        for series in base.SERIES
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.95),
        fontsize=10,
    )
    figure.suptitle(title, fontsize=16, y=0.995)
    figure.supxlabel("Arrival time scale", fontsize=12, y=0.025)
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.09,
        top=0.84,
        hspace=0.31,
        wspace=0.08,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        csv_path, run_dir = resolve_summary(args.input)
        records_by_batch_type, case_counts = load_batch_type_summary(
            csv_path, args.latency_stat
        )
        summary_records = [
            record
            for records in records_by_batch_type.values()
            for record in records
        ]
        if args.arrival_time_scales is None:
            args.arrival_time_scales = tuple(
                sorted({record.arrival_time_scale for record in summary_records})
            )
        if args.dcp_sizes is None:
            args.dcp_sizes = tuple(
                sorted({record.dcp_size for record in summary_records})
            )
        if len(args.dcp_sizes) != 3:
            raise ValueError(
                "the selected matrix must contain exactly three DCP sizes; "
                "use --dcp-sizes to select them"
            )
        points_by_batch_type = {
            batch_type: base.select_plot_points(
                batch_records,
                arrivals=args.arrival_time_scales,
                dcp_sizes=args.dcp_sizes,
                mega_num_comm_sm=args.mega_num_comm_sm,
                allow_incomplete=args.allow_incomplete,
            )
            for batch_type, batch_records in records_by_batch_type.items()
        }
        selector = (
            "best_comm_sm"
            if args.mega_num_comm_sm == "best"
            else f"comm_sm_{args.mega_num_comm_sm}"
        )
        metrics = (
            (
                "latency",
                base.MetricSpec(
                    "latency_ms",
                    base.LATENCY_LABELS[args.latency_stat],
                    1000.0,
                    False,
                ),
                args.title,
                args.output
                or run_dir
                / f"dcp_latency_{args.latency_stat}_by_batch_type_{selector}.png",
            ),
            (
                "tflops",
                base.MetricSpec(
                    "tflops_per_gpu",
                    "Workload-weighted effective TFLOPS/GPU",
                    1.0,
                    True,
                ),
                args.flops_title,
                args.flops_output
                or run_dir / f"dcp_tflops_per_gpu_by_batch_type_{selector}.png",
            ),
            (
                "bandwidth",
                base.MetricSpec(
                    "kv_bandwidth_gbps_per_gpu",
                    "Workload-weighted effective KV bandwidth (GB/s/GPU)",
                    1.0,
                    True,
                ),
                args.bandwidth_title,
                args.bandwidth_output
                or run_dir
                / f"dcp_kv_bandwidth_per_gpu_by_batch_type_{selector}.png",
            ),
        )
        for _metric_name, metric, title, output in metrics:
            plot_metric(
                points_by_batch_type,
                case_counts=case_counts,
                arrivals=args.arrival_time_scales,
                dcp_sizes=args.dcp_sizes,
                metric=metric,
                mega_num_comm_sm=args.mega_num_comm_sm,
                title=title,
                output=output,
                dpi=args.dpi,
            )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"input={csv_path}")
    print(f"latency_column={base.LATENCY_COLUMNS[args.latency_stat]}")
    print("aggregation=loaded_from_batch_type_summary_csv")
    for batch_type, points in points_by_batch_type.items():
        for dcp_size in args.dcp_sizes:
            for arrival in args.arrival_time_scales:
                point = points.get((dcp_size, arrival, "mega_eager"))
                if point is None:
                    continue
                comparisons = []
                for metric_name, metric, _title, _output in metrics:
                    comparison = base.best_baseline_comparison(
                        points,
                        dcp_size=dcp_size,
                        arrival=arrival,
                        mega=point,
                        metric=metric,
                    )
                    if comparison is None:
                        comparisons.append(
                            f"{metric_name}_best_baseline=unavailable "
                            f"{metric_name}_speedup=unavailable"
                        )
                    else:
                        comparisons.append(
                            f"{metric_name}_best_baseline={comparison.series.key} "
                            f"{metric_name}_best_value="
                            f"{comparison.value * metric.scale:.6f} "
                            f"{metric_name}_speedup={comparison.speedup:.6f}x"
                        )
                print(
                    f"mega_selection batch_type={batch_type} "
                    f"case_count={case_counts[(batch_type, dcp_size, arrival)]} "
                    f"dcp={dcp_size} "
                    f"arrival={base._decimal_label(arrival)} "
                    f"comm_sm={point.mega_num_comm_sm} "
                    f"latency_us={point.latency_ms * 1000.0:.6f} "
                    f"tflops_per_gpu={point.tflops_per_gpu:.6f} "
                    f"kv_bandwidth_gbps_per_gpu="
                    f"{point.kv_bandwidth_gbps_per_gpu:.6f} "
                    + " ".join(comparisons)
                )
    for metric_name, _metric, _title, output in metrics:
        print(f"{metric_name}_output={output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
