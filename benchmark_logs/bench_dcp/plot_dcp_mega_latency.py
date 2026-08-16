#!/usr/bin/env python3
"""Plot latency, per-GPU FLOPS, and KV bandwidth from a Mega DCP matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parent

METHOD_VLLM_AG_RS = "vllm_ag_rs_min_fa3_varlen"
METHOD_VLLM_A2A = "vllm_a2a_min_fa3_varlen"
METHOD_SGLANG = "sglang_mha_ag_ar_min_fa3_varlen"
METHOD_MEGA = "dcp_mega_varlen"

LATENCY_COLUMNS = {
    "mean": "p50_latency_ms_mean",
    "p50": "p50_latency_ms_p50",
    "min": "p50_latency_ms_min",
    "max": "p50_latency_ms_max",
}
LATENCY_LABELS = {
    "mean": "Mean per-case p50 latency (us)",
    "p50": "Median per-case p50 latency (us)",
    "min": "Minimum per-case p50 latency (us)",
    "max": "Maximum per-case p50 latency (us)",
}
TFLOPS_COLUMN = "workload_weighted_effective_tflops_per_gpu"
BANDWIDTH_COLUMN = "workload_weighted_effective_kv_bandwidth_gbps_per_gpu"
@dataclass(frozen=True)
class Series:
    key: str
    label: str
    method: str
    execution_mode: str
    color: str
    hatch: str | None = None


SERIES = (
    Series(
        "vllm_ag_rs_eager",
        "vLLM AG+RS Eager",
        METHOD_VLLM_AG_RS,
        "eager",
        "#4C78A8",
    ),
    Series(
        "vllm_ag_rs_graph",
        "vLLM AG+RS CUDA Graph",
        METHOD_VLLM_AG_RS,
        "cuda_graph",
        "#4C78A8",
        "///",
    ),
    Series(
        "vllm_a2a_eager",
        "vLLM A2A Eager",
        METHOD_VLLM_A2A,
        "eager",
        "#59A14F",
    ),
    Series(
        "vllm_a2a_graph",
        "vLLM A2A CUDA Graph",
        METHOD_VLLM_A2A,
        "cuda_graph",
        "#59A14F",
        "///",
    ),
    Series(
        "sglang_eager",
        "SGLang Eager",
        METHOD_SGLANG,
        "eager",
        "#F28E2B",
    ),
    Series(
        "sglang_graph",
        "SGLang CUDA Graph",
        METHOD_SGLANG,
        "cuda_graph",
        "#F28E2B",
        "///",
    ),
    Series(
        "mega_eager",
        "Mega DCP Eager",
        METHOD_MEGA,
        "eager",
        "#E15759",
        "xx",
    ),
)


@dataclass(frozen=True)
class LatencyRecord:
    arrival_time_scale: Decimal
    dcp_size: int
    suite: str
    execution_mode: str
    method: str
    mega_num_comm_sm: int | None
    latency_ms: float
    tflops_per_gpu: float
    kv_bandwidth_gbps_per_gpu: float


@dataclass(frozen=True)
class PlotPoint:
    latency_ms: float
    tflops_per_gpu: float
    kv_bandwidth_gbps_per_gpu: float
    mega_num_comm_sm: int | None = None


@dataclass(frozen=True)
class BaselineComparison:
    series: Series
    value: float
    speedup: float


@dataclass(frozen=True)
class MetricSpec:
    key: str
    ylabel: str
    scale: float
    higher_is_better: bool


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parse_arrivals(spec: str) -> tuple[Decimal, ...]:
    values: list[Decimal] = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        try:
            value = Decimal(token)
        except InvalidOperation as error:
            raise ValueError(
                "arrival scales must be finite positive numbers"
            ) from error
        if not value.is_finite() or value <= 0:
            raise ValueError("arrival scales must be finite positive numbers")
        if value in values:
            raise ValueError("arrival scales must not contain duplicates")
        values.append(value)
    if not values:
        raise ValueError("arrival scales must not be empty")
    return tuple(values)


def _parse_dcp_sizes(spec: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as error:
            raise ValueError("DCP sizes must be comma-separated integers") from error
        if value <= 0:
            raise ValueError("DCP sizes must be positive")
        if value in values:
            raise ValueError("DCP sizes must not contain duplicates")
        values.append(value)
    if not values:
        raise ValueError("DCP sizes must not be empty")
    return tuple(values)


def _mega_selector(value: str) -> str | int:
    if value == "best":
        return value
    return _positive_integer(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot eager/CUDA-Graph baseline and eager Mega DCP latency, "
            "per-GPU FLOPS, and KV bandwidth from matrix_summary.csv"
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help=(
            "benchmark run directory or matrix_summary.csv; defaults to the "
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
        choices=tuple(LATENCY_COLUMNS),
        default="mean",
        help="statistic over the per-case p50 latency distribution (default: mean)",
    )
    parser.add_argument(
        "--mega-num-comm-sm",
        type=_mega_selector,
        default="best",
        metavar="{best,N}",
        help=(
            "plot the lowest-latency Mega comm-SM result per workload or a "
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
        help="plot available bars and mark missing matrix points as N/A",
    )
    parser.add_argument("--title", default="Mega DCP latency by arrival scale")
    parser.add_argument(
        "--flops-title", default="Mega DCP per-GPU FLOPS by arrival scale"
    )
    parser.add_argument(
        "--bandwidth-title",
        default="Mega DCP effective KV bandwidth by arrival scale",
    )
    parser.add_argument("--dpi", type=_positive_integer, default=220)
    args = parser.parse_args(argv)
    try:
        if args.arrival_time_scales is not None:
            args.arrival_time_scales = _parse_arrivals(args.arrival_time_scales)
        if args.dcp_sizes is not None:
            args.dcp_sizes = _parse_dcp_sizes(args.dcp_sizes)
    except ValueError as error:
        parser.error(str(error))
    if args.dcp_sizes is not None and len(args.dcp_sizes) != 3:
        parser.error("--dcp-sizes must contain exactly three values for a 1x3 figure")
    return args


def _latest_run(root: Path) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"benchmark root does not exist: {root}")
    candidates = [
        path for path in root.glob("*/matrix_summary.csv") if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"no matrix_summary.csv was found below {root}")
    latest_csv = max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.parent.name),
    )
    return latest_csv.parent


def resolve_input(path: Path | None) -> tuple[Path, Path | None, Path]:
    selected = _latest_run(DEFAULT_BENCHMARK_ROOT) if path is None else path
    selected = selected.expanduser().resolve()
    if selected.is_dir():
        run_dir = selected
        csv_path = run_dir / "matrix_summary.csv"
    else:
        csv_path = selected
        run_dir = csv_path.parent
    if not csv_path.is_file():
        raise FileNotFoundError(f"matrix summary CSV does not exist: {csv_path}")
    manifest_path = run_dir / "matrix_manifest.json"
    return csv_path, manifest_path if manifest_path.is_file() else None, run_dir


def validate_manifest(path: Path | None, *, allow_incomplete: bool) -> None:
    if path is None:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read matrix manifest {path}: {error}") from error
    status = payload.get("status")
    if status == "complete" or allow_incomplete:
        return
    counts = ", ".join(
        f"{name}={payload.get(f'{name}_launch_count', '?')}"
        for name in ("completed", "running", "missing", "failed", "invalid")
    )
    raise ValueError(
        f"matrix manifest is {status!r} ({counts}); wait for completion or pass "
        "--allow-incomplete"
    )


def load_records(path: Path, latency_stat: str) -> list[LatencyRecord]:
    latency_column = LATENCY_COLUMNS[latency_stat]
    required = {
        "arrival_time_scale",
        "dcp_size",
        "suite",
        "execution_mode",
        "method",
        "mega_num_comm_sm",
        latency_column,
        TFLOPS_COLUMN,
        BANDWIDTH_COLUMN,
    }
    records: list[LatencyRecord] = []
    relevant_methods = {series.method for series in SERIES}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(required - set(reader.fieldnames or ()))
        if missing_columns:
            raise ValueError(
                f"{path} is missing columns: {', '.join(missing_columns)}"
            )
        for line_number, row in enumerate(reader, start=2):
            if row["method"] not in relevant_methods:
                continue
            try:
                arrival = Decimal(row["arrival_time_scale"])
                dcp_size = int(row["dcp_size"])
                latency_ms = float(row[latency_column])
                tflops_per_gpu = float(row[TFLOPS_COLUMN])
                kv_bandwidth_gbps_per_gpu = float(row[BANDWIDTH_COLUMN])
                comm_sm = (
                    int(row["mega_num_comm_sm"])
                    if row["mega_num_comm_sm"].strip()
                    else None
                )
            except (InvalidOperation, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid relevant row at {path}:{line_number}: {error}"
                ) from error
            if not arrival.is_finite() or arrival <= 0:
                raise ValueError(f"invalid arrival scale at {path}:{line_number}")
            metric_values = (
                latency_ms,
                tflops_per_gpu,
                kv_bandwidth_gbps_per_gpu,
            )
            if dcp_size <= 0 or any(
                not math.isfinite(value) or value <= 0 for value in metric_values
            ):
                raise ValueError(
                    f"invalid DCP size or metric at {path}:{line_number}"
                )
            method = row["method"]
            suite = row["suite"]
            execution_mode = row["execution_mode"]
            if method == METHOD_MEGA:
                if suite != "mega" or execution_mode != "eager" or comm_sm is None:
                    raise ValueError(
                        f"invalid Mega suite/mode/comm-SM at {path}:{line_number}"
                    )
            elif (
                suite != "baseline"
                or execution_mode not in {"eager", "cuda_graph"}
                or comm_sm is not None
            ):
                raise ValueError(
                    f"invalid baseline suite/mode/comm-SM at {path}:{line_number}"
                )
            records.append(
                LatencyRecord(
                    arrival_time_scale=arrival,
                    dcp_size=dcp_size,
                    suite=suite,
                    execution_mode=execution_mode,
                    method=method,
                    mega_num_comm_sm=comm_sm,
                    latency_ms=latency_ms,
                    tflops_per_gpu=tflops_per_gpu,
                    kv_bandwidth_gbps_per_gpu=kv_bandwidth_gbps_per_gpu,
                )
            )
    if not records:
        raise ValueError(f"no relevant latency rows were found in {path}")
    return records


def select_plot_points(
    records: Sequence[LatencyRecord],
    *,
    arrivals: Sequence[Decimal],
    dcp_sizes: Sequence[int],
    mega_num_comm_sm: str | int,
    allow_incomplete: bool,
) -> dict[tuple[int, Decimal, str], PlotPoint]:
    baseline: dict[tuple[int, Decimal, str, str], LatencyRecord] = {}
    mega: dict[tuple[int, Decimal, int], LatencyRecord] = {}
    requested_arrivals = set(arrivals)
    requested_dcp_sizes = set(dcp_sizes)
    for record in records:
        if (
            record.arrival_time_scale not in requested_arrivals
            or record.dcp_size not in requested_dcp_sizes
        ):
            continue
        if record.method == METHOD_MEGA:
            assert record.mega_num_comm_sm is not None
            key = (
                record.dcp_size,
                record.arrival_time_scale,
                record.mega_num_comm_sm,
            )
            if key in mega:
                raise ValueError(f"duplicate Mega row for DCP/arrival/comm-SM={key}")
            mega[key] = record
        else:
            key = (
                record.dcp_size,
                record.arrival_time_scale,
                record.method,
                record.execution_mode,
            )
            if key in baseline:
                raise ValueError(f"duplicate baseline row for {key}")
            baseline[key] = record

    points: dict[tuple[int, Decimal, str], PlotPoint] = {}
    missing: list[str] = []
    for dcp_size in dcp_sizes:
        for arrival in arrivals:
            for series in SERIES:
                point_key = (dcp_size, arrival, series.key)
                if series.method != METHOD_MEGA:
                    record = baseline.get(
                        (dcp_size, arrival, series.method, series.execution_mode)
                    )
                    if record is None:
                        missing.append(
                            f"DCP={dcp_size}/arrival={_decimal_label(arrival)}/"
                            f"{series.label}"
                        )
                    else:
                        points[point_key] = PlotPoint(
                            record.latency_ms,
                            record.tflops_per_gpu,
                            record.kv_bandwidth_gbps_per_gpu,
                        )
                    continue

                candidates = [
                    record
                    for (
                        candidate_dcp,
                        candidate_arrival,
                        candidate_sm,
                    ), record in mega.items()
                    if candidate_dcp == dcp_size
                    and candidate_arrival == arrival
                    and (
                        mega_num_comm_sm == "best"
                        or candidate_sm == mega_num_comm_sm
                    )
                ]
                if not candidates:
                    selection = (
                        "best available comm-SM"
                        if mega_num_comm_sm == "best"
                        else f"comm-SM={mega_num_comm_sm}"
                    )
                    missing.append(
                        f"DCP={dcp_size}/arrival={_decimal_label(arrival)}/"
                        f"Mega {selection}"
                    )
                else:
                    record = min(
                        candidates,
                        key=lambda item: (item.latency_ms, item.mega_num_comm_sm or 0),
                    )
                    points[point_key] = PlotPoint(
                        record.latency_ms,
                        record.tflops_per_gpu,
                        record.kv_bandwidth_gbps_per_gpu,
                        record.mega_num_comm_sm,
                    )
    if missing and not allow_incomplete:
        preview = "; ".join(missing[:6])
        suffix = f"; ... and {len(missing) - 6} more" if len(missing) > 6 else ""
        raise ValueError(
            f"matrix is missing {len(missing)} required plot points: {preview}{suffix}"
        )
    return points


def _decimal_label(value: Decimal) -> str:
    label = format(value, "f")
    if "." in label:
        label = label.rstrip("0").rstrip(".")
    return label


def _legend_label(series: Series, mega_num_comm_sm: str | int) -> str:
    if series.method != METHOD_MEGA:
        return series.label
    if mega_num_comm_sm == "best":
        return f"{series.label} (best comm SM)"
    return f"{series.label} (comm SM={mega_num_comm_sm})"


def _metric_value(point: PlotPoint, metric: MetricSpec) -> float:
    value = getattr(point, metric.key)
    assert isinstance(value, float)
    return value


def best_baseline_comparison(
    points: Mapping[tuple[int, Decimal, str], PlotPoint],
    *,
    dcp_size: int,
    arrival: Decimal,
    mega: PlotPoint,
    metric: MetricSpec,
) -> BaselineComparison | None:
    candidates = [
        (series, point)
        for series in SERIES
        if series.method != METHOD_MEGA
        and (point := points.get((dcp_size, arrival, series.key))) is not None
    ]
    if not candidates:
        return None
    if metric.higher_is_better:
        series, baseline = max(
            candidates,
            key=lambda item: (_metric_value(item[1], metric), item[0].key),
        )
    else:
        series, baseline = min(
            candidates,
            key=lambda item: (_metric_value(item[1], metric), item[0].key),
        )
    baseline_value = _metric_value(baseline, metric)
    mega_value = _metric_value(mega, metric)
    speedup = (
        mega_value / baseline_value
        if metric.higher_is_better
        else baseline_value / mega_value
    )
    return BaselineComparison(
        series=series,
        value=baseline_value,
        speedup=speedup,
    )


def plot_metric(
    points: Mapping[tuple[int, Decimal, str], PlotPoint],
    *,
    arrivals: Sequence[Decimal],
    dcp_sizes: Sequence[int],
    metric: MetricSpec,
    mega_num_comm_sm: str | int,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 5.8),
        sharey=True,
        squeeze=False,
    )
    axes_row = axes[0]
    group_width = 0.88
    bar_width = group_width / len(SERIES)
    available_values = [
        _metric_value(point, metric) * metric.scale for point in points.values()
    ]
    y_max = max(available_values, default=1.0)
    annotation_offset = y_max * 0.018
    missing_y = y_max * 0.018

    for axis, dcp_size in zip(axes_row, dcp_sizes):
        mega_annotations: list[tuple[float, Decimal, PlotPoint]] = []
        missing_positions: list[float] = []
        for series_index, series in enumerate(SERIES):
            offset = -group_width / 2 + (series_index + 0.5) * bar_width
            positions = [index + offset for index in range(len(arrivals))]
            values: list[float] = []
            alphas: list[float] = []
            selected_points: list[PlotPoint | None] = []
            for arrival in arrivals:
                point = points.get((dcp_size, arrival, series.key))
                selected_points.append(point)
                values.append(
                    _metric_value(point, metric) * metric.scale
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
                elif series.method == METHOD_MEGA:
                    mega_annotations.append((position, arrival, point))

        axis.set_title(f"DCP size = {dcp_size}", fontsize=13, pad=10)
        axis.set_xticks(
            range(len(arrivals)),
            [_decimal_label(arrival) for arrival in arrivals],
        )
        axis.tick_params(axis="x", labelsize=11)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for boundary in range(len(arrivals) - 1):
            axis.axvline(boundary + 0.5, color="#ECECEC", linewidth=0.8, zorder=1)
        for position, arrival, point in mega_annotations:
            comparison = best_baseline_comparison(
                points,
                dcp_size=dcp_size,
                arrival=arrival,
                mega=point,
                metric=metric,
            )
            speedup_label = (
                f"{comparison.speedup:.2f}x" if comparison is not None else "N/A"
            )
            axis.text(
                position,
                _metric_value(point, metric) * metric.scale + annotation_offset,
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

    axes_row[0].set_ylabel(metric.ylabel, fontsize=12)
    for axis in axes_row:
        axis.set_ylim(0, y_max * 1.22)
    legend_handles = [
        Patch(
            facecolor=series.color,
            edgecolor="#222222",
            linewidth=0.6,
            hatch=series.hatch,
            label=_legend_label(series, mega_num_comm_sm),
        )
        for series in SERIES
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=10,
    )
    figure.suptitle(title, fontsize=16, y=0.995)
    figure.supxlabel("Arrival time scale", fontsize=12, y=0.035)
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.13, top=0.77, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        csv_path, _manifest_path, run_dir = resolve_input(args.input)
        records = load_records(csv_path, args.latency_stat)
        if args.arrival_time_scales is None:
            args.arrival_time_scales = tuple(
                sorted({record.arrival_time_scale for record in records})
            )
        if args.dcp_sizes is None:
            args.dcp_sizes = tuple(sorted({record.dcp_size for record in records}))
        if len(args.dcp_sizes) != 3:
            raise ValueError(
                "the selected matrix must contain exactly three DCP sizes for a "
                "1x3 figure; use --dcp-sizes to select them"
            )
        points = select_plot_points(
            records,
            arrivals=args.arrival_time_scales,
            dcp_sizes=args.dcp_sizes,
            mega_num_comm_sm=args.mega_num_comm_sm,
            allow_incomplete=args.allow_incomplete,
        )
        selector = (
            "best_comm_sm"
            if args.mega_num_comm_sm == "best"
            else f"comm_sm_{args.mega_num_comm_sm}"
        )
        metrics = (
            (
                "latency",
                MetricSpec(
                    "latency_ms",
                    LATENCY_LABELS[args.latency_stat],
                    1000.0,
                    False,
                ),
                args.title,
                args.output
                or run_dir / f"dcp_latency_{args.latency_stat}_{selector}.png",
            ),
            (
                "tflops",
                MetricSpec(
                    "tflops_per_gpu",
                    "Workload-weighted effective TFLOPS/GPU",
                    1.0,
                    True,
                ),
                args.flops_title,
                args.flops_output
                or run_dir / f"dcp_tflops_per_gpu_{selector}.png",
            ),
            (
                "bandwidth",
                MetricSpec(
                    "kv_bandwidth_gbps_per_gpu",
                    "Workload-weighted effective KV bandwidth (GB/s/GPU)",
                    1.0,
                    True,
                ),
                args.bandwidth_title,
                args.bandwidth_output
                or run_dir / f"dcp_kv_bandwidth_per_gpu_{selector}.png",
            ),
        )
        for _metric_name, metric, title, output in metrics:
            plot_metric(
                points,
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
    print(f"latency_column={LATENCY_COLUMNS[args.latency_stat]}")
    for dcp_size in args.dcp_sizes:
        for arrival in args.arrival_time_scales:
            point = points.get((dcp_size, arrival, "mega_eager"))
            if point is None:
                continue
            comparisons = []
            for metric_name, metric, _title, _output in metrics:
                comparison = best_baseline_comparison(
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
                f"mega_selection dcp={dcp_size} "
                f"arrival={_decimal_label(arrival)} "
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
