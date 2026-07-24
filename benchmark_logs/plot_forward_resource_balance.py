#!/usr/bin/env python3
"""Plot static forward resource pressure and mega-ring KV/QO probe results.

The load-balance log provides a static model for every baseline.  Each bar in
the token, FLOP, and communication panels is the mean, over all selected
workload cases for one dataset, of the maximum physical load on one rank.

The Global KV/QO panel shows the static model for all methods.  Causal
mega-ring methods have a lower/upper scheduler range, shown as a shaded
interval.  When a dataset benchmark log with --collect-mega-ring-stats is
provided, its single post-timing device probes are overlaid for the two
mega-ring methods.  The probe range reflects all SM configurations recorded
for the case and is not part of the latency measurement.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOAD_BALANCE_LOG = (SCRIPT_DIR / "20260724-160353" / "benchmark_load_balance_forward.log")
DEFAULT_DATASET_LOG = SCRIPT_DIR / "20260724-153149" / "benchmark_dataset.log"
DEFAULT_OUTPUT = SCRIPT_DIR / "forward_resource_balance.png"

METHOD_ORDER = (
    "allgather_attention",
    "llama3_allgather_attention",
    "fa3_ring",
    "megatron_hybrid_cp",
    "magi_attention",
    "zepplin",
    "mega_ring_all_cp",
    "mega_ring_hybrid",
)
METHOD_LABELS = {
    "allgather_attention": "AllGather",
    "llama3_allgather_attention": "Llama3 AllGather",
    "fa3_ring": "FA3 Ring",
    "megatron_hybrid_cp": "Megatron Hybrid CP",
    "magi_attention": "MagiAttention",
    "zepplin": "Zeppelin",
    "mega_ring_all_cp": "Mega-Ring All-CP",
    "mega_ring_hybrid": "Mega-Ring Hybrid",
}
METHOD_COLORS = {
    "allgather_attention": "#4C78A8",
    "llama3_allgather_attention": "#59A14F",
    "fa3_ring": "#9C755F",
    "megatron_hybrid_cp": "#F28E2B",
    "magi_attention": "#17A2B8",
    "zepplin": "#ECA82C",
    "mega_ring_all_cp": "#E15759",
    "mega_ring_hybrid": "#B07AA1",
}
MEGA_RING_METHODS = frozenset(("mega_ring_all_cp", "mega_ring_hybrid"))
DATASET_LABELS = {
    "arxiv": "ArXiv",
    "freelaw": "FreeLaw",
    "github": "GitHub",
    "pile": "Pile",
    "prolong": "ProLong",
}
DATASET_ORDER = {dataset: index for index, dataset in enumerate(DATASET_LABELS)}

CASE_RE = re.compile(r"^Case (?P<case>\d+)/(?P<count>\d+):")
STATIC_CONFIG_RE = re.compile(r"^Config: source=dataset=(?P<dataset>[^,]+),")
WORKLOAD_RE = re.compile(r"^Workload: B=\d+, tokens=(?P<tokens>\d+),")
METHOD_RE = re.compile(r"^Method: (?P<method>\S+) \((?P<mode>\S+)\)$")
CUMULATIVE_SUMMARY_RE = re.compile(r"^Cumulative dataset summary$")
SUMMARY_RE = re.compile(
    r"^  (?P<label>Physical tokens|Physical FLOPs|Sent bytes)\s+"
    r"min=\s*(?P<minimum>\S+)\s+avg=\s*(?P<average>\S+)\s+"
    r"max=\s*(?P<maximum>\S+)\s+max/avg=(?P<ratio>\S+)$"
)
GLOBAL_RATIO_RE = re.compile(
    r"^  Global KV/QO\s+\[(?P<lower>[0-9.]+),(?P<upper>[0-9.]+)\]$"
)
DATASET_CASE_RE = re.compile(
    r"^Benchmark case: dataset=(?P<dataset>[^,]+), "
    r"case=(?P<case>\d+)/(?P<count>\d+)$"
)
SM_CONFIG_RE = re.compile(
    r"^SM config: num_comp_sm=(?P<compute>\d+), num_comm_sm=(?P<communication>\d+)$"
)
STATS_HEADER_RE = re.compile(
    r"^  Mega-ring stats for (?P<method>\S+) \(single post-timing probe\):$"
)
STATS_GLOBAL_RE = re.compile(
    r"^    global: qo_visits=(?P<qo_visits>\d+), "
    r"kv_tile_reads=(?P<kv_tiles>\d+), sum\(KV\)/sum\(QO\)=(?P<ratio>[0-9.]+)$"
)


@dataclass(frozen=True)
class Range:
    minimum: float
    average: float
    maximum: float


@dataclass(frozen=True)
class StaticRecord:
    dataset: str
    case: int
    case_count: int
    workload_tokens: int
    method: str
    mode: str
    physical_tokens: Range
    physical_flops: Range
    sent_bytes: Range
    kv_qo_lower: float
    kv_qo_upper: float


@dataclass(frozen=True)
class ProbeRecord:
    dataset: str
    case: int
    case_count: int
    method: str
    sm_config: str
    qo_visits: int
    kv_tile_reads: int

    @property
    def kv_qo(self) -> float:
        return self.kv_tile_reads / self.qo_visits


def parse_human_value(value: str) -> float:
    """Parse benchmark_load_balance.py's decimal and binary display values."""

    binary_scales = {
        "KiB": 1 << 10,
        "MiB": 1 << 20,
        "GiB": 1 << 30,
        "TiB": 1 << 40,
    }
    decimal_scales = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}
    match = re.fullmatch(r"(?P<number>[+-]?[0-9]+(?:\.[0-9]+)?)(?P<suffix>[A-Za-z]*)", value)
    if match is None:
        raise ValueError(f"unsupported benchmark value {value!r}")
    number = float(match.group("number"))
    suffix = match.group("suffix")
    if suffix in binary_scales:
        return number * binary_scales[suffix]
    if suffix in decimal_scales:
        return number * decimal_scales[suffix]
    raise ValueError(f"unsupported benchmark suffix {suffix!r} in {value!r}")


def parse_static_log(path: Path) -> list[StaticRecord]:
    """Extract one static resource record per method and workload case."""

    if not path.is_file():
        raise FileNotFoundError(f"load-balance log does not exist: {path}")

    records: list[StaticRecord] = []
    dataset: str | None = None
    case: int | None = None
    case_count: int | None = None
    workload_tokens: int | None = None
    method: str | None = None
    mode: str | None = None
    summaries: dict[str, Range] = {}
    in_cumulative_summary = False

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        config_match = STATIC_CONFIG_RE.match(line)
        if config_match is not None:
            dataset = config_match.group("dataset")
            case = None
            case_count = None
            workload_tokens = None
            method = None
            mode = None
            summaries = {}
            in_cumulative_summary = False
            continue

        case_match = CASE_RE.match(line)
        if case_match is not None:
            case = int(case_match.group("case"))
            case_count = int(case_match.group("count"))
            workload_tokens = None
            method = None
            mode = None
            summaries = {}
            in_cumulative_summary = False
            continue

        if CUMULATIVE_SUMMARY_RE.match(line) is not None:
            in_cumulative_summary = True
            method = None
            mode = None
            summaries = {}
            continue
        if in_cumulative_summary:
            continue

        workload_match = WORKLOAD_RE.match(line)
        if workload_match is not None:
            if case is None:
                raise ValueError(f"workload without case at {path}:{line_number}")
            workload_tokens = int(workload_match.group("tokens"))
            continue

        method_match = METHOD_RE.match(line)
        if method_match is not None:
            if (
                dataset is None
                or case is None
                or case_count is None
                or workload_tokens is None
            ):
                raise ValueError(
                    f"method without complete case metadata at {path}:{line_number}"
                )
            method = method_match.group("method")
            mode = method_match.group("mode")
            summaries = {}
            continue

        summary_match = SUMMARY_RE.match(line)
        if summary_match is not None:
            if method is None:
                raise ValueError(f"summary without method at {path}:{line_number}")
            label = summary_match.group("label")
            summaries[label] = Range(
                minimum=parse_human_value(summary_match.group("minimum")),
                average=parse_human_value(summary_match.group("average")),
                maximum=parse_human_value(summary_match.group("maximum")),
            )
            continue

        ratio_match = GLOBAL_RATIO_RE.match(line)
        if ratio_match is None:
            continue
        if (
            dataset is None
            or case is None
            or case_count is None
            or workload_tokens is None
            or method is None
            or mode is None
        ):
            raise ValueError(f"Global KV/QO without method at {path}:{line_number}")
        required_labels = ("Physical tokens", "Physical FLOPs", "Sent bytes")
        missing = [label for label in required_labels if label not in summaries]
        if missing:
            raise ValueError(
                f"Global KV/QO for {method} case {case} lacks "
                f"{', '.join(missing)} at {path}:{line_number}"
            )
        records.append(
            StaticRecord(
                dataset=dataset,
                case=case,
                case_count=case_count,
                workload_tokens=workload_tokens,
                method=method,
                mode=mode,
                physical_tokens=summaries["Physical tokens"],
                physical_flops=summaries["Physical FLOPs"],
                sent_bytes=summaries["Sent bytes"],
                kv_qo_lower=float(ratio_match.group("lower")),
                kv_qo_upper=float(ratio_match.group("upper")),
            )
        )
        method = None
        mode = None
        summaries = {}

    if not records:
        raise ValueError(f"no static forward summary records found in {path}")
    _assert_unique_static_records(records, path)
    return records


def _assert_unique_static_records(records: Iterable[StaticRecord], path: Path) -> None:
    seen: set[tuple[str, int, str, str]] = set()
    for record in records:
        key = (record.dataset, record.case, record.mode, record.method)
        if key in seen:
            raise ValueError(
                f"duplicate static record for dataset={record.dataset}, case={record.case}, "
                f"mode={record.mode}, method={record.method} in {path}"
            )
        seen.add(key)


def parse_probe_log(path: Path) -> list[ProbeRecord]:
    """Extract the global device-side counters from a dataset benchmark log."""

    if not path.is_file():
        raise FileNotFoundError(f"dataset benchmark log does not exist: {path}")

    records: list[ProbeRecord] = []
    dataset: str | None = None
    case: int | None = None
    case_count: int | None = None
    sm_config: str | None = None
    pending_method: str | None = None

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        case_match = DATASET_CASE_RE.match(line)
        if case_match is not None:
            dataset = case_match.group("dataset")
            case = int(case_match.group("case"))
            case_count = int(case_match.group("count"))
            sm_config = None
            pending_method = None
            continue

        sm_match = SM_CONFIG_RE.match(line)
        if sm_match is not None:
            sm_config = f"{sm_match.group('compute')}:{sm_match.group('communication')}"
            pending_method = None
            continue

        stats_match = STATS_HEADER_RE.match(line)
        if stats_match is not None:
            if dataset is None or case is None or case_count is None or sm_config is None:
                raise ValueError(
                    f"mega-ring stats without case/SM metadata at {path}:{line_number}"
                )
            pending_method = stats_match.group("method")
            continue

        global_match = STATS_GLOBAL_RE.match(line)
        if global_match is None or pending_method is None:
            continue
        if dataset is None or case is None or case_count is None or sm_config is None:
            raise ValueError(f"global probe without context at {path}:{line_number}")
        qo_visits = int(global_match.group("qo_visits"))
        kv_tile_reads = int(global_match.group("kv_tiles"))
        if qo_visits <= 0:
            raise ValueError(f"non-positive Q/O count at {path}:{line_number}")
        printed_ratio = float(global_match.group("ratio"))
        measured_ratio = kv_tile_reads / qo_visits
        if abs(printed_ratio - measured_ratio) > 0.00006:
            raise ValueError(
                f"inconsistent Global KV/QO at {path}:{line_number}: "
                f"printed {printed_ratio}, counters give {measured_ratio}"
            )
        records.append(
            ProbeRecord(
                dataset=dataset,
                case=case,
                case_count=case_count,
                method=pending_method,
                sm_config=sm_config,
                qo_visits=qo_visits,
                kv_tile_reads=kv_tile_reads,
            )
        )
        pending_method = None

    _assert_unique_probe_records(records, path)
    return records


def _assert_unique_probe_records(records: Iterable[ProbeRecord], path: Path) -> None:
    seen: set[tuple[str, int, str, str]] = set()
    for record in records:
        key = (record.dataset, record.case, record.method, record.sm_config)
        if key in seen:
            raise ValueError(
                f"duplicate probe for dataset={record.dataset}, case={record.case}, "
                f"method={record.method}, SM={record.sm_config} in {path}"
            )
        seen.add(key)


def select_static_records(
    records: Sequence[StaticRecord],
    datasets: set[str] | None,
    mode: str,
    selected_case: int | None,
) -> list[StaticRecord]:
    selected = [
        record
        for record in records
        if (
            (datasets is None or record.dataset in datasets)
            and record.mode == mode
            and (selected_case is None or record.case == selected_case)
        )
    ]
    if not selected:
        case_text = "all cases" if selected_case is None else f"case {selected_case}"
        dataset_text = ", ".join(sorted(datasets)) if datasets else "any dataset"
        raise ValueError(f"no {mode} static records found for {dataset_text}, {case_text}")
    case_counts = {record.case_count for record in selected}
    if len(case_counts) != 1:
        raise ValueError("selected static records disagree about the total case count")
    return selected


def select_probe_records(
    records: Sequence[ProbeRecord], datasets: set[str] | None, selected_case: int | None
) -> list[ProbeRecord]:
    return [
        record
        for record in records
        if (datasets is None or record.dataset in datasets)
        and (selected_case is None or record.case == selected_case)
    ]


def ordered_methods(records: Iterable[StaticRecord]) -> list[str]:
    present = {record.method for record in records}
    return [
        *[method for method in METHOD_ORDER if method in present],
        *sorted(present - set(METHOD_ORDER)),
    ]


def ordered_datasets(records: Iterable[StaticRecord]) -> list[str]:
    return sorted(
        {record.dataset for record in records},
        key=lambda dataset: (DATASET_ORDER.get(dataset, len(DATASET_ORDER)), dataset),
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute the mean of no values")
    return sum(values) / len(values)


def _method_offset(method_index: int, method_count: int, bar_width: float) -> float:
    return (method_index - (method_count - 1) / 2) * bar_width


def plot_resource_metric(
    axis: plt.Axes,
    *,
    datasets: Sequence[str],
    methods: Sequence[str],
    records: Sequence[StaticRecord],
    value_name: str,
    value_scale: float,
    title: str,
    ylabel: str,
) -> None:
    values_by_dataset_method: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        metric = getattr(record, value_name)
        values_by_dataset_method[(record.dataset, record.method)].append(metric.maximum)

    bar_width = 0.84 / len(methods)
    x_positions = list(range(len(datasets)))
    for method_index, method in enumerate(methods):
        offset = _method_offset(method_index, len(methods), bar_width)
        values = [
            _mean(values_by_dataset_method[(dataset, method)]) / value_scale
            if values_by_dataset_method[(dataset, method)]
            else float("nan")
            for dataset in datasets
        ]
        axis.bar(
            [position + offset for position in x_positions],
            values,
            width=bar_width * 0.92,
            color=METHOD_COLORS.get(method, "#444444"),
            edgecolor="white",
            linewidth=0.45,
            label=METHOD_LABELS.get(method, method),
        )
    axis.set_title(title, loc="left", fontsize=12.5, fontweight="bold")
    axis.set_ylabel(ylabel)


def plot_global_kv_qo(
    axis: plt.Axes,
    *,
    datasets: Sequence[str],
    methods: Sequence[str],
    static_records: Sequence[StaticRecord],
    probes: Sequence[ProbeRecord],
) -> None:
    static_lower: dict[tuple[str, str], list[float]] = defaultdict(list)
    static_upper: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in static_records:
        key = (record.dataset, record.method)
        static_lower[key].append(record.kv_qo_lower)
        static_upper[key].append(record.kv_qo_upper)

    bar_width = 0.84 / len(methods)
    x_positions = list(range(len(datasets)))
    method_positions: dict[tuple[str, str], float] = {}
    for method_index, method in enumerate(methods):
        offset = _method_offset(method_index, len(methods), bar_width)
        positions = [position + offset for position in x_positions]
        lower_values = [
            _mean(static_lower[(dataset, method)])
            if static_lower[(dataset, method)]
            else float("nan")
            for dataset in datasets
        ]
        upper_values = [
            _mean(static_upper[(dataset, method)])
            if static_upper[(dataset, method)]
            else float("nan")
            for dataset in datasets
        ]
        for dataset, position in zip(datasets, positions):
            method_positions[(dataset, method)] = position
        color = METHOD_COLORS.get(method, "#444444")
        axis.bar(
            positions,
            lower_values,
            width=bar_width * 0.92,
            color=color,
            edgecolor="white",
            linewidth=0.45,
        )
        range_values = [upper - lower for lower, upper in zip(lower_values, upper_values)]
        axis.bar(
            positions,
            range_values,
            bottom=lower_values,
            width=bar_width * 0.92,
            color=color,
            alpha=0.3,
            hatch="//",
            edgecolor=color,
            linewidth=0.45,
        )

    probes_by_dataset_method_sm: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for probe in probes:
        if probe.method in MEGA_RING_METHODS:
            probes_by_dataset_method_sm[
                (probe.dataset, probe.method, probe.sm_config)
            ].append(probe.kv_qo)

    for dataset in datasets:
        for method in methods:
            if method not in MEGA_RING_METHODS:
                continue
            config_means = [
                _mean(values)
                for (probe_dataset, probe_method, _sm_config), values in probes_by_dataset_method_sm.items()
                if probe_dataset == dataset and probe_method == method
            ]
            if not config_means:
                continue
            position = method_positions[(dataset, method)]
            color = METHOD_COLORS.get(method, "#444444")
            axis.vlines(
                position,
                min(config_means),
                max(config_means),
                color=color,
                linewidth=2.4,
                zorder=5,
            )
            axis.scatter(
                position,
                _mean(config_means),
                color=color,
                edgecolor="black",
                linewidth=0.5,
                marker="D",
                s=32,
                zorder=6,
            )

    axis.set_title(
        "Global KV/QO (Mean Across Cases)",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
    )
    axis.set_ylabel("KV tiles per Q/O visit")


def style_axes(axes: Sequence[plt.Axes], datasets: Sequence[str]) -> None:
    for axis in axes:
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.85)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks(range(len(datasets)))
        axis.set_xticklabels([DATASET_LABELS.get(dataset, dataset) for dataset in datasets])
        axis.set_xlabel("Dataset")


def make_figure(
    static_records: Sequence[StaticRecord], probes: Sequence[ProbeRecord], mode: str
) -> plt.Figure:
    datasets = ordered_datasets(static_records)
    methods = ordered_methods(static_records)
    figure, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=False)
    flat_axes = tuple(axes.flat)

    plot_resource_metric(
        flat_axes[0],
        datasets=datasets,
        methods=methods,
        records=static_records,
        value_name="physical_tokens",
        value_scale=1e3,
        title="Mean Peak Physical Token Load",
        ylabel="Mean peak rank tokens (K)",
    )
    plot_resource_metric(
        flat_axes[1],
        datasets=datasets,
        methods=methods,
        records=static_records,
        value_name="physical_flops",
        value_scale=1e12,
        title="Mean Peak Physical Attention Work",
        ylabel="Mean peak rank FLOPs (T)",
    )
    plot_resource_metric(
        flat_axes[2],
        datasets=datasets,
        methods=methods,
        records=static_records,
        value_name="sent_bytes",
        value_scale=float(1 << 30),
        title="Mean Peak Outbound Communication",
        ylabel="Mean peak rank sent (GiB)",
    )
    plot_global_kv_qo(
        flat_axes[3],
        datasets=datasets,
        methods=methods,
        static_records=static_records,
        probes=probes,
    )
    style_axes(flat_axes, datasets)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    total_cases = static_records[0].case_count
    selected_cases = sorted({record.case for record in static_records})
    selected_case_text = (
        f"case {selected_cases[0]}"
        if len(selected_cases) == 1
        else f"mean across up to {total_cases} cases per dataset"
    )
    figure.suptitle(
        f"Forward Baseline Comparison by Dataset ({mode}, {selected_case_text})",
        y=1.055,
        fontsize=16,
    )
    probe_note = (
        "Global KV/QO: solid bars are mean static lower bounds; hatching extends to "
        "mean upper bounds; diamonds and ranges are mean post-timing device probes across SM configs."
        if probes
        else "Global KV/QO: solid bars are mean static lower bounds; hatching extends to mean upper bounds."
    )
    figure.text(0.5, 0.012, probe_note, ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0, 0.05, 1, 0.89))
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean peak rank token/FLOP/communication pressure for every "
            "forward baseline by dataset, plus Global KV/QO."
        )
    )
    parser.add_argument(
        "--load-balance-log",
        type=Path,
        default=DEFAULT_LOAD_BALANCE_LOG,
        help=f"static load-balance log (default: {DEFAULT_LOAD_BALANCE_LOG})",
    )
    parser.add_argument(
        "--dataset-log",
        type=Path,
        default=DEFAULT_DATASET_LOG,
        help=(
            "dataset log containing --collect-mega-ring-stats probes "
            f"(default: {DEFAULT_DATASET_LOG})"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("causal", "noncausal"),
        default="causal",
        help="attention mode to select from the load-balance log (default: causal)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "dataset section to include; repeat to select several. Defaults to all "
            "datasets in the static log."
        ),
    )
    parser.add_argument(
        "--case",
        type=int,
        default=None,
        help="use one 1-based workload case instead of the per-dataset mean",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output PNG path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.case is not None and args.case <= 0:
        raise ValueError("--case must be a positive 1-based index")

    selected_datasets = set(args.dataset) if args.dataset else None
    probe_log_exists = args.dataset_log.is_file()
    all_probe_records = parse_probe_log(args.dataset_log) if probe_log_exists else []

    static_records = select_static_records(
        parse_static_log(args.load_balance_log),
        selected_datasets,
        args.mode,
        args.case,
    )
    probe_records = select_probe_records(all_probe_records, selected_datasets, args.case)
    figure = make_figure(static_records, probe_records, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)

    static_cases = sorted({record.case for record in static_records})
    static_datasets = ordered_datasets(static_records)
    probe_cases = sorted({record.case for record in probe_records})
    print(
        f"Loaded {len(static_records)} static method records across "
        f"{len(static_cases)} case(s) and {len(static_datasets)} dataset(s) "
        f"({', '.join(static_datasets)}): {args.load_balance_log}"
    )
    if probe_records:
        print(
            f"Loaded {len(probe_records)} mega-ring global probes across "
            f"{len(probe_cases)} case(s): {args.dataset_log}"
        )
    else:
        print(
            "Warning: "
            + (
                f"probe log is unavailable ({args.dataset_log}); "
                if not probe_log_exists
                else "no mega-ring device probes were found; "
            )
            + "Global KV/QO shows only static values.",
            file=sys.stderr,
        )
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
