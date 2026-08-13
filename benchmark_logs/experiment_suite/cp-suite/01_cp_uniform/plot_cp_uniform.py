#!/usr/bin/env python3
"""Plot forward and backward uniform-CP benchmark results from one log.

By default this script reads ``benchmark_uniform_both.log`` next to itself and
creates a 2x3 throughput overview.  The rows are forward/backward, the columns
are total-token contexts, and each panel compares batch sizes for all methods.
Mega-ring methods are tuned independently at every workload point by selecting
the Comp:Comm configuration with the highest per-GPU throughput.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "benchmark_uniform_both.log"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"

METHODS = (
    "allgather_attention",
    "llama3_allgather_attention",
    "fa3_ring",
    "megatron_hybrid_cp",
    "magi_attention",
    "zeppelin",
    "mega_ring_all_cp",
    "mega_ring_hybrid",
)
TUNED_METHODS = frozenset(("mega_ring_all_cp", "mega_ring_hybrid"))
METHOD_LABELS = {
    "allgather_attention": "AllGather",
    "llama3_allgather_attention": "Llama3 AllGather",
    "fa3_ring": "FA3 Ring",
    "megatron_hybrid_cp": "Megatron Hybrid CP",
    "magi_attention": "MagiAttention",
    "zeppelin": "Zeppelin",
    "mega_ring_all_cp": "Mega-Ring All-CP",
    "mega_ring_hybrid": "Mega-Ring Hybrid",
}
METHOD_COLORS = {
    "allgather_attention": "#4C78A8",
    "llama3_allgather_attention": "#59A14F",
    "fa3_ring": "#9C755F",
    "megatron_hybrid_cp": "#F28E2B",
    "magi_attention": "#17A2B8",
    "zeppelin": "#ECA82C",
    "mega_ring_all_cp": "#E15759",
    "mega_ring_hybrid": "#B07AA1",
}
METHOD_MARKERS = {
    "allgather_attention": "o",
    "llama3_allgather_attention": "s",
    "fa3_ring": "^",
    "megatron_hybrid_cp": "D",
    "magi_attention": "v",
    "zeppelin": "P",
    "mega_ring_all_cp": "X",
    "mega_ring_hybrid": "*",
}
METRICS = {
    "avg-gpu-tflops": ("avg_gpu_tflops", "TFLOPS / GPU", False),
    "agg-tflops": ("agg_tflops", "Aggregate TFLOPS", False),
    "latency-ms": ("latency_ms", "Latency (ms, log scale)", True),
}

DIRECTION_RE = re.compile(r"^\[uniform_(forward|backward)\b")
CASE_RE = re.compile(
    r"^(?:Benchmark case:|Workload:)\s+uniform "
    r"context=(?P<context>\d+), batch=(?P<batch>\d+), "
    r"seqlen=(?P<seqlen>\d+), hybrid=(?P<hybrid>G\d+)"
)
FORWARD_SM_RE = re.compile(
    r"^SM config: num_comp_sm=(?P<compute>\d+), "
    r"num_comm_sm=(?P<communication>\d+)$"
)
WORLD_SIZE_RE = re.compile(r"--nproc_per_node(?:=|\s+)(?P<size>\d+)")
NUMBER = r"[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
RESULT_RE = re.compile(
    rf"^(?P<method>{'|'.join(map(re.escape, METHODS))})\s+"
    rf"(?:(?P<sm>-|\d+:\d+)\s+)?"
    rf"t0=.*?\|\s*max_across_ranks=(?P<latency>{NUMBER})\s+"
    rf"(?P<aggregate>{NUMBER})\s+(?P<average>{NUMBER})\s+"
    r"(?P<check>skip|pass|fail)\b"
)


@dataclass(frozen=True)
class Result:
    direction: str
    context: int
    batch: int
    seqlen: int
    hybrid: str
    method: str
    sm_config: str
    latency_ms: float
    agg_tflops: float
    avg_gpu_tflops: float
    line_number: int


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def comma_list(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("must not be empty")
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("must not contain duplicates")
    return items


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the uniform CP forward/backward workload matrix"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="combined benchmark log (default: next to this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output PNG (default: figures/cp_uniform_both_<metric>.png)",
    )
    parser.add_argument(
        "--direction",
        choices=("both", "forward", "backward"),
        default="both",
        help="direction rows to include (default: both)",
    )
    parser.add_argument(
        "--metric",
        choices=tuple(METRICS),
        default="avg-gpu-tflops",
        help="value to plot (default: avg-gpu-tflops)",
    )
    parser.add_argument(
        "--mega-sm-config",
        default="best",
        metavar="best|COMP:COMM",
        help="per-point best or a fixed mega-ring SM configuration (default: best)",
    )
    parser.add_argument(
        "--methods",
        type=comma_list,
        default=METHODS,
        metavar="NAME,...",
        help="comma-separated methods to plot (default: all)",
    )
    parser.add_argument("--dpi", type=positive_int, default=220)
    parser.add_argument(
        "--title",
        default="Uniform Context-Parallel Attention",
        help="figure title",
    )
    return parser.parse_args(argv)


def finite_positive(value: str, *, field: str, path: Path, line: int) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{path}:{line}: {field} must be finite and positive")
    return parsed


def parse_log(path: Path) -> tuple[list[Result], int]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error

    results: list[Result] = []
    direction: str | None = None
    case: tuple[int, int, int, str] | None = None
    forward_sm: str | None = None
    world_sizes: set[int] = set()
    exact_keys: dict[tuple[str, int, int, str, str], int] = {}

    for line_number, line in enumerate(lines, start=1):
        world_match = WORLD_SIZE_RE.search(line)
        if world_match is not None:
            world_sizes.add(int(world_match.group("size")))

        direction_match = DIRECTION_RE.match(line)
        if direction_match is not None:
            direction = direction_match.group(1)
            case = None
            forward_sm = None
            continue

        case_match = CASE_RE.match(line)
        if case_match is not None:
            if direction is None:
                raise ValueError(f"{path}:{line_number}: workload has no direction")
            case = (
                int(case_match.group("context")),
                int(case_match.group("batch")),
                int(case_match.group("seqlen")),
                case_match.group("hybrid"),
            )
            forward_sm = None
            continue

        sm_match = FORWARD_SM_RE.match(line)
        if sm_match is not None:
            forward_sm = (
                f"{sm_match.group('compute')}:{sm_match.group('communication')}"
            )
            continue

        result_match = RESULT_RE.match(line)
        if result_match is None:
            continue
        if direction is None or case is None:
            raise ValueError(f"{path}:{line_number}: result has no workload metadata")

        method = result_match.group("method")
        embedded_sm = result_match.group("sm")
        if direction == "forward":
            if embedded_sm is not None or forward_sm is None:
                raise ValueError(
                    f"{path}:{line_number}: malformed forward SM configuration"
                )
            sm_config = forward_sm
        else:
            if embedded_sm is None:
                raise ValueError(
                    f"{path}:{line_number}: missing backward Comp:Comm value"
                )
            sm_config = embedded_sm

        context, batch, seqlen, hybrid = case
        key = (direction, context, batch, method, sm_config)
        if key in exact_keys:
            raise ValueError(
                f"{path}:{line_number}: duplicate result; first seen on line "
                f"{exact_keys[key]} for {key}"
            )
        exact_keys[key] = line_number
        results.append(
            Result(
                direction=direction,
                context=context,
                batch=batch,
                seqlen=seqlen,
                hybrid=hybrid,
                method=method,
                sm_config=sm_config,
                latency_ms=finite_positive(
                    result_match.group("latency"),
                    field="latency",
                    path=path,
                    line=line_number,
                ),
                agg_tflops=finite_positive(
                    result_match.group("aggregate"),
                    field="aggregate TFLOPS",
                    path=path,
                    line=line_number,
                ),
                avg_gpu_tflops=finite_positive(
                    result_match.group("average"),
                    field="average GPU TFLOPS",
                    path=path,
                    line=line_number,
                ),
                line_number=line_number,
            )
        )

    if not results:
        raise ValueError(f"{path}: no uniform benchmark result rows found")
    if len(world_sizes) != 1:
        found = ", ".join(map(str, sorted(world_sizes))) or "none"
        raise ValueError(f"{path}: expected one torchrun world size, found {found}")
    return results, world_sizes.pop()


def select_results(
    results: Sequence[Result],
    *,
    directions: Sequence[str],
    methods: Sequence[str],
    mega_sm_config: str,
) -> dict[tuple[str, int, int, str], Result]:
    grouped: dict[tuple[str, int, int, str], list[Result]] = defaultdict(list)
    for result in results:
        if result.direction in directions and result.method in methods:
            grouped[(result.direction, result.context, result.batch, result.method)].append(
                result
            )

    selected: dict[tuple[str, int, int, str], Result] = {}
    for key, candidates in grouped.items():
        method = key[-1]
        if method not in TUNED_METHODS:
            if len(candidates) != 1:
                raise ValueError(f"expected one untuned result for {key}, found {len(candidates)}")
            selected[key] = candidates[0]
            continue

        if mega_sm_config == "best":
            selected[key] = max(candidates, key=lambda item: item.avg_gpu_tflops)
            continue
        matching = [item for item in candidates if item.sm_config == mega_sm_config]
        if len(matching) != 1:
            raise ValueError(
                f"expected one SM={mega_sm_config} result for {key}, found {len(matching)}"
            )
        selected[key] = matching[0]
    return selected


def validate_matrix(
    selected: dict[tuple[str, int, int, str], Result],
    directions: Sequence[str],
    methods: Sequence[str],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    contexts = tuple(sorted({key[1] for key in selected}))
    batches = tuple(sorted({key[2] for key in selected}))
    if not contexts or not batches:
        raise ValueError("no results remain after applying the filters")

    missing = [
        (direction, context, batch, method)
        for direction in directions
        for context in contexts
        for batch in batches
        for method in methods
        if (direction, context, batch, method) not in selected
    ]
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(f"incomplete uniform matrix; missing {preview}{suffix}")

    case_shapes: dict[tuple[str, int, int], tuple[int, str]] = {}
    for (direction, context, batch, _method), result in selected.items():
        case_key = (direction, context, batch)
        shape = (result.seqlen, result.hybrid)
        previous = case_shapes.setdefault(case_key, shape)
        if previous != shape:
            raise ValueError(f"inconsistent workload metadata for {case_key}")
        if result.seqlen * batch != context:
            raise ValueError(
                f"invalid uniform workload {case_key}: seqlen*batch != context"
            )
    return contexts, batches


def context_label(context: int) -> str:
    if context % 1024 == 0:
        return f"{context // 1024}K"
    return f"{context:,}"


def make_figure(
    selected: dict[tuple[str, int, int, str], Result],
    *,
    directions: Sequence[str],
    contexts: Sequence[int],
    batches: Sequence[int],
    methods: Sequence[str],
    metric: str,
    mega_sm_config: str,
    world_size: int,
    title: str,
) -> plt.Figure:
    attribute, ylabel, log_scale = METRICS[metric]
    width = max(12.0, 5.2 * len(contexts))
    height = 4.1 * len(directions) + 1.8
    fig, axes = plt.subplots(
        len(directions),
        len(contexts),
        figsize=(width, height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_positions = list(range(len(batches)))

    for row, direction in enumerate(directions):
        for column, context in enumerate(contexts):
            axis = axes[row, column]
            for method in methods:
                values = [
                    getattr(selected[(direction, context, batch, method)], attribute)
                    for batch in batches
                ]
                is_mega = method in TUNED_METHODS
                axis.plot(
                    x_positions,
                    values,
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    markersize=7.5 if method == "mega_ring_hybrid" else 5.5,
                    linewidth=2.6 if is_mega else 1.7,
                    linestyle="--" if method == "mega_ring_all_cp" else "-",
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    label=METHOD_LABELS[method],
                    zorder=3 if is_mega else 2,
                )

            if log_scale:
                axis.set_yscale("log")
            axis.set_title(
                f"{context_label(context)} total tokens",
                fontsize=12.5,
                fontweight="bold",
            )
            axis.set_xticks(x_positions, [str(batch) for batch in batches])
            axis.grid(axis="both", color="#D8D8D8", linewidth=0.8, alpha=0.75)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.margins(x=0.04, y=0.10)
            if column == 0:
                axis.set_ylabel(f"{direction.capitalize()}\n{ylabel}")
            if row == len(directions) - 1:
                axis.set_xlabel("Batch size")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=min(4, len(methods)),
        frameon=False,
        fontsize=9.5,
        columnspacing=1.5,
    )
    tuning = (
        "best Comp:Comm selected independently at each point"
        if mega_sm_config == "best"
        else f"fixed Comp:Comm={mega_sm_config}"
    )
    fig.suptitle(
        f"{title} ({world_size} GPUs)",
        y=0.995,
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        f"Uniform causal workloads; mega-ring uses {tuning}.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.88))
    return fig


def print_summary(
    *,
    input_path: Path,
    raw_results: Sequence[Result],
    selected: dict[tuple[str, int, int, str], Result],
    mega_sm_config: str,
) -> None:
    cases = {(item.direction, item.context, item.batch) for item in selected.values()}
    print(
        f"Parsed {len(raw_results)} result rows from {input_path.resolve()}; "
        f"selected {len(selected)} rows across {len(cases)} workload cases."
    )
    if mega_sm_config != "best":
        return
    for method in METHODS:
        if method not in TUNED_METHODS:
            continue
        counts = Counter(
            item.sm_config for item in selected.values() if item.method == method
        )
        if counts:
            rendered = ", ".join(
                f"{sm}={count}" for sm, count in sorted(counts.items())
            )
            print(f"Best-SM selections for {METHOD_LABELS[method]}: {rendered}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    unknown_methods = set(args.methods).difference(METHODS)
    if unknown_methods:
        raise ValueError(f"unknown methods: {sorted(unknown_methods)}")
    if args.mega_sm_config != "best" and re.fullmatch(
        r"\d+:\d+", args.mega_sm_config
    ) is None:
        raise ValueError("--mega-sm-config must be 'best' or COMP:COMM")

    directions = (
        ("forward", "backward")
        if args.direction == "both"
        else (args.direction,)
    )
    raw_results, world_size = parse_log(args.input)
    selected = select_results(
        raw_results,
        directions=directions,
        methods=args.methods,
        mega_sm_config=args.mega_sm_config,
    )
    contexts, batches = validate_matrix(selected, directions, args.methods)
    figure = make_figure(
        selected,
        directions=directions,
        contexts=contexts,
        batches=batches,
        methods=args.methods,
        metric=args.metric,
        mega_sm_config=args.mega_sm_config,
        world_size=world_size,
        title=args.title,
    )
    output = args.output or (
        DEFAULT_OUTPUT_DIR / f"cp_uniform_{args.direction}_{args.metric}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print_summary(
        input_path=args.input,
        raw_results=raw_results,
        selected=selected,
        mega_sm_config=args.mega_sm_config,
    )
    print(f"Saved {output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
