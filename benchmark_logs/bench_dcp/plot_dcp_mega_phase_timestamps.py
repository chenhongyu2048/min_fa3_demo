#!/usr/bin/env python3
"""Plot paired Mega DCP phase-timestamp summaries for DCP sizes 2, 4, and 8."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parent
METHOD_MEGA = "dcp_mega_varlen"
METHOD_VLLM_A2A = "vllm_a2a_min_fa3_varlen"
DEFAULT_DCP_SIZES = (2, 4, 8)
DEFAULT_COMM_SMS = (4, 8, 12, 16, 20)
CASE_NAME_PATTERN = re.compile(r"^(case_\d+)_dcp\d+_.*\.json$")
CASE_ID_PATTERN = re.compile(r"^case_\d+$")

KIND_DECODE = "decode"
KIND_MIXED = "mixed"
KIND_LABELS = {
    KIND_DECODE: "Decode-only batches",
    KIND_MIXED: "Mixed batches containing chunk prefill",
}


@dataclass(frozen=True)
class PhaseSpec:
    key: str
    label: str
    color: str
    marker: str


PHASES = (
    PhaseSpec("q_allgather_done", "Q all-gather done", "#4C78A8", "o"),
    PhaseSpec("attention_done", "Attention done", "#59A14F", "s"),
    PhaseSpec(
        "history_combine_done",
        "History / publish done",
        "#F28E2B",
        "D",
    ),
    PhaseSpec("receive_done", "Receive done", "#B279A2", "P"),
    PhaseSpec("final_combine_done", "Final combine done", "#E15759", "v"),
    PhaseSpec("kernel_done", "Kernel done", "#2F2F2F", "X"),
)


@dataclass(frozen=True)
class TailSpec:
    key: str
    label: str


TAILS = (
    TailSpec("kernel_start_to_attention_done", "Start -> attention"),
    TailSpec("attention_done_to_publish_done", "Attention -> publish"),
    TailSpec("publish_done_to_receive_done", "Publish -> receive"),
    TailSpec(
        "receive_done_to_final_combine_done",
        "Receive -> final",
    ),
)
KERNEL_TOTAL = TailSpec("kernel_start_to_kernel_done", "Start -> kernel done")
TAIL_ROWS = (*TAILS, KERNEL_TOTAL)


@dataclass(frozen=True)
class BaselineStageSpec:
    key: str
    label: str
    color: str


BASELINE_STAGES = (
    BaselineStageSpec("q_allgather_and_reorder_ms", "Q AG done", "#1F77B4"),
    BaselineStageSpec(
        "local_history_attention_ms", "History attention done", "#9467BD"
    ),
    BaselineStageSpec("a2a_pack_ms", "A2A pack done", "#BCBD22"),
    BaselineStageSpec("a2a_all_to_all_ms", "All-to-all done", "#FF7F0E"),
    BaselineStageSpec(
        "a2a_unpack_combine_ms", "Unpack + combine done", "#D62728"
    ),
    BaselineStageSpec(
        "local_chunk_attention_ms", "Chunk attention done", "#17BECF"
    ),
    BaselineStageSpec("state_merge_ms", "State merge done", "#8C564B"),
    BaselineStageSpec(
        "attention_end_to_end_ms", "Measured Graph E2E", "#111111"
    ),
)

COMM_SM_COLORS = {
    4: "#1F77B4",
    8: "#4C78A8",
    12: "#59A14F",
    16: "#F28E2B",
    20: "#E15759",
}


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    dcp_size: int
    comm_sm: int
    workload_kind: str
    batch_size: int
    total_q_tokens: int
    global_effective_flops: int
    iterations: int
    workload_signature: tuple[tuple[int, ...], tuple[int, ...], int]
    e2e_latency_us: float
    milestones_us: Mapping[str, float]
    tails_us: Mapping[str, float]
    source: Path


Dataset = dict[int, dict[int, dict[str, CaseRecord]]]


@dataclass(frozen=True)
class BaselineCaseRecord:
    case_id: str
    dcp_size: int
    workload_kind: str
    iterations: int
    workload_signature: tuple[tuple[int, ...], tuple[int, ...], int]
    stages_us: Mapping[str, float]
    source: Path


BaselineDataset = dict[int, dict[str, BaselineCaseRecord]]


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _percentile(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or not 90.0 <= parsed <= 100.0:
        raise argparse.ArgumentTypeError("must be in [90, 100]")
    return parsed


def _integer_list(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for token in (item.strip() for item in value.split(",")):
        if not token:
            continue
        try:
            parsed = int(token)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "must contain comma-separated integers"
            ) from error
        if parsed <= 0 or parsed in result:
            raise argparse.ArgumentTypeError(
                "values must be positive and must not contain duplicates"
            )
        result.append(parsed)
    if not result:
        raise argparse.ArgumentTypeError("must not be empty")
    return tuple(result)


def _case_id(value: str) -> str:
    parsed = value.strip()
    if CASE_ID_PATTERN.fullmatch(parsed) is None:
        raise argparse.ArgumentTypeError("must have the form case_<integer>")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a 4x3 overview of Mega DCP %globaltimer phase milestones "
            "overlaid with reconstructed vLLM A2A CUDA Graph phase completion "
            "timestamps, paired "
            "comm-SM slowdowns, and explicit Mega tail durations"
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help=(
            "phase-enabled benchmark run directory; defaults to the newest "
            "compatible run below benchmark_logs/bench_dcp"
        ),
    )
    parser.add_argument(
        "--arrival-time-scale",
        default=None,
        help="arrival scale to plot; inferred when the run contains one scale",
    )
    parser.add_argument(
        "--dcp-sizes",
        type=_integer_list,
        default=DEFAULT_DCP_SIZES,
        help="exactly three comma-separated DCP sizes (default: 2,4,8)",
    )
    parser.add_argument(
        "--comm-sms",
        type=_integer_list,
        default=DEFAULT_COMM_SMS,
        help="comma-separated communication-SM settings (default: 4,8,12,16,20)",
    )
    parser.add_argument(
        "--stat",
        choices=("p50", "p90"),
        default="p50",
        help="within-case iteration statistic to plot (default: p50)",
    )
    parser.add_argument(
        "--exclude-case-id",
        action="append",
        default=[],
        type=_case_id,
        metavar="CASE_ID",
        help=(
            "exclude one paired workload case from every panel and aggregate; "
            "repeat to exclude multiple cases"
        ),
    )
    parser.add_argument(
        "--rolling-window",
        type=_positive_integer,
        default=5,
        metavar="N",
        help="centered rolling-median window for milestone trends (default: 5)",
    )
    parser.add_argument(
        "--clip-percentile",
        type=_percentile,
        default=99.0,
        metavar="P",
        help="row-level milestone y-axis clipping percentile (default: 99)",
    )
    parser.add_argument("--output", type=Path, default=None, help="PNG output path")
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="vector PDF output path",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="do not write the default PDF alongside the PNG",
    )
    parser.add_argument("--dpi", type=_positive_integer, default=220)
    parser.add_argument(
        "--title",
        default="Mega DCP and vLLM A2A CUDA Graph Phase Latencies",
    )
    args = parser.parse_args(argv)
    if len(args.dcp_sizes) != 3:
        parser.error("--dcp-sizes must contain exactly three values")
    if len(args.comm_sms) < 2:
        parser.error("--comm-sms must contain at least two values")
    if args.rolling_window < 3:
        parser.error("--rolling-window must be at least 3")
    return args


def _run_has_phase_records(path: Path) -> bool:
    mega_pattern = "results/arrival_*/dcp_*/mega/comm_sm_*/case_*.json"
    baseline_pattern = "results/arrival_*/dcp_*/baseline_graph/case_*.json"
    mega_sample = next(path.glob(mega_pattern), None)
    baseline_sample = next(path.glob(baseline_pattern), None)
    if mega_sample is None or baseline_sample is None:
        return False
    try:
        mega_payload = json.loads(mega_sample.read_text(encoding="utf-8"))
        baseline_payload = json.loads(
            baseline_sample.read_text(encoding="utf-8")
        )
        baseline_report = baseline_payload["methods"][METHOD_VLLM_A2A]
        return (
            bool(mega_payload["parameters"]["mega_phase_timestamps"])
            and mega_payload["methods"][METHOD_MEGA]["execution"]["phase_profile"]
            is not None
            and bool(baseline_payload["parameters"]["baseline_phase_timing"])
            and baseline_report["execution"]["execution_mode"] == "cuda_graph"
            and bool(
                baseline_report["execution"][
                    "cuda_event_phase_timing_enabled"
                ]
            )
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False


def _latest_phase_run(root: Path) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"benchmark root does not exist: {root}")
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and _run_has_phase_records(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no Mega and baseline phase-enabled run was found below {root}"
        )

    def completion_time(path: Path) -> tuple[int, str]:
        summary = path / "matrix_summary.csv"
        target = summary if summary.is_file() else path
        return target.stat().st_mtime_ns, path.name

    return max(candidates, key=completion_time)


def resolve_run_dir(path: Path | None) -> Path:
    selected = _latest_phase_run(DEFAULT_BENCHMARK_ROOT) if path is None else path
    selected = selected.expanduser().resolve()
    if selected.is_file():
        selected = selected.parent
    if not selected.is_dir():
        raise FileNotFoundError(f"benchmark run directory does not exist: {selected}")
    if not _run_has_phase_records(selected):
        raise ValueError(
            f"no compatible Mega and vLLM A2A phase records were found below "
            f"{selected}"
        )
    return selected


def _arrival_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"invalid arrival time scale: {value!r}") from error
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"arrival time scale must be finite and positive: {value!r}")
    return parsed


def resolve_arrival_dir(run_dir: Path, requested: str | None) -> tuple[Path, str]:
    result_root = run_dir / "results"
    candidates = [
        path
        for path in result_root.glob("arrival_*")
        if path.is_dir() and path.name.removeprefix("arrival_")
    ]
    if not candidates:
        raise FileNotFoundError(f"no arrival result directories were found in {result_root}")
    if requested is None:
        if len(candidates) != 1:
            labels = ", ".join(sorted(path.name for path in candidates))
            raise ValueError(
                f"run contains multiple arrival scales ({labels}); pass "
                "--arrival-time-scale"
            )
        selected = candidates[0]
    else:
        target = _arrival_decimal(requested)
        matches = [
            path
            for path in candidates
            if _arrival_decimal(path.name.removeprefix("arrival_")) == target
        ]
        if len(matches) != 1:
            raise ValueError(
                f"arrival scale {requested!r} does not identify one directory below "
                f"{result_root}"
            )
        selected = matches[0]
    return selected, selected.name.removeprefix("arrival_")


def _finite_positive(value: object, *, field: str, source: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field} in {source}: {value!r}") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field} must be finite and positive in {source}")
    return parsed


def _finite_nonnegative(value: object, *, field: str, source: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field} in {source}: {value!r}") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and nonnegative in {source}")
    return parsed


def load_case(path: Path, *, dcp_size: int, comm_sm: int, stat: str) -> CaseRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    match = CASE_NAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"unexpected Mega case filename: {path.name}")
    case_id = match.group(1)
    try:
        parameters = payload["parameters"]
        lengths = payload["lengths"]
        report = payload["methods"][METHOD_MEGA]
        execution = report["execution"]
        profile = execution["phase_profile"]
        milestone_payload = profile["milestones_us"]
        tail_payload = profile["post_global_completion_tails_us"]
        q_lengths = tuple(int(value) for value in lengths["q_global"])
        history_lengths = tuple(
            int(value) for value in lengths["history_or_cache_global"]
        )
        global_flops = int(payload["global_effective_flops"])
        iterations = int(parameters["iters"])
        e2e_latency_us = float(
            report["stages_ms"]["attention_end_to_end_ms"][stat]
        ) * 1000.0
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid Mega phase schema in {path}: {error}") from error
    if int(parameters["dcp_size"]) != dcp_size:
        raise ValueError(f"DCP size mismatch in {path}")
    if int(parameters["mega_num_comm_sm"]) != comm_sm:
        raise ValueError(f"comm-SM mismatch in {path}")
    if not parameters.get("mega_phase_timestamps") or profile is None:
        raise ValueError(f"Mega phase timestamps are absent in {path}")
    if not q_lengths or len(q_lengths) != len(history_lengths):
        raise ValueError(f"invalid packed batch lengths in {path}")
    if any(value <= 0 for value in q_lengths) or any(
        value < 0 for value in history_lengths
    ):
        raise ValueError(f"non-positive Q or negative history length in {path}")
    if global_flops <= 0:
        raise ValueError(f"global_effective_flops must be positive in {path}")
    if iterations <= 0:
        raise ValueError(f"iters must be positive in {path}")
    _finite_positive(e2e_latency_us, field="E2E latency", source=path)

    milestones = {
        phase.key: _finite_positive(
            milestone_payload[phase.key][stat],
            field=f"{phase.key}/{stat}",
            source=path,
        )
        for phase in PHASES
    }
    publish = _finite_positive(
        milestone_payload["publish_done"][stat],
        field=f"publish_done/{stat}",
        source=path,
    )
    history = milestones["history_combine_done"]
    if not math.isclose(publish, history, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"history and publish timestamps differ in {path}")
    tails = {
        "kernel_start_to_attention_done": milestones["attention_done"],
        "attention_done_to_publish_done": _finite_positive(
            tail_payload["attention_done_to_history_combine_done"][stat],
            field=f"attention_done_to_publish_done/{stat}",
            source=path,
        ),
        "publish_done_to_receive_done": _finite_positive(
            tail_payload["publish_done_to_receive_done"][stat],
            field=f"publish_done_to_receive_done/{stat}",
            source=path,
        ),
        "receive_done_to_final_combine_done": _finite_positive(
            milestones["final_combine_done"] - milestones["receive_done"],
            field=f"receive_done_to_final_combine_done/{stat}",
            source=path,
        ),
        "kernel_start_to_kernel_done": milestones["kernel_done"],
    }
    workload_kind = (
        KIND_DECODE if max(q_lengths) <= 16 else KIND_MIXED
    )
    return CaseRecord(
        case_id=case_id,
        dcp_size=dcp_size,
        comm_sm=comm_sm,
        workload_kind=workload_kind,
        batch_size=len(q_lengths),
        total_q_tokens=sum(q_lengths),
        global_effective_flops=global_flops,
        iterations=iterations,
        workload_signature=(q_lengths, history_lengths, global_flops),
        e2e_latency_us=e2e_latency_us,
        milestones_us=milestones,
        tails_us=tails,
        source=path,
    )


def load_dataset(
    arrival_dir: Path,
    *,
    dcp_sizes: Sequence[int],
    comm_sms: Sequence[int],
    stat: str,
) -> Dataset:
    dataset: Dataset = {}
    for dcp_size in dcp_sizes:
        dataset[dcp_size] = {}
        for comm_sm in comm_sms:
            case_dir = arrival_dir / f"dcp_{dcp_size}" / "mega" / f"comm_sm_{comm_sm}"
            paths = sorted(case_dir.glob("case_*.json"))
            if not paths:
                raise FileNotFoundError(f"no Mega case JSON files were found in {case_dir}")
            records: dict[str, CaseRecord] = {}
            for path in paths:
                record = load_case(
                    path,
                    dcp_size=dcp_size,
                    comm_sm=comm_sm,
                    stat=stat,
                )
                if record.case_id in records:
                    raise ValueError(f"duplicate case {record.case_id} in {case_dir}")
                records[record.case_id] = record
            dataset[dcp_size][comm_sm] = records
    validate_pairing(dataset, dcp_sizes=dcp_sizes, comm_sms=comm_sms)
    return dataset


def load_baseline_case(
    path: Path,
    *,
    dcp_size: int,
    stat: str,
) -> BaselineCaseRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    match = CASE_NAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"unexpected baseline case filename: {path.name}")
    case_id = match.group(1)
    try:
        parameters = payload["parameters"]
        lengths = payload["lengths"]
        report = payload["methods"][METHOD_VLLM_A2A]
        execution = report["execution"]
        stage_payload = report["stages_ms"]
        q_lengths = tuple(int(value) for value in lengths["q_global"])
        history_lengths = tuple(
            int(value) for value in lengths["history_or_cache_global"]
        )
        global_flops = int(payload["global_effective_flops"])
        iterations = int(parameters["iters"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"missing or invalid vLLM A2A phase schema in {path}: {error}"
        ) from error
    if int(parameters["dcp_size"]) != dcp_size:
        raise ValueError(f"DCP size mismatch in {path}")
    if not parameters.get("cuda_graph"):
        raise ValueError(f"vLLM A2A baseline is not CUDA Graph mode in {path}")
    if not parameters.get("baseline_phase_timing"):
        raise ValueError(f"baseline phase timing is disabled in {path}")
    if execution.get("execution_mode") != "cuda_graph":
        raise ValueError(f"vLLM A2A execution mode is not cuda_graph in {path}")
    if not execution.get("cuda_event_phase_timing_enabled"):
        raise ValueError(f"vLLM A2A CUDA Event phase timing is absent in {path}")
    if report.get("output_collective_kind") != "bf16_packed_all_to_all":
        raise ValueError(f"unexpected vLLM A2A output collective in {path}")
    if not q_lengths or len(q_lengths) != len(history_lengths):
        raise ValueError(f"invalid packed batch lengths in {path}")
    if any(value <= 0 for value in q_lengths) or any(
        value < 0 for value in history_lengths
    ):
        raise ValueError(f"non-positive Q or negative history length in {path}")
    if global_flops <= 0 or iterations <= 0:
        raise ValueError(f"invalid FLOPs or iteration count in {path}")
    try:
        stages = {
            stage.key: _finite_nonnegative(
                stage_payload[stage.key][stat],
                field=f"{METHOD_VLLM_A2A}/{stage.key}/{stat}",
                source=path,
            )
            * 1000.0
            for stage in BASELINE_STAGES
        }
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"missing vLLM A2A stage statistic in {path}: {error}"
        ) from error
    _finite_positive(
        stages["attention_end_to_end_ms"],
        field=f"{METHOD_VLLM_A2A}/attention_end_to_end_ms/{stat}",
        source=path,
    )
    workload_kind = KIND_DECODE if max(q_lengths) <= 16 else KIND_MIXED
    return BaselineCaseRecord(
        case_id=case_id,
        dcp_size=dcp_size,
        workload_kind=workload_kind,
        iterations=iterations,
        workload_signature=(q_lengths, history_lengths, global_flops),
        stages_us=stages,
        source=path,
    )


def load_baseline_dataset(
    arrival_dir: Path,
    *,
    dcp_sizes: Sequence[int],
    stat: str,
) -> BaselineDataset:
    dataset: BaselineDataset = {}
    for dcp_size in dcp_sizes:
        case_dir = arrival_dir / f"dcp_{dcp_size}" / "baseline_graph"
        paths = sorted(case_dir.glob("case_*.json"))
        if not paths:
            raise FileNotFoundError(
                f"no baseline Graph case JSON files were found in {case_dir}"
            )
        records: dict[str, BaselineCaseRecord] = {}
        for path in paths:
            record = load_baseline_case(path, dcp_size=dcp_size, stat=stat)
            if record.case_id in records:
                raise ValueError(f"duplicate case {record.case_id} in {case_dir}")
            records[record.case_id] = record
        dataset[dcp_size] = records
    return dataset


def validate_pairing(
    dataset: Dataset,
    *,
    dcp_sizes: Sequence[int],
    comm_sms: Sequence[int],
) -> None:
    reference = dataset[dcp_sizes[0]][comm_sms[0]]
    reference_ids = set(reference)
    for dcp_size in dcp_sizes:
        for comm_sm in comm_sms:
            records = dataset[dcp_size][comm_sm]
            if set(records) != reference_ids:
                missing = sorted(reference_ids - set(records))
                extra = sorted(set(records) - reference_ids)
                raise ValueError(
                    f"unpaired cases for DCP={dcp_size}, comm-SM={comm_sm}: "
                    f"missing={missing[:4]} extra={extra[:4]}"
                )
            for case_id, expected in reference.items():
                actual = records[case_id]
                if actual.workload_signature != expected.workload_signature:
                    raise ValueError(
                        f"workload mismatch for {case_id}, DCP={dcp_size}, "
                        f"comm-SM={comm_sm}"
                    )
                if actual.iterations != expected.iterations:
                    raise ValueError(
                        f"iteration-count mismatch for {case_id}, DCP={dcp_size}, "
                        f"comm-SM={comm_sm}"
                    )
    kinds = {record.workload_kind for record in reference.values()}
    if kinds != {KIND_DECODE, KIND_MIXED}:
        raise ValueError(
            "the overview requires both decode-only and mixed-prefill batches; "
            f"found {sorted(kinds)}"
        )


def validate_baseline_pairing(
    dataset: Dataset,
    baseline_dataset: BaselineDataset,
    *,
    dcp_sizes: Sequence[int],
    reference_sm: int,
) -> None:
    for dcp_size in dcp_sizes:
        mega_records = dataset[dcp_size][reference_sm]
        baseline_records = baseline_dataset[dcp_size]
        if set(baseline_records) != set(mega_records):
            missing = sorted(set(mega_records) - set(baseline_records))
            extra = sorted(set(baseline_records) - set(mega_records))
            raise ValueError(
                f"unpaired vLLM A2A Graph cases for DCP={dcp_size}: "
                f"missing={missing[:4]} extra={extra[:4]}"
            )
        for case_id, mega_record in mega_records.items():
            baseline_record = baseline_records[case_id]
            if baseline_record.workload_signature != mega_record.workload_signature:
                raise ValueError(
                    f"vLLM A2A workload mismatch for {case_id}, DCP={dcp_size}"
                )
            if baseline_record.iterations != mega_record.iterations:
                raise ValueError(
                    f"vLLM A2A iteration-count mismatch for {case_id}, "
                    f"DCP={dcp_size}"
                )
            if baseline_record.workload_kind != mega_record.workload_kind:
                raise ValueError(
                    f"vLLM A2A workload-kind mismatch for {case_id}, "
                    f"DCP={dcp_size}"
                )


def exclude_paired_cases(
    dataset: Dataset,
    baseline_dataset: BaselineDataset,
    *,
    case_ids: Sequence[str],
    dcp_sizes: Sequence[int],
    comm_sms: Sequence[int],
) -> tuple[str, ...]:
    excluded = tuple(dict.fromkeys(case_ids))
    if not excluded:
        return excluded
    reference_ids = set(dataset[dcp_sizes[0]][comm_sms[0]])
    missing = sorted(set(excluded) - reference_ids)
    if missing:
        raise ValueError(f"excluded case IDs were not found: {missing}")
    if len(excluded) == len(reference_ids):
        raise ValueError("cannot exclude every workload case")
    for dcp_size in dcp_sizes:
        for comm_sm in comm_sms:
            records = dataset[dcp_size][comm_sm]
            for case_id in excluded:
                del records[case_id]
        baseline_records = baseline_dataset[dcp_size]
        for case_id in excluded:
            del baseline_records[case_id]
    return excluded


def select_best_comm_sms(
    dataset: Dataset,
    *,
    dcp_sizes: Sequence[int],
    comm_sms: Sequence[int],
) -> dict[int, int]:
    return {
        dcp_size: min(
            comm_sms,
            key=lambda comm_sm: (
                statistics.fmean(
                    record.e2e_latency_us
                    for record in dataset[dcp_size][comm_sm].values()
                ),
                comm_sm,
            ),
        )
        for dcp_size in dcp_sizes
    }


def select_best_records_by_case(
    dataset: Dataset,
    *,
    dcp_sizes: Sequence[int],
    comm_sms: Sequence[int],
) -> dict[int, dict[str, CaseRecord]]:
    selected: dict[int, dict[str, CaseRecord]] = {}
    for dcp_size in dcp_sizes:
        case_ids = dataset[dcp_size][comm_sms[0]].keys()
        selected[dcp_size] = {
            case_id: min(
                (
                    dataset[dcp_size][comm_sm][case_id]
                    for comm_sm in comm_sms
                ),
                key=lambda record: (record.e2e_latency_us, record.comm_sm),
            )
            for case_id in case_ids
        }
    return selected


def paired_winner_counts(
    dataset: Dataset,
    *,
    dcp_size: int,
    comm_sms: Sequence[int],
) -> dict[int, int]:
    counts = {comm_sm: 0 for comm_sm in comm_sms}
    selected = select_best_records_by_case(
        dataset,
        dcp_sizes=(dcp_size,),
        comm_sms=comm_sms,
    )[dcp_size]
    for record in selected.values():
        counts[record.comm_sm] += 1
    return counts


def _case_order(
    dataset: Dataset,
    *,
    dcp_size: int,
    comm_sm: int,
    workload_kind: str,
) -> list[str]:
    records = dataset[dcp_size][comm_sm]
    return sorted(
        (
            case_id
            for case_id, record in records.items()
            if record.workload_kind == workload_kind
        ),
        key=lambda case_id: (
            records[case_id].global_effective_flops,
            records[case_id].total_q_tokens,
            records[case_id].batch_size,
            case_id,
        ),
    )


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("rolling median expects one-dimensional values")
    radius = window // 2
    result = np.empty_like(values, dtype=np.float64)
    for index in range(values.size):
        begin = max(0, index - radius)
        end = min(values.size, index + radius + 1)
        result[index] = float(np.median(values[begin:end]))
    return result


def _expanded_limits(values: Sequence[float], *, log_scale: bool) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if log_scale:
        return low / 1.12, high * 1.12
    span = max(high - low, 0.05)
    return max(0.0, low - span * 0.04), high + span * 0.04


def _milestone_row_cap(
    mega_records_by_dcp: Mapping[int, Mapping[str, CaseRecord]],
    baseline_dataset: BaselineDataset,
    *,
    dcp_sizes: Sequence[int],
    workload_kind: str,
    percentile: float,
) -> float:
    comparable_e2e_values = [
        record.milestones_us["kernel_done"]
        for dcp_size in dcp_sizes
        for record in mega_records_by_dcp[dcp_size].values()
        if record.workload_kind == workload_kind
    ]
    comparable_e2e_values.extend(
        _baseline_max_completion_us(record)
        for dcp_size in dcp_sizes
        for record in baseline_dataset[dcp_size].values()
        if record.workload_kind == workload_kind
    )
    return float(np.percentile(comparable_e2e_values, percentile)) * 1.06


def _baseline_max_completion_us(record: BaselineCaseRecord) -> float:
    reconstructed_end = sum(
        record.stages_us[stage.key] for stage in BASELINE_STAGES[:-1]
    )
    return max(
        reconstructed_end,
        record.stages_us["attention_end_to_end_ms"],
    )


def _baseline_completion_matrix(
    records: Mapping[str, BaselineCaseRecord],
    case_order: Sequence[str],
) -> np.ndarray:
    component_stages = BASELINE_STAGES[:-1]
    component_durations = np.asarray(
        [
            [records[case_id].stages_us[stage.key] for stage in component_stages]
            for case_id in case_order
        ],
        dtype=np.float64,
    )
    cumulative = np.cumsum(component_durations, axis=1)
    measured_e2e = np.asarray(
        [
            records[case_id].stages_us["attention_end_to_end_ms"]
            for case_id in case_order
        ],
        dtype=np.float64,
    )
    return np.column_stack((cumulative, measured_e2e))


def _plot_milestone_panel(
    axis: plt.Axes,
    records: Mapping[str, CaseRecord],
    baseline_records: Mapping[str, BaselineCaseRecord],
    case_order: Sequence[str],
    *,
    y_cap: float,
    x_limits: tuple[float, float],
    rolling_window: int,
) -> None:
    x = np.asarray(
        [records[case_id].global_effective_flops / 1.0e9 for case_id in case_order],
        dtype=np.float64,
    )
    clipped_cases = {
        case_id
        for case_id in case_order
        if records[case_id].milestones_us["kernel_done"] > y_cap
    }
    clipped_baseline_cases = {
        case_id
        for case_id in case_order
        if _baseline_max_completion_us(baseline_records[case_id]) > y_cap
    }
    for phase in PHASES:
        values = np.asarray(
            [records[case_id].milestones_us[phase.key] for case_id in case_order],
            dtype=np.float64,
        )
        visible = values <= y_cap
        axis.scatter(
            x[visible],
            values[visible],
            s=13,
            marker=phase.marker,
            color=phase.color,
            alpha=0.28,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        clipped = ~visible
        if bool(clipped.any()):
            axis.scatter(
                x[clipped],
                np.full(int(clipped.sum()), y_cap * 0.992),
                s=23,
                marker="^",
                color=phase.color,
                edgecolors="white",
                linewidths=0.35,
                alpha=0.9,
                rasterized=True,
                zorder=4,
            )
        trend = _rolling_median(values, rolling_window)
        axis.plot(
            x,
            trend,
            color=phase.color,
            linewidth=1.45,
            alpha=0.96,
            zorder=3,
        )
    baseline_completions = _baseline_completion_matrix(
        baseline_records,
        case_order,
    )
    for stage_index, stage in enumerate(BASELINE_STAGES):
        values = baseline_completions[:, stage_index]
        trend = _rolling_median(values, rolling_window)
        is_total = stage.key == "attention_end_to_end_ms"
        axis.plot(
            x,
            trend,
            color=stage.color,
            linewidth=2.15 if is_total else 1.25,
            linestyle=(0, (7, 2)) if is_total else (0, (4, 2)),
            alpha=0.98 if is_total else 0.9,
            zorder=4 if is_total else 3.5,
        )
    if clipped_baseline_cases:
        clipped_x = np.asarray(
            [
                baseline_records[case_id].workload_signature[2] / 1.0e9
                for case_id in case_order
                if case_id in clipped_baseline_cases
            ],
            dtype=np.float64,
        )
        axis.scatter(
            clipped_x,
            np.full(clipped_x.size, y_cap * 0.992),
            s=28,
            marker="^",
            facecolors="none",
            edgecolors="#111111",
            linewidths=0.8,
            rasterized=True,
            zorder=5,
        )
    axis.set_xscale("log")
    axis.set_xlim(*x_limits)
    axis.set_ylim(0.0, y_cap * 1.015)
    axis.grid(True, which="major", color="#DDDDDD", linewidth=0.7, alpha=0.8)
    axis.grid(True, which="minor", axis="x", color="#EEEEEE", linewidth=0.45)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.tick_params(labelsize=9)
    axis.text(
        0.02,
        0.97,
        f"n={len(case_order)} | clipped Mega/A2A="
        f"{len(clipped_cases)}/{len(clipped_baseline_cases)}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#555555",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
            "pad": 1.5,
        },
    )


def _ecdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    y = np.arange(1, x.size + 1, dtype=np.float64) / x.size
    return x, y


def _paired_slowdowns(
    dataset: Dataset,
    *,
    dcp_size: int,
    comm_sm: int,
    reference_sm: int,
    workload_kind: str,
) -> list[float]:
    reference = dataset[dcp_size][reference_sm]
    current = dataset[dcp_size][comm_sm]
    return [
        current[case_id].milestones_us["kernel_done"]
        / reference[case_id].milestones_us["kernel_done"]
        for case_id, record in reference.items()
        if record.workload_kind == workload_kind
    ]


def _plot_ecdf_panel(
    axis: plt.Axes,
    dataset: Dataset,
    *,
    dcp_size: int,
    comm_sms: Sequence[int],
    reference_sm: int,
    x_limits: tuple[float, float],
) -> None:
    axis.axvline(1.0, color="#555555", linewidth=1.1, linestyle=":", zorder=1)
    for comm_sm in comm_sms:
        if comm_sm == reference_sm:
            continue
        color = COMM_SM_COLORS.get(comm_sm, plt.get_cmap("tab10")(comm_sms.index(comm_sm)))
        for workload_kind, linestyle in ((KIND_DECODE, "-"), (KIND_MIXED, "--")):
            values = _paired_slowdowns(
                dataset,
                dcp_size=dcp_size,
                comm_sm=comm_sm,
                reference_sm=reference_sm,
                workload_kind=workload_kind,
            )
            x, y = _ecdf(values)
            axis.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=1.65,
                drawstyle="steps-post",
                alpha=0.95,
            )
    axis.set_xlim(*x_limits)
    axis.set_ylim(0.0, 1.01)
    axis.grid(True, color="#E0E0E0", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=9)
    axis.text(
        0.02,
        0.96,
        f"reference: comm SM {reference_sm}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#444444",
    )


def _baseline_stage_matrix(
    dataset: BaselineDataset,
    *,
    dcp_size: int,
) -> np.ndarray:
    rows: list[list[float]] = []
    for workload_kind in (KIND_DECODE, KIND_MIXED):
        records = [
            record
            for record in dataset[dcp_size].values()
            if record.workload_kind == workload_kind
        ]
        if not records:
            raise ValueError(
                f"no vLLM A2A Graph records for {workload_kind}, DCP={dcp_size}"
            )
        rows.append(
            [
                float(np.median([record.stages_us[stage.key] for record in records]))
                for stage in BASELINE_STAGES
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _tail_matrix(
    dataset: Dataset,
    *,
    dcp_size: int,
    comm_sms: Sequence[int],
) -> np.ndarray:
    rows: list[list[float]] = []
    for workload_kind in (KIND_DECODE, KIND_MIXED):
        for tail in TAIL_ROWS:
            row = []
            for comm_sm in comm_sms:
                values = [
                    record.tails_us[tail.key]
                    for record in dataset[dcp_size][comm_sm].values()
                    if record.workload_kind == workload_kind
                ]
                row.append(float(np.median(values)))
            rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _tail_row_labels() -> list[str]:
    labels = []
    for workload_label in ("Decode", "Mixed"):
        labels.extend(f"{workload_label}: {tail.label}" for tail in TAIL_ROWS)
    return labels


def _plot_tail_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    comm_sms: Sequence[int],
    best_comm_sm: int,
    norm: LogNorm,
    show_y_labels: bool,
) -> matplotlib.image.AxesImage:
    total_rows = np.zeros(matrix.shape, dtype=bool)
    total_rows[len(TAILS) :: len(TAIL_ROWS), :] = True
    heatmap_values = np.ma.array(matrix, mask=total_rows)
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#ECEFF1")
    image = axis.imshow(
        heatmap_values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    axis.set_xticks(
        range(len(comm_sms)),
        [f"{comm_sm}{'*' if comm_sm == best_comm_sm else ''}" for comm_sm in comm_sms],
    )
    axis.set_yticks(range(matrix.shape[0]))
    if show_y_labels:
        axis.set_yticklabels(_tail_row_labels(), fontsize=8.2)
    else:
        axis.set_yticklabels([])
        axis.tick_params(axis="y", length=0)
    axis.tick_params(axis="x", labelsize=9)
    axis.set_xlabel("Communication SMs (* aggregate best)", fontsize=9.5)
    axis.axhline(len(TAIL_ROWS) - 0.5, color="white", linewidth=2.0)
    best_index = comm_sms.index(best_comm_sm)
    axis.add_patch(
        Rectangle(
            (best_index - 0.5, -0.5),
            1.0,
            matrix.shape[0],
            fill=False,
            edgecolor="#FFEE88",
            linewidth=2.0,
        )
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            if total_rows[row, column]:
                text_color = "#111111"
            else:
                red, green, blue, _alpha = cmap(norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                text_color = "#111111" if luminance > 0.57 else "white"
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color=text_color,
            )
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def plot_overview(
    dataset: Dataset,
    baseline_dataset: BaselineDataset,
    *,
    run_dir: Path,
    arrival_label: str,
    dcp_sizes: Sequence[int],
    comm_sms: Sequence[int],
    best_comm_sms: Mapping[int, int],
    stat: str,
    rolling_window: int,
    clip_percentile: float,
    excluded_case_ids: Sequence[str],
    title: str,
    output: Path,
    pdf_output: Path | None,
    dpi: int,
) -> None:
    reference_records = dataset[dcp_sizes[0]][comm_sms[0]]
    per_case_best_records = select_best_records_by_case(
        dataset,
        dcp_sizes=dcp_sizes,
        comm_sms=comm_sms,
    )
    case_orders = {
        workload_kind: _case_order(
            dataset,
            dcp_size=dcp_sizes[0],
            comm_sm=comm_sms[0],
            workload_kind=workload_kind,
        )
        for workload_kind in (KIND_DECODE, KIND_MIXED)
    }
    x_limits = {}
    for workload_kind, case_order in case_orders.items():
        flops_g = [
            reference_records[case_id].global_effective_flops / 1.0e9
            for case_id in case_order
        ]
        x_limits[workload_kind] = _expanded_limits(flops_g, log_scale=True)
    y_caps = {
        workload_kind: _milestone_row_cap(
            per_case_best_records,
            baseline_dataset,
            dcp_sizes=dcp_sizes,
            workload_kind=workload_kind,
            percentile=clip_percentile,
        )
        for workload_kind in (KIND_DECODE, KIND_MIXED)
    }

    slowdown_values = [
        value
        for dcp_size in dcp_sizes
        for comm_sm in comm_sms
        if comm_sm != best_comm_sms[dcp_size]
        for workload_kind in (KIND_DECODE, KIND_MIXED)
        for value in _paired_slowdowns(
            dataset,
            dcp_size=dcp_size,
            comm_sm=comm_sm,
            reference_sm=best_comm_sms[dcp_size],
            workload_kind=workload_kind,
        )
    ]
    slowdown_limits = _expanded_limits(slowdown_values + [1.0], log_scale=False)

    tail_matrices = {
        dcp_size: _tail_matrix(dataset, dcp_size=dcp_size, comm_sms=comm_sms)
        for dcp_size in dcp_sizes
    }
    positive_tail_values = np.concatenate(
        [
            matrix.reshape(2, len(TAIL_ROWS), len(comm_sms))[:, : len(TAILS), :]
            .reshape(-1)
            for matrix in tail_matrices.values()
        ]
    )
    tail_norm = LogNorm(
        vmin=float(positive_tail_values.min()),
        vmax=float(positive_tail_values.max()),
    )

    figure = plt.figure(figsize=(24.0, 18.5))
    grid = figure.add_gridspec(
        4,
        3,
        left=0.085,
        right=0.91,
        bottom=0.075,
        top=0.82,
        wspace=0.13,
        hspace=0.36,
        height_ratios=(1.14, 1.14, 0.88, 1.16),
    )
    axes = [
        [figure.add_subplot(grid[row, column]) for column in range(3)]
        for row in range(4)
    ]

    for column, dcp_size in enumerate(dcp_sizes):
        best_comm_sm = best_comm_sms[dcp_size]
        records = per_case_best_records[dcp_size]
        axes[0][column].set_title(
            f"DCP size {dcp_size} | per-case best comm SM by E2E",
            fontsize=13.5,
            pad=10,
        )
        for row, workload_kind in enumerate((KIND_DECODE, KIND_MIXED)):
            _plot_milestone_panel(
                axes[row][column],
                records,
                baseline_dataset[dcp_size],
                case_orders[workload_kind],
                y_cap=y_caps[workload_kind],
                x_limits=x_limits[workload_kind],
                rolling_window=rolling_window,
            )
            axes[row][column].set_xlabel(
                "Global effective workload (GFLOP, log scale)",
                fontsize=9.5,
            )
            if column == 0:
                axes[row][column].set_ylabel(
                    f"{KIND_LABELS[workload_kind]}\n"
                    f"{stat} completion timestamp / E2E (us)",
                    fontsize=10.5,
                )
            else:
                axes[row][column].tick_params(labelleft=False)

        _plot_ecdf_panel(
            axes[2][column],
            dataset,
            dcp_size=dcp_size,
            comm_sms=comm_sms,
            reference_sm=best_comm_sm,
            x_limits=slowdown_limits,
        )
        axes[2][column].set_xlabel(
            f"Paired kernel-done slowdown vs comm SM {best_comm_sm}",
            fontsize=9.5,
        )
        if column == 0:
            axes[2][column].set_ylabel("Fraction of workload batches", fontsize=10.5)
        else:
            axes[2][column].tick_params(labelleft=False)

    heatmap_image = None
    for column, dcp_size in enumerate(dcp_sizes):
        heatmap_image = _plot_tail_heatmap(
            axes[3][column],
            tail_matrices[dcp_size],
            comm_sms=comm_sms,
            best_comm_sm=best_comm_sms[dcp_size],
            norm=tail_norm,
            show_y_labels=column == 0,
        )
    assert heatmap_image is not None
    colorbar = figure.colorbar(
        heatmap_image,
        ax=axes[3],
        location="right",
        fraction=0.025,
        pad=0.018,
    )
    colorbar.set_label(f"Median per-case {stat} tail duration (us)", fontsize=10)
    colorbar.ax.tick_params(labelsize=8.5)

    phase_handles = [
        Line2D(
            [0],
            [0],
            color=phase.color,
            marker=phase.marker,
            markersize=5,
            linewidth=1.6,
            label=phase.label,
        )
        for phase in PHASES
    ]
    figure.legend(
        handles=phase_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(PHASES),
        frameon=False,
        fontsize=9.5,
        handlelength=2.2,
        columnspacing=1.5,
        title=(
            "Mega per-case lowest-E2E comm-SM selection: cumulative completion "
            "timestamps (solid + per-case points)"
        ),
        title_fontsize=9.5,
    )
    baseline_handles = [
        Line2D(
            [0],
            [0],
            color=stage.color,
            linewidth=(2.15 if stage.key == "attention_end_to_end_ms" else 1.4),
            linestyle=(
                (0, (7, 2))
                if stage.key == "attention_end_to_end_ms"
                else (0, (4, 2))
            ),
            label=stage.label,
        )
        for stage in BASELINE_STAGES
    ]
    figure.legend(
        handles=baseline_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.872),
        ncol=len(BASELINE_STAGES),
        frameon=False,
        fontsize=8.6,
        handlelength=2.5,
        columnspacing=1.15,
        title=(
            "vLLM A2A reconstructed completion timestamps "
            "(dashed rolling medians)"
        ),
        title_fontsize=9.2,
    )
    comm_handles = [
        Line2D(
            [0],
            [0],
            color=COMM_SM_COLORS.get(comm_sm, plt.get_cmap("tab10")(index)),
            linewidth=1.8,
            label=f"comm SM {comm_sm}",
        )
        for index, comm_sm in enumerate(comm_sms)
    ]
    kind_handles = [
        Line2D([0], [0], color="#555555", linewidth=1.8, label="Decode-only"),
        Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=1.8,
            linestyle="--",
            label="Mixed-prefill",
        ),
    ]
    axes[2][1].legend(
        handles=comm_handles + kind_handles,
        loc="lower right",
        ncol=2,
        frameon=False,
        fontsize=8.2,
        columnspacing=1.0,
        handlelength=2.2,
    )

    case_count = len(reference_records)
    iteration_counts = {record.iterations for record in reference_records.values()}
    if len(iteration_counts) != 1:
        raise ValueError(
            f"paired cases use different iteration counts: {sorted(iteration_counts)}"
        )
    iterations = next(iter(iteration_counts))
    decode_count = sum(
        record.workload_kind == KIND_DECODE for record in reference_records.values()
    )
    mixed_count = case_count - decode_count
    exclusion_text = ""
    if excluded_case_ids:
        exclusion_text = (
            f" | excluded by request: {','.join(excluded_case_ids)}"
        )
    figure.suptitle(title, fontsize=20, y=0.982)
    figure.text(
        0.5,
        0.953,
        (
            f"Arrival scale {arrival_label} | {case_count} paired workload batches "
            f"({decode_count} decode-only, {mixed_count} mixed-prefill) | "
            f"{iterations} measured iterations per case | Mega uses each case's "
            f"lowest-E2E comm SM | Mega and vLLM A2A Graph paired by case"
            f"{exclusion_text} | source: {run_dir.name}"
        ),
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
    )
    figure.text(
        0.5,
        0.025,
        (
            "Milestones are per-case values after an iteration-wise MAX across DCP "
            "ranks; they are percentile summaries, not one serialized execution. "
            "vLLM dashed milestones are per-case prefix sums of CUDA Event phase "
            "durations in runner order; Graph E2E is independently measured and "
            "need not equal the final prefix sum. Mega milestones are in-kernel "
            "timestamps. Mega tail heatmap cells use explicitly recorded "
            "differences; gray Start-to-kernel-done rows are totals excluded "
            "from the heatmap color scale. "
            "Upward triangles mark "
            f"values above the shared row-level p{clip_percentile:g} display limit."
        ),
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if pdf_output is not None:
        pdf_output = pdf_output.expanduser().resolve()
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(pdf_output, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = resolve_run_dir(args.input)
        arrival_dir, arrival_label = resolve_arrival_dir(
            run_dir, args.arrival_time_scale
        )
        dataset = load_dataset(
            arrival_dir,
            dcp_sizes=args.dcp_sizes,
            comm_sms=args.comm_sms,
            stat=args.stat,
        )
        baseline_dataset = load_baseline_dataset(
            arrival_dir,
            dcp_sizes=args.dcp_sizes,
            stat=args.stat,
        )
        validate_baseline_pairing(
            dataset,
            baseline_dataset,
            dcp_sizes=args.dcp_sizes,
            reference_sm=args.comm_sms[0],
        )
        excluded_case_ids = exclude_paired_cases(
            dataset,
            baseline_dataset,
            case_ids=args.exclude_case_id,
            dcp_sizes=args.dcp_sizes,
            comm_sms=args.comm_sms,
        )
        validate_pairing(
            dataset,
            dcp_sizes=args.dcp_sizes,
            comm_sms=args.comm_sms,
        )
        validate_baseline_pairing(
            dataset,
            baseline_dataset,
            dcp_sizes=args.dcp_sizes,
            reference_sm=args.comm_sms[0],
        )
        best_comm_sms = select_best_comm_sms(
            dataset,
            dcp_sizes=args.dcp_sizes,
            comm_sms=args.comm_sms,
        )
        suffix = f"arrival_{arrival_label}_{args.stat}"
        output = args.output or run_dir / f"mega_phase_timestamp_overview_{suffix}.png"
        pdf_output = None
        if not args.no_pdf:
            pdf_output = args.pdf_output or run_dir / (
                f"mega_phase_timestamp_overview_{suffix}.pdf"
            )
        elif args.pdf_output is not None:
            raise ValueError("--pdf-output cannot be combined with --no-pdf")
        plot_overview(
            dataset,
            baseline_dataset,
            run_dir=run_dir,
            arrival_label=arrival_label,
            dcp_sizes=args.dcp_sizes,
            comm_sms=args.comm_sms,
            best_comm_sms=best_comm_sms,
            stat=args.stat,
            rolling_window=args.rolling_window,
            clip_percentile=args.clip_percentile,
            excluded_case_ids=excluded_case_ids,
            title=args.title,
            output=output,
            pdf_output=pdf_output,
            dpi=args.dpi,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    reference = dataset[args.dcp_sizes[0]][args.comm_sms[0]]
    kind_counts = {
        workload_kind: sum(
            record.workload_kind == workload_kind for record in reference.values()
        )
        for workload_kind in (KIND_DECODE, KIND_MIXED)
    }
    print(f"input={run_dir}")
    print(f"arrival_time_scale={arrival_label}")
    print(f"case_count={len(reference)}")
    print(
        f"decode_case_count={kind_counts[KIND_DECODE]} "
        f"mixed_prefill_case_count={kind_counts[KIND_MIXED]}"
    )
    print(
        f"baseline_method={METHOD_VLLM_A2A} "
        "baseline_execution_mode=cuda_graph "
        "baseline_phase_timing=cuda_event"
    )
    if excluded_case_ids:
        print(f"excluded_case_ids={','.join(excluded_case_ids)}")
    for dcp_size in args.dcp_sizes:
        counts = paired_winner_counts(
            dataset,
            dcp_size=dcp_size,
            comm_sms=args.comm_sms,
        )
        winner_text = ",".join(
            f"sm{comm_sm}:{counts[comm_sm]}" for comm_sm in args.comm_sms
        )
        print(
            f"dcp={dcp_size} aggregate_best_comm_sm={best_comm_sms[dcp_size]} "
            f"per_case_e2e_winners={winner_text}"
        )
        baseline_matrix = _baseline_stage_matrix(
            baseline_dataset,
            dcp_size=dcp_size,
        )
        for row, workload_kind in enumerate((KIND_DECODE, KIND_MIXED)):
            stage_text = ",".join(
                f"{stage.key.removesuffix('_ms')}={baseline_matrix[row, column]:.3f}"
                for column, stage in enumerate(BASELINE_STAGES)
            )
            print(
                f"vllm_a2a_graph_phase_durations_us dcp={dcp_size} "
                f"batch_type={workload_kind} {stage_text}"
            )
    print(f"png_output={output.expanduser().resolve()}")
    if pdf_output is not None:
        print(f"pdf_output={pdf_output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
