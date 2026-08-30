"""Run a matrix of packed-varlen Mega DCP benchmarks in one torchrun."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import torch.distributed as dist

from dcp_test import benchmark_dcp_varlen
from dcp_test.trace.models import (
    SCHEMA_VERSION as TRACE_SCHEMA_VERSION,
    ReplayConfig,
    load_config as load_trace_config,
)
from dcp_test.utils import (
    DCPGroup,
    initialize_distributed_sm90,
    make_dcp_groups,
    require_world_size,
)
from min_fa3_dcp import DCPMegaAttentionRunner, make_topology


CONFIG_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
SWEEP_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "dcp_mega_six_loads.json"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class BatchWorkload:
    name: str
    batch_size: int
    q_lengths: tuple[int, ...]
    history_lengths: tuple[int, ...]
    trace_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class BatchTopology:
    name: str
    dcp_size: int
    kv_heads: int


@dataclass(frozen=True)
class BatchConfig:
    name: str
    tp_size: int
    q_heads: int
    head_dim: int
    workloads: tuple[BatchWorkload, ...]
    topologies: tuple[BatchTopology, ...]


@dataclass(frozen=True)
class BatchCase:
    case_id: str
    workload: BatchWorkload
    topology: BatchTopology


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _check_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    context: str,
) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required)
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {', '.join(unknown)}")


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _safe_name(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
        raise ValueError(
            f"{context} must match {_SAFE_NAME.pattern!r} for safe result filenames"
        )
    return value


def _lengths(value: Any, batch_size: int, context: str) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a positive integer or a length-B list")
    if isinstance(value, int):
        length = _positive_int(value, context)
        return (length,) * batch_size
    if not isinstance(value, list) or len(value) != batch_size:
        raise ValueError(
            f"{context} must be a positive integer or contain exactly B={batch_size} values"
        )
    return tuple(
        _positive_int(length, f"{context}[{index}]")
        for index, length in enumerate(value)
    )


def _unique_names(values: Sequence[Any], context: str) -> None:
    names = [value.name for value in values]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate {context} names: {', '.join(duplicates)}")


def load_batch_config(path: Path) -> BatchConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read batch config {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in batch config {path}: {error}") from error

    root = _mapping(raw, "batch config")
    required = {
        "schema_version",
        "name",
        "tp_size",
        "qhead",
        "headdim",
        "workloads",
        "topologies",
    }
    _check_keys(root, required=required, context="batch config")
    if root["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"batch config schema_version must be {CONFIG_SCHEMA_VERSION}, "
            f"got {root['schema_version']!r}"
        )

    name = _safe_name(root["name"], "batch config name")
    tp_size = _positive_int(root["tp_size"], "tp_size")
    q_heads = _positive_int(root["qhead"], "qhead")
    head_dim = _positive_int(root["headdim"], "headdim")
    if tp_size not in (2, 4, 8):
        raise ValueError("Mega DCP batch configs require tp_size=2, 4, or 8")
    if head_dim != 128:
        raise ValueError("Mega DCP batch configs require headdim=128")

    raw_workloads = root["workloads"]
    if not isinstance(raw_workloads, list) or not raw_workloads:
        raise ValueError("workloads must be a nonempty JSON list")
    workloads: list[BatchWorkload] = []
    for index, item in enumerate(raw_workloads):
        context = f"workloads[{index}]"
        workload = _mapping(item, context)
        _check_keys(
            workload,
            required={"name", "b", "sq", "seqlen"},
            context=context,
        )
        batch_size = _positive_int(workload["b"], f"{context}.b")
        workloads.append(
            BatchWorkload(
                name=_safe_name(workload["name"], f"{context}.name"),
                batch_size=batch_size,
                q_lengths=_lengths(workload["sq"], batch_size, f"{context}.sq"),
                history_lengths=_lengths(
                    workload["seqlen"], batch_size, f"{context}.seqlen"
                ),
            )
        )
    _unique_names(workloads, "workload")

    raw_topologies = root["topologies"]
    if not isinstance(raw_topologies, list) or not raw_topologies:
        raise ValueError("topologies must be a nonempty JSON list")
    topologies: list[BatchTopology] = []
    for index, item in enumerate(raw_topologies):
        context = f"topologies[{index}]"
        topology = _mapping(item, context)
        _check_keys(
            topology,
            required={"name", "dcp_size", "kvhead"},
            context=context,
        )
        dcp_size = _positive_int(topology["dcp_size"], f"{context}.dcp_size")
        kv_heads = _positive_int(topology["kvhead"], f"{context}.kvhead")
        if dcp_size not in (2, 4, 8):
            raise ValueError(f"{context}.dcp_size must be 2, 4, or 8")
        try:
            resolved = make_topology(q_heads, kv_heads, tp_size, dcp_size)
        except ValueError as error:
            raise ValueError(f"{context} is invalid: {error}") from error
        if resolved.q_heads_local not in (4, 8):
            raise ValueError(
                f"{context} gives Hq_local={resolved.q_heads_local}; Mega requires 4 or 8"
            )
        topologies.append(
            BatchTopology(
                name=_safe_name(topology["name"], f"{context}.name"),
                dcp_size=dcp_size,
                kv_heads=kv_heads,
            )
        )
    _unique_names(topologies, "topology")

    return BatchConfig(
        name=name,
        tp_size=tp_size,
        q_heads=q_heads,
        head_dim=head_dim,
        workloads=tuple(workloads),
        topologies=tuple(topologies),
    )


def load_trace_workloads(
    path: Path,
    trace_config: ReplayConfig,
) -> tuple[BatchWorkload, ...]:
    """Load trace snapshots and validate them against the effective replay config."""
    workloads: list[BatchWorkload] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read trace cases {path}: {error}") from error

    with handle:
        for line_number, text in enumerate(handle, start=1):
            if not text.strip():
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"trace cases line {line_number} contains invalid JSON: {error}"
                ) from error
            context = f"trace cases line {line_number}"
            case = _mapping(raw, context)
            required = {
                "schema_version",
                "case_id",
                "trace_sha256",
                "config_sha256",
                "batch_size",
                "q_lens",
                "logical_q_lens",
                "q_len_alignment",
                "history_lens",
                "total_kv_lens",
            }
            missing = sorted(required - case.keys())
            if missing:
                raise ValueError(
                    f"{context} is missing required fields: {', '.join(missing)}"
                )
            if case["schema_version"] != TRACE_SCHEMA_VERSION:
                raise ValueError(
                    f"{context}.schema_version must be {TRACE_SCHEMA_VERSION!r}"
                )
            if case["trace_sha256"] != trace_config.trace_sha256:
                raise ValueError(f"{context}.trace_sha256 does not match trace config")
            if case["config_sha256"] != trace_config.config_sha256:
                raise ValueError(
                    f"{context}.config_sha256 does not match the effective trace config"
                )

            batch_size = _positive_int(case["batch_size"], f"{context}.batch_size")
            q_lengths = _lengths(case["q_lens"], batch_size, f"{context}.q_lens")
            logical_q_lengths = _lengths(
                case["logical_q_lens"],
                batch_size,
                f"{context}.logical_q_lens",
            )
            history_lengths = _lengths(
                case["history_lens"], batch_size, f"{context}.history_lens"
            )
            total_kv_lengths = _lengths(
                case["total_kv_lens"], batch_size, f"{context}.total_kv_lens"
            )
            expected_total_kv = tuple(
                history + query
                for history, query in zip(history_lengths, q_lengths)
            )
            if total_kv_lengths != expected_total_kv:
                raise ValueError(
                    f"{context}.total_kv_lens must equal history_lens + q_lens"
                )
            alignment = _positive_int(
                case["q_len_alignment"], f"{context}.q_len_alignment"
            )
            if alignment != trace_config.q_len_alignment:
                raise ValueError(
                    f"{context}.q_len_alignment does not match trace config"
                )
            if any(
                physical % alignment != 0
                or logical > physical
                or physical - logical >= alignment
                for physical, logical in zip(q_lengths, logical_q_lengths)
            ):
                raise ValueError(
                    f"{context}.q_lens must be the aligned physical form of "
                    "logical_q_lens"
                )
            if any(length < trace_config.dcp_size for length in history_lengths):
                raise ValueError(
                    f"{context}.history_lens must be at least DCP={trace_config.dcp_size}"
                )

            metadata = {
                key: value
                for key, value in case.items()
                if key
                not in {
                    "batch_size",
                    "q_lens",
                    "history_lens",
                    "total_kv_lens",
                }
            }
            workloads.append(
                BatchWorkload(
                    name=_safe_name(case["case_id"], f"{context}.case_id"),
                    batch_size=batch_size,
                    q_lengths=q_lengths,
                    history_lengths=history_lengths,
                    trace_metadata=metadata,
                )
            )

    if len(workloads) != trace_config.num_cases:
        raise ValueError(
            f"trace cases contains {len(workloads)} cases, but the effective trace "
            f"config requests {trace_config.num_cases}"
        )
    _unique_names(workloads, "trace case")
    return tuple(workloads)


def _csv_tokens(spec: str, option: str) -> tuple[str, ...] | None:
    tokens = tuple(token.strip() for token in spec.split(",") if token.strip())
    if not tokens:
        raise ValueError(f"{option} must not be empty")
    if "all" in tokens:
        if len(tokens) != 1:
            raise ValueError(f"{option}=all cannot be combined with other values")
        return None
    return tuple(dict.fromkeys(tokens))


def expand_cases(
    config: BatchConfig,
    *,
    workloads: str = "all",
    dcp_sizes: str = "all",
) -> tuple[BatchCase, ...]:
    workload_filter = _csv_tokens(workloads, "--workloads")
    dcp_filter_tokens = _csv_tokens(dcp_sizes, "--dcp-sizes")
    known_workloads = {workload.name for workload in config.workloads}
    if workload_filter is not None:
        unknown = sorted(set(workload_filter) - known_workloads)
        if unknown:
            raise ValueError(f"unknown workloads: {', '.join(unknown)}")

    dcp_filter = None
    if dcp_filter_tokens is not None:
        try:
            dcp_filter = tuple(int(token) for token in dcp_filter_tokens)
        except ValueError as error:
            raise ValueError("--dcp-sizes must contain comma-separated integers") from error
        known_dcp_sizes = {topology.dcp_size for topology in config.topologies}
        unknown = sorted(set(dcp_filter) - known_dcp_sizes)
        if unknown:
            raise ValueError(
                "DCP sizes are not present in the config: "
                + ", ".join(str(value) for value in unknown)
            )

    cases = tuple(
        BatchCase(
            case_id=f"{workload.name}_{topology.name}",
            workload=workload,
            topology=topology,
        )
        for workload in config.workloads
        if workload_filter is None or workload.name in workload_filter
        for topology in config.topologies
        if dcp_filter is None or topology.dcp_size in dcp_filter
    )
    if not cases:
        raise ValueError("the selected workload/topology matrix is empty")
    case_ids = [case.case_id for case in cases]
    duplicate_ids = sorted(
        {case_id for case_id in case_ids if case_ids.count(case_id) > 1}
    )
    if duplicate_ids:
        raise ValueError(
            "workload/topology names produce duplicate case IDs: "
            + ", ".join(duplicate_ids)
        )
    return cases


def _arg_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _arg_nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _arg_num_splits(value: str) -> int:
    parsed = _arg_nonnegative_int(value)
    if parsed > 128:
        raise argparse.ArgumentTypeError("value must be at most 128")
    return parsed


def _arg_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def _arg_positive_int_list(value: str) -> tuple[int, ...]:
    tokens = tuple(token.strip() for token in value.split(",") if token.strip())
    if not tokens:
        raise argparse.ArgumentTypeError("value must not be empty")
    try:
        parsed = tuple(_arg_positive_int(token) for token in tokens)
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError(
            "value must contain comma-separated positive integers"
        ) from error
    return tuple(dict.fromkeys(parsed))


def default_output_dir() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return Path("benchmarks/results") / f"dcp_mega_batch_{timestamp}"


def apply_topology_override(
    config: BatchConfig,
    *,
    tp_size: int | None,
    dcp_size: int | None,
    q_heads: int | None,
    kv_heads: int | None,
    dcp_sizes: str,
) -> tuple[BatchConfig, str]:
    """Replace a config's topology matrix with one explicitly requested topology."""
    values = {
        "--tp-size": tp_size,
        "--dcp-size": dcp_size,
        "--qhead": q_heads,
        "--kvhead": kv_heads,
    }
    provided = tuple(name for name, value in values.items() if value is not None)
    if not provided:
        return config, dcp_sizes
    if len(provided) != len(values):
        missing = ", ".join(name for name, value in values.items() if value is None)
        raise ValueError(
            "topology override requires --tp-size, --dcp-size, --qhead, and "
            f"--kvhead together; missing {missing}"
        )

    assert tp_size is not None
    assert dcp_size is not None
    assert q_heads is not None
    assert kv_heads is not None
    if tp_size not in (2, 4, 8):
        raise ValueError("--tp-size must be 2, 4, or 8")
    if dcp_size not in (2, 4, 8):
        raise ValueError("--dcp-size must be 2, 4, or 8")
    try:
        topology = make_topology(q_heads, kv_heads, tp_size, dcp_size)
    except ValueError as error:
        raise ValueError(f"CLI topology override is invalid: {error}") from error
    if topology.q_heads_local not in (4, 8):
        raise ValueError(
            "CLI topology override gives "
            f"Hq_local={topology.q_heads_local}; Mega requires 4 or 8"
        )

    if dcp_sizes != "all":
        tokens = _csv_tokens(dcp_sizes, "--dcp-sizes")
        if tokens is not None:
            try:
                selected = {int(token) for token in tokens}
            except ValueError as error:
                raise ValueError(
                    "--dcp-sizes must contain comma-separated integers"
                ) from error
            if selected != {dcp_size}:
                raise ValueError(
                    "--dcp-sizes conflicts with the CLI topology override: "
                    f"expected only {dcp_size}, got {sorted(selected)}"
                )

    topology_name = f"dcp{dcp_size}_hkv{kv_heads}"
    return (
        replace(
            config,
            tp_size=tp_size,
            q_heads=q_heads,
            topologies=(BatchTopology(topology_name, dcp_size, kv_heads),),
        ),
        str(dcp_size),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a JSON-defined packed-varlen Mega DCP case matrix"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--trace-cases",
        type=Path,
        default=None,
        help="use Mega DCP workload snapshots from this trace JSONL",
    )
    parser.add_argument(
        "--trace-config",
        type=Path,
        default=None,
        help="effective replay config used to generate --trace-cases",
    )
    parser.add_argument(
        "--num-cases",
        type=_arg_positive_int,
        default=None,
        help="trace num_cases override used when generating the input JSONL",
    )
    parser.add_argument(
        "--trace-arrival-time-scale",
        type=_arg_positive_float,
        default=None,
        help="arrival_time_scale override used to generate --trace-cases",
    )
    parser.add_argument(
        "--trace-dcp-size",
        type=int,
        choices=(2, 4, 8),
        default=None,
        help="dcp_size override used to generate --trace-cases",
    )
    parser.add_argument("--workloads", default="all")
    parser.add_argument("--dcp-sizes", default="all")
    parser.add_argument(
        "--tp-size",
        type=_arg_positive_int,
        default=None,
        help="physical TP/world size for a single CLI topology override",
    )
    parser.add_argument(
        "--dcp-size",
        type=_arg_positive_int,
        default=None,
        help="DCP group size for a single CLI topology override",
    )
    parser.add_argument(
        "--qhead",
        type=_arg_positive_int,
        default=None,
        help="global Q-head count for a single CLI topology override",
    )
    parser.add_argument(
        "--kvhead",
        type=_arg_positive_int,
        default=None,
        help="global KV-head count for a single CLI topology override",
    )
    parser.add_argument(
        "--implementations", default="mega,ours,vllm,sglang"
    )
    parser.add_argument("--num-splits", type=_arg_num_splits, default=0)
    parser.add_argument(
        "--mega-scheduler-heuristic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use critical-wave split selection instead of FA3 native split "
            "selection (default for Mega with --num-splits 0 or 1)"
        ),
    )
    parser.add_argument(
        "--mega-history-order",
        choices=("auto", "fifo", "release-lpt"),
        default="auto",
        help=(
            "Mega history order: auto uses release-LPT only for critical-wave "
            "decode-only split plans; fifo and release-lpt force that order"
        ),
    )
    parser.add_argument("--mega-num-comm-sm", type=_arg_positive_int, default=8)
    parser.add_argument(
        "--mega-num-comm-sms",
        type=_arg_positive_int_list,
        default=None,
        help="Mega-only eager comm-SM sweep executed in one torchrun",
    )
    parser.add_argument(
        "--mega-block-n",
        type=benchmark_dcp_varlen.parse_mega_block_n,
        default=None,
        metavar="{auto,128,176}",
        help=(
            "Mega BlockN policy: auto uses 176 for critical-wave NoSplit and "
            "128 for selected split=2/4; 128 or 176 fixes the kernel variant"
        ),
    )
    parser.add_argument(
        "--mega-phase-timestamps",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--baseline-phase-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Record non-Mega CUDA-event phase breakdowns; disabling keeps "
            "end-to-end CUDA-event timing"
        ),
    )
    parser.add_argument("--warmup", type=_arg_nonnegative_int, default=5)
    parser.add_argument("--iters", type=_arg_positive_int, default=20)
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--print-cases",
        action="store_true",
        help="Validate and print the expanded case matrix without initializing CUDA",
    )
    args = parser.parse_args(argv)
    implementation_tokens = {
        token.strip() for token in args.implementations.split(",")
    }
    if args.mega_scheduler_heuristic is None:
        args.mega_scheduler_heuristic = (
            "mega" in implementation_tokens
            and args.num_splits in (0, 1)
        )
    if (args.trace_cases is None) != (args.trace_config is None):
        parser.error("--trace-cases and --trace-config must be provided together")
    if args.trace_cases is None and args.num_cases is not None:
        parser.error("--num-cases is valid only with --trace-cases")
    if args.trace_cases is None and (
        args.trace_arrival_time_scale is not None or args.trace_dcp_size is not None
    ):
        parser.error("trace overrides are valid only with --trace-cases")
    if args.trace_cases is not None and args.workloads != "all":
        parser.error("--workloads is not supported with --trace-cases")
    if args.mega_num_comm_sm >= 132:
        parser.error("--mega-num-comm-sm must leave at least one of 132 SMs for compute")
    if args.mega_scheduler_heuristic and args.num_splits not in (0, 1):
        parser.error("--mega-scheduler-heuristic requires --num-splits 0 or 1")
    if args.mega_scheduler_heuristic and "mega" not in implementation_tokens:
        parser.error("--mega-scheduler-heuristic requires --implementations mega")
    if args.mega_history_order != "auto" and "mega" not in implementation_tokens:
        parser.error(
            "--mega-history-order fifo/release-lpt requires "
            "--implementations mega"
        )
    if args.mega_num_comm_sms is not None and any(
        value >= 132 for value in args.mega_num_comm_sms
    ):
        parser.error(
            "--mega-num-comm-sms values must leave at least one of 132 SMs "
            "for compute"
        )
    if args.output_dir is None:
        args.output_dir = default_output_dir()
    if args.manifest is None:
        args.manifest = args.output_dir / "manifest.json"
    return args


def _length_spec(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _result_path(
    args: argparse.Namespace,
    case: BatchCase,
    mega_num_comm_sm: int | None = None,
) -> Path:
    mode = "graph" if args.cuda_graph else "eager"
    if args.mega_num_comm_sms is not None:
        if mega_num_comm_sm is None:
            raise ValueError("comm-SM sweep result paths require a comm-SM value")
        return (
            args.output_dir
            / f"comm_sm_{mega_num_comm_sm}"
            / f"{case.case_id}_{mode}.json"
        )
    return args.output_dir / f"{case.case_id}_{mode}.json"


def _case_argv(
    args: argparse.Namespace,
    config: BatchConfig,
    case: BatchCase,
    output_path: Path,
    mega_num_comm_sm: int | None = None,
) -> list[str]:
    comm_sm = (
        args.mega_num_comm_sm if mega_num_comm_sm is None else mega_num_comm_sm
    )
    case_argv = [
        "--b",
        str(case.workload.batch_size),
        "--sq",
        _length_spec(case.workload.q_lengths),
        "--seqlen",
        _length_spec(case.workload.history_lengths),
        "--qhead",
        str(config.q_heads),
        "--kvhead",
        str(case.topology.kv_heads),
        "--headdim",
        str(config.head_dim),
        "--tp-size",
        str(config.tp_size),
        "--dcp-size",
        str(case.topology.dcp_size),
        "--workload",
        "chunk",
        "--implementations",
        args.implementations,
        "--num-splits",
        str(args.num_splits),
        (
            "--mega-scheduler-heuristic"
            if args.mega_scheduler_heuristic
            else "--no-mega-scheduler-heuristic"
        ),
        "--mega-num-comm-sm",
        str(comm_sm),
        "--mega-block-n",
        benchmark_dcp_varlen.mega_block_n_spec(args.mega_block_n),
        "--mega-history-order",
        args.mega_history_order,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--output-json",
        str(output_path),
        "--cuda-graph" if args.cuda_graph else "--no-cuda-graph",
        "--check" if args.check else "--no-check",
        (
            "--mega-phase-timestamps"
            if args.mega_phase_timestamps
            else "--no-mega-phase-timestamps"
        ),
        (
            "--baseline-phase-timing"
            if args.baseline_phase_timing
            else "--no-baseline-phase-timing"
        ),
    ]
    return case_argv


def _case_description(case: BatchCase) -> str:
    return (
        f"{case.case_id}: B={case.workload.batch_size}, "
        f"Sq={list(case.workload.q_lengths)}, "
        f"Sk={list(case.workload.history_lengths)}, "
        f"DCP={case.topology.dcp_size}, Hkv={case.topology.kv_heads}"
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _method_summary(result: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    methods = result["methods"]
    assert isinstance(methods, dict)
    return {
        method: {
            "p50_ms": float(report["stages_ms"]["attention_end_to_end_ms"]["p50"]),
            "p90_ms": float(report["stages_ms"]["attention_end_to_end_ms"]["p90"]),
            "effective_tflops": float(report["effective_tflops"]),
            "average_logical_kv_bytes_per_gpu": float(
                report["logical_kv_read"]["average_bytes_per_gpu"]
            ),
            "effective_kv_bandwidth_gbps_per_gpu": float(
                report["logical_kv_read"]["effective_bandwidth_gbps_per_gpu"]
            ),
        }
        for method, report in methods.items()
    }


def _manifest_case(
    case: BatchCase,
    output_path: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    entry = {
        "case_id": case.case_id,
        "workload": case.workload.name,
        "topology": case.topology.name,
        "batch_size": case.workload.batch_size,
        "q_lengths": list(case.workload.q_lengths),
        "history_lengths": list(case.workload.history_lengths),
        "dcp_size": case.topology.dcp_size,
        "kv_heads": case.topology.kv_heads,
        "global_effective_flops": int(result["global_effective_flops"]),
        "output_json": str(output_path),
        "methods": _method_summary(result),
    }
    if case.workload.trace_metadata is not None:
        entry["trace"] = dict(case.workload.trace_metadata)
    return entry


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "p50": median(values),
        "max": max(values),
    }


def _weighted_summary(
    entries: Sequence[Mapping[str, Any]],
    tp_size: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        for method, values in entry["methods"].items():
            grouped[
                (
                    str(entry["topology"]),
                    int(entry["dcp_size"]),
                    int(entry["kv_heads"]),
                    str(method),
                )
            ].append(
                {
                    "global_effective_flops": int(entry["global_effective_flops"]),
                    **values,
                }
            )

    topologies: dict[str, dict[str, Any]] = {}
    for (topology, dcp_size, kv_heads, method), records in grouped.items():
        p50_values = [float(record["p50_ms"]) for record in records]
        p90_values = [float(record["p90_ms"]) for record in records]
        tflops_values = [float(record["effective_tflops"]) for record in records]
        bandwidth_values = [
            float(record["effective_kv_bandwidth_gbps_per_gpu"])
            for record in records
        ]
        total_flops = sum(int(record["global_effective_flops"]) for record in records)
        total_logical_kv_bytes_per_gpu = sum(
            float(record["average_logical_kv_bytes_per_gpu"])
            for record in records
        )
        total_p50_ms = sum(p50_values)
        if total_p50_ms <= 0:
            raise ValueError(
                f"cannot aggregate non-positive p50 latency for {topology}/{method}"
            )
        weighted_tflops = total_flops / (total_p50_ms * 1.0e9)
        weighted_bandwidth = total_logical_kv_bytes_per_gpu / (
            total_p50_ms * 1.0e6
        )
        topology_summary = topologies.setdefault(
            topology,
            {
                "dcp_size": dcp_size,
                "kv_heads": kv_heads,
                "methods": {},
            },
        )
        topology_summary["methods"][method] = {
            "case_count": len(records),
            "p50_latency_ms": _latency_summary(p50_values),
            "p90_latency_ms": _latency_summary(p90_values),
            "mean_effective_tflops": sum(tflops_values) / len(tflops_values),
            "workload_weighted_effective_tflops": weighted_tflops,
            "workload_weighted_effective_tflops_per_gpu": weighted_tflops
            / tp_size,
            "mean_effective_kv_bandwidth_gbps_per_gpu": sum(bandwidth_values)
            / len(bandwidth_values),
            "workload_weighted_effective_kv_bandwidth_gbps_per_gpu": (
                weighted_bandwidth
            ),
            "total_effective_flops": total_flops,
            "total_logical_kv_bytes_per_gpu": total_logical_kv_bytes_per_gpu,
            "total_p50_latency_ms": total_p50_ms,
        }

    return {
        "latency_basis": "per-case p50 of the global-rank-max CUDA-event latency",
        "weighting": (
            "sum(global_effective_flops) / sum(p50_latency_seconds), equivalent "
            "to a p50-latency-weighted mean of per-case effective TFLOPS"
        ),
        "bandwidth_weighting": (
            "sum(average logical BF16 K+V bytes per GPU) / "
            "sum(p50_latency_seconds); this is effective payload bandwidth, "
            "not hardware-counter HBM traffic"
        ),
        "topologies": topologies,
    }


def _print_summary(entries: Sequence[Mapping[str, Any]]) -> None:
    print("\nMega DCP batch summary")
    rows = []
    for entry in entries:
        for method, values in entry["methods"].items():
            rows.append(
                (
                    entry["case_id"],
                    entry["dcp_size"],
                    method,
                    values["p50_ms"],
                    values["p90_ms"],
                    values["effective_tflops"],
                )
            )
    case_width = max(20, *(len(str(row[0])) for row in rows))
    method_width = max(24, *(len(str(row[2])) for row in rows))
    print(
        f"{'Case':<{case_width}} {'DCP':>4} {'Method':<{method_width}} "
        f"{'p50 ms':>10} {'p90 ms':>10} {'Agg TFLOPS':>12}"
    )
    for case_id, dcp_size, method, p50, p90, tflops in rows:
        print(
            f"{case_id:<{case_width}} {dcp_size:>4} {method:<{method_width}} "
            f"{p50:>10.3f} {p90:>10.3f} {tflops:>12.1f}"
        )


def _print_weighted_summary(summary: Mapping[str, Any]) -> None:
    print("\nMega DCP workload-weighted summary")
    rows = []
    for topology, topology_summary in summary["topologies"].items():
        for method, values in topology_summary["methods"].items():
            rows.append(
                (
                    topology,
                    topology_summary["dcp_size"],
                    method,
                    values["case_count"],
                    values["p50_latency_ms"]["mean"],
                    values["mean_effective_tflops"],
                    values["workload_weighted_effective_tflops"],
                    values["workload_weighted_effective_tflops_per_gpu"],
                )
            )
    topology_width = max(14, *(len(str(row[0])) for row in rows))
    method_width = max(24, *(len(str(row[2])) for row in rows))
    print(
        f"{'Topology':<{topology_width}} {'DCP':>4} "
        f"{'Method':<{method_width}} {'Cases':>7} {'Mean ms':>10} "
        f"{'Mean TFLOPS':>12} {'Weighted':>12} {'Weighted/GPU':>13}"
    )
    for topology, dcp_size, method, count, mean_ms, mean_tflops, weighted, per_gpu in rows:
        print(
            f"{topology:<{topology_width}} {dcp_size:>4} "
            f"{method:<{method_width}} {count:>7} {mean_ms:>10.3f} "
            f"{mean_tflops:>12.1f} {weighted:>12.1f} {per_gpu:>13.1f}"
        )


def _trace_manifest(
    args: argparse.Namespace,
    trace_config: ReplayConfig | None,
) -> dict[str, Any] | None:
    if trace_config is None:
        return None
    return {
        "cases_jsonl": str(args.trace_cases),
        "replay_config": str(args.trace_config),
        "trace_sha256": trace_config.trace_sha256,
        "config_sha256": trace_config.config_sha256,
        "num_cases": trace_config.num_cases,
        "arrival_time_scale": str(trace_config.arrival_time_scale),
        "dcp_size": trace_config.dcp_size,
        "fixed_step_us": trace_config.fixed_step_us,
        "sampling_start_ms": trace_config.sampling_start_ms,
        "sampling_end_ms": trace_config.sampling_end_ms,
        "seed": trace_config.seed,
    }


def _manifest_parameters(
    args: argparse.Namespace,
    config: BatchConfig,
    implementations: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "tp_size": config.tp_size,
        "q_heads": config.q_heads,
        "head_dim": config.head_dim,
        "implementations": list(implementations),
        "method_labels": list(
            benchmark_dcp_varlen.expanded_method_labels(implementations)
        ),
        "num_splits": args.num_splits,
        "mega_scheduler_heuristic": args.mega_scheduler_heuristic,
        "mega_history_order": args.mega_history_order,
        "mega_block_n": benchmark_dcp_varlen.mega_block_n_spec(
            args.mega_block_n
        ),
        "mega_phase_timestamps": args.mega_phase_timestamps,
        "baseline_phase_timing": args.baseline_phase_timing,
        "warmup": args.warmup,
        "iters": args.iters,
        "check": args.check,
    }


def _manifest_common(
    args: argparse.Namespace,
    config: BatchConfig,
    selected_dcp_sizes: str,
    implementations: tuple[str, ...],
    trace_config: ReplayConfig | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "running",
        "name": config.name,
        "source_config": str(args.config),
        "started_at": datetime.now().astimezone().isoformat(),
        "completed_at": None,
        "execution_mode": "cuda_graph" if args.cuda_graph else "eager",
        "selection": {
            "workloads": args.workloads,
            "dcp_sizes": selected_dcp_sizes,
            "requested_dcp_sizes": args.dcp_sizes,
        },
        "parameters": _manifest_parameters(args, config, implementations),
    }
    trace = _trace_manifest(args, trace_config)
    if trace is not None:
        payload["trace"] = trace
    return payload


def _run_case(
    args: argparse.Namespace,
    config: BatchConfig,
    case: BatchCase,
    output_path: Path,
    device: torch.device,
    dcp_group: DCPGroup,
    *,
    mega_num_comm_sm: int | None = None,
    shared_mega_runner: DCPMegaAttentionRunner | None = None,
) -> dict[str, object]:
    return benchmark_dcp_varlen.main(
        _case_argv(
            args,
            config,
            case,
            output_path,
            mega_num_comm_sm=mega_num_comm_sm,
        ),
        device=device,
        dcp_group=dcp_group,
        manage_process_group=False,
        shared_mega_runner=shared_mega_runner,
    )


def _make_shared_mega_runner(
    args: argparse.Namespace,
    config: BatchConfig,
    cases: Sequence[BatchCase],
    comm_sm: int,
    local_groups: Mapping[int, DCPGroup],
) -> DCPMegaAttentionRunner | None:
    topology_keys = {
        (case.topology.dcp_size, case.topology.kv_heads) for case in cases
    }
    if len(topology_keys) != 1:
        return None
    dcp_size, kv_heads = next(iter(topology_keys))
    topology = make_topology(config.q_heads, kv_heads, config.tp_size, dcp_size)
    return DCPMegaAttentionRunner(
        local_groups[dcp_size].process_group,
        dist.group.WORLD,
        max_total_q=max(sum(case.workload.q_lengths) for case in cases),
        max_batch=max(case.workload.batch_size for case in cases),
        Hq_local=topology.q_heads_local,
        max_num_splits=128,
        num_comm_sm=comm_sm,
        block_n_override=args.mega_block_n,
        record_phase_timestamps=args.mega_phase_timestamps,
    )


def _build_manifest(
    args: argparse.Namespace,
    config: BatchConfig,
    selected_dcp_sizes: str,
    implementations: tuple[str, ...],
    trace_config: ReplayConfig | None,
    cases: Sequence[BatchCase],
    comm_sm_sweep: tuple[int, ...] | None,
) -> dict[str, Any]:
    common = _manifest_common(
        args,
        config,
        selected_dcp_sizes,
        implementations,
        trace_config,
    )
    if comm_sm_sweep is None:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            **common,
            "parameters": {
                **common["parameters"],
                "mega_num_comm_sm": args.mega_num_comm_sm,
            },
            "case_total": len(cases),
            "completed_case_count": 0,
            "cases": [],
            "weighted_summary": None,
        }
    return {
        "schema_version": SWEEP_MANIFEST_SCHEMA_VERSION,
        "manifest_kind": "mega_comm_sm_sweep",
        **common,
        "parameters": {
            **common["parameters"],
            "mega_num_comm_sms": list(comm_sm_sweep),
        },
        "case_total": len(cases) * len(comm_sm_sweep),
        "completed_case_count": 0,
        "variant_total": len(comm_sm_sweep),
        "completed_variant_count": 0,
        "variants": [
            {
                "variant_id": f"comm_sm_{comm_sm}",
                "mega_num_comm_sm": comm_sm,
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "case_total": len(cases),
                "completed_case_count": 0,
                "cases": [],
                "weighted_summary": None,
            }
            for comm_sm in comm_sm_sweep
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    trace_config: ReplayConfig | None = None
    try:
        config = load_batch_config(args.config)
        config, selected_dcp_sizes = apply_topology_override(
            config,
            tp_size=args.tp_size,
            dcp_size=args.dcp_size,
            q_heads=args.qhead,
            kv_heads=args.kvhead,
            dcp_sizes=args.dcp_sizes,
        )
        if args.trace_cases is not None:
            assert args.trace_config is not None
            trace_config = load_trace_config(
                args.trace_config,
                num_cases=args.num_cases,
                arrival_time_scale=args.trace_arrival_time_scale,
                dcp_size=(
                    args.trace_dcp_size
                    if args.trace_dcp_size is not None
                    else args.dcp_size
                ),
            )
            config = replace(
                config,
                name="dcp_mega_trace",
                workloads=load_trace_workloads(args.trace_cases, trace_config),
            )
            if selected_dcp_sizes == "all":
                selected_dcp_sizes = str(trace_config.dcp_size)
        cases = expand_cases(
            config,
            workloads=args.workloads,
            dcp_sizes=selected_dcp_sizes,
        )
        if trace_config is not None:
            selected = {case.topology.dcp_size for case in cases}
            if selected != {trace_config.dcp_size}:
                raise ValueError(
                    "trace benchmark must select exactly its configured "
                    f"DCP={trace_config.dcp_size}, got {sorted(selected)}"
                )
        implementations = benchmark_dcp_varlen.parse_implementations(
            args.implementations
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    comm_sm_sweep = args.mega_num_comm_sms
    if comm_sm_sweep is not None:
        if implementations != ("mega",):
            raise SystemExit(
                "--mega-num-comm-sms requires --implementations mega"
            )
        if args.cuda_graph:
            raise SystemExit("--mega-num-comm-sms supports eager execution only")

    mode = "graph" if args.cuda_graph else "eager"
    if args.print_cases:
        if comm_sm_sweep is None:
            print(
                f"Batch config: name={config.name}, mode={mode}, "
                f"cases={len(cases)}, implementations={list(implementations)}, "
                f"baseline_phase_timing={args.baseline_phase_timing}"
            )
            for index, case in enumerate(cases, start=1):
                print(
                    f"[{index}/{len(cases)}] {_case_description(case)}; "
                    f"output={_result_path(args, case)}"
                )
            return
        total = len(cases) * len(comm_sm_sweep)
        print(
            f"Batch config: name={config.name}, mode={mode}, "
            f"cases={len(cases)}, executions={total}, "
            f"implementations={list(implementations)}, "
            f"baseline_phase_timing={args.baseline_phase_timing}"
        )
        execution_index = 0
        for comm_sm in comm_sm_sweep:
            for case in cases:
                execution_index += 1
                print(
                    f"[{execution_index}/{total}] {_case_description(case)}; "
                    f"comm_sm={comm_sm}; "
                    f"output={_result_path(args, case, comm_sm)}"
                )
        return

    device = initialize_distributed_sm90("Mega DCP batch benchmark")
    rank = dist.get_rank()
    require_world_size(config.tp_size)
    local_groups = make_dcp_groups(
        (case.topology.dcp_size for case in cases), device
    )
    manifest = _build_manifest(
        args,
        config,
        selected_dcp_sizes,
        implementations,
        trace_config,
        cases,
        comm_sm_sweep,
    )
    if rank == 0:
        _write_json(args.manifest, manifest)
        print(
            f"Mega DCP batch: name={config.name}, mode={mode}, cases={len(cases)}, "
            f"methods={list(benchmark_dcp_varlen.expanded_method_labels(implementations))}, "
            f"comm_sm_sweep={list(comm_sm_sweep) if comm_sm_sweep else None}, "
            f"baseline_phase_timing={args.baseline_phase_timing}"
        )

    active_case: BatchCase | None = None
    active_variant: dict[str, Any] | None = None
    try:
        if comm_sm_sweep is None:
            for case_index, case in enumerate(cases, start=1):
                active_case = case
                output_path = _result_path(args, case)
                if rank == 0:
                    print("\n" + "=" * 80)
                    print(
                        f"Batch case {case_index}/{len(cases)}: "
                        f"{_case_description(case)}",
                        flush=True,
                    )
                    print("=" * 80)
                result = _run_case(
                    args,
                    config,
                    case,
                    output_path,
                    device,
                    local_groups[case.topology.dcp_size],
                )
                if rank == 0:
                    manifest["cases"].append(
                        _manifest_case(case, output_path, result)
                    )
                    manifest["completed_case_count"] = len(manifest["cases"])
                    _write_json(args.manifest, manifest)
            if rank == 0:
                manifest["weighted_summary"] = _weighted_summary(
                    manifest["cases"], config.tp_size
                )
                manifest["status"] = "complete"
                manifest["completed_at"] = datetime.now().astimezone().isoformat()
                _write_json(args.manifest, manifest)
                _print_summary(manifest["cases"])
                _print_weighted_summary(manifest["weighted_summary"])
        else:
            for variant in manifest["variants"]:
                active_variant = variant
                comm_sm = int(variant["mega_num_comm_sm"])
                if rank == 0:
                    variant["status"] = "running"
                    variant["started_at"] = datetime.now().astimezone().isoformat()
                    _write_json(args.manifest, manifest)
                shared_mega_runner = _make_shared_mega_runner(
                    args, config, cases, comm_sm, local_groups
                )
                try:
                    for case_index, case in enumerate(cases, start=1):
                        active_case = case
                        output_path = _result_path(args, case, comm_sm)
                        if rank == 0:
                            print("\n" + "=" * 80)
                            print(
                                f"Mega comm_sm={comm_sm}, case "
                                f"{case_index}/{len(cases)}: "
                                f"{_case_description(case)}",
                                flush=True,
                            )
                            print("=" * 80)
                        result = _run_case(
                            args,
                            config,
                            case,
                            output_path,
                            device,
                            local_groups[case.topology.dcp_size],
                            mega_num_comm_sm=comm_sm,
                            shared_mega_runner=shared_mega_runner,
                        )
                        if rank == 0:
                            variant["cases"].append(
                                _manifest_case(case, output_path, result)
                            )
                            variant["completed_case_count"] = len(variant["cases"])
                            manifest["completed_case_count"] += 1
                            _write_json(args.manifest, manifest)
                finally:
                    if shared_mega_runner is not None:
                        shared_mega_runner.close()
                if rank == 0:
                    variant["weighted_summary"] = _weighted_summary(
                        variant["cases"], config.tp_size
                    )
                    variant["status"] = "complete"
                    variant["completed_at"] = datetime.now().astimezone().isoformat()
                    manifest["completed_variant_count"] += 1
                    _write_json(args.manifest, manifest)
                    print(f"\nMega comm_sm={comm_sm} summary")
                    _print_summary(variant["cases"])
                    _print_weighted_summary(variant["weighted_summary"])
            if rank == 0:
                manifest["status"] = "complete"
                manifest["completed_at"] = datetime.now().astimezone().isoformat()
                _write_json(args.manifest, manifest)
        if rank == 0:
            print(f"Wrote batch manifest to {args.manifest}", flush=True)
    except BaseException as error:
        if rank == 0:
            manifest["status"] = "failed"
            manifest["completed_at"] = datetime.now().astimezone().isoformat()
            if active_variant is not None:
                active_variant["status"] = "failed"
                active_variant["completed_at"] = manifest["completed_at"]
                active_variant["error"] = f"{type(error).__name__}: {error}"
                manifest["failed_variant"] = active_variant["variant_id"]
            manifest["failed_case"] = (
                active_case.case_id if active_case is not None else None
            )
            manifest["error"] = f"{type(error).__name__}: {error}"
            _write_json(args.manifest, manifest)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
