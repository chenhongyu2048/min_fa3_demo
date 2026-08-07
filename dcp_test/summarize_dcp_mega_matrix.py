"""Summarize the trace-driven Mega DCP arrival/DCP benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
BASELINE_METHODS = {
    "ours_no_overlap_varlen",
    "ours_overlap_varlen",
    "vllm_ag_rs_min_fa3_varlen",
    "vllm_a2a_min_fa3_varlen",
    "sglang_mha_ag_ar_min_fa3_varlen",
    "full_kv_min_fa3_varlen",
}
MEGA_METHOD = "dcp_mega_varlen"
CSV_FIELDS = (
    "arrival_time_scale",
    "dcp_size",
    "suite",
    "execution_mode",
    "method",
    "mega_num_comm_sm",
    "case_count",
    "p50_latency_ms_min",
    "p50_latency_ms_mean",
    "p50_latency_ms_p50",
    "p50_latency_ms_max",
    "p90_latency_ms_mean",
    "mean_effective_tflops",
    "workload_weighted_effective_tflops",
    "workload_weighted_effective_tflops_per_gpu",
    "mean_effective_kv_bandwidth_gbps_per_gpu",
    "workload_weighted_effective_kv_bandwidth_gbps_per_gpu",
    "total_effective_flops",
    "total_logical_kv_bytes_per_gpu",
    "total_p50_latency_ms",
    "trace_config_sha256",
    "manifest",
)


def _csv_tokens(spec: str, option: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in spec.split(",") if token.strip())
    if not tokens:
        raise ValueError(f"{option} must not be empty")
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"{option} must not contain duplicates")
    return tokens


def _int_tokens(spec: str, option: str) -> tuple[int, ...]:
    tokens = _csv_tokens(spec, option)
    try:
        values = tuple(int(token) for token in tokens)
    except ValueError as error:
        raise ValueError(f"{option} must contain comma-separated integers") from error
    if any(value <= 0 for value in values):
        raise ValueError(f"{option} values must be positive")
    return values


def _arrival_tokens(spec: str) -> tuple[str, ...]:
    tokens = _csv_tokens(spec, "--arrival-time-scales")
    numeric: list[Decimal] = []
    for token in tokens:
        try:
            value = Decimal(token)
        except InvalidOperation as error:
            raise ValueError(
                "--arrival-time-scales must contain positive finite numbers"
            ) from error
        if not value.is_finite() or value <= 0:
            raise ValueError(
                "--arrival-time-scales must contain positive finite numbers"
            )
        if value in numeric:
            raise ValueError(
                "--arrival-time-scales must not contain numerically duplicate values"
            )
        numeric.append(value)
    return tokens


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build JSON and CSV summaries for the Mega DCP matrix"
    )
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--arrival-time-scales", default="1,2,4")
    parser.add_argument("--dcp-sizes", default="2,4,8")
    parser.add_argument("--mega-num-comm-sms", default="4,8,12,16,20")
    parser.add_argument("--num-cases", type=int, default=100)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write partial summaries and return success for missing/failed runs",
    )
    args = parser.parse_args(argv)
    if args.num_cases <= 0:
        parser.error("--num-cases must be positive")
    try:
        args.arrival_time_scales = _arrival_tokens(args.arrival_time_scales)
        args.dcp_sizes = _int_tokens(args.dcp_sizes, "--dcp-sizes")
        args.mega_num_comm_sms = _int_tokens(
            args.mega_num_comm_sms, "--mega-num-comm-sms"
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"manifest root must be an object: {path}")
    return payload


def _resolve_case_output(path_value: Any, manifest_path: Path) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"case in {manifest_path} has no output_json path")
    path = Path(path_value).expanduser()
    if path.is_absolute():
        candidates = (path,)
    else:
        candidates = (Path.cwd() / path, manifest_path.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(
        f"case output JSON referenced by {manifest_path} does not exist: {path_value}"
    )


def _backfill_bandwidth_metrics(
    case_entries: Sequence[Mapping[str, Any]],
    methods: set[str],
    manifest_path: Path,
) -> dict[str, dict[str, float]]:
    if not case_entries:
        raise ValueError(
            f"weighted summary in {manifest_path} lacks bandwidth fields and cases"
        )
    bandwidth_values = {method: [] for method in methods}
    logical_byte_values = {method: [] for method in methods}
    for entry in case_entries:
        output_path = _resolve_case_output(entry.get("output_json"), manifest_path)
        output = _read_manifest(output_path)
        reports = output.get("methods")
        if not isinstance(reports, dict):
            raise ValueError(f"case output is missing methods: {output_path}")
        for method in methods:
            report = reports.get(method)
            if not isinstance(report, dict):
                raise ValueError(
                    f"case output {output_path} is missing method {method}"
                )
            logical_kv = report.get("logical_kv_read")
            if not isinstance(logical_kv, dict):
                raise ValueError(
                    f"case output {output_path} is missing logical_kv_read for {method}"
                )
            try:
                average_bytes = float(logical_kv["average_bytes_per_gpu"])
                bandwidth = float(
                    logical_kv["effective_bandwidth_gbps_per_gpu"]
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid logical KV bandwidth for {method} in {output_path}"
                ) from error
            if (
                not math.isfinite(average_bytes)
                or average_bytes < 0
                or not math.isfinite(bandwidth)
                or bandwidth < 0
            ):
                raise ValueError(
                    f"non-finite or negative bandwidth for {method} in {output_path}"
                )
            logical_byte_values[method].append(average_bytes)
            bandwidth_values[method].append(bandwidth)

    return {
        method: {
            "mean_effective_kv_bandwidth_gbps_per_gpu": (
                sum(bandwidth_values[method]) / len(bandwidth_values[method])
            ),
            "total_logical_kv_bytes_per_gpu": sum(logical_byte_values[method]),
        }
        for method in methods
    }


def _summary_rows(
    summary: Mapping[str, Any],
    *,
    arrival: str,
    dcp_size: int,
    suite: str,
    execution_mode: str,
    comm_sm: int | None,
    config_sha256: str,
    manifest_path: Path,
    case_entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    topologies = summary.get("topologies")
    if not isinstance(topologies, dict) or len(topologies) != 1:
        raise ValueError("weighted_summary must contain exactly one topology")
    topology = next(iter(topologies.values()))
    if not isinstance(topology, dict) or topology.get("dcp_size") != dcp_size:
        raise ValueError("weighted_summary DCP does not match the matrix combination")
    methods = topology.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("weighted_summary topology is missing methods")

    missing_bandwidth_methods = {
        method
        for method, values in methods.items()
        if not isinstance(values, dict)
        or "mean_effective_kv_bandwidth_gbps_per_gpu" not in values
        or "workload_weighted_effective_kv_bandwidth_gbps_per_gpu" not in values
        or "total_logical_kv_bytes_per_gpu" not in values
    }
    backfilled = (
        _backfill_bandwidth_metrics(
            case_entries, missing_bandwidth_methods, manifest_path
        )
        if missing_bandwidth_methods
        else {}
    )

    rows: list[dict[str, Any]] = []
    for method in sorted(methods):
        values = methods[method]
        if not isinstance(values, dict):
            raise ValueError(f"invalid weighted summary for method {method}")
        p50 = values["p50_latency_ms"]
        p90 = values["p90_latency_ms"]
        if method in backfilled:
            bandwidth = backfilled[method]
            mean_bandwidth = bandwidth[
                "mean_effective_kv_bandwidth_gbps_per_gpu"
            ]
            total_logical_bytes = bandwidth["total_logical_kv_bytes_per_gpu"]
            weighted_bandwidth = total_logical_bytes / (
                float(values["total_p50_latency_ms"]) * 1.0e6
            )
        else:
            mean_bandwidth = values[
                "mean_effective_kv_bandwidth_gbps_per_gpu"
            ]
            weighted_bandwidth = values[
                "workload_weighted_effective_kv_bandwidth_gbps_per_gpu"
            ]
            total_logical_bytes = values["total_logical_kv_bytes_per_gpu"]
        rows.append(
            {
                "arrival_time_scale": arrival,
                "dcp_size": dcp_size,
                "suite": suite,
                "execution_mode": execution_mode,
                "method": method,
                "mega_num_comm_sm": "" if comm_sm is None else comm_sm,
                "case_count": values["case_count"],
                "p50_latency_ms_min": p50["min"],
                "p50_latency_ms_mean": p50["mean"],
                "p50_latency_ms_p50": p50["p50"],
                "p50_latency_ms_max": p50["max"],
                "p90_latency_ms_mean": p90["mean"],
                "mean_effective_tflops": values["mean_effective_tflops"],
                "workload_weighted_effective_tflops": values[
                    "workload_weighted_effective_tflops"
                ],
                "workload_weighted_effective_tflops_per_gpu": values[
                    "workload_weighted_effective_tflops_per_gpu"
                ],
                "mean_effective_kv_bandwidth_gbps_per_gpu": mean_bandwidth,
                "workload_weighted_effective_kv_bandwidth_gbps_per_gpu": (
                    weighted_bandwidth
                ),
                "total_effective_flops": values["total_effective_flops"],
                "total_logical_kv_bytes_per_gpu": total_logical_bytes,
                "total_p50_latency_ms": values["total_p50_latency_ms"],
                "trace_config_sha256": config_sha256,
                "manifest": str(manifest_path),
            }
        )
    return rows


def _validate_trace(
    manifest: Mapping[str, Any],
    arrival: str,
    dcp_size: int,
    num_cases: int,
) -> str:
    trace = manifest.get("trace")
    if not isinstance(trace, dict):
        raise ValueError("manifest is missing trace provenance")
    try:
        trace_arrival = Decimal(str(trace.get("arrival_time_scale")))
    except InvalidOperation as error:
        raise ValueError("manifest arrival_time_scale is invalid") from error
    if trace_arrival != Decimal(arrival):
        raise ValueError("manifest arrival_time_scale does not match its directory")
    if trace.get("dcp_size") != dcp_size:
        raise ValueError("manifest trace DCP does not match its directory")
    if trace.get("num_cases") != num_cases:
        raise ValueError("manifest trace case count does not match --num-cases")
    config_sha256 = trace.get("config_sha256")
    if not isinstance(config_sha256, str) or re.fullmatch(
        r"[0-9a-fA-F]{64}", config_sha256
    ) is None:
        raise ValueError("manifest trace config SHA is invalid")
    trace_sha256 = trace.get("trace_sha256")
    if not isinstance(trace_sha256, str) or re.fullmatch(
        r"[0-9a-fA-F]{64}", trace_sha256
    ) is None:
        raise ValueError("manifest source trace SHA is invalid")
    if not isinstance(trace.get("cases_jsonl"), str) or not trace["cases_jsonl"]:
        raise ValueError("manifest trace cases path is invalid")
    return config_sha256


def _standard_run(
    path: Path,
    *,
    arrival: str,
    dcp_size: int,
    execution_mode: str,
    num_cases: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_manifest(path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported baseline manifest schema in {path}")
    config_sha256 = _validate_trace(manifest, arrival, dcp_size, num_cases)
    if manifest.get("execution_mode") != execution_mode:
        raise ValueError(f"execution mode mismatch in {path}")
    if manifest.get("status") != "complete":
        return manifest, []
    if (
        manifest.get("case_total") != num_cases
        or manifest.get("completed_case_count") != num_cases
    ):
        raise ValueError(f"completed case count mismatch in {path}")
    rows = _summary_rows(
        manifest["weighted_summary"],
        arrival=arrival,
        dcp_size=dcp_size,
        suite="baseline",
        execution_mode=execution_mode,
        comm_sm=None,
        config_sha256=config_sha256,
        manifest_path=path,
        case_entries=manifest.get("cases", []),
    )
    if {row["method"] for row in rows} != BASELINE_METHODS:
        raise ValueError(f"baseline method set mismatch in {path}")
    if any(row["case_count"] != num_cases for row in rows):
        raise ValueError(f"weighted summary case count mismatch in {path}")
    return manifest, rows


def _mega_run(
    path: Path,
    *,
    arrival: str,
    dcp_size: int,
    comm_sms: tuple[int, ...],
    num_cases: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_manifest(path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported Mega sweep manifest schema in {path}")
    config_sha256 = _validate_trace(manifest, arrival, dcp_size, num_cases)
    if manifest.get("manifest_kind") != "mega_comm_sm_sweep":
        raise ValueError(f"not a Mega comm-SM sweep manifest: {path}")
    if manifest.get("execution_mode") != "eager":
        raise ValueError(f"Mega sweep must use eager execution: {path}")
    variants = manifest.get("variants")
    if not isinstance(variants, list):
        raise ValueError(f"Mega sweep is missing variants: {path}")
    if tuple(variant.get("mega_num_comm_sm") for variant in variants) != comm_sms:
        raise ValueError(f"Mega comm-SM variants do not match the requested matrix: {path}")
    if manifest.get("status") == "complete" and (
        manifest.get("variant_total") != len(comm_sms)
        or manifest.get("completed_variant_count") != len(comm_sms)
        or manifest.get("case_total") != num_cases * len(comm_sms)
        or manifest.get("completed_case_count") != num_cases * len(comm_sms)
        or any(variant.get("status") != "complete" for variant in variants)
    ):
        raise ValueError(f"completed Mega sweep counts are inconsistent: {path}")

    rows: list[dict[str, Any]] = []
    for variant in variants:
        if variant.get("status") != "complete":
            continue
        if variant.get("completed_case_count") != num_cases:
            raise ValueError(f"Mega variant case count mismatch in {path}")
        variant_rows = _summary_rows(
            variant["weighted_summary"],
            arrival=arrival,
            dcp_size=dcp_size,
            suite="mega",
            execution_mode="eager",
            comm_sm=int(variant["mega_num_comm_sm"]),
            config_sha256=config_sha256,
            manifest_path=path,
            case_entries=variant.get("cases", []),
        )
        if len(variant_rows) != 1 or variant_rows[0]["method"] != MEGA_METHOD:
            raise ValueError(f"Mega variant method mismatch in {path}")
        if variant_rows[0]["case_count"] != num_cases:
            raise ValueError(f"Mega weighted summary case count mismatch in {path}")
        rows.extend(variant_rows)
    return manifest, rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    combination_shas: dict[tuple[str, int], set[str]] = {}

    for arrival in args.arrival_time_scales:
        for dcp_size in args.dcp_sizes:
            combo_dir = args.result_dir / f"arrival_{arrival}" / f"dcp_{dcp_size}"
            specs = (
                ("mega", "eager", combo_dir / "mega" / "manifest.json"),
                (
                    "baseline",
                    "eager",
                    combo_dir / "baseline_eager" / "manifest.json",
                ),
                (
                    "baseline",
                    "cuda_graph",
                    combo_dir / "baseline_graph" / "manifest.json",
                ),
            )
            for suite, execution_mode, path in specs:
                run = {
                    "run_id": f"arrival_{arrival}_dcp_{dcp_size}_{suite}_{execution_mode}",
                    "arrival_time_scale": arrival,
                    "dcp_size": dcp_size,
                    "suite": suite,
                    "execution_mode": execution_mode,
                    "manifest": str(path),
                    "status": "missing",
                    "error": None,
                    "summary_row_count": 0,
                    "trace_sha256": None,
                    "trace_config_sha256": None,
                    "cases_jsonl": None,
                }
                if not path.is_file():
                    runs.append(run)
                    continue
                try:
                    if suite == "mega":
                        manifest, run_rows = _mega_run(
                            path,
                            arrival=arrival,
                            dcp_size=dcp_size,
                            comm_sms=args.mega_num_comm_sms,
                            num_cases=args.num_cases,
                        )
                    else:
                        manifest, run_rows = _standard_run(
                            path,
                            arrival=arrival,
                            dcp_size=dcp_size,
                            execution_mode=execution_mode,
                            num_cases=args.num_cases,
                        )
                    run["status"] = str(manifest.get("status", "invalid"))
                    run["summary_row_count"] = len(run_rows)
                    config_sha = manifest["trace"]["config_sha256"]
                    run["trace_sha256"] = manifest["trace"].get("trace_sha256")
                    run["trace_config_sha256"] = config_sha
                    run["cases_jsonl"] = manifest["trace"].get("cases_jsonl")
                    combination_shas.setdefault((arrival, dcp_size), set()).add(
                        config_sha
                    )
                    rows.extend(run_rows)
                except (KeyError, TypeError, ValueError) as error:
                    run["status"] = "invalid"
                    run["error"] = str(error)
                runs.append(run)

    for (arrival, dcp_size), shas in combination_shas.items():
        if len(shas) != 1:
            message = (
                f"arrival={arrival}, DCP={dcp_size} uses different trace config SHAs"
            )
            for run in runs:
                if (
                    run["arrival_time_scale"] == arrival
                    and run["dcp_size"] == dcp_size
                ):
                    run["status"] = "invalid"
                    run["error"] = message

    completed = sum(run["status"] == "complete" for run in runs)
    failed = sum(run["status"] == "failed" for run in runs)
    missing = sum(run["status"] == "missing" for run in runs)
    invalid = sum(run["status"] == "invalid" for run in runs)
    running = sum(run["status"] == "running" for run in runs)
    expected_rows = len(args.arrival_time_scales) * len(args.dcp_sizes) * (
        len(args.mega_num_comm_sms) + 2 * len(BASELINE_METHODS)
    )
    complete = completed == len(runs) and len(rows) == expected_rows
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if complete else "incomplete",
        "matrix": {
            "arrival_time_scales": list(args.arrival_time_scales),
            "dcp_sizes": list(args.dcp_sizes),
            "mega_num_comm_sms": list(args.mega_num_comm_sms),
            "num_cases": args.num_cases,
            "baseline_methods": sorted(BASELINE_METHODS),
        },
        "expected_launch_count": len(runs),
        "completed_launch_count": completed,
        "failed_launch_count": failed,
        "missing_launch_count": missing,
        "invalid_launch_count": invalid,
        "running_launch_count": running,
        "expected_summary_row_count": expected_rows,
        "summary_row_count": len(rows),
        "summary_rows": rows,
        "runs": runs,
    }
    _write_json(args.output_json, payload)
    _write_csv(args.output_csv, rows)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = summarize(args)
    summary_keys = (
        "status",
        "expected_launch_count",
        "completed_launch_count",
        "failed_launch_count",
        "missing_launch_count",
        "invalid_launch_count",
        "running_launch_count",
        "expected_summary_row_count",
        "summary_row_count",
    )
    print(
        json.dumps({key: payload[key] for key in summary_keys}, sort_keys=True),
        file=sys.stderr,
    )
    if payload["status"] != "complete" and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
