"""Run paired D1 Mega/vLLM-A2A Graph traces for frozen decode workloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcp_test.benchmark_dcp_varlen import STAGES, build_inputs, capture_runner, measure_mega, runner_call
from dcp_test.utils import DCPGroup, make_dcp_group, make_runner_set, quantiles, synchronize_before_samples
from min_fa3_dcp import DCPMegaAttentionRunner, make_topology
from motivation.common import environment, init_distributed_sm90, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--num-comm-sm", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_dcp_size = payload.get("dcp_size", payload.get("selection_dcp_size"))
    if source_dcp_size != 4:
        raise SystemExit(
            "D1 currently requires cases selected from the historical DCP4 run"
        )
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("D1 case manifest has no cases")
    return cases


def _resolve_topology(args: argparse.Namespace):
    if args.tp_size not in (4, 8):
        raise SystemExit("D1 supports TP/CP world size 4 or 8")
    allowed = {4: (1, 2), 8: (1, 2, 4)}
    if args.kvhead not in allowed[args.tp_size]:
        raise SystemExit(
            f"D1 CP{args.tp_size} supports KVH in {allowed[args.tp_size]}, "
            f"got {args.kvhead}"
        )
    dcp_size = args.tp_size // args.kvhead
    topology = make_topology(
        args.qhead,
        args.kvhead,
        args.tp_size,
        dcp_size,
    )
    if topology.q_heads_local not in (4, 8):
        raise SystemExit(
            "D1 Mega requires Hq_local in {4, 8}; "
            f"got QH={args.qhead}, TP={args.tp_size}, "
            f"Hq_local={topology.q_heads_local}"
        )
    return topology


def _runner_args(case: dict[str, Any], args: argparse.Namespace, *, optimized: bool) -> SimpleNamespace:
    q_lengths = [int(value) for value in case["q_lengths"]]
    history_lengths = [int(value) for value in case["history_lengths"]]
    return SimpleNamespace(
        b=len(q_lengths), sq=",".join(str(value) for value in q_lengths),
        seqlen=",".join(str(value) for value in history_lengths), qhead=args.qhead,
        kvhead=args.kvhead, headdim=128, tp_size=args.tp_size, dcp_size=args.dcp_size,
        workload="chunk", num_splits=0, mega_scheduler_heuristic=optimized,
        mega_history_order="auto" if optimized else "fifo", mega_num_comm_sm=args.num_comm_sm,
        mega_block_n=None, mega_phase_timestamps=True, baseline_phase_timing=True,
        warmup=args.warmup, iters=args.iters, cuda_graph=True, check=False, seed=args.seed,
    )


def _trace_rows(runner: DCPMegaAttentionRunner) -> list[list[int]]:
    destination = torch.empty_like(runner._cta_trace)
    runner.copy_last_cta_trace(destination)
    return [row for row in destination.cpu().tolist() if row[0] != 0 or row[1] != 0]


def _phase_summary(rows: list[list[int]]) -> dict[str, dict[str, int]]:
    phase_names = {0: "q_allgather", 1: "attention", 2: "history_combine", 3: "receive", 4: "final_combine"}
    result: dict[str, dict[str, int]] = {}
    for phase_id, name in phase_names.items():
        ends = sorted(row[1] for row in rows if row[6] == phase_id and row[7] > 0)
        if not ends:
            continue
        p50 = ends[(len(ends) - 1) // 2]
        p90 = ends[min(len(ends) - 1, round((len(ends) - 1) * 0.9))]
        result[name] = {"count": len(ends), "p50_useful_end_ns": p50, "p90_useful_end_ns": p90, "max_useful_end_ns": ends[-1], "tail_ns": ends[-1] - p50}
    return result


def _measure_vllm_a2a_graph(
    runner,
    inputs,
    runner_args: SimpleNamespace,
    device: torch.device,
) -> dict[str, Any]:
    call = lambda: runner_call(runner, inputs, runner_args, False)
    captured = capture_runner(runner, inputs, runner_args, False)
    try:
        for _ in range(runner_args.warmup):
            captured.replay()
        synchronize_before_samples(device)
        critical_samples: list[dict[str, Any]] = []
        for iteration in range(runner_args.iters):
            captured.replay()
            local_timing = runner.last_timing_ms(synchronize=True)
            # ``overlap_hidden_time_ms`` is a derived stage added by
            # ``measure_runner``.  The D1 baseline is intentionally measured
            # with ``overlap=False``, so no hidden overlap time exists, but
            # keep the common ``STAGES`` schema complete for aggregation.
            local_timing["overlap_hidden_time_ms"] = 0.0
            rank_timings: list[dict[str, float] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(rank_timings, local_timing)
            valid = [timing for timing in rank_timings if timing is not None]
            critical_rank = max(
                range(len(valid)),
                key=lambda rank: valid[rank]["attention_end_to_end_ms"],
            )
            critical_samples.append(
                {
                    "iteration": iteration,
                    "critical_rank": critical_rank,
                    "stages_ms": valid[critical_rank],
                }
            )
        summary = {
            stage: quantiles(
                [sample["stages_ms"][stage] for sample in critical_samples]
            )
            for stage in STAGES
        }
        return {
            "timing": summary,
            "critical_rank_samples": critical_samples,
            "execution": {
                "execution_mode": "cuda_graph",
                "post_capture_warmup": runner_args.warmup,
                "cuda_event_phase_timing_enabled": True,
                "stream_policy": "single_stream",
                "graph_static_signature": captured.signature,
                "timeline_sample_semantics": (
                    "same iteration and the rank with maximum end-to-end time"
                ),
            },
        }
    finally:
        captured.close()


def _run_case(
    case: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    dcp_group: DCPGroup,
    *,
    optimized: bool,
    baseline_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runner_args = _runner_args(case, args, optimized=optimized)
    topology = make_topology(args.qhead, args.kvhead, args.tp_size, args.dcp_size)
    inputs = build_inputs(runner_args, topology, device, materialize_reference=False)
    baseline_runners = (
        make_runner_set(
            dcp_group.process_group,
            ("vllm_a2a",),
            timed=True,
            varlen=True,
            phase_timing=True,
        )
        if baseline_report is None
        else {}
    )
    mega = DCPMegaAttentionRunner(
        dcp_group.process_group, dist.group.WORLD, max_total_q=sum(inputs.q_lengths),
        max_batch=len(inputs.q_lengths), Hq_local=topology.q_heads_local,
        num_comm_sm=args.num_comm_sm, record_phase_timestamps=True, record_cta_trace=True,
    )
    try:
        if baseline_report is None:
            baseline = next(iter(baseline_runners.values()))
            baseline_report = _measure_vllm_a2a_graph(
                baseline, inputs, runner_args, device
            )
        mega_q = mega.q_local(sum(inputs.q_lengths))
        mega_q.copy_(inputs.q_local)

        def mega_call():
            return mega.forward_chunk_prefill_varlen(
                mega_q, inputs.k_history_local, inputs.v_history_local, inputs.k_chunk, inputs.v_chunk,
                inputs.cu_q, inputs.cu_history_local, max(inputs.q_lengths), max(inputs.local_history_lengths),
                cu_seqlens_q_host=inputs.cu_q_host, cu_seqlens_history_local_host=inputs.cu_history_local_host,
                num_splits=0, scheduler_heuristic=optimized,
                reorder_history_override=None if optimized else False, return_lse=False,
            )

        mega_stages, mega_execution = measure_mega(mega, mega_call, runner_args, device, expected=None)
        local_trace = _trace_rows(mega)
        gathered: list[list[list[int]] | None] | None = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(local_trace, gathered, dst=0)
        if dist.get_rank() != 0:
            return {}
        trace_by_rank = {str(rank): rows for rank, rows in enumerate(gathered or []) if rows is not None}
        queue_counts = mega.last_queue_counts or {}
        expected_split = "critical_wave" if optimized else "fa3_native"
        expected_order = (
            "release_lpt"
            if optimized and case["workload_kind"] == "decode_only_q16"
            else "fifo"
        )
        if queue_counts.get("split_policy") != expected_split:
            raise RuntimeError(
                f"D1 policy mismatch for {case['case_id']}: "
                f"expected split={expected_split}, got {queue_counts.get('split_policy')}"
            )
        if queue_counts.get("history_order_policy") != expected_order:
            raise RuntimeError(
                f"D1 policy mismatch for {case['case_id']}: "
                f"expected history order={expected_order}, got {queue_counts.get('history_order_policy')}"
            )
        return {
            "case_id": case["case_id"], "workload_kind": case["workload_kind"],
            "optimized": optimized,
            "policy": "critical_wave_auto" if optimized else "fa3_native_fifo",
            "topology": topology.to_dict(),
            "dcp_groups": [list(group) for group in topology.all_dcp_groups()],
            "scheduler_policy": queue_counts.get("scheduler_policy"),
            "split_policy": queue_counts.get("split_policy"),
            "history_order_policy": queue_counts.get("history_order_policy"),
            "mega": {
                "timing": mega_stages, "execution": mega_execution, "queue_counts": queue_counts,
                "trace_schema": ["start_ns", "useful_work_end_ns", "exit_ns", "iteration", "cta_id", "sm_id", "phase_id", "valid_work_count"],
                "trace_by_rank": trace_by_rank,
                "phase_summary_by_rank": {rank: _phase_summary(rows) for rank, rows in trace_by_rank.items()},
            },
            "vllm_a2a_graph": baseline_report,
            "manifest": {
                "batch_size": case["batch_size"],
                "q_lengths": case["q_lengths"],
                "logical_q_lengths": case["logical_q_lengths"],
                "history_lengths": case["history_lengths"],
                "selection_metric": case.get("selection_metric"),
                "selection_metric_value": case.get("selection_metric_value"),
                "selection_median_value": case.get("selection_median_value"),
                "selection_distance": case.get("selection_distance"),
            },
        }
    finally:
        mega.close()
        dist.barrier()


def main() -> None:
    args = parse_args()
    rank, world_size, device = init_distributed_sm90("motivation D1")
    if world_size != args.tp_size:
        raise SystemExit("D1 requires torchrun world size equal to --tp-size")
    topology = _resolve_topology(args)
    args.dcp_size = topology.dcp_size
    cases = _load_manifest(args.case_manifest)
    dcp_group = make_dcp_group(args.dcp_size, device)
    records: list[dict[str, Any]] = []
    for case in cases:
        optimized_record = _run_case(
            case, args, device, dcp_group, optimized=True
        )
        baseline_holder: list[object] = [
            optimized_record.get("vllm_a2a_graph") if rank == 0 else None
        ]
        dist.broadcast_object_list(baseline_holder, src=0)
        if rank == 0:
            records.append(optimized_record)
        unoptimized_record = _run_case(
            case,
            args,
            device,
            dcp_group,
            optimized=False,
            baseline_report=baseline_holder[0],
        )
        if rank == 0:
            records.append(unoptimized_record)
    if rank == 0:
        output = {
            "schema_version": 3,
            "experiment": "D1_decode_sm_trace",
            "config": vars(args) | {"world_size": world_size, "rank": rank},
            "environment": environment(device),
            "trace_scope": "all DCP ranks gathered to rank 0; one trace buffer per rank and policy",
            "cases": records,
            "notes": [
                "Mega and vLLM A2A use the same frozen cases and CUDA Graph mode.",
                "Mega critical_wave_auto explicitly enables the critical-wave split heuristic; fa3_native_fifo explicitly disables it and forces FIFO history order.",
                "NCCL internal CTA/SM intervals are not fabricated; CTA traces cover the Mega persistent kernel only.",
            ],
        }
        write_json(args.output_json, output)
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as handle:
            for record in records:
                for rank_name, rows in record["mega"]["trace_by_rank"].items():
                    for row in rows:
                        handle.write(json.dumps({"case_id": record["case_id"], "optimized": record["optimized"], "rank": int(rank_name), **dict(zip(record["mega"]["trace_schema"], row))}) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
