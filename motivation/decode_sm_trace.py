"""D1: uninstrumented Graph comparisons and separate phase diagnostics."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .config import D1_SELECTION_RULE, d1_config, provenance


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, choices=(4, 8), required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist
    from dcp_test.benchmark_dcp_varlen import build_inputs, capture_runner
    from dcp_test.utils import make_dcp_group, make_runner_set
    from min_fa3_dcp import DCPMegaAttentionRunner, make_topology
    from .common import init_distributed_sm90, require_homogeneous_devices, timed_call, environment

    rank, world, device = init_distributed_sm90("motivation D1")
    if world != args.gpus:
        raise SystemExit("--gpus must match torchrun world size")
    require_homogeneous_devices("motivation D1", world, device)
    config = d1_config(world)
    topology = make_topology(32, config["kvhead"], world, 2)
    group = make_dcp_group(2, device)
    manifest = json.loads(args.case_manifest.read_text())
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output = args.output_jsonl.open("x") if rank == 0 else None
    origin = provenance()
    try:
        for case in manifest["cases"][:args.case_limit]:
            query = case["q_lengths"]
            history = case["history_lengths"]
            runner_args = SimpleNamespace(b=len(query), workload="chunk",
                sq=",".join(map(str, query)), seqlen=",".join(map(str, history)),
                qhead=32, kvhead=config["kvhead"], headdim=128, tp_size=world,
                dcp_size=2, num_splits=0)
            inputs = build_inputs(runner_args, topology, device, materialize_reference=False)

            def baseline(diagnostic=False):
                runners = make_runner_set(group.process_group, ("vllm_a2a",),
                                          timed=diagnostic, varlen=True, phase_timing=diagnostic)
                runner = next(iter(runners.values()))
                graph = capture_runner(runner, inputs, runner_args, False)
                try:
                    expected = graph.replay().clone()
                    for _ in range(3):
                        torch.testing.assert_close(graph.replay(), expected, atol=0.02, rtol=0.02)
                    if diagnostic:
                        graph.replay()
                        local = runner.last_timing_ms(synchronize=True)
                        gathered = [None] * world
                        dist.all_gather_object(gathered, local)
                        critical = max(range(world), key=lambda r: gathered[r]["attention_end_to_end_ms"])
                        return {"critical_rank": critical, "stages_ms": gathered[critical],
                                "semantics": "one replay, same critical rank; CUDA events, not per-SM trace"}
                    timing = timed_call(graph.replay, args.warmup, args.iters, device)
                    return expected, timing
                finally:
                    graph.close()

            expected, baseline_timing = baseline()

            def mega(optimized, trace=False):
                runner = DCPMegaAttentionRunner(group.process_group, dist.group.WORLD,
                    max_total_q=sum(query), max_batch=len(query), Hq_local=32 // world,
                    num_comm_sm=4, record_phase_timestamps=False, record_cta_trace=trace)
                graph = None
                try:
                    q = runner.q_local(sum(query))
                    q.copy_(inputs.q_local)
                    runner.forward_chunk_prefill_varlen(
                        q, inputs.k_history_local, inputs.v_history_local, inputs.k_chunk, inputs.v_chunk,
                        inputs.cu_q, inputs.cu_history_local, max(query), max(inputs.local_history_lengths),
                        cu_seqlens_q_host=inputs.cu_q_host,
                        cu_seqlens_history_local_host=inputs.cu_history_local_host,
                        scheduler_heuristic=optimized, reorder_history_override=None if optimized else False)
                    graph = runner.capture_last_forward()
                    for _ in range(3):
                        torch.testing.assert_close(graph.replay(), expected, atol=0.02, rtol=0.02)
                    counts = runner.last_queue_counts
                    wanted = "critical_wave" if optimized else "fa3_native"
                    if counts["split_policy"] != wanted or (not optimized and counts["history_order_policy"] != "fifo"):
                        raise RuntimeError("D1 scheduler control does not match requested policy")
                    timing = timed_call(graph.replay, 2 if trace else args.warmup,
                                        3 if trace else args.iters, device)
                    result = {"timing": timing, "queue_counts": counts, "trace_enabled": trace}
                    if trace:
                        graph.replay()
                        copied = torch.empty_like(runner._cta_trace)
                        runner.copy_last_cta_trace(copied)
                        rows = [row for row in copied.cpu().tolist() if row[1] > 0]
                        gathered = [None] * world
                        dist.all_gather_object(gathered, rows)
                        result.update(trace_by_rank={str(i): rows for i, rows in enumerate(gathered)},
                                      trace_schema=["start_ns", "end_ns", "cta_id", "sm_id", "phase_id"],
                                      trace_semantics="thread-zero phase checkpoints; may include wait/synchronization; no useful-work counts")
                    return result
                finally:
                    if graph is not None:
                        graph.close()
                    runner.close()

            # Alternate scheduler order between independent runs without
            # selecting a different resource configuration for each case.
            modes = (True, False) if sum(args.run_id.encode()) % 2 == 0 else (False, True)
            results = {"vllm_a2a": {"timing": baseline_timing, "trace_enabled": False}}
            for optimized in modes:
                results["critical_wave" if optimized else "fa3_native"] = mega(optimized)
            diagnostics = {}
            if args.trace:
                diagnostics["vllm_a2a"] = baseline(diagnostic=True)
                for optimized in modes:
                    diagnostics["critical_wave" if optimized else "fa3_native"] = mega(optimized, trace=True)
            record = {"schema": "motivation.v2.D1", **origin, "run_id": args.run_id,
                      "config": config, "topology": topology.to_dict(), "environment": environment(device),
                      "case_id": case["case_id"], "workload_kind": case["workload_kind"],
                      "case": case, "warmup": args.warmup, "iters": args.iters,
                      "selection_status": manifest.get("selection_status", "pending_gpu_measurement"),
                      "selection_rule": manifest.get("selection_rule", D1_SELECTION_RULE),
                      "execution_mode": "cuda_graph", "results": results, "diagnostics": diagnostics}
            if output:
                output.write(json.dumps(record) + "\n")
                output.flush()
            del expected, inputs
            torch.cuda.empty_cache()
        dist.destroy_process_group()
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    main()
