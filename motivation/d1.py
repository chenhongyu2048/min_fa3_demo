"""D1: fixed native task image, mixed/phased megakernels and vLLM A2A graphs."""

import argparse
from types import SimpleNamespace

from .config import PHASES, TRACE_FIELDS, add_sampling_args, d1_cases, d1_topology, validate_sampling


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_sampling_args(parser)
    parser.add_argument("--smoke", action="store_true", help="4-GPU TP=4, DCP=2, QH=16, KVH=2 smoke configuration")
    parser.add_argument("--check-rank-delay", action="store_true",
                        help="separate untuned diagnostic: delay rank 0 before a traced replay")
    args = parser.parse_args(argv)
    validate_sampling(args)

    import torch
    import torch.distributed as dist
    from dcp_test.benchmark_dcp_varlen import build_inputs
    from dcp_test.utils import initialize_distributed_sm90, make_dcp_group, make_runner_set
    from min_fa3_dcp import DCPMegaAttentionRunner, make_topology
    from .common import check_all_ranks, environment, measure, write_json
    from .trace import decode_trace

    device = initialize_distributed_sm90("motivation D1")
    rank = dist.get_rank()
    world = dist.get_world_size()
    topology_config = d1_topology(world, args.smoke)
    group = make_dcp_group(2, device)
    topology = make_topology(topology_config["q_heads"], topology_config["kv_heads"], world, 2)
    config = {**topology_config,
              "head_dim": 128, "num_comm_sm_mixed": 4, "split_policy": "fa3_native",
              "history_order_policy": "fifo", "dtype": "bfloat16",
              "return_lse": True, "execution": "CUDA Graph",
              "input_generator": "build_inputs(materialize_reference=False)"}
    results = []

    def check_result(actual, expected):
        def check():
            for actual_tensor, expected_tensor in zip(actual, expected):
                torch.testing.assert_close(actual_tensor, expected_tensor, atol=0.02, rtol=0.02)
        check_all_ranks(check)

    try:
        for case in d1_cases():
            if rank == 0:
                print(f'D1 {case["case_id"]}: preparing inputs', flush=True)
            query, history = case["q_lens"], case["history_lens"]
            input_args = SimpleNamespace(
                b=case["batch_size"], workload="chunk", sq=",".join(map(str, query)),
                seqlen=",".join(map(str, history)), qhead=config["q_heads"], kvhead=config["kv_heads"], headdim=128,
                tp_size=world, dcp_size=2, num_splits=0)
            inputs = build_inputs(input_args, topology, device, materialize_reference=False)
            forward_args = (
                inputs.q_local, inputs.k_history_local, inputs.v_history_local,
                inputs.k_chunk, inputs.v_chunk, inputs.cu_q, inputs.cu_history_local,
                max(query), max(inputs.local_history_lengths),
            )
            forward_kwargs = dict(
                cu_seqlens_q_host=inputs.cu_q_host,
                cu_seqlens_history_local_host=inputs.cu_history_local_host,
                num_splits=0, return_lse=True,
            )
            timings, expected = {}, None
            rank_data = {"rank": rank, "dcp_ranks": list(group.ranks), "case": case,
                         "trace_fields": TRACE_FIELDS, "phases": PHASES}
            for diagnostic in (False, True):
                if rank == 0:
                    print(f'D1 {case["case_id"]}: vllm_a2a events={diagnostic}', flush=True)
                runners = make_runner_set(
                    group.process_group, ("vllm_a2a",), timed=diagnostic,
                    varlen=True, phase_timing=diagnostic)
                baseline = next(iter(runners.values()))
                graph = baseline.capture_chunk_prefill_varlen(
                    *forward_args, **forward_kwargs, overlap_q_allgather=False, capture_warmup=3)
                try:
                    actual = graph.replay()
                    if expected is None:
                        expected = tuple(tensor.clone() for tensor in actual)
                    check_result(actual, expected)
                    timing, observations = measure(
                        graph.replay, device, args.warmup, args.iters,
                        observe=baseline.last_timing_ms if diagnostic else None)
                    key = "vllm_a2a_events_on" if diagnostic else "vllm_a2a_events_off"
                    timings[key] = timing
                    if diagnostic:
                        rank_data["vllm_phase_samples_ms"] = observations
                finally:
                    graph.close()
                del graph, baseline, runners

            reference_metadata = None
            for mode, trace in (("mixed", False), ("phased", False), ("phased", True)):
                if rank == 0:
                    print(f'D1 {case["case_id"]}: {mode} trace={trace}', flush=True)
                with DCPMegaAttentionRunner(
                    group.process_group, dist.group.WORLD,
                    max_total_q=sum(query), max_batch=len(query), Hq_local=4,
                    num_comm_sm=4, execution_mode=mode, record_sm_trace=trace,
                ) as runner:
                    q = runner.q_local(sum(query))
                    q.copy_(inputs.q_local)
                    actual = runner.forward_chunk_prefill_varlen(
                        q, *forward_args[1:], **forward_kwargs,
                        scheduler_heuristic=False, reorder_history_override=False)
                    check_result(actual, expected)
                    # Compare the actual serialized task image, not only counts.
                    # Header slots 19/20 are replay epochs, not task planning.
                    packed = runner._last_replay.backend_args
                    metadata = packed[27][:packed[29]].tolist()
                    metadata[19:21] = [0, 0]
                    counts = runner.last_queue_counts

                    def check_plan():
                        if counts["split_policy"] != "fa3_native" or counts["history_order_policy"] != "fifo":
                            raise ValueError("D1 requires native splits and FIFO ordering")
                        if reference_metadata is not None and metadata != reference_metadata:
                            raise ValueError("mixed/phased task images differ")
                    check_all_ranks(check_plan)
                    if reference_metadata is None:
                        reference_metadata = metadata
                        rank_data["metadata_image"] = metadata
                        rank_data["queue_counts"] = counts

                    graph = runner.capture_last_forward()
                    trace_host = torch.empty((6, runner.num_sms, 7), dtype=torch.int64) if trace else None

                    def observe_trace():
                        runner.copy_last_sm_trace(trace_host)
                        return trace_host.tolist()

                    try:
                        for _ in range(3):
                            check_result(graph.replay(), expected)
                        timing, observations = measure(
                            graph.replay, device, args.warmup, args.iters,
                            observe=observe_trace if trace else None)
                        key = f'{mode}_trace_{"on" if trace else "off"}'
                        timings[key] = timing
                        if trace:
                            rank_data["sm_trace_samples"] = observations
                            check_all_ranks(lambda: [decode_trace(sample, runner.num_sms)
                                                     for sample in observations])
                        check_result(graph.replay(), expected)
                    finally:
                        graph.close()
                    if trace and args.check_rank_delay:
                        # This separate graph omits only the outer pre-barrier: otherwise
                        # it absorbs the injected skew before the internal Q barrier.
                        graph = runner.capture_last_forward(run_pre_barrier=False)
                        try:
                            _, normal = measure(graph.replay, device, 0, 1, observe_trace)
                            def delayed_replay():
                                if rank == 0:
                                    torch.cuda._sleep(20_000_000)
                                return graph.replay()
                            _, delayed = measure(delayed_replay, device, 0, 1, observe_trace)
                            normal_wait = decode_trace(normal[0], runner.num_sms)[0]["rank_wait_ns"]
                            delayed_wait = decode_trace(delayed[0], runner.num_sms)[0]["rank_wait_ns"]
                            waits = [None] * world
                            dist.all_gather_object(waits, delayed_wait - normal_wait)
                            def check_delay():
                                if waits[1] <= max(1_000_000, max(waits[2:]) + 1_000_000):
                                    raise ValueError(f"rank 1 did not show isolated DCP wait: {waits}")
                            check_all_ranks(check_delay)
                            rank_data["delayed_sm_trace"] = delayed[0]
                            rank_data["rank_delay_wait_increment_ns"] = waits
                            check_result(graph.replay(), expected)
                        finally:
                            graph.close()
                del graph, runner, q, actual, packed

            case_id = case["case_id"]
            write_json(args.output_dir / f"{case_id}.rank{rank}.json", rank_data)
            results.append({"case_id": case_id, "case": case, "timings": timings})
            if rank == 0:
                for key, timing in timings.items():
                    print(f'D1 {case_id}/{key}: p50={timing["p50_ms"]:.6f} ms', flush=True)
        if rank == 0:
            write_json(args.output_dir / "d1.json", {
                "schema": "motivation.v3.D1", "config": config, "environment": environment(device),
                "warmup": args.warmup, "iters": args.iters, "records": results})
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
