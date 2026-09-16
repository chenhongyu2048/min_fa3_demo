"""Motivation T1: communication/compute overlap on uniform all-CP inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RING_TEST = ROOT / "ring_test"
if str(RING_TEST) not in sys.path:
    sys.path.insert(0, str(RING_TEST))

import min_fa3_op
from motivation.allgather_control import AllGatherControl
from motivation.ring_control import zigzag_control
from motivation.common import cuda_barrier, environment, init_distributed_sm90, make_cu_seqlens, randn_bf16, require_homogeneous_devices, timed_call, write_json
from ring_common import RingComm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--mode", choices=("causal", "noncausal"), default="causal")
    parser.add_argument("--methods", default="ring,allgather,mega")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--num-comp-sm", type=int, default=0)
    parser.add_argument("--num-comm-sm", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--case-manifest", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _arena(k: torch.Tensor, v: torch.Tensor, rank: int, world_size: int):
    capacity = ((k.size(0) + 127) // 128) * 128
    remote_k = min_fa3_op.TKParallelTensor([world_size * capacity, k.size(1), 128], torch.bfloat16, rank, world_size, False)
    remote_v = min_fa3_op.TKParallelTensor([world_size * capacity, v.size(1), 128], torch.bfloat16, rank, world_size, False)
    remote_k.data_.zero_(); remote_v.data_.zero_()
    local_ks = [torch.empty_like(k) for _ in range(world_size)]
    local_vs = [torch.empty_like(v) for _ in range(world_size)]
    dist.all_gather(local_ks, k); dist.all_gather(local_vs, v)
    for source, (source_k, source_v) in enumerate(zip(local_ks, local_vs)):
        remote_k.data_[source * capacity:source * capacity + k.size(0)].copy_(source_k)
        remote_v.data_[source * capacity:source * capacity + v.size(0)].copy_(source_v)
    return remote_k, remote_v, capacity


def _communication_only(k: torch.Tensor, v: torch.Tensor) -> None:
    comm = RingComm(dist.group.WORLD)
    current_k, current_v = k, v
    for step in range(comm.world_size - 1):
        next_k, next_v = comm.send_recv_kv(current_k, current_v)
        comm.wait()
        current_k, current_v = next_k, next_v


def _min_fa3_block_varlen(
    q, k, v, cu_q, cu_k, cu_q_host, cu_k_host, max_q, max_k, causal
):
    result = min_fa3_op.forward_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        causal,
        cu_seqlens_q_host=cu_q_host,
        cu_seqlens_k_host=cu_k_host,
        return_lse=True,
    )
    return result[0], result[1]


def _sample_summary(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    return {
        "rank_max_ms": values,
        "p50_rank_max_ms": ordered[round((len(ordered) - 1) * 0.50)],
        "p90_rank_max_ms": ordered[round((len(ordered) - 1) * 0.90)],
    }


def _timed_corun(
    communication: Callable[[], object],
    compute: Callable[[], object],
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, object]:
    """Time a controlled communication/compute co-run on separate streams."""
    communication_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)

    def once() -> tuple[float, float, float]:
        cuda_barrier()
        gate = torch.cuda.Event(enable_timing=True)
        communication_start = torch.cuda.Event(enable_timing=True)
        communication_end = torch.cuda.Event(enable_timing=True)
        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        gate.record()
        host_gate = threading.Barrier(3)
        errors: list[BaseException] = []

        def worker(fn, stream, start, end) -> None:
            torch.cuda.set_device(device)
            host_gate.wait()
            try:
                with torch.cuda.stream(stream):
                    stream.wait_event(gate)
                    start.record()
                    fn()
                    end.record()
            except BaseException as error:
                errors.append(error)

        threads = (
            threading.Thread(
                target=worker,
                args=(communication, communication_stream, communication_start, communication_end),
            ),
            threading.Thread(
                target=worker,
                args=(compute, compute_stream, compute_start, compute_end),
            ),
        )
        for thread in threads:
            thread.start()
        host_gate.wait()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        current_stream = torch.cuda.current_stream(device)
        current_stream.wait_stream(communication_stream)
        current_stream.wait_stream(compute_stream)
        total_end.record(current_stream)
        total_end.synchronize()
        return (
            communication_start.elapsed_time(communication_end),
            compute_start.elapsed_time(compute_end),
            gate.elapsed_time(total_end),
        )

    for _ in range(warmup):
        once()
    samples = {"communication_ms": [], "compute_ms": [], "total_ms": []}
    for _ in range(iters):
        local = torch.tensor(once(), dtype=torch.float64, device=device)
        dist.all_reduce(local, op=dist.ReduceOp.MAX)
        for name, value in zip(samples, local.tolist()):
            samples[name].append(float(value))
    return {
        name: _sample_summary(values) for name, values in samples.items()
    } | {
        "timing_boundary": (
            "controlled co-run: pre-ready COMM-ONLY and COMP-ONLY replay on "
            "independent CUDA streams; per-stream events, rank-max per iteration"
        )
    }


def main() -> None:
    args = parse_args()
    input_case = None
    if args.case_manifest is not None:
        if args.case_id is None:
            raise SystemExit("--case-manifest requires --case-id")
        manifest = json.loads(args.case_manifest.read_text(encoding="utf-8"))
        if int(manifest["world_size"]) != int(os.environ["LOCAL_WORLD_SIZE"]):
            raise SystemExit("T1 case manifest world size does not match torchrun")
        matches = [case for case in manifest["cases"] if case["case_id"] == args.case_id]
        if len(matches) != 1:
            raise SystemExit(f"T1 case manifest does not contain exactly one {args.case_id}")
        input_case = matches[0]
        args.b = int(input_case["batch"])
        args.seqlen = int(input_case["local_seqlen"])
        args.case_id = str(matches[0]["case_id"])
        if int(input_case["context"]) != args.b * args.seqlen * int(manifest["world_size"]):
            raise SystemExit("T1 case manifest context does not match B*local_seqlen*CP")
    if args.headdim != 128 or args.qhead % args.kvhead:
        raise SystemExit("T1 requires D=128 and QH divisible by KVH")
    rank, world_size, device = init_distributed_sm90("motivation T1")
    sm_count = require_homogeneous_devices("motivation T1", world_size, device)
    if args.num_comp_sm <= 0:
        args.num_comp_sm = sm_count - args.num_comm_sm
    if args.num_comp_sm + args.num_comm_sm > sm_count:
        raise SystemExit("requested T1 SM allocation exceeds the visible device SM count")
    if args.seqlen % 2 and args.mode == "causal":
        raise SystemExit("causal T1 requires an even local sequence length")
    causal = args.mode == "causal"
    total = args.b * args.seqlen
    q = randn_bf16((total, args.qhead, args.headdim), args.seed + rank, device)
    k = randn_bf16((total, args.kvhead, args.headdim), args.seed + 1000 + rank, device)
    v = randn_bf16((total, args.kvhead, args.headdim), args.seed + 2000 + rank, device)
    cu, cu_host = make_cu_seqlens([args.seqlen] * args.b, device)
    remote_k, remote_v, capacity = _arena(k, v, rank, world_size)
    backend = "min_fa3"
    allgather = AllGatherControl(
        dist.group.WORLD, q, k, v, args.b, args.seqlen, causal, backend,
        heads_k_stride=1,
    )
    allgather_send_k = torch.empty(
        (total, 1, args.headdim), device=device, dtype=torch.bfloat16
    )
    allgather_send_v = torch.empty_like(allgather_send_k)
    allgather_receive_k = torch.empty(
        (world_size * total, 1, args.headdim),
        device=device,
        dtype=torch.bfloat16,
    )
    allgather_receive_v = torch.empty_like(allgather_receive_k)
    global_host = torch.tensor([args.seqlen * world_size] * args.b, dtype=torch.int32)
    ring_sizes = torch.full((args.b,), world_size, dtype=torch.int32)
    ring_starts = torch.zeros(args.b, dtype=torch.int32)

    def mega_call():
        return min_fa3_op.forward_varlen_mega_ring(
            q, remote_k.data_, remote_v.data_, cu, cu, args.seqlen, args.seqlen,
            causal, cu_seqlens_q_host=cu_host, cu_seqlens_k_host=cu_host,
            remote_k=remote_k, remote_v=remote_v, num_comp_sm=args.num_comp_sm,
            num_comm_sm=args.num_comm_sm, global_seqlens_host=global_host,
            ring_sizes_host=ring_sizes, ring_starts_host=ring_starts,
        )

    if not causal:
        raise SystemExit("motivation T1 uses causal zigzag inputs")
    preloaded = [(remote_k.data_[i * capacity:i * capacity + total],
                  remote_v.data_[i * capacity:i * capacity + total]) for i in range(world_size)]
    allgather.prepare_compute()

    def ring_call(*, overlap=True, compute_only=False):
        return zigzag_control(
            dist.group.WORLD, q, k, v, cu, cu_host, args.seqlen,
            _min_fa3_block_varlen, overlap=overlap,
            preloaded=preloaded if compute_only else None,
        )

    def allgather_call():
        return allgather.forward()

    def allgather_comm():
        for kv_head in range(args.kvhead):
            allgather_send_k.copy_(k[:, kv_head : kv_head + 1])
            allgather_send_v.copy_(v[:, kv_head : kv_head + 1])
            k_work = dist.all_gather_into_tensor(
                allgather_receive_k,
                allgather_send_k,
                async_op=True,
            )
            v_work = dist.all_gather_into_tensor(
                allgather_receive_v,
                allgather_send_v,
                async_op=True,
            )
            k_work.wait()
            v_work.wait()

    calls: dict[str, Callable[[], object]] = {}
    requested = {item.strip() for item in args.methods.split(",")}
    if "ring" in requested:
        calls.update({"ring_comm_only": lambda: _communication_only(k, v), "ring_comp_only": lambda: ring_call(compute_only=True), "ring_serial": lambda: ring_call(overlap=False), "ring_overlap": ring_call})
    if "allgather" in requested:
        calls.update({"allgather_comm_only": allgather_comm, "allgather_comp_only": lambda: allgather.forward(compute_only=True), "allgather_serial": lambda: allgather.forward(overlap=False), "allgather_overlap": allgather_call})
    if "mega" in requested:
        calls["mega_complete"] = mega_call
    reference = ring_call().clone()
    for name, fn in calls.items():
        if not name.endswith("comm_only"):
            torch.testing.assert_close(fn(), reference, atol=0.02, rtol=0.02)
    results = {name: timed_call(fn, args.warmup, args.iters, device) for name, fn in calls.items()}
    corun: dict[str, object] = {}
    if "ring" in requested:
        corun["ring"] = _timed_corun(
            lambda: _communication_only(k, v),
            lambda: ring_call(compute_only=True),
            args.warmup,
            args.iters,
            device,
        )
    if "allgather" in requested:
        corun["allgather"] = _timed_corun(
            allgather_comm,
            lambda: allgather.forward(compute_only=True),
            args.warmup,
            args.iters,
            device,
        )
    from motivation.config import provenance
    payload = {
        "schema": "motivation.v2.T1", **provenance(),
        "schema_version": 2,
        "experiment": "T1_training_overlap",
        "config": vars(args) | {"world_size": world_size, "rank_capacity": capacity, "fa_backend": backend, "input_mode": "uniform", "dataset": "cp_uniform", "input_case": input_case},
        "environment": environment(device),
        "timing_boundary": "CUDA events around complete requested call; rank-max per iteration",
        "results": results,
        "corun": corun,
        "notes": [
            "COMM/COMP controls are diagnostic replay controls; Mega is never split into fake pure modes.",
            "Co-run component durations are controlled resource-contention diagnostics, not a decomposition of the original ring/all-gather critical path.",
        ],
    }
    if rank == 0:
        write_json(args.output_json or Path("benchmark_logs/motivation/t1_training_overlap.json"), payload)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
