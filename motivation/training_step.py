"""Motivation T2: compare ring-step execution with one continuous queue.

This entry point reuses the existing forward-ablation profiles.  It deliberately
keeps the scope to the two motivation profiles: step_fused_reduce and
linear_queue_recycle.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import min_fa3_op
from motivation.common import cuda_barrier, environment, init_distributed_sm90, randn_bf16, require_homogeneous_devices, timed_call, write_json
from ring_test.forward_ablation import ForwardAblationPlan, make_cu_seqlens as make_ablation_cu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--num-comp-sm", type=int, default=0)
    parser.add_argument("--num-comm-sm", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--case-manifest", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _make_arena(local_k: torch.Tensor, local_v: torch.Tensor, rank: int, world_size: int):
    capacity = ((local_k.size(0) + 127) // 128) * 128
    remote_k = min_fa3_op.TKParallelTensor([world_size * capacity, local_k.size(1), 128], torch.bfloat16, rank, world_size, False)
    remote_v = min_fa3_op.TKParallelTensor([world_size * capacity, local_v.size(1), 128], torch.bfloat16, rank, world_size, False)
    remote_k.data_.zero_()
    remote_v.data_.zero_()
    remote_k.data_[rank * capacity : rank * capacity + local_k.size(0)].copy_(local_k)
    remote_v.data_[rank * capacity : rank * capacity + local_v.size(0)].copy_(local_v)
    gathered_k = [torch.empty_like(remote_k.data_[rank * capacity : rank * capacity + local_k.size(0)]) for _ in range(world_size)]
    gathered_v = [torch.empty_like(gathered_k[0]) for _ in range(world_size)]
    dist.all_gather(gathered_k, local_k)
    dist.all_gather(gathered_v, local_v)
    for source, (source_k, source_v) in enumerate(zip(gathered_k, gathered_v)):
        remote_k.data_[source * capacity : source * capacity + source_k.size(0)].copy_(source_k)
        remote_v.data_[source * capacity : source * capacity + source_v.size(0)].copy_(source_v)
    return remote_k, remote_v, capacity


def main() -> None:
    args = parse_args()
    input_case = None
    if args.case_manifest is not None:
        if args.case_id is None:
            raise SystemExit("--case-manifest requires --case-id")
        manifest = json.loads(args.case_manifest.read_text(encoding="utf-8"))
        if int(manifest["world_size"]) != int(os.environ["LOCAL_WORLD_SIZE"]):
            raise SystemExit("T2 case manifest world size does not match torchrun")
        matches = [case for case in manifest["cases"] if case["case_id"] == args.case_id]
        if len(matches) != 1:
            raise SystemExit(f"T2 case manifest does not contain exactly one {args.case_id}")
        input_case = matches[0]
        args.b = int(input_case["batch"])
        args.seqlen = int(input_case["local_seqlen"])
        if int(input_case["context"]) != args.b * args.seqlen * int(manifest["world_size"]):
            raise SystemExit("T2 case manifest context does not match B*local_seqlen*CP")
    if args.headdim != 128 or args.kvhead != 8 or args.qhead % args.kvhead:
        raise SystemExit("T2 currently requires QH divisible by KVH=8 and D=128")
    rank, world_size, device = init_distributed_sm90("motivation T2")
    if world_size not in (2, 4, 8):
        raise SystemExit("motivation T2 requires CP world size 2, 4, or 8")
    sm_count = require_homogeneous_devices("motivation T2", world_size, device)
    if args.num_comp_sm <= 0:
        args.num_comp_sm = sm_count - args.num_comm_sm
    if args.num_comp_sm + args.num_comm_sm > sm_count:
        raise SystemExit(
            f"requested COMP:COMM={args.num_comp_sm}:{args.num_comm_sm} exceeds "
            f"the device SM count {sm_count}; use --num-comp-sm 0 for auto"
        )
    if args.seqlen % 256:
        raise SystemExit("--seqlen must be divisible by 256 for causal CP step ablation")
    lengths = [args.seqlen] * args.b
    q = randn_bf16((args.b * args.seqlen, args.qhead, args.headdim), args.seed + rank, device)
    local_k = randn_bf16((args.b * args.seqlen, args.kvhead, args.headdim), args.seed + 1000 + rank, device)
    local_v = randn_bf16((args.b * args.seqlen, args.kvhead, args.headdim), args.seed + 2000 + rank, device)
    cu, cu_host = make_ablation_cu(lengths, device)
    remote_k, remote_v, capacity = _make_arena(local_k, local_v, rank, world_size)
    results: dict[str, object] = {}
    for profile_name in (
        "step_external_reduce",
        "step_fused_reduce",
        "linear_queue_recycle",
    ):
        plan = ForwardAblationPlan(
            q, remote_k, remote_v, cu, cu_host, args.seqlen,
            [args.seqlen * world_size] * args.b,
            [world_size] * args.b,
            [0] * args.b,
            profile_name, num_comp_sm=args.num_comp_sm, num_comm_sm=args.num_comm_sm,
            collect_stats=True, compute_only=True,
        )
        timing = timed_call(plan.run, args.warmup, args.iters, device)
        probe = plan.probe()
        results[profile_name] = {"timing": timing, "stats": probe}
        del plan
        cuda_barrier()
    work_signatures = {
        (
            result["stats"]["qo_visits"],
            result["stats"]["kv_tile_reads"],
        )
        for result in results.values()
    }
    if len(work_signatures) != 1:
        raise RuntimeError(
            "T2 profiles do not report the same Q/O visits and KV tile reads"
        )
    payload = {
        "schema_version": 2,
        "experiment": "T2_ring_step_compute",
        "config": vars(args) | {"world_size": world_size, "rank_capacity": capacity, "input_mode": "uniform", "input_case": input_case},
        "environment": environment(device),
        "timing_boundary": "preloaded full K/V arena; compute-only profile call",
        "results": results,
    }
    if rank == 0:
        write_json(args.output_json or Path("benchmark_logs/motivation/t2_training_step.json"), payload)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
