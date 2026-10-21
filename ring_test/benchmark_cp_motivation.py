"""Dataset-shaped forward/backward controls for Megatron hybrid CP."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Sequence


DEMO_DIR = Path(__file__).resolve().parent.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import balancer


MODES = ("normal", "compute-only", "one-step")
PHASES = ("forward", "backward", "forward+backward")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(balancer.DATASET_WEIGHTS),
                        default=["arxiv", "freelaw", "github", "pile", "prolong"])
    parser.add_argument("--target-tokens", type=int, default=131072)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8),
                        default=int(os.environ.get("WORLD_SIZE", "8")))
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, choices=(128,), default=128)
    parser.add_argument("--megatron-max-seqlen-per-rank", type=int, default=8192)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=40)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--print-workload", action="store_true")
    parser.add_argument("--output", type=Path, help="Rank-0 JSONL measurements and summaries")
    args = parser.parse_args(argv)
    if min(args.target_tokens, args.num_cases, args.num_iters, args.qhead, args.kvhead,
           args.megatron_max_seqlen_per_rank) <= 0 or args.warmup_iters < 0:
        parser.error("sizes and iteration counts must be positive; warmup may be zero")
    if args.qhead % args.kvhead:
        parser.error("qhead must be divisible by kvhead")
    return args


def make_cases(args: argparse.Namespace) -> list[dict]:
    from baseline.megatron_hybrid_cp import build_hybrid_cp_plan_for_fa3_ring

    cases = []
    for dataset in args.datasets:
        lengths_list = balancer.generate_dataset_length_cases(
            dataset, args.target_tokens, args.seed, args.num_cases,
        )
        for case_index, lengths in enumerate(lengths_list):
            plan = build_hybrid_cp_plan_for_fa3_ring(
                lengths, args.world_size, True,
                max_seqlen_per_rank=args.megatron_max_seqlen_per_rank,
            )
            cases.append({
                "dataset": dataset, "case_index": case_index, "plan": plan,
                "global_lengths": lengths,
                "execution_lengths": list(plan.global_lengths),
                "original_tokens": sum(lengths), "execution_tokens": sum(plan.global_lengths),
                "cp_counts": dict(sorted(Counter(a.cp_size for a in plan.assignments).items())),
                "execution_groups": plan.num_execution_groups,
                "rank_samples": [list(plan.sample_ids_for_rank(rank)) for rank in range(args.world_size)],
                "assignments": [
                    {"sample_id": a.sample_id, "cp_size": a.cp_size,
                     "rank_start": a.rank_start, "group": a.execution_group_id}
                    for a in plan.assignments
                ],
            })
    return cases


def measure(runner, phase: str, args: argparse.Namespace) -> dict:
    import torch
    import torch.distributed as dist

    def prepare() -> None:
        if phase != "forward" and hasattr(runner, "prepare_backward"):
            runner.prepare_backward()
        if phase == "backward":
            runner.forward_all()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()

    def launch() -> None:
        if phase != "backward":
            runner.forward_all()
        if phase != "forward":
            runner.backward_all()

    samples = []
    for iteration in range(args.warmup_iters + args.num_iters):
        prepare()
        start = time.perf_counter()
        launch()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if iteration >= args.warmup_iters:
            samples.append(elapsed)
    local = torch.tensor(samples, device="cuda", dtype=torch.float64)
    parts = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, local)
    per_rank = [part.tolist() for part in parts]
    maxima = [max(values) for values in zip(*per_rank)]
    return {"latency_ms": mean(maxima), "max_rank_samples_ms": maxima,
            "rank_mean_ms": [mean(values) for values in per_rank]}


def check_runner(runner, expected: tuple | None, args: argparse.Namespace) -> tuple:
    import torch
    from ring_test.ring_common import raise_if_any_rank_failed

    if hasattr(runner, "prepare_backward"):
        runner.prepare_backward()
    out = runner.forward_all()
    gradients = runner.backward_all()
    if hasattr(runner, "reduce_dkv_for_check"):
        gradients = runner.reduce_dkv_for_check()
    actual = (out, runner.lse, *gradients)
    error = None
    if expected is not None:
        try:
            for label, value, reference in zip(("O", "LSE", "dQ", "dK", "dV"), actual, expected):
                torch.testing.assert_close(value, reference, atol=args.atol, rtol=args.rtol, msg=label)
        except AssertionError as exc:
            error = str(exc)
    raise_if_any_rank_failed(error, runner.process_groups.world_group)
    return tuple(value.clone() for value in actual) if expected is None else expected


def summarize(records: list[dict]) -> list[dict]:
    summaries = []
    for dataset in dict.fromkeys(row["dataset"] for row in records):
        for phase in PHASES:
            times = {
                mode: mean(row["latency_ms"] for row in records
                           if row["dataset"] == dataset and row["phase"] == phase and row["mode"] == mode)
                for mode in MODES
            }
            summaries.append({
                "type": "summary", "dataset": dataset, "phase": phase,
                "mean_ms": times,
                "normal_over_compute_only": times["normal"] / times["compute-only"],
                "compute_only_over_one_step": times["compute-only"] / times["one-step"],
            })
    return summaries


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cases = make_cases(args)
    if args.print_workload:
        for case in cases:
            print(json.dumps({key: value for key, value in case.items() if key != "plan"}))
        return

    import torch
    import torch.distributed as dist
    from baseline.megatron_hybrid_cp import (
        MegatronHybridCPAttention, create_hybrid_cp_process_groups, make_packed_hybrid_cp_inputs,
    )
    from baseline.megatron_hybrid_cp.motivation import MegatronHybridCPComputeAttention
    from allgather_attention import select_fa3_backend

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    if dist.get_world_size() != args.world_size:
        raise ValueError("--world-size must match the torchrun world size")
    groups = create_hybrid_cp_process_groups(dist.group.WORLD)
    backend = select_fa3_backend(dist.group.WORLD, require_backward=True)
    output = None
    records = []
    try:
        if rank == 0 and args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = args.output.open("w", encoding="utf-8")
        environment = {
            "type": "config", "world_size": args.world_size, "backend": backend,
            "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "datasets": args.datasets, "seed": args.seed, "num_cases": args.num_cases,
            "target_tokens": args.target_tokens, "qhead": args.qhead, "kvhead": args.kvhead,
            "headdim": args.headdim, "max_seqlen_per_rank": args.megatron_max_seqlen_per_rank,
            "warmup_iters": args.warmup_iters, "num_iters": args.num_iters, "check": args.check,
            "atol": args.atol, "rtol": args.rtol,
            "timing": "mean of per-iteration maximum rank wall time; CUDA synchronized",
        }

        def emit(record: dict) -> None:
            if rank == 0:
                print(json.dumps(record), flush=True)
                if output is not None:
                    output.write(json.dumps(record) + "\n")
                    output.flush()

        emit(environment)
        with torch.no_grad():
            for case in cases:
                plan = case["plan"]
                inputs = make_packed_hybrid_cp_inputs(
                    plan, rank, args.qhead, args.kvhead, args.headdim,
                    torch.device("cuda", torch.cuda.current_device()),
                    is_causal=True, seed=args.seed + case["case_index"], with_dout=True,
                )
                expected = None
                for mode in MODES:
                    runner_type = MegatronHybridCPAttention if mode == "normal" else MegatronHybridCPComputeAttention
                    kwargs = {} if mode == "normal" else {"one_step": mode == "one-step"}
                    runner = runner_type(
                        plan, groups, inputs.q, inputs.k, inputs.v, True, backend,
                        dout=inputs.dout, **kwargs,
                    )
                    if args.check:
                        expected = check_runner(runner, expected, args)
                    for phase in PHASES:
                        timing = measure(runner, phase, args)
                        record = {"type": "measurement", **{k: v for k, v in case.items() if k != "plan"},
                                  "mode": mode, "phase": phase, "check_passed": args.check, **timing}
                        records.append(record)
                        emit(record)
                    del runner
                del expected, inputs
        for summary in summarize(records):
            emit(summary)
        if rank == 0:
            print(f"{'Dataset':<10} {'Phase':<18} {'Normal ms':>10} {'Compute ms':>12} {'One-step ms':>12} {'Norm/comp':>9} {'Comp/one':>9}")
            for row in summarize(records):
                times = row["mean_ms"]
                print(f"{row['dataset']:<10} {row['phase']:<18} {times['normal']:>10.3f} "
                      f"{times['compute-only']:>12.3f} {times['one-step']:>12.3f} "
                      f"{row['normal_over_compute_only']:>9.3f} {row['compute_only_over_one_step']:>9.3f}")
    finally:
        if output is not None:
            output.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
