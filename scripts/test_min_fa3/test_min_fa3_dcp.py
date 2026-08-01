"""Multi-GPU correctness matrix for production-shaped GQA DCP topologies."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.distributed as dist

import min_fa3_op
from min_fa3_dcp import (
    DCPAttentionRunner,
    DCPTopology,
    SGLangDCPAttentionRunner,
    VLLMDCPAttentionRunner,
    make_topology,
    validate_topology,
)


METHOD_OURS_OVERLAP = "ours_overlap"
METHOD_OURS_NO_OVERLAP = "ours_no_overlap"
METHOD_VLLM = "vllm_ag_rs_min_fa3"
METHOD_SGLANG = "sglang_mha_ag_ar_min_fa3"


@dataclass(frozen=True)
class DCPGroup:
    size: int
    start_rank: int
    ranks: tuple[int, ...]
    process_group: dist.ProcessGroup


@dataclass
class CorrectnessInputs:
    q_local: torch.Tensor
    k_history_full: torch.Tensor
    v_history_full: torch.Tensor
    k_history_local: torch.Tensor
    v_history_local: torch.Tensor
    history_lengths: list[int]
    history_lengths_local: torch.Tensor
    k_chunk: torch.Tensor | None = None
    v_chunk: torch.Tensor | None = None
    k_reference: torch.Tensor | None = None
    v_reference: torch.Tensor | None = None
    reference_lengths: torch.Tensor | None = None


def parse_int_list(spec: str, name: str) -> list[int]:
    try:
        values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    except ValueError as error:
        raise SystemExit(f"{name} must be a comma-separated integer list") from error
    if not values:
        raise SystemExit(f"{name} must contain at least one integer")
    return values


def make_dcp_groups(sizes: Iterable[int], device: torch.device) -> dict[int, DCPGroup]:
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    local_groups: dict[int, DCPGroup] = {}
    for size in sorted(set(sizes)):
        if size <= 0 or size > world_size or world_size % size:
            raise SystemExit(
                f"every DCP size must divide torchrun world size {world_size}, got {size}"
            )
        for start_rank in range(0, world_size, size):
            ranks = tuple(range(start_rank, start_rank + size))
            process_group = dist.new_group(
                list(ranks), backend="nccl", device_id=device
            )
            if global_rank in ranks:
                local_groups[size] = DCPGroup(
                    size, start_rank, ranks, process_group
                )
    return local_groups


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def randn_bf16(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)


def local_length(global_length: int, dcp_rank: int, dcp_size: int) -> int:
    return max(0, (global_length + dcp_size - 1 - dcp_rank) // dcp_size)


def make_full_chunk_cache(
    k_history: torch.Tensor,
    v_history: torch.Tensor,
    history_lengths: list[int],
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, sq, _, d = k_chunk.shape
    capacity = max(history_lengths) + sq
    k_full = torch.zeros(
        (b, capacity, 1, d), device=k_history.device, dtype=torch.bfloat16
    )
    v_full = torch.zeros_like(k_full)
    full_lengths: list[int] = []
    for batch_idx, history_length in enumerate(history_lengths):
        k_full[batch_idx, :history_length].copy_(
            k_history[batch_idx, :history_length]
        )
        v_full[batch_idx, :history_length].copy_(
            v_history[batch_idx, :history_length]
        )
        k_full[batch_idx, history_length : history_length + sq].copy_(
            k_chunk[batch_idx]
        )
        v_full[batch_idx, history_length : history_length + sq].copy_(
            v_chunk[batch_idx]
        )
        full_lengths.append(history_length + sq)
    return (
        k_full,
        v_full,
        torch.tensor(full_lengths, device=k_history.device, dtype=torch.int32),
    )


def build_inputs(
    topology: DCPTopology,
    workload: str,
    sq: int,
    num_splits: int,
    tp_rank: int,
    device: torch.device,
) -> CorrectnessInputs:
    b, d = 2, 128
    history_lengths = [257, 262] if workload == "decode" else [129, 258]
    history_capacity = max(history_lengths) + 1
    seed = (
        31013
        + topology.q_heads * 1009
        + topology.kv_heads * 503
        + topology.dcp_size * 211
        + sq * 53
        + num_splits * 17
        + (1 if workload == "chunk" else 0)
    )
    q_generator = make_generator(seed + tp_rank * 100_003, device)
    q_local = randn_bf16(
        (b, sq, topology.q_heads_local, d), q_generator, device
    )
    kv_head = topology.kv_head_for_rank(tp_rank)
    kv_generator = make_generator(seed + kv_head * 1_000_003, device)
    k_history_full = randn_bf16(
        (b, history_capacity, 1, d), kv_generator, device
    )
    v_history_full = randn_bf16(
        (b, history_capacity, 1, d), kv_generator, device
    )
    dcp_rank = topology.dcp_rank(tp_rank)
    k_history_local = k_history_full[:, dcp_rank::topology.dcp_size].contiguous()
    v_history_local = v_history_full[:, dcp_rank::topology.dcp_size].contiguous()
    history_lengths_local = torch.tensor(
        [
            local_length(length, dcp_rank, topology.dcp_size)
            for length in history_lengths
        ],
        device=device,
        dtype=torch.int32,
    )
    inputs = CorrectnessInputs(
        q_local=q_local,
        k_history_full=k_history_full,
        v_history_full=v_history_full,
        k_history_local=k_history_local,
        v_history_local=v_history_local,
        history_lengths=history_lengths,
        history_lengths_local=history_lengths_local,
    )
    if workload == "decode":
        inputs.k_reference = k_history_full
        inputs.v_reference = v_history_full
        inputs.reference_lengths = torch.tensor(
            history_lengths, device=device, dtype=torch.int32
        )
        return inputs

    k_chunk = randn_bf16((b, sq, 1, d), kv_generator, device)
    v_chunk = randn_bf16((b, sq, 1, d), kv_generator, device)
    inputs.k_chunk = k_chunk
    inputs.v_chunk = v_chunk
    (
        inputs.k_reference,
        inputs.v_reference,
        inputs.reference_lengths,
    ) = make_full_chunk_cache(
        k_history_full,
        v_history_full,
        history_lengths,
        k_chunk,
        v_chunk,
    )
    return inputs


def global_error_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, float | bool]:
    difference = (actual.float() - expected.float()).abs()
    max_error = difference.max().to(torch.float64)
    error_sum = difference.sum(dtype=torch.float64)
    count = torch.tensor(difference.numel(), device=actual.device, dtype=torch.float64)
    close = torch.tensor(
        int(torch.isclose(actual.float(), expected.float(), atol=atol, rtol=rtol).all()),
        device=actual.device,
        dtype=torch.int32,
    )
    dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
    dist.all_reduce(error_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    dist.all_reduce(close, op=dist.ReduceOp.MIN)
    return {
        "max_abs_error": float(max_error.item()),
        "mean_abs_error": float((error_sum / count).item()),
        "close": bool(close.item()),
    }


def check_result(
    method: str,
    output: torch.Tensor,
    lse: torch.Tensor,
    reference_output: torch.Tensor,
    reference_lse: torch.Tensor,
) -> dict[str, object]:
    if output.shape != reference_output.shape or output.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{method} output contract failed: shape={tuple(output.shape)}, "
            f"dtype={output.dtype}, expected_shape={tuple(reference_output.shape)}, "
            "expected_dtype=torch.bfloat16"
        )
    if lse.shape != reference_lse.shape or lse.dtype != torch.float32:
        raise RuntimeError(
            f"{method} LSE contract failed: shape={tuple(lse.shape)}, "
            f"dtype={lse.dtype}, expected_shape={tuple(reference_lse.shape)}, "
            "expected_dtype=torch.float32"
        )
    output_stats = global_error_stats(output, reference_output, 3.0e-2, 3.0e-2)
    lse_stats = global_error_stats(lse, reference_lse, 3.0e-3, 3.0e-3)
    if not output_stats["close"] or not lse_stats["close"]:
        raise RuntimeError(
            f"{method} correctness failed: output={output_stats}, lse={lse_stats}"
        )
    return {"output": output_stats, "lse": lse_stats}


def method_calls(
    workload: str,
    inputs: CorrectnessInputs,
    runners: dict[str, DCPAttentionRunner],
    num_splits: int,
) -> list[tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]]:
    ours = runners[METHOD_OURS_OVERLAP]
    vllm = runners[METHOD_VLLM]
    sglang = runners[METHOD_SGLANG]
    if workload == "decode":
        return [
            (
                METHOD_OURS_OVERLAP,
                lambda: ours.forward_decode(
                    inputs.q_local,
                    inputs.k_history_local,
                    inputs.v_history_local,
                    inputs.history_lengths_local,
                    num_splits=num_splits,
                    return_lse=True,
                ),
            ),
            (
                METHOD_VLLM,
                lambda: vllm.forward_decode(
                    inputs.q_local,
                    inputs.k_history_local,
                    inputs.v_history_local,
                    inputs.history_lengths_local,
                    num_splits=num_splits,
                    return_lse=True,
                ),
            ),
            (
                METHOD_SGLANG,
                lambda: sglang.forward_decode(
                    inputs.q_local,
                    inputs.k_history_local,
                    inputs.v_history_local,
                    inputs.history_lengths_local,
                    num_splits=num_splits,
                    return_lse=True,
                ),
            ),
        ]

    assert inputs.k_chunk is not None and inputs.v_chunk is not None
    return [
        (
            METHOD_OURS_OVERLAP,
            lambda: ours.forward_chunk_prefill(
                inputs.q_local,
                inputs.k_history_local,
                inputs.v_history_local,
                inputs.history_lengths_local,
                inputs.k_chunk,
                inputs.v_chunk,
                num_splits=num_splits,
                return_lse=True,
                overlap_q_allgather=True,
            ),
        ),
        (
            METHOD_OURS_NO_OVERLAP,
            lambda: ours.forward_chunk_prefill(
                inputs.q_local,
                inputs.k_history_local,
                inputs.v_history_local,
                inputs.history_lengths_local,
                inputs.k_chunk,
                inputs.v_chunk,
                num_splits=num_splits,
                return_lse=True,
                overlap_q_allgather=False,
            ),
        ),
        (
            METHOD_VLLM,
            lambda: vllm.forward_chunk_prefill(
                inputs.q_local,
                inputs.k_history_local,
                inputs.v_history_local,
                inputs.history_lengths_local,
                inputs.k_chunk,
                inputs.v_chunk,
                num_splits=num_splits,
                return_lse=True,
            ),
        ),
        (
            METHOD_SGLANG,
            lambda: sglang.forward_chunk_prefill(
                inputs.q_local,
                inputs.k_history_local,
                inputs.v_history_local,
                inputs.history_lengths_local,
                inputs.k_chunk,
                inputs.v_chunk,
                num_splits=num_splits,
                return_lse=True,
            ),
        ),
    ]


def run_case(
    topology: DCPTopology,
    workload: str,
    sq: int,
    num_splits: int,
    runners: dict[str, DCPAttentionRunner],
    repeat: int,
    device: torch.device,
) -> dict[str, object]:
    inputs = build_inputs(
        topology, workload, sq, num_splits, dist.get_rank(), device
    )
    assert inputs.k_reference is not None
    assert inputs.v_reference is not None
    assert inputs.reference_lengths is not None
    reference_output, reference_lse = min_fa3_op.forward_kvcache(
        inputs.q_local,
        inputs.k_reference,
        inputs.v_reference,
        inputs.reference_lengths,
        num_splits=num_splits,
        return_lse=True,
        is_causal=workload == "chunk",
    )
    methods: dict[str, object] = {}
    for method, call in method_calls(workload, inputs, runners, num_splits):
        stats: dict[str, object] | None = None
        if method == METHOD_OURS_NO_OVERLAP and repeat > 1:
            caller_stream = torch.cuda.current_stream(device)
            alternate_stream = torch.cuda.Stream(device=device)
            queued_results: list[tuple[torch.Tensor, torch.Tensor]] = []
            for repeat_idx in range(repeat):
                stream = caller_stream if repeat_idx % 2 == 0 else alternate_stream
                with torch.cuda.stream(stream):
                    queued_results.append(call())
            caller_stream.wait_stream(alternate_stream)
            for output, lse in queued_results:
                stats = check_result(
                    method, output, lse, reference_output, reference_lse
                )
        else:
            for _ in range(repeat):
                output, lse = call()
                stats = check_result(
                    method, output, lse, reference_output, reference_lse
                )
        assert stats is not None
        methods[method] = stats
    return {
        "status": "ok",
        "topology": topology.to_dict(),
        "workload": workload,
        "sq": sq,
        "ragged_history_lengths": inputs.history_lengths,
        "num_splits": num_splits,
        "repeat": repeat,
        "methods": methods,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-GPU correctness matrix for min FA3 DCP orchestrations."
    )
    parser.add_argument("--qhead", type=str, default="32,64")
    parser.add_argument("--kvhead", type=str, default="2,4")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--dcp-sizes", type=str, default="2,4,8")
    parser.add_argument("--sq", type=str, default="2,8,32,128")
    parser.add_argument("--num-splits", type=str, default="0,1,2")
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help=(
            "Repeat the largest chunk case across alternating compute streams "
            "to validate workspace reuse"
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise SystemExit("This test requires SM90 Hopper")
        if args.tp_size != world_size:
            raise SystemExit(
                f"--tp-size ({args.tp_size}) must equal torchrun world size ({world_size})"
            )
        if args.repeat <= 0:
            raise SystemExit("--repeat must be positive")

        q_heads = parse_int_list(args.qhead, "--qhead")
        kv_heads = parse_int_list(args.kvhead, "--kvhead")
        dcp_sizes = parse_int_list(args.dcp_sizes, "--dcp-sizes")
        sq_values = parse_int_list(args.sq, "--sq")
        split_values = parse_int_list(args.num_splits, "--num-splits")
        if any(value <= 0 for value in q_heads + kv_heads + sq_values):
            raise SystemExit("head counts and Sq values must be positive")
        if any(value < 0 or value > 128 for value in split_values):
            raise SystemExit("--num-splits values must be in [0, 128]")

        groups = make_dcp_groups(dcp_sizes, device)
        topologies: list[DCPTopology] = []
        skipped: list[dict[str, object]] = []
        for hq in q_heads:
            for hkv in kv_heads:
                for dcp_size in dcp_sizes:
                    issues = validate_topology(hq, hkv, args.tp_size, dcp_size)
                    if issues:
                        skipped.append(
                            {
                                "status": "skipped_topology",
                                "q_heads": hq,
                                "kv_heads": hkv,
                                "tp_size": args.tp_size,
                                "dcp_size": dcp_size,
                                "reasons": [issue.to_dict() for issue in issues],
                            }
                        )
                    else:
                        topologies.append(
                            make_topology(hq, hkv, args.tp_size, dcp_size)
                        )

        runner_cache: dict[int, dict[str, DCPAttentionRunner]] = {}

        def runners_for(dcp_size: int) -> dict[str, DCPAttentionRunner]:
            cached = runner_cache.get(dcp_size)
            if cached is None:
                group = groups[dcp_size].process_group
                cached = {
                    METHOD_OURS_OVERLAP: DCPAttentionRunner(group),
                    METHOD_VLLM: VLLMDCPAttentionRunner(group),
                    METHOD_SGLANG: SGLangDCPAttentionRunner(group),
                }
                runner_cache[dcp_size] = cached
            return cached

        cases: list[dict[str, object]] = []
        for topology in topologies:
            expected_group = topology.dcp_group_ranks(global_rank)
            actual_group = groups[topology.dcp_size].ranks
            if actual_group != expected_group:
                raise RuntimeError(
                    f"DCP group {actual_group} crosses the topology's KV replica "
                    f"boundary; expected {expected_group} for TP rank {global_rank}"
                )
            runners = runners_for(topology.dcp_size)
            for num_splits in split_values:
                cases.append(
                    run_case(
                        topology,
                        "decode",
                        1,
                        num_splits,
                        runners,
                        1,
                        device,
                    )
                )
                for sq in sq_values:
                    representative = (
                        topology == topologies[-1]
                        and num_splits == split_values[-1]
                        and sq == sq_values[-1]
                    )
                    cases.append(
                        run_case(
                            topology,
                            "chunk",
                            sq,
                            num_splits,
                            runners,
                            args.repeat if representative else 1,
                            device,
                        )
                    )
            if global_rank == 0:
                print(
                    f"DCP correctness ok: Hq={topology.q_heads} "
                    f"Hkv={topology.kv_heads} DCP={topology.dcp_size}",
                    flush=True,
                )

        result = {
            "comparison_scope": (
                "Same min_fa3_op.forward_kvcache kernel; DCP orchestration only."
            ),
            "world_size": world_size,
            "cases": cases,
            "skipped_topologies": skipped,
        }
        if global_rank == 0:
            if args.output_json is not None:
                args.output_json.parent.mkdir(parents=True, exist_ok=True)
                args.output_json.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(
                f"DCP correctness matrix: ok ({len(cases)} cases, "
                f"{len(topologies)} valid topologies, {len(skipped)} skipped)",
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
