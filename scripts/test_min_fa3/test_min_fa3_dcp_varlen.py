"""Eight-GPU correctness suite for packed-varlen DCP orchestration siblings."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

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


METHOD_OURS_OVERLAP = "ours_overlap_varlen"
METHOD_OURS_NO_OVERLAP = "ours_no_overlap_varlen"
METHOD_VLLM = "vllm_ag_rs_min_fa3_varlen"
METHOD_SGLANG = "sglang_mha_ag_ar_min_fa3_varlen"


@dataclass(frozen=True)
class DCPGroup:
    ranks: tuple[int, ...]
    process_group: dist.ProcessGroup


@dataclass
class PackedInputs:
    q_local: torch.Tensor
    k_history_full: torch.Tensor
    v_history_full: torch.Tensor
    k_history_local: torch.Tensor
    v_history_local: torch.Tensor
    k_chunk: torch.Tensor | None
    v_chunk: torch.Tensor | None
    k_reference: torch.Tensor
    v_reference: torch.Tensor
    q_lengths: list[int]
    history_lengths: list[int]
    history_lengths_local: list[int]
    reference_lengths: list[int]
    cu_q: torch.Tensor
    cu_q_host: torch.Tensor
    cu_history_local: torch.Tensor
    cu_history_local_host: torch.Tensor
    cu_reference: torch.Tensor
    cu_reference_host: torch.Tensor


def parse_int_list(spec: str, name: str) -> list[int]:
    try:
        values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    except ValueError as error:
        raise SystemExit(f"{name} must be a comma-separated integer list") from error
    if not values:
        raise SystemExit(f"{name} must contain at least one integer")
    return values


def parse_lengths(spec: str, batch_size: int, name: str) -> list[int]:
    values = parse_int_list(spec, name)
    if len(values) == 1:
        values *= batch_size
    if len(values) != batch_size:
        raise SystemExit(f"{name} must contain one value or exactly B={batch_size} values")
    if any(value <= 0 for value in values):
        raise SystemExit(f"{name} values must be positive")
    return values


def make_cu(lengths: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    host = torch.tensor(offsets, dtype=torch.int32)
    return host.to(device), host


def make_dcp_groups(sizes: Iterable[int], device: torch.device) -> dict[int, DCPGroup]:
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    local_groups: dict[int, DCPGroup] = {}
    for size in sorted(set(sizes)):
        if size <= 0 or size > world_size or world_size % size:
            raise SystemExit(f"DCP size {size} must divide world size {world_size}")
        for start in range(0, world_size, size):
            ranks = tuple(range(start, start + size))
            group = dist.new_group(list(ranks), backend="nccl", device_id=device)
            if global_rank in ranks:
                local_groups[size] = DCPGroup(ranks, group)
    return local_groups


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def randn_bf16(
    shape: tuple[int, ...], seed: int, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        shape,
        generator=make_generator(seed, device),
        device=device,
        dtype=torch.bfloat16,
    )


def shard_packed_interleaved(
    tensor: torch.Tensor,
    lengths: list[int],
    dcp_rank: int,
    dcp_size: int,
) -> tuple[torch.Tensor, list[int]]:
    pieces: list[torch.Tensor] = []
    local_lengths: list[int] = []
    start = 0
    for length in lengths:
        piece = tensor[start : start + length][dcp_rank::dcp_size]
        pieces.append(piece)
        local_lengths.append(piece.shape[0])
        start += length
    return torch.cat(pieces, dim=0).contiguous(), local_lengths


def append_packed_chunk(
    history: torch.Tensor,
    chunk: torch.Tensor,
    history_lengths: list[int],
    q_lengths: list[int],
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    history_start = 0
    chunk_start = 0
    for history_length, q_length in zip(history_lengths, q_lengths):
        pieces.append(history[history_start : history_start + history_length])
        pieces.append(chunk[chunk_start : chunk_start + q_length])
        history_start += history_length
        chunk_start += q_length
    return torch.cat(pieces, dim=0).contiguous()


def build_inputs(
    topology: DCPTopology,
    workload: str,
    q_lengths: list[int],
    history_lengths: list[int],
    num_splits: int,
    tp_rank: int,
    device: torch.device,
) -> PackedInputs:
    d = 128
    seed = (
        71023
        + topology.q_heads * 1009
        + topology.kv_heads * 503
        + topology.dcp_size * 211
        + sum(q_lengths) * 53
        + sum(history_lengths) * 7
        + num_splits * 17
        + (workload == "chunk")
    )
    q_local = randn_bf16(
        (sum(q_lengths), topology.q_heads_local, d),
        seed + tp_rank * 100_003,
        device,
    )
    kv_head = topology.kv_head_for_rank(tp_rank)
    k_history_full = randn_bf16(
        (sum(history_lengths), 1, d), seed + kv_head * 1_000_003, device
    )
    v_history_full = randn_bf16(
        (sum(history_lengths), 1, d), seed + kv_head * 1_000_003 + 1, device
    )
    dcp_rank = topology.dcp_rank(tp_rank)
    k_history_local, history_lengths_local = shard_packed_interleaved(
        k_history_full, history_lengths, dcp_rank, topology.dcp_size
    )
    v_history_local, v_lengths_local = shard_packed_interleaved(
        v_history_full, history_lengths, dcp_rank, topology.dcp_size
    )
    assert history_lengths_local == v_lengths_local
    if any(length <= 0 for length in history_lengths_local):
        raise RuntimeError("test case produced an empty local sequence")

    k_chunk: torch.Tensor | None = None
    v_chunk: torch.Tensor | None = None
    if workload == "chunk":
        k_chunk = randn_bf16(
            (sum(q_lengths), 1, d), seed + kv_head * 1_000_003 + 2, device
        )
        v_chunk = randn_bf16(
            (sum(q_lengths), 1, d), seed + kv_head * 1_000_003 + 3, device
        )
        k_reference = append_packed_chunk(
            k_history_full, k_chunk, history_lengths, q_lengths
        )
        v_reference = append_packed_chunk(
            v_history_full, v_chunk, history_lengths, q_lengths
        )
        reference_lengths = [
            history + query for history, query in zip(history_lengths, q_lengths)
        ]
    else:
        k_reference = k_history_full
        v_reference = v_history_full
        reference_lengths = list(history_lengths)

    cu_q, cu_q_host = make_cu(q_lengths, device)
    cu_history_local, cu_history_local_host = make_cu(history_lengths_local, device)
    cu_reference, cu_reference_host = make_cu(reference_lengths, device)
    return PackedInputs(
        q_local,
        k_history_full,
        v_history_full,
        k_history_local,
        v_history_local,
        k_chunk,
        v_chunk,
        k_reference,
        v_reference,
        q_lengths,
        history_lengths,
        history_lengths_local,
        reference_lengths,
        cu_q,
        cu_q_host,
        cu_history_local,
        cu_history_local_host,
        cu_reference,
        cu_reference_host,
    )


def reference(inputs: PackedInputs, workload: str, num_splits: int):
    return min_fa3_op.forward_kvcache_varlen(
        inputs.q_local,
        inputs.k_reference,
        inputs.v_reference,
        inputs.cu_q,
        inputs.cu_reference,
        max(inputs.q_lengths),
        max(inputs.reference_lengths),
        cu_seqlens_q_host=inputs.cu_q_host,
        cu_seqlens_k_host=inputs.cu_reference_host,
        num_splits=num_splits,
        return_lse=True,
        is_causal=workload == "chunk",
    )


def method_calls(
    workload: str,
    inputs: PackedInputs,
    runners: dict[str, DCPAttentionRunner],
    num_splits: int,
) -> list[tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]]:
    common = dict(
        cu_seqlens_q_host=inputs.cu_q_host,
        num_splits=num_splits,
        return_lse=True,
    )
    if workload == "decode":
        calls = []
        for method in (METHOD_OURS_OVERLAP, METHOD_VLLM, METHOD_SGLANG):
            runner = runners[method]
            calls.append(
                (
                    method,
                    lambda runner=runner: runner.forward_decode_varlen(
                        inputs.q_local,
                        inputs.k_history_local,
                        inputs.v_history_local,
                        inputs.cu_q,
                        inputs.cu_history_local,
                        max(inputs.q_lengths),
                        max(inputs.history_lengths_local),
                        cu_seqlens_k_local_host=inputs.cu_history_local_host,
                        **common,
                    ),
                )
            )
        return calls

    assert inputs.k_chunk is not None and inputs.v_chunk is not None
    calls = []
    for method, overlap in (
        (METHOD_OURS_OVERLAP, True),
        (METHOD_OURS_NO_OVERLAP, False),
        (METHOD_VLLM, False),
        (METHOD_SGLANG, False),
    ):
        runner = runners[method]
        calls.append(
            (
                method,
                lambda runner=runner, overlap=overlap: runner.forward_chunk_prefill_varlen(
                    inputs.q_local,
                    inputs.k_history_local,
                    inputs.v_history_local,
                    inputs.k_chunk,
                    inputs.v_chunk,
                    inputs.cu_q,
                    inputs.cu_history_local,
                    max(inputs.q_lengths),
                    max(inputs.history_lengths_local),
                    cu_seqlens_history_local_host=inputs.cu_history_local_host,
                    overlap_q_allgather=overlap,
                    **common,
                ),
            )
        )
    return calls


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    difference = (actual.float() - expected.float()).abs()
    max_error = difference.max().double()
    total_error = difference.sum(dtype=torch.float64)
    count = torch.tensor(difference.numel(), device=actual.device, dtype=torch.float64)
    close = torch.tensor(
        int(torch.isclose(actual.float(), expected.float(), atol=3e-2, rtol=3e-2).all()),
        device=actual.device,
        dtype=torch.int32,
    )
    for value, op in (
        (max_error, dist.ReduceOp.MAX),
        (total_error, dist.ReduceOp.SUM),
        (count, dist.ReduceOp.SUM),
        (close, dist.ReduceOp.MIN),
    ):
        dist.all_reduce(value, op=op)
    return {
        "max_abs_error": float(max_error.item()),
        "mean_abs_error": float((total_error / count).item()),
        "close": bool(close.item()),
    }


def check_result(
    method: str,
    result: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, object]:
    output, lse = result
    expected_output, expected_lse = expected
    if output.shape != expected_output.shape or output.dtype != torch.bfloat16:
        raise RuntimeError(f"{method}: invalid output contract {output.shape}, {output.dtype}")
    if lse.shape != expected_lse.shape or lse.dtype != torch.float32:
        raise RuntimeError(f"{method}: invalid LSE contract {lse.shape}, {lse.dtype}")
    output_report = error_stats(output, expected_output)
    lse_report = error_stats(lse, expected_lse)
    if not output_report["close"] or not lse_report["close"]:
        raise RuntimeError(
            f"{method} correctness failed: output={output_report}, lse={lse_report}"
        )
    return {"output": output_report, "lse": lse_report}


def run_case(
    topology: DCPTopology,
    workload: str,
    q_lengths: list[int],
    history_lengths: list[int],
    num_splits: int,
    runners: dict[str, DCPAttentionRunner],
    repeat: int,
    device: torch.device,
) -> dict[str, object]:
    inputs = build_inputs(
        topology,
        workload,
        q_lengths,
        history_lengths,
        num_splits,
        dist.get_rank(),
        device,
    )
    expected = reference(inputs, workload, num_splits)
    reports: dict[str, object] = {}
    for method, call in method_calls(workload, inputs, runners, num_splits):
        report = None
        if method == METHOD_OURS_OVERLAP and repeat > 1:
            caller_stream = torch.cuda.current_stream(device)
            alternate_stream = torch.cuda.Stream(device=device)
            queued = []
            for index in range(repeat):
                stream = caller_stream if index % 2 == 0 else alternate_stream
                with torch.cuda.stream(stream):
                    queued.append(call())
            caller_stream.wait_stream(alternate_stream)
            for result in queued:
                report = check_result(method, result, expected)
        else:
            for _ in range(repeat):
                report = check_result(method, call(), expected)
        reports[method] = report
    return {
        "topology": topology.to_dict(),
        "workload": workload,
        "q_lengths": q_lengths,
        "history_lengths": history_lengths,
        "local_history_lengths": inputs.history_lengths_local,
        "total_q": sum(q_lengths),
        "total_k_local": sum(inputs.history_lengths_local),
        "num_splits": num_splits,
        "methods": reports,
    }


def check_dense_parity(
    topology: DCPTopology,
    runners: dict[str, DCPAttentionRunner],
    device: torch.device,
) -> None:
    q_lengths = [8, 8, 8]
    history_lengths = [129, 129, 129]
    inputs = build_inputs(
        topology, "chunk", q_lengths, history_lengths, 1, dist.get_rank(), device
    )
    assert inputs.k_chunk is not None and inputs.v_chunk is not None
    b, sq = len(q_lengths), q_lengths[0]
    local_k = inputs.history_lengths_local[0]
    dense_lengths = torch.full((b,), local_k, device=device, dtype=torch.int32)
    for method, packed_call in method_calls("chunk", inputs, runners, 1):
        packed = packed_call()
        runner = runners[method]
        dense = runner.forward_chunk_prefill(
            inputs.q_local.view(b, sq, inputs.q_local.shape[1], 128),
            inputs.k_history_local.view(b, local_k, 1, 128),
            inputs.v_history_local.view(b, local_k, 1, 128),
            dense_lengths,
            inputs.k_chunk.view(b, sq, 1, 128),
            inputs.v_chunk.view(b, sq, 1, 128),
            num_splits=1,
            return_lse=True,
            overlap_q_allgather=method == METHOD_OURS_OVERLAP,
        )
        dense_out, dense_lse = dense
        expected = (
            dense_out.reshape_as(packed[0]),
            dense_lse.permute(1, 0, 2).reshape_as(packed[1]),
        )
        check_result(f"{method}_dense_parity", packed, expected)


def expect_error(name: str, expected: str, call: Callable[[], object]) -> None:
    try:
        call()
    except (RuntimeError, TypeError, ValueError) as error:
        if expected not in str(error):
            raise AssertionError(
                f"{name}: expected error containing {expected!r}, got {error!r}"
            ) from error
        return
    raise AssertionError(f"{name}: expected an exception")


def check_failures(
    topology: DCPTopology,
    runner: DCPAttentionRunner,
    device: torch.device,
) -> None:
    inputs = build_inputs(
        topology,
        "chunk",
        [1, 8, 32],
        [129, 258, 515],
        1,
        dist.get_rank(),
        device,
    )
    assert inputs.k_chunk is not None and inputs.v_chunk is not None

    def call(**overrides):
        values = dict(
            q_local=inputs.q_local,
            k_history_local=inputs.k_history_local,
            v_history_local=inputs.v_history_local,
            k_chunk=inputs.k_chunk,
            v_chunk=inputs.v_chunk,
            cu_seqlens_q=inputs.cu_q,
            cu_seqlens_history_local=inputs.cu_history_local,
            max_seqlen_q=max(inputs.q_lengths),
            max_seqlen_history_local=max(inputs.history_lengths_local),
            cu_seqlens_q_host=inputs.cu_q_host,
            cu_seqlens_history_local_host=inputs.cu_history_local_host,
            num_splits=1,
        )
        values.update(overrides)
        return runner.forward_chunk_prefill_varlen(**values)

    expect_error("q rank", "shape [total_tokens, H, 128]", lambda: call(q_local=inputs.q_local.unsqueeze(0)))
    expect_error("q dtype", "CUDA BF16", lambda: call(q_local=inputs.q_local.half()))
    expect_error("q device", "CUDA BF16", lambda: call(q_local=inputs.q_local.cpu()))
    noncontiguous = torch.empty(
        (inputs.q_local.shape[0], inputs.q_local.shape[1], 256),
        device=device,
        dtype=torch.bfloat16,
    )[:, :, ::2]
    expect_error("q contiguity", "must be contiguous", lambda: call(q_local=noncontiguous))
    expect_error("K/V shape", "identical shapes", lambda: call(v_history_local=inputs.v_history_local[:-1]))
    expect_error("chunk tokens", "k_chunk and v_chunk must have shape", lambda: call(k_chunk=inputs.k_chunk[:-1], v_chunk=inputs.v_chunk[:-1]))
    expect_error("cu dtype", "CUDA int32", lambda: call(cu_seqlens_q=inputs.cu_q.long()))
    expect_error("cu device", "CUDA int32", lambda: call(cu_seqlens_q=inputs.cu_q_host))
    expect_error("host device", "CPU int32", lambda: call(cu_seqlens_q_host=inputs.cu_q))
    expect_error("batch mismatch", "same batch size", lambda: call(cu_seqlens_history_local=inputs.cu_history_local[:-1]))
    bad_start = inputs.cu_q_host.clone()
    bad_start[0] = 1
    expect_error("cu start", "start with 0", lambda: call(cu_seqlens_q_host=bad_start))
    empty = inputs.cu_history_local_host.clone()
    empty[1] = empty[0]
    expect_error("local empty", "strictly increasing", lambda: call(cu_seqlens_history_local_host=empty))
    bad_end = inputs.cu_q_host.clone()
    bad_end[-1] += 1
    expect_error("cu endpoint", "total token count", lambda: call(cu_seqlens_q_host=bad_end))
    expect_error("max length", "max length", lambda: call(max_seqlen_q=max(inputs.q_lengths) + 1))
    expect_error("negative split", "num_splits", lambda: call(num_splits=-1))
    expect_error("large split", "num_splits", lambda: call(num_splits=129))

    expect_error(
        "decode mixed q",
        "every q_len == 1",
        lambda: runner.forward_decode_varlen(
            inputs.q_local,
            inputs.k_history_local,
            inputs.v_history_local,
            inputs.cu_q,
            inputs.cu_history_local,
            max(inputs.q_lengths),
            max(inputs.history_lengths_local),
            cu_seqlens_q_host=inputs.cu_q_host,
            cu_seqlens_k_local_host=inputs.cu_history_local_host,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Packed-varlen DCP correctness matrix on SM90."
    )
    parser.add_argument("--b", type=int, default=3)
    parser.add_argument("--sq", type=str, default="1,8,32")
    parser.add_argument("--seqlen", type=str, default="129,258,515")
    parser.add_argument("--qhead", type=str, default="32")
    parser.add_argument("--kvhead", type=str, default="1,2")
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--dcp-sizes", type=str, default="2,4,8")
    parser.add_argument("--num-splits", type=str, default="0,1,2,8")
    parser.add_argument("--repeat", type=int, default=2)
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
            raise SystemExit(f"--tp-size must equal torchrun world size {world_size}")
        if args.headdim != 128 or args.b <= 0 or args.repeat <= 0:
            raise SystemExit("--headdim must be 128 and B/repeat must be positive")

        q_lengths_chunk = parse_lengths(args.sq, args.b, "--sq")
        history_lengths = parse_lengths(args.seqlen, args.b, "--seqlen")
        q_lengths_decode = [1] * args.b
        q_heads = parse_int_list(args.qhead, "--qhead")
        kv_heads = parse_int_list(args.kvhead, "--kvhead")
        dcp_sizes = parse_int_list(args.dcp_sizes, "--dcp-sizes")
        split_values = parse_int_list(args.num_splits, "--num-splits")
        if any(split < 0 or split > 128 for split in split_values):
            raise SystemExit("--num-splits values must be in [0, 128]")

        groups = make_dcp_groups(dcp_sizes, device)
        topologies = [
            make_topology(hq, hkv, args.tp_size, dcp_size)
            for hq in q_heads
            for hkv in kv_heads
            for dcp_size in dcp_sizes
            if not validate_topology(hq, hkv, args.tp_size, dcp_size)
        ]
        runner_cache: dict[int, dict[str, DCPAttentionRunner]] = {}

        def runners_for(dcp_size: int) -> dict[str, DCPAttentionRunner]:
            if dcp_size not in runner_cache:
                group = groups[dcp_size].process_group
                ours = DCPAttentionRunner(group)
                runner_cache[dcp_size] = {
                    METHOD_OURS_OVERLAP: ours,
                    METHOD_OURS_NO_OVERLAP: ours,
                    METHOD_VLLM: VLLMDCPAttentionRunner(group),
                    METHOD_SGLANG: SGLangDCPAttentionRunner(group),
                }
            return runner_cache[dcp_size]

        cases = []
        for topology in topologies:
            if groups[topology.dcp_size].ranks != topology.dcp_group_ranks(global_rank):
                raise RuntimeError("DCP group crosses a KV replica boundary")
            runners = runners_for(topology.dcp_size)
            for num_splits in split_values:
                cases.append(
                    run_case(
                        topology,
                        "decode",
                        q_lengths_decode,
                        history_lengths,
                        num_splits,
                        runners,
                        1,
                        device,
                    )
                )
                cases.append(
                    run_case(
                        topology,
                        "chunk",
                        q_lengths_chunk,
                        history_lengths,
                        num_splits,
                        runners,
                        args.repeat if num_splits == split_values[-1] else 1,
                        device,
                    )
                )
            if global_rank == 0:
                print(
                    f"packed DCP ok: Hq={topology.q_heads} Hkv={topology.kv_heads} "
                    f"DCP={topology.dcp_size}",
                    flush=True,
                )

        parity_topology = topologies[-1]
        check_dense_parity(parity_topology, runners_for(parity_topology.dcp_size), device)
        failure_candidates = [topology for topology in topologies if topology.dcp_size == max(dcp_sizes)]
        if failure_candidates:
            failure_topology = failure_candidates[0]
            check_failures(
                failure_topology,
                runners_for(failure_topology.dcp_size)[METHOD_OURS_OVERLAP],
                device,
            )

        result = {
            "comparison_scope": (
                "Same min_fa3_op.forward_kvcache_varlen kernel; DCP orchestration only."
            ),
            "world_size": world_size,
            "cases": cases,
            "dense_parity": "ok",
            "failure_contracts": "ok",
        }
        if global_rank == 0:
            if args.output_json is not None:
                args.output_json.parent.mkdir(parents=True, exist_ok=True)
                args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"packed DCP correctness: ok ({len(cases)} cases)", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
