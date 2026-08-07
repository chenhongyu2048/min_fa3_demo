"""Multi-rank correctness test for the batched varlen DCP mega forward path.

Run with, for example:

    torchrun --standalone --nproc-per-node=8 \
        scripts/test_min_fa3/test_dcp_mega_varlen_multi_rank.py --dcp-size 2
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist

import min_fa3_op
from min_fa3_dcp import DCPMegaAttentionRunner


@dataclass(frozen=True)
class CorrectnessCase:
    name: str
    dcp_size: int
    q_lengths: tuple[int, ...]
    history_lengths: tuple[int, ...]
    hq_local: int
    num_splits: int
    block_n: int
    repeat: int
    num_comm_sm: int


def parse_lengths(value: str, name: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise SystemExit(f"{name} must be a comma-separated integer list") from error
    if not result or any(length <= 0 for length in result):
        raise SystemExit(f"{name} must contain positive lengths")
    return result


def make_cu(lengths: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    host = torch.tensor(offsets, dtype=torch.int32)
    return host.to(device), host


def randn_bf16(
    shape: tuple[int, ...], seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)


def shard_history(
    tensor: torch.Tensor,
    lengths: list[int],
    dcp_rank: int,
    dcp_size: int,
) -> tuple[torch.Tensor, list[int]]:
    pieces: list[torch.Tensor] = []
    local_lengths: list[int] = []
    offset = 0
    for length in lengths:
        piece = tensor[offset : offset + length][dcp_rank::dcp_size]
        pieces.append(piece)
        local_lengths.append(piece.shape[0])
        offset += length
    return torch.cat(pieces).contiguous(), local_lengths


def append_chunk(
    history: torch.Tensor,
    chunk: torch.Tensor,
    history_lengths: list[int],
    q_lengths: list[int],
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    history_offset = 0
    q_offset = 0
    for history_length, q_length in zip(history_lengths, q_lengths):
        pieces.append(history[history_offset : history_offset + history_length])
        pieces.append(chunk[q_offset : q_offset + q_length])
        history_offset += history_length
        q_offset += q_length
    return torch.cat(pieces).contiguous()


def make_local_group(
    dcp_size: int,
    device: torch.device,
) -> tuple[dist.ProcessGroup, int, int]:
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    if world_size % dcp_size:
        raise SystemExit(f"DCP size {dcp_size} must divide world size {world_size}")
    local_group = None
    local_rank = -1
    group_index = -1
    for start in range(0, world_size, dcp_size):
        ranks = list(range(start, start + dcp_size))
        group = dist.new_group(ranks, backend="nccl", device_id=device)
        if global_rank in ranks:
            local_group = group
            local_rank = global_rank - start
            group_index = start // dcp_size
    assert local_group is not None
    return local_group, local_rank, group_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DCP mega packed-varlen correctness")
    parser.add_argument("--dcp-size", type=int, default=2, choices=(2, 4, 8))
    parser.add_argument("--q-lengths", default="1,8,32")
    parser.add_argument("--history-lengths", default="129,258,515")
    parser.add_argument("--hq-local", type=int, default=4)
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--block-n", type=int, default=128, choices=(128, 176))
    parser.add_argument("--num-comm-sm", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--test-phase-wrap",
        action="store_true",
        help="Force the first forward through the synchronized int32 phase reset",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run the fixed six-case DCP mega correctness matrix",
    )
    return parser.parse_args()


def run_case(
    case: CorrectnessCase,
    device: torch.device,
    test_phase_wrap: bool,
) -> None:
    dcp_group, dcp_rank, group_index = make_local_group(case.dcp_size, device)
    q_lengths = list(case.q_lengths)
    history_lengths = list(case.history_lengths)
    total_q = sum(q_lengths)
    runner = DCPMegaAttentionRunner(
        dcp_group,
        dist.group.WORLD,
        max_total_q=total_q,
        max_batch=len(q_lengths),
        Hq_local=case.hq_local,
        max_num_splits=128,
        num_comm_sm=case.num_comm_sm,
        block_n_override=case.block_n,
        record_phase_timestamps=True,
    )
    try:
        seed = 410_003 + group_index * 100_019
        q = runner.q_local(total_q)
        local_history_lengths = [
            len(range(dcp_rank, length, case.dcp_size))
            for length in history_lengths
        ]
        if any(length <= 0 for length in local_history_lengths):
            raise SystemExit("every rank must own positive history length per sequence")

        cu_q, cu_q_host = make_cu(q_lengths, device)
        cu_local, cu_local_host = make_cu(local_history_lengths, device)
        reference_lengths = [
            history + query
            for history, query in zip(history_lengths, q_lengths)
        ]
        cu_reference, cu_reference_host = make_cu(reference_lengths, device)
        timestamps = torch.empty(8, device=device, dtype=torch.int64)
        if test_phase_wrap:
            runner._phase = (1 << 31) - 4

        for iteration in range(case.repeat):
            iteration_seed = seed + iteration * 1_000_003
            q.copy_(
                randn_bf16(
                    (total_q, case.hq_local, 128),
                    iteration_seed + dist.get_rank() * 1_009,
                    device,
                )
            )
            history_k = randn_bf16(
                (sum(history_lengths), 1, 128), iteration_seed + 1, device
            )
            history_v = randn_bf16(
                (sum(history_lengths), 1, 128), iteration_seed + 2, device
            )
            local_k, local_k_lengths = shard_history(
                history_k, history_lengths, dcp_rank, case.dcp_size
            )
            local_v, local_v_lengths = shard_history(
                history_v, history_lengths, dcp_rank, case.dcp_size
            )
            assert local_k_lengths == local_history_lengths
            assert local_v_lengths == local_history_lengths
            chunk_k = randn_bf16((total_q, 1, 128), iteration_seed + 3, device)
            chunk_v = randn_bf16((total_q, 1, 128), iteration_seed + 4, device)
            reference_k = append_chunk(
                history_k, chunk_k, history_lengths, q_lengths
            )
            reference_v = append_chunk(
                history_v, chunk_v, history_lengths, q_lengths
            )
            expected_o, expected_lse = min_fa3_op.forward_kvcache_varlen(
                q,
                reference_k,
                reference_v,
                cu_q,
                cu_reference,
                max(q_lengths),
                max(reference_lengths),
                cu_seqlens_q_host=cu_q_host,
                cu_seqlens_k_host=cu_reference_host,
                num_splits=case.num_splits,
                return_lse=True,
                is_causal=True,
            )
            actual_o, actual_lse = runner.forward_chunk_prefill_varlen(
                q,
                local_k,
                local_v,
                chunk_k,
                chunk_v,
                cu_q,
                cu_local,
                max(q_lengths),
                max(local_history_lengths),
                cu_seqlens_q_host=cu_q_host,
                cu_seqlens_history_local_host=cu_local_host,
                num_splits=case.num_splits,
                return_lse=True,
            )
            runner.copy_last_phase_timestamps(timestamps)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(
                actual_o, expected_o, atol=3.0e-2, rtol=3.0e-2
            )
            torch.testing.assert_close(
                actual_lse, expected_lse, atol=3.0e-2, rtol=3.0e-2
            )
            if timestamps[3].item() != timestamps[4].item():
                raise AssertionError("fused history/publish timestamps differ")
            expected_phase = runner._phase - 1
            ready = runner._ipc_tile_ready.data_[
                : case.dcp_size, : (total_q + 15) // 16
            ]
            for source in range(case.dcp_size):
                if source != dcp_rank:
                    torch.testing.assert_close(
                        ready[source],
                        torch.full_like(ready[source], expected_phase),
                        rtol=0,
                        atol=0,
                    )

            runner.prepare_last_forward_replay()
            replay_o, replay_lse = runner.replay_last_forward()
            runner.copy_last_phase_timestamps(timestamps)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(
                replay_o, expected_o, atol=3.0e-2, rtol=3.0e-2
            )
            torch.testing.assert_close(
                replay_lse, expected_lse, atol=3.0e-2, rtol=3.0e-2
            )
            if timestamps[3].item() != timestamps[4].item():
                raise AssertionError("fused replay history/publish timestamps differ")
            replay_phase = runner._phase - 1
            for source in range(case.dcp_size):
                if source != dcp_rank:
                    torch.testing.assert_close(
                        ready[source],
                        torch.full_like(ready[source], replay_phase),
                        rtol=0,
                        atol=0,
                    )

            graph = runner.capture_last_forward(capture_warmup=1)
            try:
                for graph_replay in range(2):
                    graph_o, graph_lse = graph.replay()
                    runner.copy_last_phase_timestamps(timestamps)
                    torch.cuda.synchronize(device)
                    torch.testing.assert_close(
                        graph_o, expected_o, atol=3.0e-2, rtol=3.0e-2
                    )
                    torch.testing.assert_close(
                        graph_lse, expected_lse, atol=3.0e-2, rtol=3.0e-2
                    )
                    if timestamps[3].item() != timestamps[4].item():
                        raise AssertionError(
                            "fused CUDA Graph history/publish timestamps differ "
                            f"at replay {graph_replay}"
                        )
            finally:
                graph.close()
            graph_phase = runner._phase - 1
            for source in range(case.dcp_size):
                if source != dcp_rank:
                    torch.testing.assert_close(
                        ready[source],
                        torch.full_like(ready[source], graph_phase),
                        rtol=0,
                        atol=0,
                    )
            if dist.get_rank() == 0:
                dispatch = runner.last_dispatch
                print(
                    f"DCP mega ok: case={case.name} iteration={iteration} "
                    f"DCP={case.dcp_size} split={case.num_splits} "
                    f"Pack={dispatch.pack_gqa} BlockN={dispatch.block_n} "
                    "CUDA_Graph=ok",
                    flush=True,
                )
    finally:
        runner.close()
    dist.barrier()


def main() -> None:
    args = parse_args()
    local_device_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_device_rank)
    device = torch.device("cuda", local_device_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise SystemExit("This test requires SM90 Hopper")
        if args.matrix:
            if dist.get_world_size() != 8:
                raise SystemExit("--matrix requires exactly 8 processes on one node")
            cases = (
                CorrectnessCase(
                    "dcp2_h4_bn128_split1", 2, (1, 8, 19), (129, 258, 515),
                    4, 1, 128, 2, 8,
                ),
                CorrectnessCase(
                    "dcp4_h8_bn176_split2", 4, (3, 16, 31), (257, 518, 1031),
                    8, 2, 176, 2, 8,
                ),
                CorrectnessCase(
                    "dcp8_h4_bn176_auto", 8, (1, 17, 33), (515, 1030, 2061),
                    4, 0, 176, 2, 8,
                ),
                CorrectnessCase(
                    "dcp2_h8_bn128_split2_tail", 2,
                    (5, 16, 23), (129, 258, 515),
                    8, 2, 128, 2, 8,
                ),
                CorrectnessCase(
                    "case007_dcp8_h4_bn128_auto", 8,
                    (16, 16, 16), (1139, 44536, 3167),
                    4, 0, 128, 1, 8,
                ),
                CorrectnessCase(
                    "case007_dcp8_h4_bn128_split16", 8,
                    (16, 16, 16), (1139, 44536, 3167),
                    4, 16, 128, 1, 8,
                ),
            )
        else:
            q_lengths = tuple(parse_lengths(args.q_lengths, "--q-lengths"))
            history_lengths = tuple(
                parse_lengths(args.history_lengths, "--history-lengths")
            )
            if len(q_lengths) != len(history_lengths):
                raise SystemExit(
                    "Q and history length lists must have the same batch size"
                )
            if args.hq_local <= 0 or not 0 <= args.num_splits <= 128:
                raise SystemExit(
                    "hq-local must be positive and num-splits must be in [0, 128]"
                )
            if args.repeat <= 0 or args.num_comm_sm <= 0:
                raise SystemExit("repeat and num-comm-sm must be positive")
            cases = (
                CorrectnessCase(
                    "single",
                    args.dcp_size,
                    q_lengths,
                    history_lengths,
                    args.hq_local,
                    args.num_splits,
                    args.block_n,
                    args.repeat,
                    args.num_comm_sm,
                ),
            )
        for case in cases:
            run_case(case, device, args.test_phase_wrap or args.matrix)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
