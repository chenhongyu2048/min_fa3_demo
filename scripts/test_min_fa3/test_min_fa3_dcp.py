import argparse
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

import min_fa3_op
from min_fa3_dcp import DCPAttentionRunner


@dataclass(frozen=True)
class DCPGroup:
    size: int
    start_rank: int
    process_group: dist.ProcessGroup


def parse_int_list(spec: str, name: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise SystemExit(f"{name} must contain at least one integer")
    return values


def make_dcp_groups(sizes: list[int], device: torch.device) -> list[DCPGroup]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_groups: list[DCPGroup] = []
    for size in sizes:
        if size <= 0 or size > world_size or world_size % size:
            raise SystemExit(
                f"every DCP size must divide torchrun world size {world_size}, got {size}"
            )
        for start_rank in range(0, world_size, size):
            ranks = list(range(start_rank, start_rank + size))
            process_group = dist.new_group(ranks, backend="nccl", device_id=device)
            if start_rank <= rank < start_rank + size:
                local_groups.append(DCPGroup(size, start_rank, process_group))
    return local_groups


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def randn_bf16(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)


def shard_interleaved(
    tensor: torch.Tensor, rank: int, world_size: int
) -> torch.Tensor:
    return tensor[:, rank::world_size].contiguous()


def local_lengths(
    global_lengths: list[int], rank: int, world_size: int, device: torch.device
) -> torch.Tensor:
    values = [
        max(0, (length + world_size - 1 - rank) // world_size)
        for length in global_lengths
    ]
    return torch.tensor(values, device=device, dtype=torch.int32)


def local_q_heads(q_group: torch.Tensor, rank: int, h_local: int) -> torch.Tensor:
    return q_group[:, :, rank * h_local : (rank + 1) * h_local].contiguous()


def full_kvcache_reference(
    q_local: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_lengths: list[int],
    num_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor(cache_lengths, device=q_local.device, dtype=torch.int32)
    return min_fa3_op.forward_kvcache(
        q_local,
        k_cache,
        v_cache,
        lengths,
        num_splits=num_splits,
        return_lse=True,
    )


def torch_chunk_reference(
    q: torch.Tensor,
    k_history: torch.Tensor,
    v_history: torch.Tensor,
    history_lengths: list[int],
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, sq, hq, d = q.shape
    repeats = hq // k_history.shape[2]
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for batch_idx in range(b):
        history_len = history_lengths[batch_idx]
        k = torch.cat((k_history[batch_idx, :history_len], k_chunk[batch_idx]), dim=0)
        v = torch.cat((v_history[batch_idx, :history_len], v_chunk[batch_idx]), dim=0)
        k = k.float().repeat_interleave(repeats, dim=1)
        v = v.float().repeat_interleave(repeats, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q[batch_idx].float(), k) / math.sqrt(d)
        query_idx = torch.arange(sq, device=q.device)[:, None]
        key_idx = torch.arange(history_len + sq, device=q.device)[None, :]
        scores.masked_fill_(key_idx > history_len + query_idx, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probabilities, v))
        lses.append(torch.logsumexp(scores, dim=-1))
    return torch.stack(outputs), torch.stack(lses)


def make_chunk_full_cache(
    k_history: torch.Tensor,
    v_history: torch.Tensor,
    history_lengths: list[int],
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    b, sq, h_kv, d = k_chunk.shape
    capacity = k_history.shape[1] + sq
    k_full = torch.zeros(
        (b, capacity, h_kv, d), device=k_history.device, dtype=k_history.dtype
    )
    v_full = torch.zeros_like(k_full)
    lengths: list[int] = []
    for batch_idx, history_len in enumerate(history_lengths):
        k_full[batch_idx, :history_len].copy_(k_history[batch_idx, :history_len])
        v_full[batch_idx, :history_len].copy_(v_history[batch_idx, :history_len])
        k_full[batch_idx, history_len : history_len + sq].copy_(k_chunk[batch_idx])
        v_full[batch_idx, history_len : history_len + sq].copy_(v_chunk[batch_idx])
        lengths.append(history_len + sq)
    return k_full, v_full, lengths


def assert_attention_close(
    out: torch.Tensor,
    lse: torch.Tensor,
    out_ref: torch.Tensor,
    lse_ref: torch.Tensor,
    label: str,
) -> None:
    torch.testing.assert_close(
        out.float(), out_ref.float(), atol=3e-2, rtol=3e-2, msg=lambda msg: f"{label}: {msg}"
    )
    torch.testing.assert_close(
        lse, lse_ref, atol=3e-3, rtol=3e-3, msg=lambda msg: f"{label}: {msg}"
    )


def run_decode_case(
    runner: DCPAttentionRunner,
    start_rank: int,
    h_kv: int,
    num_splits: int,
    h_local: int,
    device: torch.device,
) -> None:
    b, sq, d = 2, 1, 128
    capacity = 263
    lengths = [257, 262]
    generator = make_generator(
        1000 + start_rank * 97 + h_kv * 11 + num_splits, device
    )
    q_group = randn_bf16(
        (b, sq, h_local * runner.world_size, d), generator, device
    )
    q_local = local_q_heads(q_group, runner.rank, h_local)
    k_full = randn_bf16((b, capacity, h_kv, d), generator, device)
    v_full = randn_bf16((b, capacity, h_kv, d), generator, device)
    k_local = shard_interleaved(k_full, runner.rank, runner.world_size)
    v_local = shard_interleaved(v_full, runner.rank, runner.world_size)
    lengths_local = local_lengths(lengths, runner.rank, runner.world_size, device)

    out_ref, lse_ref = full_kvcache_reference(
        q_local, k_full, v_full, lengths, num_splits
    )
    out, lse = runner.forward_decode(
        q_local,
        k_local,
        v_local,
        lengths_local,
        num_splits=num_splits,
        return_lse=True,
    )
    assert_attention_close(
        out,
        lse,
        out_ref,
        lse_ref,
        f"decode N={runner.world_size} Hkv={h_kv} split={num_splits}",
    )


def run_chunk_case(
    runner: DCPAttentionRunner,
    start_rank: int,
    sq: int,
    h_kv: int,
    num_splits: int,
    h_local: int,
    device: torch.device,
    *,
    repeat_overlap: int,
    nondefault_stream: bool,
    check_torch_reference: bool,
) -> None:
    b, d = 2, 128
    history_capacity = 259
    history_lengths = [129, 258]
    generator = make_generator(
        2000 + start_rank * 193 + sq * 17 + h_kv * 7 + num_splits, device
    )
    q_group = randn_bf16(
        (b, sq, h_local * runner.world_size, d), generator, device
    )
    q_local = local_q_heads(q_group, runner.rank, h_local)
    k_history = randn_bf16((b, history_capacity, h_kv, d), generator, device)
    v_history = randn_bf16((b, history_capacity, h_kv, d), generator, device)
    k_chunk = randn_bf16((b, sq, h_kv, d), generator, device)
    v_chunk = randn_bf16((b, sq, h_kv, d), generator, device)
    k_local = shard_interleaved(k_history, runner.rank, runner.world_size)
    v_local = shard_interleaved(v_history, runner.rank, runner.world_size)
    lengths_local = local_lengths(
        history_lengths, runner.rank, runner.world_size, device
    )
    k_full, v_full, full_lengths = make_chunk_full_cache(
        k_history, v_history, history_lengths, k_chunk, v_chunk
    )
    out_ref, lse_ref = full_kvcache_reference(
        q_local, k_full, v_full, full_lengths, num_splits
    )
    label = f"chunk N={runner.world_size} Sq={sq} Hkv={h_kv} split={num_splits}"

    sequential = runner.forward_chunk_prefill(
        q_local,
        k_local,
        v_local,
        lengths_local,
        k_chunk,
        v_chunk,
        num_splits=num_splits,
        return_lse=True,
        overlap_q_allgather=False,
    )
    assert_attention_close(*sequential, out_ref, lse_ref, f"{label} sequential")

    for repeat_idx in range(repeat_overlap):
        overlapped = runner.forward_chunk_prefill(
            q_local,
            k_local,
            v_local,
            lengths_local,
            k_chunk,
            v_chunk,
            num_splits=num_splits,
            return_lse=True,
            overlap_q_allgather=True,
        )
        assert_attention_close(
            *overlapped, out_ref, lse_ref, f"{label} overlap repeat={repeat_idx}"
        )

    if nondefault_stream:
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            side_stream_result = runner.forward_chunk_prefill(
                q_local,
                k_local,
                v_local,
                lengths_local,
                k_chunk,
                v_chunk,
                num_splits=num_splits,
                return_lse=True,
                overlap_q_allgather=True,
            )
            assert_attention_close(
                *side_stream_result, out_ref, lse_ref, f"{label} nondefault stream"
            )
        stream.synchronize()

    if check_torch_reference:
        torch_out, torch_lse = torch_chunk_reference(
            q_local,
            k_history,
            v_history,
            history_lengths,
            k_chunk,
            v_chunk,
        )
        assert_attention_close(out_ref, lse_ref, torch_out, torch_lse, f"{label} FP32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-GPU correctness test for minimal FA3 DCP decode/chunk prefill."
    )
    parser.add_argument(
        "--dcp-sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated subgroup sizes; every size must divide torchrun world size",
    )
    parser.add_argument(
        "--sq", type=str, default="2,8,32,128", help="Chunk query lengths"
    )
    parser.add_argument(
        "--num-splits",
        type=str,
        default="0,1,2",
        help="0=auto, 1=NoSplit, values >=2 force Split",
    )
    parser.add_argument(
        "--kvheads", type=str, default="1,2", help="MQA/GQA KV-head counts"
    )
    parser.add_argument("--qhead-local", type=int, default=8)
    parser.add_argument(
        "--repeat-overlap",
        type=int,
        default=1,
        help="Ordinary overlap repetitions; one representative case is always repeated five times",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise SystemExit(
                f"This test requires SM90 Hopper, got {torch.cuda.get_device_capability(device)}"
            )
        dcp_sizes = parse_int_list(args.dcp_sizes, "--dcp-sizes")
        sq_values = parse_int_list(args.sq, "--sq")
        split_values = parse_int_list(args.num_splits, "--num-splits")
        kv_heads = parse_int_list(args.kvheads, "--kvheads")
        if args.qhead_local <= 0 or any(
            h_kv <= 0 or args.qhead_local % h_kv for h_kv in kv_heads
        ):
            raise SystemExit("--qhead-local must be positive and divisible by every --kvheads value")
        if any(value < 0 or value > 128 for value in split_values):
            raise SystemExit("--num-splits values must be in [0, 128]")
        if any(sq <= 0 for sq in sq_values):
            raise SystemExit("--sq values must be positive")

        local_groups = make_dcp_groups(dcp_sizes, device)
        for group_info in local_groups:
            runner = DCPAttentionRunner(group_info.process_group)
            for h_kv in kv_heads:
                for num_splits in split_values:
                    run_decode_case(
                        runner,
                        group_info.start_rank,
                        h_kv,
                        num_splits,
                        args.qhead_local,
                        device,
                    )
                    for sq in sq_values:
                        representative = (
                            group_info.size == max(dcp_sizes)
                            and sq == max(sq_values)
                            and h_kv == kv_heads[-1]
                            and num_splits == split_values[-1]
                        )
                        run_chunk_case(
                            runner,
                            group_info.start_rank,
                            sq,
                            h_kv,
                            num_splits,
                            args.qhead_local,
                            device,
                            repeat_overlap=max(args.repeat_overlap, 5 if representative else 1),
                            nondefault_stream=representative,
                            check_torch_reference=sq == min(sq_values) and num_splits == 1,
                        )
            dist.barrier(group=group_info.process_group)
            if runner.rank == 0:
                print(
                    f"DCP correctness: ok (size={group_info.size}, "
                    f"Sq={sq_values}, Hkv={kv_heads}, splits={split_values})",
                    flush=True,
                )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
