import argparse
import os

import torch
import torch.distributed as dist

import min_fa3_op


SENTINEL = -123.0
BASE_SEED = 20260713


def parse_int_list(spec: str, name: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise SystemExit(f"{name} must provide at least one integer")
    return values


def init_distributed() -> tuple[int, int]:
    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit("Run this test with torchrun")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", rank))
    if dist.get_world_size() != world_size or world_size not in (2, 4, 8):
        raise SystemExit(f"hierarchical mega ring requires one node with 2, 4, or 8 ranks, got {world_size}")
    return rank, world_size


def make_cu_seqlens(lengths: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.zeros((len(lengths) + 1,), dtype=torch.int32)
    for idx, length in enumerate(lengths):
        host[idx + 1] = host[idx] + length
    return host.to(device=device), host


def local_lengths_for_rank(
    global_lengths: list[int], ring_sizes: list[int], ring_starts: list[int], rank: int
) -> list[int]:
    return [
        global_len // ring_size if ring_start <= rank < ring_start + ring_size else 0
        for global_len, ring_size, ring_start in zip(global_lengths, ring_sizes, ring_starts)
    ]


def make_local_qkv(
    total_tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    rank: int,
    is_causal: bool,
    device: torch.device,
    base_seed: int = BASE_SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Keep each mode independent of execution order. Large rank-wise K offsets
    # make logits nearly tie when sum(q) is close to zero, while large V offsets
    # turn a normal floating-point tie break into a false correctness failure.
    seed = base_seed + rank * 1009 + int(is_causal) * 1_000_003
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(
        (total_tokens, q_heads, head_dim), device=device, dtype=torch.float32, generator=generator
    ).to(torch.bfloat16)
    k = torch.randn(
        (total_tokens, kv_heads, head_dim), device=device, dtype=torch.float32, generator=generator
    ).to(torch.bfloat16)
    v = torch.randn(
        (total_tokens, kv_heads, head_dim), device=device, dtype=torch.float32, generator=generator
    ).mul_(0.5).add_(rank * 0.125).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), v.contiguous()


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_positions: torch.Tensor | None,
    key_positions: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    qf, kf, vf = q.float(), k.float(), v.float()
    repeat = q.size(1) // k.size(1)
    if repeat != 1:
        kf = kf.repeat_interleave(repeat, dim=1)
        vf = vf.repeat_interleave(repeat, dim=1)
    scores = torch.einsum("qhd,khd->hqk", qf, kf) * (q.size(-1) ** -0.5)
    if query_positions is not None:
        mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", probs, vf).to(torch.bfloat16)
    return out, lse


def hierarchical_reference(
    q: torch.Tensor,
    gathered_k: torch.Tensor,
    gathered_v: torch.Tensor,
    all_rank_lengths: list[list[int]],
    local_cu: torch.Tensor,
    global_lengths: list[int],
    ring_sizes: list[int],
    ring_starts: list[int],
    rank: int,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for batch_idx, (global_len, ring_size, ring_start) in enumerate(
        zip(global_lengths, ring_sizes, ring_starts)
    ):
        q_begin = int(local_cu[batch_idx])
        q_end = int(local_cu[batch_idx + 1])
        if q_begin == q_end:
            continue
        q_batch = q[q_begin:q_end]
        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        key_positions: list[torch.Tensor] = []
        local_len = global_len // ring_size
        half_len = local_len // 2
        for source_rank in range(ring_start, ring_start + ring_size):
            source_offset = sum(all_rank_lengths[source_rank][:batch_idx])
            k_parts.append(gathered_k[source_rank, source_offset:source_offset + local_len])
            v_parts.append(gathered_v[source_rank, source_offset:source_offset + local_len])
            if is_causal and ring_size > 1:
                subgroup_rank = source_rank - ring_start
                front = torch.arange(half_len, device=q.device) + subgroup_rank * half_len
                back = torch.arange(half_len, device=q.device) + (2 * ring_size - 1 - subgroup_rank) * half_len
                key_positions.append(torch.cat((front, back)))
        k_batch = torch.cat(k_parts)
        v_batch = torch.cat(v_parts)
        if is_causal and ring_size > 1:
            subgroup_rank = rank - ring_start
            query_front = torch.arange(half_len, device=q.device) + subgroup_rank * half_len
            query_back = torch.arange(half_len, device=q.device) + (2 * ring_size - 1 - subgroup_rank) * half_len
            query_positions = torch.cat((query_front, query_back))
            key_position_tensor = torch.cat(key_positions)
        elif is_causal:
            query_positions = torch.arange(local_len, device=q.device)
            key_position_tensor = torch.arange(local_len, device=q.device)
        else:
            query_positions = None
            key_position_tensor = None
        out, lse = attention_reference(
            q_batch, k_batch, v_batch, query_positions, key_position_tensor
        )
        outputs.append(out)
        lses.append(lse)
    if not outputs:
        return q.new_empty(q.shape), torch.empty((q.size(1), 0), device=q.device, dtype=torch.float32)
    return torch.cat(outputs), torch.cat(lses, dim=1)


def expected_loaded_mask(
    all_rank_lengths: list[list[int]],
    rank_capacity: int,
    global_lengths: list[int],
    ring_sizes: list[int],
    ring_starts: list[int],
    rank: int,
    is_causal: bool,
    device: torch.device,
) -> torch.Tensor:
    world_size = len(all_rank_lengths)
    mask = torch.zeros((world_size * rank_capacity,), device=device, dtype=torch.bool)
    local_total = sum(all_rank_lengths[rank])
    mask[rank * rank_capacity:rank * rank_capacity + local_total] = True
    for batch_idx, (global_len, ring_size, ring_start) in enumerate(
        zip(global_lengths, ring_sizes, ring_starts)
    ):
        if ring_size == 1 or not (ring_start <= rank < ring_start + ring_size):
            continue
        local_rank = rank - ring_start
        local_len = global_len // ring_size
        destination_offset = sum(all_rank_lengths[rank][:batch_idx])
        for step in range(1, ring_size):
            source_rank = ring_start + (local_rank - step + ring_size) % ring_size
            source_offset = sum(all_rank_lengths[source_rank][:batch_idx])
            if source_offset != destination_offset:
                raise AssertionError("batch ordering did not preserve subgroup row offsets")
            copied_len = local_len // 2 if is_causal and step <= local_rank else local_len
            begin = source_rank * rank_capacity + source_offset
            mask[begin:begin + copied_len] = True
    return mask


def assert_all_ranks(local_error: str | None) -> None:
    failed = torch.tensor([local_error is not None], device="cuda", dtype=torch.int32)
    dist.all_reduce(failed)
    if failed.item() == 0:
        return
    if local_error is not None:
        raise AssertionError(local_error)
    raise AssertionError("another rank failed")


def close_error(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    count_output_rows: bool = False,
) -> str | None:
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        details = [f"{name} check failed"]
        if count_output_rows:
            matched = torch.isclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
            mismatched = ~matched
            details.append(f"mismatched_elements={int(mismatched.sum().item())}")
            details.append(
                f"mismatched_token_head_rows={int(mismatched.any(dim=-1).sum().item())}"
            )
        details.append(str(exc))
        return "\n".join(details)
    return None


def run_case(args: argparse.Namespace, rank: int, world_size: int, is_causal: bool) -> None:
    global_lengths = parse_int_list(args.global_seqlens, "--global-seqlens")
    ring_sizes = parse_int_list(args.ring_sizes, "--ring-sizes")
    ring_starts = parse_int_list(args.ring_starts, "--ring-starts")
    if not (len(global_lengths) == len(ring_sizes) == len(ring_starts)):
        raise SystemExit("global lengths, ring sizes, and ring starts must have the same length")

    all_rank_lengths = [
        local_lengths_for_rank(global_lengths, ring_sizes, ring_starts, source_rank)
        for source_rank in range(world_size)
    ]
    local_lengths = all_rank_lengths[rank]
    local_total = sum(local_lengths)
    device = torch.device("cuda", rank)
    cu_seqlens, cu_seqlens_host = make_cu_seqlens(local_lengths, device)
    q, local_k, local_v = make_local_qkv(
        local_total, args.qhead, args.kvhead, args.headdim, rank, is_causal, device
    )

    capacity_tensor = torch.tensor([local_total], device=device, dtype=torch.int32)
    dist.all_reduce(capacity_tensor, op=dist.ReduceOp.MAX)
    rank_capacity = int(capacity_tensor.item())
    padded_k = torch.full(
        (rank_capacity, args.kvhead, args.headdim), SENTINEL, device=device, dtype=torch.bfloat16
    )
    padded_v = torch.full_like(padded_k, SENTINEL)
    padded_k[:local_total].copy_(local_k)
    padded_v[:local_total].copy_(local_v)
    gathered_k_parts = [torch.empty_like(padded_k) for _ in range(world_size)]
    gathered_v_parts = [torch.empty_like(padded_v) for _ in range(world_size)]
    dist.all_gather(gathered_k_parts, padded_k)
    dist.all_gather(gathered_v_parts, padded_v)
    gathered_k = torch.stack(gathered_k_parts)
    gathered_v = torch.stack(gathered_v_parts)

    arena_shape = [world_size * rank_capacity, args.kvhead, args.headdim]
    remote_k = min_fa3_op.TKParallelTensor(arena_shape, torch.bfloat16, rank, world_size, False)
    remote_v = min_fa3_op.TKParallelTensor(arena_shape, torch.bfloat16, rank, world_size, False)
    k, v = remote_k.data_, remote_v.data_
    k.fill_(SENTINEL)
    v.fill_(SENTINEL)
    owner_begin = rank * rank_capacity
    k[owner_begin:owner_begin + local_total].copy_(local_k)
    v[owner_begin:owner_begin + local_total].copy_(local_v)

    global_host = torch.tensor(global_lengths, dtype=torch.int32)
    ring_sizes_host = torch.tensor(ring_sizes, dtype=torch.int32)
    ring_starts_host = torch.tensor(ring_starts, dtype=torch.int32)
    max_local_len = max(max(lengths) for lengths in all_rank_lengths)
    torch.cuda.synchronize()
    dist.barrier()

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        return min_fa3_op.forward_varlen_mega_ring(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_local_len,
            max_local_len,
            is_causal,
            cu_seqlens_q_host=cu_seqlens_host,
            cu_seqlens_k_host=cu_seqlens_host,
            remote_k=remote_k,
            remote_v=remote_v,
            num_comp_sm=args.num_comp_sm,
            num_comm_sm=args.num_comm_sm,
            global_seqlens_host=global_host,
            ring_sizes_host=ring_sizes_host,
            ring_starts_host=ring_starts_host,
            return_lse=True,
        )

    out = None
    lse = None
    for _ in range(args.repeat):
        out, lse = launch()
    torch.cuda.synchronize()
    dist.barrier()

    expected_out, expected_lse = hierarchical_reference(
        q,
        gathered_k,
        gathered_v,
        all_rank_lengths,
        cu_seqlens_host,
        global_lengths,
        ring_sizes,
        ring_starts,
        rank,
        is_causal,
    )
    errors = [
        close_error(
            "output",
            out.float(),
            expected_out.float(),
            atol=2e-1,
            rtol=2e-1,
            count_output_rows=True,
        ),
        close_error("LSE", lse, expected_lse, atol=2e-1, rtol=2e-1),
    ]
    if args.check_arena:
        expected_arena_k = gathered_k.reshape_as(k)
        expected_arena_v = gathered_v.reshape_as(v)
        loaded = expected_loaded_mask(
            all_rank_lengths,
            rank_capacity,
            global_lengths,
            ring_sizes,
            ring_starts,
            rank,
            is_causal,
            device,
        )
        errors.extend(
            (
                close_error("arena K", k[loaded], expected_arena_k[loaded], atol=0.0, rtol=0.0),
                close_error("arena V", v[loaded], expected_arena_v[loaded], atol=0.0, rtol=0.0),
            )
        )
        padding = torch.zeros((world_size * rank_capacity,), device=device, dtype=torch.bool)
        for source_rank, lengths in enumerate(all_rank_lengths):
            begin = source_rank * rank_capacity + sum(lengths)
            padding[begin:(source_rank + 1) * rank_capacity] = True
        if padding.any():
            errors.extend(
                (
                    close_error(
                        "arena K padding",
                        k[padding],
                        torch.full_like(k[padding], SENTINEL),
                        atol=0.0,
                        rtol=0.0,
                    ),
                    close_error(
                        "arena V padding",
                        v[padding],
                        torch.full_like(v[padding], SENTINEL),
                        atol=0.0,
                        rtol=0.0,
                    ),
                )
            )
    error_details = [error for error in errors if error is not None]
    local_error: str | None = None
    if error_details:
        local_error = (
            f"rank={rank}, causal={is_causal}, local_lengths={local_lengths}, "
            f"global_lengths={global_lengths}, ring_sizes={ring_sizes}, ring_starts={ring_starts}\n"
            + "\n\n".join(error_details)
        )
    assert_all_ranks(local_error)
    if rank == 0:
        print(
            f"hierarchical mega ring causal={is_causal}: ok "
            f"(global={global_lengths}, rings={ring_sizes}, starts={ring_starts}, repeat={args.repeat})",
            flush=True,
        )
    dist.barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hierarchical hybrid mega-ring forward checks")
    parser.add_argument("--global-seqlens", default="8192,4096,2048,1024")
    parser.add_argument("--ring-sizes", default="8,4,2,1")
    parser.add_argument("--ring-starts", default="0,4,2,7")
    parser.add_argument("--qhead", type=int, default=16)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--num-comp-sm", type=int, default=116)
    parser.add_argument("--num-comm-sm", type=int, default=16)
    parser.add_argument("--mode", choices=("noncausal", "causal", "both"), default="both")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--check-arena", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_metadata_args(args: argparse.Namespace, world_size: int) -> None:
    global_lengths = parse_int_list(args.global_seqlens, "--global-seqlens")
    ring_sizes = parse_int_list(args.ring_sizes, "--ring-sizes")
    ring_starts = parse_int_list(args.ring_starts, "--ring-starts")
    if not (len(global_lengths) == len(ring_sizes) == len(ring_starts)):
        raise SystemExit("global lengths, ring sizes, and ring starts must have the same length")
    previous_size = 8
    for idx, (global_len, ring_size, ring_start) in enumerate(
        zip(global_lengths, ring_sizes, ring_starts)
    ):
        if ring_size not in (1, 2, 4, 8) or ring_size > previous_size:
            raise SystemExit(f"invalid ring size/order at batch {idx}")
        if ring_start < 0 or ring_start % ring_size or ring_start + ring_size > world_size:
            raise SystemExit(f"invalid ring start at batch {idx}")
        if global_len <= 0 or global_len % ring_size:
            raise SystemExit(f"invalid global length at batch {idx}")
        local_len = global_len // ring_size
        if local_len % 128:
            raise SystemExit(f"local length is not 128-aligned at batch {idx}")
        if args.mode in ("causal", "both") and ring_size > 1 and (
            local_len % 2 or (local_len // 2) % 128
        ):
            raise SystemExit(f"causal local half length is not 128-aligned at batch {idx}")
        previous_size = ring_size


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("SM90 Hopper is required")
    if args.headdim != 128 or args.kvhead * args.headdim != 1024:
        raise SystemExit("This path requires D=128 and KVH * D == 1024")
    if args.qhead % args.kvhead != 0:
        raise SystemExit("qhead must be divisible by kvhead")
    rank, world_size = init_distributed()
    validate_metadata_args(args, world_size)
    try:
        if args.mode in ("noncausal", "both"):
            run_case(args, rank, world_size, False)
        if args.mode in ("causal", "both"):
            run_case(args, rank, world_size, True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
