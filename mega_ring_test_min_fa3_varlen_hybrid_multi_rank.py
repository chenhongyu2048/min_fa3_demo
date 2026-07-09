import argparse
import os

import torch
import torch.distributed as dist

import min_fa3_op


def parse_lengths(spec: str, name: str) -> list[int]:
    lengths = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not lengths:
        raise SystemExit(f"{name} must provide at least one length")
    return lengths


def init_distributed() -> tuple[int, int]:
    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit("Run this test with torchrun so LOCAL_RANK and LOCAL_WORLD_SIZE are set")

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    if dist.get_world_size() != local_world_size:
        raise SystemExit(
            "ThunderKittens mega-ring attention demo is single-node only: "
            f"world_size={dist.get_world_size()}, local_world_size={local_world_size}"
        )

    return local_rank, local_world_size


def make_cu_seqlens(lengths: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.zeros((len(lengths) + 1,), dtype=torch.int32)
    for idx, length in enumerate(lengths):
        host[idx + 1] = host[idx] + int(length)
    return host.to(device=device), host


def gather_rank_blocks(local_tensor: torch.Tensor, local_rank: int, local_world_size: int) -> torch.Tensor:
    blocks = [torch.empty_like(local_tensor) for _ in range(local_world_size)]
    dist.all_gather(blocks, local_tensor)
    blocks[local_rank] = local_tensor
    return torch.cat(blocks, dim=0).contiguous()


def raise_if_any_rank_failed(local_error: str | None, local_rank: int) -> None:
    failed = torch.tensor([1 if local_error is not None else 0], device="cuda", dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM)
    if failed.item() == 0:
        return
    if local_error is not None:
        raise AssertionError(local_error)
    raise AssertionError(f"another rank failed this case; rank {local_rank} had no local assertion failure")


def assert_close_named(name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> None:
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(f"{name} mismatch\n{exc}") from exc


def make_rank_local_qkv(
    total_tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    local_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(total_tokens, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    base_k = torch.arange(total_tokens * kv_heads * head_dim, device="cuda", dtype=torch.float32)
    base_v = torch.arange(total_tokens * kv_heads * head_dim, device="cuda", dtype=torch.float32)
    k = ((base_k % 16).reshape(total_tokens, kv_heads, head_dim) + local_rank * 16.0).to(torch.bfloat16).contiguous()
    v = ((base_v % 16).reshape(total_tokens, kv_heads, head_dim) + local_rank * 16.0 + 7.0).to(torch.bfloat16).contiguous()
    return q, k, v


def reference_attention(
    q_i: torch.Tensor,
    k_i: torch.Tensor,
    v_i: torch.Tensor,
    is_causal: bool,
    query_pos: torch.Tensor | None = None,
    key_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    q_i = q_i.float()
    k_i = k_i.float()
    v_i = v_i.float()
    qhead_per_kvhead = q_i.size(1) // k_i.size(1)
    if qhead_per_kvhead != 1:
        k_i = k_i.repeat_interleave(qhead_per_kvhead, dim=1)
        v_i = v_i.repeat_interleave(qhead_per_kvhead, dim=1)

    scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * (q_i.size(-1) ** -0.5)
    if is_causal:
        if query_pos is None:
            query_pos = torch.arange(q_i.size(0), device=q_i.device, dtype=torch.int64)
        if key_pos is None:
            key_pos = torch.arange(k_i.size(0), device=q_i.device, dtype=torch.int64)
        causal_mask = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v_i).to(dtype=torch.bfloat16)


def reference_hybrid_varlen(
    q: torch.Tensor,
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    gathered_k: torch.Tensor,
    gathered_v: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    global_lengths: list[int],
    cp_threshold: int,
    is_causal: bool,
    local_rank: int,
    local_world_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    total_tokens = cu_seqlens_host[-1].item()
    batch_size = len(global_lengths)

    for batch_idx in range(batch_size):
        q_start = cu_seqlens_host[batch_idx].item()
        q_end = cu_seqlens_host[batch_idx + 1].item()
        local_len = q_end - q_start
        q_i = q[q_start:q_end]

        if global_lengths[batch_idx] <= cp_threshold:
            k_i = local_k[q_start:q_end]
            v_i = local_v[q_start:q_end]
            outputs.append(reference_attention(q_i, k_i, v_i, is_causal))
            continue

        if global_lengths[batch_idx] != local_len * local_world_size:
            raise AssertionError(
                "CP batch requires global length == local length * world_size in this test. "
                f"batch={batch_idx}, global={global_lengths[batch_idx]}, "
                f"local={local_len}, world_size={local_world_size}"
            )
        if is_causal and (local_len % 2 != 0 or (local_len // 2) % 128 != 0):
            raise AssertionError(
                "causal CP batch requires local_len / 2 to be 128-aligned. "
                f"batch={batch_idx}, local_len={local_len}"
            )

        k_blocks = []
        v_blocks = []
        key_positions = []
        half_len = local_len // 2
        for rank_idx in range(local_world_size):
            rank_offset = rank_idx * total_tokens
            k_blocks.append(gathered_k[rank_offset + q_start:rank_offset + q_end])
            v_blocks.append(gathered_v[rank_offset + q_start:rank_offset + q_end])
            if is_causal:
                front_pos = torch.arange(half_len, device=q.device, dtype=torch.int64) + rank_idx * half_len
                back_pos = torch.arange(half_len, device=q.device, dtype=torch.int64) + (2 * local_world_size - 1 - rank_idx) * half_len
                key_positions.append(torch.cat([front_pos, back_pos], dim=0))

        k_i = torch.cat(k_blocks, dim=0)
        v_i = torch.cat(v_blocks, dim=0)
        if is_causal:
            query_front = torch.arange(half_len, device=q.device, dtype=torch.int64) + local_rank * half_len
            query_back = torch.arange(half_len, device=q.device, dtype=torch.int64) + (2 * local_world_size - 1 - local_rank) * half_len
            query_pos = torch.cat([query_front, query_back], dim=0)
            key_pos = torch.cat(key_positions, dim=0)
        else:
            query_pos = None
            key_pos = None
        outputs.append(reference_attention(q_i, k_i, v_i, is_causal, query_pos, key_pos))

    return torch.cat(outputs, dim=0)


def expected_loaded_row_mask(
    local_lengths: list[int],
    global_lengths: list[int],
    cp_threshold: int,
    is_causal: bool,
    local_rank: int,
    local_world_size: int,
    device: torch.device,
) -> torch.Tensor:
    total_tokens = sum(local_lengths)
    cu = [0]
    for length in local_lengths:
        cu.append(cu[-1] + length)

    mask = torch.zeros((local_world_size * total_tokens,), device=device, dtype=torch.bool)
    local_start = local_rank * total_tokens
    mask[local_start:local_start + total_tokens] = True

    for step in range(1, local_world_size):
        kv_rank = (local_rank - step + local_world_size) % local_world_size
        rank_start = kv_rank * total_tokens
        for batch_idx, global_len in enumerate(global_lengths):
            if global_len <= cp_threshold:
                continue
            seq_start = rank_start + cu[batch_idx]
            seq_end = rank_start + cu[batch_idx + 1]
            if is_causal and step <= local_rank:
                half_len = local_lengths[batch_idx] // 2
                mask[seq_start:seq_start + half_len] = True
            else:
                mask[seq_start:seq_end] = True
    return mask


def run_case(
    local_lengths: list[int],
    global_lengths: list[int],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    is_causal: bool,
    num_comp_sm: int,
    num_comm_sm: int,
    cp_threshold: int,
    local_rank: int,
    local_world_size: int,
) -> None:
    if len(local_lengths) != len(global_lengths):
        raise SystemExit("--local-seqlens and --global-seqlens must have the same number of entries")

    total_tokens = sum(local_lengths)
    max_seqlen = max(local_lengths)
    cu_seqlens, cu_seqlens_host = make_cu_seqlens(local_lengths, torch.device("cuda"))
    global_seqlens_host = torch.tensor(global_lengths, dtype=torch.int32)
    cp_count = sum(int(length > cp_threshold) for length in global_lengths)

    q, local_k, local_v = make_rank_local_qkv(total_tokens, q_heads, kv_heads, head_dim, local_rank)
    expected_k = gather_rank_blocks(local_k, local_rank, local_world_size)
    expected_v = gather_rank_blocks(local_v, local_rank, local_world_size)

    remote_k = min_fa3_op.TKParallelTensor(
        list(expected_k.shape),
        torch.bfloat16,
        local_rank,
        local_world_size,
        False,
    )
    remote_v = min_fa3_op.TKParallelTensor(
        list(expected_v.shape),
        torch.bfloat16,
        local_rank,
        local_world_size,
        False,
    )
    k = remote_k.data_
    v = remote_v.data_
    k.zero_()
    v.zero_()
    local_block_start = local_rank * total_tokens
    local_block_end = local_block_start + total_tokens
    k[local_block_start:local_block_end].copy_(local_k)
    v[local_block_start:local_block_end].copy_(local_v)

    torch.cuda.synchronize()
    dist.barrier()
    out = min_fa3_op.forward_varlen_mega_ring(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        is_causal,
        cu_seqlens_q_host=cu_seqlens_host,
        cu_seqlens_k_host=cu_seqlens_host,
        remote_k=remote_k,
        remote_v=remote_v,
        num_comp_sm=num_comp_sm,
        num_comm_sm=num_comm_sm,
        global_seqlens_host=global_seqlens_host,
        cp_threshold=cp_threshold,
    )
    torch.cuda.synchronize()
    dist.barrier()

    ref = reference_hybrid_varlen(
        q,
        local_k,
        local_v,
        expected_k,
        expected_v,
        cu_seqlens_host,
        global_lengths,
        cp_threshold,
        is_causal,
        local_rank,
        local_world_size,
    )

    local_error: str | None = None
    try:
        if num_comm_sm > 0:
            loaded_rows = expected_loaded_row_mask(
                local_lengths,
                global_lengths,
                cp_threshold,
                is_causal,
                local_rank,
                local_world_size,
                k.device,
            )
            assert_close_named(
                "loaded K rows",
                k[loaded_rows].float(),
                expected_k[loaded_rows].float(),
                atol=0.0,
                rtol=0.0,
            )
            assert_close_named(
                "loaded V rows",
                v[loaded_rows].float(),
                expected_v[loaded_rows].float(),
                atol=0.0,
                rtol=0.0,
            )
        assert_close_named("output", out.float(), ref.float(), atol=2e-1, rtol=2e-1)
    except AssertionError as exc:
        local_error = (
            f"case causal={is_causal}, world_size={local_world_size}, rank={local_rank}, "
            f"local_lengths={local_lengths}, global_lengths={global_lengths}, cp_count={cp_count}, "
            f"threshold={cp_threshold}, QH={q_heads}, KVH={kv_heads}, D={head_dim}, "
            f"num_comp_sm={num_comp_sm}, num_comm_sm={num_comm_sm}\n{exc}"
        )

    raise_if_any_rank_failed(local_error, local_rank)

    if local_rank == 0:
        print(
            f"distributed mega ring hybrid case causal={is_causal}: ok "
            f"(world_size={local_world_size}, local_lengths={local_lengths}, "
            f"global_lengths={global_lengths}, cp_count={cp_count}, threshold={cp_threshold}, "
            f"QH={q_heads}, KVH={kv_heads}, D={head_dim}, "
            f"num_comp_sm={num_comp_sm}, num_comm_sm={num_comm_sm})",
            flush=True,
        )
    torch.cuda.synchronize()
    dist.barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distributed hybrid mega-ring varlen correctness checks.")
    parser.add_argument("--local-seqlens", type=str, default="1152,2048,1408")
    parser.add_argument("--global-seqlens", type=str, default="1152,4096,1408")
    parser.add_argument("--all-local-seqlens", type=str, default="1152,1408")
    parser.add_argument("--all-cp-local-seqlens", type=str, default="2048,2304")
    parser.add_argument("--all-cp-global-seqlens", type=str, default="4096,4608")
    parser.add_argument("--qhead", type=int, default=16)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--num-comp-sm", type=int, default=1)
    parser.add_argument("--num-comm-sm", type=int, default=1)
    parser.add_argument("--cp-threshold", type=int, default=2048)
    parser.add_argument("--mode", choices=("noncausal", "causal", "both"), default="both")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (9, 0):
        raise SystemExit(f"This demo requires SM90 Hopper, got {(major, minor)}")
    if args.headdim != 128:
        raise SystemExit(f"This demo requires D=128, got D={args.headdim}")
    if args.qhead % args.kvhead != 0:
        raise SystemExit(f"qhead must be divisible by kvhead, got {args.qhead} and {args.kvhead}")
    if args.kvhead * args.headdim != 1024:
        raise SystemExit("Mega ring communication path requires kvhead * headdim == 1024")

    local_rank, local_world_size = init_distributed()
    mixed_local_lengths = parse_lengths(args.local_seqlens, "--local-seqlens")
    mixed_global_lengths = parse_lengths(args.global_seqlens, "--global-seqlens")
    all_local_lengths = parse_lengths(args.all_local_seqlens, "--all-local-seqlens")
    all_cp_local_lengths = parse_lengths(args.all_cp_local_seqlens, "--all-cp-local-seqlens")
    all_cp_global_lengths = parse_lengths(args.all_cp_global_seqlens, "--all-cp-global-seqlens")

    cases = [
        (mixed_local_lengths, mixed_global_lengths),
        (all_local_lengths, all_local_lengths),
        (all_cp_local_lengths, all_cp_global_lengths),
    ]

    try:
        for local_lengths, global_lengths in cases:
            if args.mode in ("noncausal", "both"):
                run_case(
                    local_lengths,
                    global_lengths,
                    args.qhead,
                    args.kvhead,
                    args.headdim,
                    False,
                    args.num_comp_sm,
                    args.num_comm_sm,
                    args.cp_threshold,
                    local_rank,
                    local_world_size,
                )
            if args.mode in ("causal", "both"):
                run_case(
                    local_lengths,
                    global_lengths,
                    args.qhead,
                    args.kvhead,
                    args.headdim,
                    True,
                    args.num_comp_sm,
                    args.num_comm_sm,
                    args.cp_threshold,
                    local_rank,
                    local_world_size,
                )
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
