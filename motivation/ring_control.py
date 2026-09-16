"""T1 controls copied from the production zigzag loop, with identical masks and merges."""
from typing import Optional
import torch
import torch.distributed as dist
from ring_test.ring_common import RingComm, get_half_index, update_out_and_lse, ZigzagBlockAttention


def zigzag_control(
    process_group: Optional[dist.ProcessGroup],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    max_seqlen: int,
    block_attention: ZigzagBlockAttention,
    *,
    overlap: bool = True,
    preloaded=None,
    return_lse: bool = False,
    ring_members: tuple[int, ...] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run load-balanced causal zigzag ring attention for one backend.

    Each rank-local sequence is interpreted as [front half | back half]. Step 0
    computes the local full-sequence causal block. Earlier global KV ranks use
    full-Q/half-KV dense blocks, while later global KV ranks use half-Q/full-KV
    dense blocks and update only the back-half output rows.
    """
    if max_seqlen % 2 != 0:
        raise RuntimeError(f"zigzag causal ring requires an even max_seqlen, got {max_seqlen}")

    comm = RingComm(process_group, ring_members)
    half_index0 = get_half_index(cu_seqlens, front=True)
    half_index1 = get_half_index(cu_seqlens, front=False)
    half_cu_seqlens = cu_seqlens // 2
    half_cu_seqlens_host = cu_seqlens_host // 2
    half_max_seqlen = max_seqlen // 2

    out = None
    lse = None
    q1 = q[half_index1].contiguous()
    cur_k = k.contiguous()
    cur_v = v.contiguous()

    for step in range(comm.world_size):
        if preloaded is not None:
            cur_k, cur_v = preloaded[(comm.rank - step) % comm.world_size]
        if preloaded is None and overlap and step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(cur_k, cur_v)
        else:
            next_k, next_v = None, None

        if step == 0:
            block_out, block_lse = block_attention(
                q,
                cur_k,
                cur_v,
                cu_seqlens,
                cu_seqlens,
                cu_seqlens_host,
                cu_seqlens_host,
                max_seqlen,
                max_seqlen,
                True,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0 = cur_k[half_index0].contiguous()
            v0 = cur_v[half_index0].contiguous()
            block_out, block_lse = block_attention(
                q,
                k0,
                v0,
                cu_seqlens,
                half_cu_seqlens,
                cu_seqlens_host,
                half_cu_seqlens_host,
                max_seqlen,
                half_max_seqlen,
                False,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = block_attention(
                q1,
                cur_k,
                cur_v,
                half_cu_seqlens,
                cu_seqlens,
                half_cu_seqlens_host,
                cu_seqlens_host,
                half_max_seqlen,
                max_seqlen,
                False,
            )
            out1, lse1 = update_out_and_lse(out[half_index1], lse[half_index1], block_out, block_lse)
            out[half_index1] = out1
            lse[half_index1] = lse1

        if preloaded is None and step + 1 != comm.world_size:
            if not overlap:
                next_k, next_v = comm.send_recv_kv(cur_k, cur_v)
            comm.wait()
            cur_k, cur_v = next_k, next_v

    if out is None:
        raise RuntimeError("zigzag ring attention produced no output blocks")
    output = out.to(q.dtype)
    if return_lse:
        return output, lse.squeeze(-1).transpose(0, 1).contiguous()
    return output
