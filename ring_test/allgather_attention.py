"""All-CP all-gather attention baseline for the varlen ring benchmarks.

The all-gather/FlashAttention/reduce-scatter structure is adapted from
ring-flash-attention's ``llama3_flash_attn_varlen.py``. The zigzag sequence
packing is local to this demo and matches its existing ring benchmarks.

The original runner zigzag-partitions every sequence independently. The
Llama3-style runner instead partitions the complete packed varlen batch into
two non-adjacent blocks per rank; either block may cross sequence boundaries.
Both paths express causal positions with bottom-right-aligned varlen calls
after K/V have been gathered and reordered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

import min_fa3_op
from ring_common import get_default_args

try:
    from flash_attn_interface import (
        _flash_attn_varlen_forward as _fa3_varlen_forward,
    )
except ImportError:
    _fa3_varlen_forward = None

try:
    from flash_attn_interface import (
        _flash_attn_varlen_backward as _fa3_varlen_backward,
    )
except ImportError:
    _fa3_varlen_backward = None


EXTERNAL_FA3_FORWARD_AVAILABLE = _fa3_varlen_forward is not None
EXTERNAL_FA3_BACKWARD_AVAILABLE = _fa3_varlen_backward is not None
EXTERNAL_FA3_AVAILABLE = (
    EXTERNAL_FA3_FORWARD_AVAILABLE and EXTERNAL_FA3_BACKWARD_AVAILABLE
)


def select_fa3_backend(
    process_group: Optional[dist.ProcessGroup],
    *,
    require_backward: bool,
) -> str:
    """Prefer external FA3 on every rank, otherwise use the in-repo FA3 op."""
    local_available = EXTERNAL_FA3_FORWARD_AVAILABLE and (
        not require_backward or EXTERNAL_FA3_BACKWARD_AVAILABLE
    )
    available = torch.tensor(
        [1 if local_available else 0],
        device=torch.device("cuda", torch.cuda.current_device()),
        dtype=torch.int32,
    )
    dist.all_reduce(available, op=dist.ReduceOp.MIN, group=process_group)
    return "external_fa3" if bool(available.item()) else "min_fa3"


def select_allgather_backend(process_group: Optional[dist.ProcessGroup]) -> str:
    """Compatibility wrapper for callers that may execute backward."""
    return select_fa3_backend(process_group, require_backward=True)


def backend_note(backend: str) -> str:
    if backend == "external_fa3":
        return "all-gather; external FA3; zigzag causal"
    if backend == "min_fa3":
        return "all-gather; in-repo min_fa3 fallback; zigzag causal"
    raise ValueError(f"unknown all-gather attention backend: {backend}")


def _external_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _fa3_varlen_forward is None:
        raise RuntimeError("external FA3 varlen forward is unavailable")
    params = get_default_args(_fa3_varlen_forward).copy()
    params.update(
        {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": max_q,
            "max_seqlen_k": max_k,
            "dropout_p": 0.0,
            "softmax_scale": q.shape[-1] ** -0.5,
            "causal": causal,
            "alibi_slopes": None,
            "return_softmax": False,
        }
    )
    result = _fa3_varlen_forward(**params)
    if len(result) == 8:
        return result[0], result[5]
    if len(result) == 4:
        return result[0], result[1]
    raise RuntimeError(f"unsupported external FA3 forward result length: {len(result)}")


def _external_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    causal: bool,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _fa3_varlen_backward is None:
        raise RuntimeError("external FA3 varlen backward is unavailable")
    params = get_default_args(_fa3_varlen_backward).copy()
    params.update(
        {
            "dout": dout,
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "softmax_lse": lse,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": max_q,
            "max_seqlen_k": max_k,
            "dropout_p": 0.0,
            "softmax_scale": q.shape[-1] ** -0.5,
            "causal": causal,
            "alibi_slopes": None,
            "deterministic": False,
        }
    )
    _fa3_varlen_backward(**params)
    return dq, dk, dv


def _local_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    cu_q_host: torch.Tensor,
    cu_k_host: torch.Tensor,
    max_q: int,
    max_k: int,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("min_fa3 varlen forward did not return output and LSE")
    return result[0], result[1]


def _local_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    causal: bool,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return min_fa3_op.backward_varlen(
        dout,
        q,
        k,
        v,
        out,
        lse,
        cu_q,
        cu_k,
        max_q,
        max_k,
        causal,
        dq=dq,
        dk=dk,
        dv=dv,
    )


def _uniform_cu(batch_size: int, seqlen: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.arange(0, (batch_size + 1) * seqlen, seqlen, dtype=torch.int32)
    return host.to(device=device), host


def sequence_shards_to_global_order(
    global_seqlens: list[int], world_size: int, causal: bool
) -> list[int]:
    """Map rank-major per-sequence shards to the original packed sequence order."""
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if any(length <= 0 or length % world_size for length in global_seqlens):
        raise ValueError("every global sequence length must be positive and divisible by world_size")

    local_lengths = [length // world_size for length in global_seqlens]
    local_total = sum(local_lengths)
    order: list[int] = []
    local_offset = 0
    for local_len in local_lengths:
        if causal:
            if local_len % 2:
                raise ValueError("causal per-sequence shards require even local sequence lengths")
            half = local_len // 2
            for source_rank in range(world_size):
                source = source_rank * local_total + local_offset
                order.extend(range(source, source + half))
            for source_rank in reversed(range(world_size)):
                source = source_rank * local_total + local_offset + half
                order.extend(range(source, source + half))
        else:
            for source_rank in range(world_size):
                source = source_rank * local_total + local_offset
                order.extend(range(source, source + local_len))
        local_offset += local_len
    return order


def llama3_rank_local_global_indices(
    total_tokens: int, world_size: int, rank: int
) -> list[int]:
    """Return the two global packed intervals owned by one Llama3-zigzag rank."""
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"invalid rank/world_size pair: rank={rank}, world_size={world_size}")
    if total_tokens <= 0 or total_tokens % (2 * world_size):
        raise ValueError(
            f"total token count must be divisible by 2 * world_size, got {total_tokens}"
        )
    chunk = total_tokens // (2 * world_size)
    front = range(rank * chunk, (rank + 1) * chunk)
    back_block = 2 * world_size - 1 - rank
    back = range(back_block * chunk, (back_block + 1) * chunk)
    return [*front, *back]


def llama3_rank_major_to_global_order(total_tokens: int, world_size: int) -> list[int]:
    """Map gathered Llama3-zigzag rank blocks back to global packed order."""
    if total_tokens <= 0 or total_tokens % (2 * world_size):
        raise ValueError(
            f"total token count must be divisible by 2 * world_size, got {total_tokens}"
        )
    chunk = total_tokens // (2 * world_size)
    local_tokens = 2 * chunk
    order: list[int] = []
    for source_rank in range(world_size):
        source = source_rank * local_tokens
        order.extend(range(source, source + chunk))
    for source_rank in reversed(range(world_size)):
        source = source_rank * local_tokens + chunk
        order.extend(range(source, source + chunk))
    return order


def repartition_sequence_shards_to_llama3(
    process_group: Optional[dist.ProcessGroup],
    tensor: torch.Tensor,
    global_seqlens: list[int],
    causal: bool,
) -> torch.Tensor:
    """Repartition existing per-sequence rank shards into whole-packed zigzag shards."""
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    expected_local = sum(global_seqlens) // world_size
    if tensor.size(0) != expected_local:
        raise ValueError(
            f"local tensor has {tensor.size(0)} tokens, expected {expected_local}"
        )

    gathered = torch.empty(
        (world_size * tensor.size(0), *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(gathered, tensor, group=process_group)
    sequence_order = torch.tensor(
        sequence_shards_to_global_order(global_seqlens, world_size, causal),
        dtype=torch.int64,
        device=tensor.device,
    )
    packed = torch.index_select(gathered, 0, sequence_order)
    local_order = torch.tensor(
        llama3_rank_local_global_indices(sum(global_seqlens), world_size, rank),
        dtype=torch.int64,
        device=tensor.device,
    )
    return torch.index_select(packed, 0, local_order).contiguous()


@dataclass(frozen=True)
class _Llama3BlockMetadata:
    local_slice: slice
    global_k_slice: slice
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    cu_q_host: torch.Tensor
    cu_k_host: torch.Tensor
    max_q: int
    max_k: int


def _cumulative_lengths(lengths: list[int]) -> torch.Tensor:
    result = torch.zeros((len(lengths) + 1,), dtype=torch.int32)
    for idx, length in enumerate(lengths):
        result[idx + 1] = result[idx] + length
    return result


def _llama3_block_metadata(
    global_seqlens: list[int],
    global_begin: int,
    global_end: int,
    local_begin: int,
    causal: bool,
    device: torch.device,
) -> _Llama3BlockMetadata:
    q_lengths: list[int] = []
    k_lengths: list[int] = []
    sequence_begin = 0
    k_begin = None
    k_end = None
    for sequence_length in global_seqlens:
        sequence_end = sequence_begin + sequence_length
        q_begin = max(sequence_begin, global_begin)
        q_end = min(sequence_end, global_end)
        if q_begin < q_end:
            q_lengths.append(q_end - q_begin)
            k_lengths.append(q_end - sequence_begin if causal else sequence_length)
            if k_begin is None:
                k_begin = sequence_begin
            k_end = q_end if causal else sequence_end
        sequence_begin = sequence_end

    if not q_lengths or k_begin is None or k_end is None:
        raise ValueError(f"packed block [{global_begin}, {global_end}) intersects no sequence")
    block_tokens = global_end - global_begin
    if sum(q_lengths) != block_tokens:
        raise ValueError("global sequence metadata does not cover the packed block")

    cu_q_host = _cumulative_lengths(q_lengths)
    cu_k_host = _cumulative_lengths(k_lengths)
    if int(cu_k_host[-1]) != k_end - k_begin:
        raise ValueError("K/V metadata is not contiguous in the global packed tensor")
    return _Llama3BlockMetadata(
        local_slice=slice(local_begin, local_begin + block_tokens),
        global_k_slice=slice(k_begin, k_end),
        cu_q=cu_q_host.to(device=device),
        cu_k=cu_k_host.to(device=device),
        cu_q_host=cu_q_host,
        cu_k_host=cu_k_host,
        max_q=max(q_lengths),
        max_k=max(k_lengths),
    )


class AllGatherAttention:
    """Per-sequence zigzag all-gather attention with KV-head pipelining."""

    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        local_seqlen: int,
        causal: bool,
        backend: str,
        *,
        heads_k_stride: int = 1,
        enable_backward: bool = False,
    ) -> None:
        if backend not in ("external_fa3", "min_fa3"):
            raise ValueError(f"unsupported all-gather attention backend: {backend}")
        if causal and local_seqlen % 2 != 0:
            raise ValueError(f"zigzag causal all-gather requires an even local seqlen, got {local_seqlen}")
        expected_tokens = batch_size * local_seqlen
        if q.size(0) != expected_tokens or k.size(0) != expected_tokens or v.size(0) != expected_tokens:
            raise ValueError("Q/K/V token counts must equal batch_size * local_seqlen")
        if k.shape != v.shape or q.size(2) != k.size(2):
            raise ValueError("K/V shapes and Q/K/V head dimensions must match")
        if q.size(1) % k.size(1):
            raise ValueError(
                f"Q head count must divide by KV head count, got QH={q.size(1)}, KVH={k.size(1)}"
            )
        if not 0 < heads_k_stride <= k.size(1) or k.size(1) % heads_k_stride:
            raise ValueError(
                "heads_k_stride must be a positive divisor of the KV head count, "
                f"got heads_k_stride={heads_k_stride}, KVH={k.size(1)}"
            )

        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.q = q
        self.k = k
        self.v = v
        self.batch_size = batch_size
        self.local_seqlen = local_seqlen
        self.causal = causal
        self.backend = backend
        self.enable_backward = enable_backward
        self.heads_k_stride = heads_k_stride
        self.q_heads_per_kv_head = q.size(1) // k.size(1)
        self.q_heads_per_chunk = heads_k_stride * self.q_heads_per_kv_head
        self.half = local_seqlen // 2
        self.local_tokens = expected_tokens
        self.global_seqlen = self.world_size * local_seqlen
        self._forward_ready = False

        gathered_shape = (
            2,
            2,
            self.world_size * expected_tokens,
            heads_k_stride,
            k.size(2),
        )
        ordered_shape = (
            2,
            2,
            batch_size * self.global_seqlen,
            heads_k_stride,
            k.size(2),
        )
        self.kv_gather = torch.empty(gathered_shape, dtype=k.dtype, device=k.device)
        self.kv_ordered = torch.empty(ordered_shape, dtype=k.dtype, device=k.device)
        self.kv_send = torch.empty(
            (2, expected_tokens, heads_k_stride, k.size(2)),
            dtype=k.dtype,
            device=k.device,
        )
        self.q_chunk = torch.empty(
            (expected_tokens, self.q_heads_per_chunk, q.size(2)),
            dtype=q.dtype,
            device=q.device,
        )
        self.out = torch.empty_like(q)
        if enable_backward:
            self.local_dk_fp32 = torch.empty_like(k, dtype=torch.float32)
            self.local_dv_fp32 = torch.empty_like(v, dtype=torch.float32)
            self.local_dk = torch.empty_like(k)
            self.local_dv = torch.empty_like(v)
            self.dq = torch.empty_like(q)
            self.backward_dout_chunk = torch.empty_like(self.q_chunk)
            self.backward_out_chunk = torch.empty_like(self.q_chunk)
            self.backward_dq_chunk = torch.empty_like(self.q_chunk)
            self.backward_block_dkv = torch.empty(
                (
                    2,
                    batch_size * self.global_seqlen,
                    heads_k_stride,
                    k.size(2),
                ),
                dtype=k.dtype,
                device=k.device,
            )
            self.backward_ordered_dkv = torch.empty(
                (
                    2,
                    batch_size * self.global_seqlen,
                    heads_k_stride,
                    k.size(2),
                ),
                dtype=torch.float32,
                device=k.device,
            )
            self.backward_rank_major_dkv = torch.empty(
                (2, self.world_size * expected_tokens, heads_k_stride, k.size(2)),
                dtype=torch.float32,
                device=k.device,
            )
            self.backward_local_dkv_fp32 = torch.empty(
                (2, expected_tokens, heads_k_stride, k.size(2)),
                dtype=torch.float32,
                device=k.device,
            )

        self.full_cu_q, self.full_cu_q_host = _uniform_cu(batch_size, local_seqlen, q.device)
        self.global_cu_k, self.global_cu_k_host = _uniform_cu(batch_size, self.global_seqlen, q.device)

        if causal:
            front_k_len = (self.rank + 1) * self.half
            back_k_len = (2 * self.world_size - self.rank) * self.half
            q_half_shape = (batch_size * self.half, self.q_heads_per_chunk, q.size(2))
            front_k_shape = (batch_size * front_k_len, heads_k_stride, k.size(2))
            back_k_shape = (batch_size * back_k_len, heads_k_stride, k.size(2))
            self.q_front = torch.empty(q_half_shape, dtype=q.dtype, device=q.device)
            self.q_back = torch.empty_like(self.q_front)
            self.k_front = torch.empty(front_k_shape, dtype=k.dtype, device=k.device)
            self.v_front = torch.empty_like(self.k_front)
            self.k_back = torch.empty(back_k_shape, dtype=k.dtype, device=k.device)
            self.v_back = torch.empty_like(self.k_back)
            if enable_backward:
                self.dout_front = torch.empty_like(self.q_front)
                self.dout_back = torch.empty_like(self.q_back)
                self.backward_out_front = torch.empty_like(self.q_front)
                self.backward_out_back = torch.empty_like(self.q_back)
                self.dq_front = torch.empty_like(self.q_front)
                self.dq_back = torch.empty_like(self.q_back)
                self.dk_front = torch.empty_like(self.k_front)
                self.dv_front = torch.empty_like(self.v_front)
                self.dk_back = torch.empty_like(self.k_back)
                self.dv_back = torch.empty_like(self.v_back)
            self.half_cu_q, self.half_cu_q_host = _uniform_cu(batch_size, self.half, q.device)
            self.front_cu_k, self.front_cu_k_host = _uniform_cu(batch_size, front_k_len, q.device)
            self.back_cu_k, self.back_cu_k_host = _uniform_cu(batch_size, back_k_len, q.device)

        lse_tokens = batch_size * (self.half if causal else local_seqlen)
        self.forward_lse_front = torch.empty(
            (q.size(1), lse_tokens), dtype=torch.float32, device=q.device
        )
        self.forward_lse_back = (
            torch.empty_like(self.forward_lse_front) if causal else None
        )

    @property
    def note(self) -> str:
        backend = (
            "external FA3"
            if self.backend == "external_fa3"
            else "in-repo min_fa3 fallback"
        )
        mode = "zigzag causal" if self.causal else "noncausal"
        return (
            "per-sequence KV-head-sharded all-gather "
            f"({self.heads_k_stride} KVH/chunk, comm/compute overlap); "
            f"{backend}; {mode}"
        )

    def _q_head_slice(self, kv_head_start: int) -> slice:
        q_head_start = kv_head_start * self.q_heads_per_kv_head
        return slice(q_head_start, q_head_start + self.q_heads_per_chunk)

    def _start_kv_all_gather(
        self, buffer_idx: int, kv_head_start: int
    ) -> tuple[object, object]:
        kv_head_slice = slice(kv_head_start, kv_head_start + self.heads_k_stride)
        self.kv_send[0].copy_(self.k[:, kv_head_slice])
        self.kv_send[1].copy_(self.v[:, kv_head_slice])
        k_work = dist.all_gather_into_tensor(
            self.kv_gather[buffer_idx, 0],
            self.kv_send[0],
            group=self.process_group,
            async_op=True,
        )
        v_work = dist.all_gather_into_tensor(
            self.kv_gather[buffer_idx, 1],
            self.kv_send[1],
            group=self.process_group,
            async_op=True,
        )
        if k_work is None or v_work is None:
            raise RuntimeError("asynchronous K/V all-gather returned no Work handle")
        return k_work, v_work

    @staticmethod
    def _wait_kv_all_gather(work: tuple[object, object]) -> None:
        for item in work:
            item.wait()  # type: ignore[attr-defined]

    def _order_kv_chunk(self, buffer_idx: int) -> None:
        gathered_k = self.kv_gather[buffer_idx, 0].view(
            self.world_size,
            self.batch_size,
            self.local_seqlen,
            self.heads_k_stride,
            self.k.size(2),
        )
        gathered_v = self.kv_gather[buffer_idx, 1].view_as(gathered_k)
        ordered_k = self.kv_ordered[buffer_idx, 0].view(
            self.batch_size,
            self.global_seqlen,
            self.heads_k_stride,
            self.k.size(2),
        )
        ordered_v = self.kv_ordered[buffer_idx, 1].view_as(ordered_k)

        for batch_idx in range(self.batch_size):
            if not self.causal:
                for rank_idx in range(self.world_size):
                    dst = slice(rank_idx * self.local_seqlen, (rank_idx + 1) * self.local_seqlen)
                    ordered_k[batch_idx, dst].copy_(gathered_k[rank_idx, batch_idx])
                    ordered_v[batch_idx, dst].copy_(gathered_v[rank_idx, batch_idx])
                continue

            for rank_idx in range(self.world_size):
                front_dst = slice(rank_idx * self.half, (rank_idx + 1) * self.half)
                back_block = 2 * self.world_size - 1 - rank_idx
                back_dst = slice(back_block * self.half, (back_block + 1) * self.half)
                ordered_k[batch_idx, front_dst].copy_(gathered_k[rank_idx, batch_idx, : self.half])
                ordered_v[batch_idx, front_dst].copy_(gathered_v[rank_idx, batch_idx, : self.half])
                ordered_k[batch_idx, back_dst].copy_(gathered_k[rank_idx, batch_idx, self.half :])
                ordered_v[batch_idx, back_dst].copy_(gathered_v[rank_idx, batch_idx, self.half :])

    def _run_forward_block(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        cu_q_host: torch.Tensor,
        cu_k_host: torch.Tensor,
        max_q: int,
        max_k: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "external_fa3":
            return _external_forward(q, k, v, cu_q, cu_k, max_q, max_k, causal)
        return _local_forward(
            q, k, v, cu_q, cu_k, cu_q_host, cu_k_host, max_q, max_k, causal
        )

    def forward(self) -> torch.Tensor:
        self._forward_ready = False
        current_buffer = 0
        current_work = self._start_kv_all_gather(current_buffer, 0)
        for kv_head_start in range(0, self.k.size(1), self.heads_k_stride):
            self._wait_kv_all_gather(current_work)
            self._order_kv_chunk(current_buffer)

            q_head_slice = self._q_head_slice(kv_head_start)
            self.q_chunk.copy_(self.q[:, q_head_slice])
            next_kv_head_start = kv_head_start + self.heads_k_stride
            if next_kv_head_start < self.k.size(1):
                next_buffer = 1 - current_buffer
                current_work = self._start_kv_all_gather(
                    next_buffer, next_kv_head_start
                )

            if not self.causal:
                out, lse = self._run_forward_block(
                    self.q_chunk,
                    self.kv_ordered[current_buffer, 0],
                    self.kv_ordered[current_buffer, 1],
                    self.full_cu_q,
                    self.global_cu_k,
                    self.full_cu_q_host,
                    self.global_cu_k_host,
                    self.local_seqlen,
                    self.global_seqlen,
                    False,
                )
                self.out[:, q_head_slice].copy_(out)
                self.forward_lse_front[q_head_slice].copy_(lse)
                current_buffer = 1 - current_buffer
                continue

            ordered_k = self.kv_ordered[current_buffer, 0].view(
                self.batch_size,
                self.global_seqlen,
                self.heads_k_stride,
                self.k.size(2),
            )
            ordered_v = self.kv_ordered[current_buffer, 1].view_as(ordered_k)
            q = self.q_chunk.view(
                self.batch_size, self.local_seqlen, self.q_heads_per_chunk, self.q.size(2)
            )
            q_front = self.q_front.view(
                self.batch_size, self.half, self.q_heads_per_chunk, self.q.size(2)
            )
            q_back = self.q_back.view_as(q_front)
            k_front = self.k_front.view(
                self.batch_size, -1, self.heads_k_stride, self.k.size(2)
            )
            v_front = self.v_front.view_as(k_front)
            k_back = self.k_back.view(
                self.batch_size, -1, self.heads_k_stride, self.k.size(2)
            )
            v_back = self.v_back.view_as(k_back)
            for batch_idx in range(self.batch_size):
                q_front[batch_idx].copy_(q[batch_idx, : self.half])
                q_back[batch_idx].copy_(q[batch_idx, self.half :])
                k_front[batch_idx].copy_(ordered_k[batch_idx, : k_front.size(1)])
                v_front[batch_idx].copy_(ordered_v[batch_idx, : v_front.size(1)])
                k_back[batch_idx].copy_(ordered_k[batch_idx, : k_back.size(1)])
                v_back[batch_idx].copy_(ordered_v[batch_idx, : v_back.size(1)])

            out_front, lse_front = self._run_forward_block(
                self.q_front,
                self.k_front,
                self.v_front,
                self.half_cu_q,
                self.front_cu_k,
                self.half_cu_q_host,
                self.front_cu_k_host,
                self.half,
                self.k_front.size(0) // self.batch_size,
                True,
            )
            out_back, lse_back = self._run_forward_block(
                self.q_back,
                self.k_back,
                self.v_back,
                self.half_cu_q,
                self.back_cu_k,
                self.half_cu_q_host,
                self.back_cu_k_host,
                self.half,
                self.k_back.size(0) // self.batch_size,
                True,
            )
            out_view = self.out.view(
                self.batch_size, self.local_seqlen, self.q.size(1), self.q.size(2)
            )
            packed_front = out_front.view(
                self.batch_size, self.half, self.q_heads_per_chunk, self.q.size(2)
            )
            packed_back = out_back.view_as(packed_front)
            for batch_idx in range(self.batch_size):
                out_view[batch_idx, : self.half, q_head_slice].copy_(
                    packed_front[batch_idx]
                )
                out_view[batch_idx, self.half :, q_head_slice].copy_(
                    packed_back[batch_idx]
                )
            self.forward_lse_front[q_head_slice].copy_(lse_front)
            assert self.forward_lse_back is not None
            self.forward_lse_back[q_head_slice].copy_(lse_back)
            current_buffer = 1 - current_buffer

        self._forward_ready = True
        return self.out

    def _run_backward_block(
        self,
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        causal: bool,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.backend == "external_fa3":
            return _external_backward(
                dout, q, k, v, out, lse, cu_q, cu_k, max_q, max_k, causal, dq, dk, dv
            )
        return _local_backward(
            dout, q, k, v, out, lse, cu_q, cu_k, max_q, max_k, causal, dq, dk, dv
        )

    def _ordered_grads_to_rank_major(self, ordered: torch.Tensor, rank_major: torch.Tensor) -> None:
        ordered_view = ordered.view(
            self.batch_size,
            self.global_seqlen,
            self.heads_k_stride,
            self.k.size(2),
        )
        rank_view = rank_major.view(
            self.world_size,
            self.batch_size,
            self.local_seqlen,
            self.heads_k_stride,
            self.k.size(2),
        )
        for batch_idx in range(self.batch_size):
            for rank_idx in range(self.world_size):
                if not self.causal:
                    src = slice(rank_idx * self.local_seqlen, (rank_idx + 1) * self.local_seqlen)
                    rank_view[rank_idx, batch_idx].copy_(ordered_view[batch_idx, src])
                    continue
                front_src = slice(rank_idx * self.half, (rank_idx + 1) * self.half)
                back_block = 2 * self.world_size - 1 - rank_idx
                back_src = slice(back_block * self.half, (back_block + 1) * self.half)
                rank_view[rank_idx, batch_idx, : self.half].copy_(ordered_view[batch_idx, front_src])
                rank_view[rank_idx, batch_idx, self.half :].copy_(ordered_view[batch_idx, back_src])

    def backward(self, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enable_backward:
            raise RuntimeError("all-gather attention runner was created without backward workspaces")
        if not self._forward_ready:
            raise RuntimeError("all-gather attention backward requires a prepared forward")
        if dout.shape != self.q.shape:
            raise ValueError("dout must match the local Q shape")

        current_buffer = 0
        current_work = self._start_kv_all_gather(current_buffer, 0)
        for kv_head_start in range(0, self.k.size(1), self.heads_k_stride):
            self._wait_kv_all_gather(current_work)
            self._order_kv_chunk(current_buffer)

            q_head_slice = self._q_head_slice(kv_head_start)
            self.q_chunk.copy_(self.q[:, q_head_slice])
            self.backward_dout_chunk.copy_(dout[:, q_head_slice])
            self.backward_out_chunk.copy_(self.out[:, q_head_slice])
            next_kv_head_start = kv_head_start + self.heads_k_stride
            if next_kv_head_start < self.k.size(1):
                next_buffer = 1 - current_buffer
                current_work = self._start_kv_all_gather(
                    next_buffer, next_kv_head_start
                )

            self.backward_ordered_dkv.zero_()
            if not self.causal:
                dk = self.backward_block_dkv[0]
                dv = self.backward_block_dkv[1]
                dq, dk, dv = self._run_backward_block(
                    self.backward_dout_chunk,
                    self.q_chunk,
                    self.kv_ordered[current_buffer, 0],
                    self.kv_ordered[current_buffer, 1],
                    self.backward_out_chunk,
                    self.forward_lse_front[q_head_slice],
                    self.full_cu_q,
                    self.global_cu_k,
                    self.local_seqlen,
                    self.global_seqlen,
                    False,
                    self.backward_dq_chunk,
                    dk,
                    dv,
                )
                self.backward_ordered_dkv[0].copy_(dk)
                self.backward_ordered_dkv[1].copy_(dv)
                self.dq[:, q_head_slice].copy_(dq)
            else:
                assert self.forward_lse_back is not None
                q = self.q_chunk.view(
                    self.batch_size,
                    self.local_seqlen,
                    self.q_heads_per_chunk,
                    self.q.size(2),
                )
                dout_view = self.backward_dout_chunk.view_as(q)
                out_view = self.backward_out_chunk.view_as(q)
                q_front = self.q_front.view(
                    self.batch_size,
                    self.half,
                    self.q_heads_per_chunk,
                    self.q.size(2),
                )
                q_back = self.q_back.view_as(q_front)
                dout_front = self.dout_front.view_as(q_front)
                dout_back = self.dout_back.view_as(q_front)
                out_front = self.backward_out_front.view_as(q_front)
                out_back = self.backward_out_back.view_as(q_front)
                ordered_k = self.kv_ordered[current_buffer, 0].view(
                    self.batch_size,
                    self.global_seqlen,
                    self.heads_k_stride,
                    self.k.size(2),
                )
                ordered_v = self.kv_ordered[current_buffer, 1].view_as(ordered_k)
                k_front = self.k_front.view(
                    self.batch_size, -1, self.heads_k_stride, self.k.size(2)
                )
                v_front = self.v_front.view_as(k_front)
                k_back = self.k_back.view(
                    self.batch_size, -1, self.heads_k_stride, self.k.size(2)
                )
                v_back = self.v_back.view_as(k_back)
                for batch_idx in range(self.batch_size):
                    q_front[batch_idx].copy_(q[batch_idx, : self.half])
                    q_back[batch_idx].copy_(q[batch_idx, self.half :])
                    dout_front[batch_idx].copy_(dout_view[batch_idx, : self.half])
                    dout_back[batch_idx].copy_(dout_view[batch_idx, self.half :])
                    out_front[batch_idx].copy_(out_view[batch_idx, : self.half])
                    out_back[batch_idx].copy_(out_view[batch_idx, self.half :])
                    k_front[batch_idx].copy_(ordered_k[batch_idx, : k_front.size(1)])
                    v_front[batch_idx].copy_(ordered_v[batch_idx, : v_front.size(1)])
                    k_back[batch_idx].copy_(ordered_k[batch_idx, : k_back.size(1)])
                    v_back[batch_idx].copy_(ordered_v[batch_idx, : v_back.size(1)])

                dq_front, dk_front, dv_front = self._run_backward_block(
                    self.dout_front,
                    self.q_front,
                    self.k_front,
                    self.v_front,
                    out_front.reshape_as(self.q_front),
                    self.forward_lse_front[q_head_slice],
                    self.half_cu_q,
                    self.front_cu_k,
                    self.half,
                    self.k_front.size(0) // self.batch_size,
                    True,
                    self.dq_front,
                    self.dk_front,
                    self.dv_front,
                )
                dq_back, dk_back, dv_back = self._run_backward_block(
                    self.dout_back,
                    self.q_back,
                    self.k_back,
                    self.v_back,
                    out_back.reshape_as(self.q_back),
                    self.forward_lse_back[q_head_slice],
                    self.half_cu_q,
                    self.back_cu_k,
                    self.half,
                    self.k_back.size(0) // self.batch_size,
                    True,
                    self.dq_back,
                    self.dk_back,
                    self.dv_back,
                )
                dq_view = self.dq.view(
                    self.batch_size, self.local_seqlen, self.q.size(1), self.q.size(2)
                )
                dq_front_view = dq_front.view_as(q_front)
                dq_back_view = dq_back.view_as(q_back)
                ordered_dk = self.backward_ordered_dkv[0].view(
                    self.batch_size,
                    self.global_seqlen,
                    self.heads_k_stride,
                    self.k.size(2),
                )
                ordered_dv = self.backward_ordered_dkv[1].view_as(ordered_dk)
                dk_front = dk_front.view_as(k_front)
                dv_front = dv_front.view_as(v_front)
                dk_back = dk_back.view_as(k_back)
                dv_back = dv_back.view_as(v_back)
                for batch_idx in range(self.batch_size):
                    dq_view[batch_idx, : self.half, q_head_slice].copy_(
                        dq_front_view[batch_idx]
                    )
                    dq_view[batch_idx, self.half :, q_head_slice].copy_(
                        dq_back_view[batch_idx]
                    )
                    ordered_dk[batch_idx, : dk_front.size(1)].add_(dk_front[batch_idx])
                    ordered_dv[batch_idx, : dv_front.size(1)].add_(dv_front[batch_idx])
                    ordered_dk[batch_idx, : dk_back.size(1)].add_(dk_back[batch_idx])
                    ordered_dv[batch_idx, : dv_back.size(1)].add_(dv_back[batch_idx])

            self._ordered_grads_to_rank_major(
                self.backward_ordered_dkv[0], self.backward_rank_major_dkv[0]
            )
            self._ordered_grads_to_rank_major(
                self.backward_ordered_dkv[1], self.backward_rank_major_dkv[1]
            )
            dist.reduce_scatter_tensor(
                self.backward_local_dkv_fp32[0],
                self.backward_rank_major_dkv[0],
                group=self.process_group,
            )
            dist.reduce_scatter_tensor(
                self.backward_local_dkv_fp32[1],
                self.backward_rank_major_dkv[1],
                group=self.process_group,
            )
            kv_head_slice = slice(
                kv_head_start, kv_head_start + self.heads_k_stride
            )
            self.local_dk_fp32[:, kv_head_slice].copy_(
                self.backward_local_dkv_fp32[0]
            )
            self.local_dv_fp32[:, kv_head_slice].copy_(
                self.backward_local_dkv_fp32[1]
            )
            current_buffer = 1 - current_buffer

        self.local_dk.copy_(self.local_dk_fp32)
        self.local_dv.copy_(self.local_dv_fp32)
        return self.dq, self.local_dk, self.local_dv


class Llama3AllGatherAttention:
    """Whole-packed zigzag all-gather attention with KV-head pipelining.

    Each iteration gathers one contiguous KV-head slice.  Once that slice is
    ready, the next slice's all-gather is launched before FlashAttention runs
    over the current slice.  The two all-gather receive buffers are ping-ponged
    so NCCL never overwrites data that the CUDA compute stream still consumes.
    """

    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_seqlens: list[int],
        causal: bool,
        backend: str,
        *,
        heads_k_stride: int = 1,
        enable_backward: bool = False,
    ) -> None:
        if backend not in ("external_fa3", "min_fa3"):
            raise ValueError(f"unsupported all-gather attention backend: {backend}")
        if not global_seqlens or any(length <= 0 for length in global_seqlens):
            raise ValueError("global_seqlens must contain positive lengths")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("Q/K/V must use flattened [tokens, heads, head_dim] layout")
        if k.shape != v.shape:
            raise ValueError("K and V must have matching shapes")
        if q.size(2) != k.size(2):
            raise ValueError("Q, K, and V must have matching head dimensions")
        if q.size(1) % k.size(1):
            raise ValueError(
                f"Q head count must divide by KV head count, got QH={q.size(1)}, KVH={k.size(1)}"
            )
        if not 0 < heads_k_stride <= k.size(1) or k.size(1) % heads_k_stride:
            raise ValueError(
                "heads_k_stride must be a positive divisor of the KV head count, "
                f"got heads_k_stride={heads_k_stride}, KVH={k.size(1)}"
            )

        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.global_seqlens = list(global_seqlens)
        self.total_tokens = sum(global_seqlens)
        if self.total_tokens % (2 * self.world_size):
            raise ValueError(
                "whole-packed zigzag requires total tokens divisible by 2 * world_size: "
                f"total={self.total_tokens}, world_size={self.world_size}"
            )

        self.chunk = self.total_tokens // (2 * self.world_size)
        self.local_tokens = 2 * self.chunk
        if q.size(0) != self.local_tokens or k.size(0) != self.local_tokens:
            raise ValueError(
                f"Q/K/V must each contain {self.local_tokens} local tokens"
            )

        self.q = q
        self.k = k
        self.v = v
        self.causal = causal
        self.backend = backend
        self.enable_backward = enable_backward
        self.heads_k_stride = heads_k_stride
        self.q_heads_per_kv_head = q.size(1) // k.size(1)
        self.q_heads_per_chunk = heads_k_stride * self.q_heads_per_kv_head
        self.out = torch.empty_like(q)

        global_order = torch.tensor(
            llama3_rank_major_to_global_order(self.total_tokens, self.world_size),
            dtype=torch.int64,
            device=q.device,
        )
        self.global_order = global_order
        self.inverse_global_order = torch.argsort(global_order)

        # [ping-pong slot, K/V, gathered tokens, KV heads in this slice, D].
        # The ordered buffers are contiguous, which is required by the local
        # min_fa3 fallback after a head slice is extracted from Q/K/V.
        kv_chunk_shape = (2, 2, self.total_tokens, heads_k_stride, k.size(2))
        self.kv_gather = torch.empty(kv_chunk_shape, dtype=k.dtype, device=k.device)
        self.kv_ordered = torch.empty_like(self.kv_gather)
        self.kv_send = torch.empty(
            (2, self.local_tokens, heads_k_stride, k.size(2)),
            dtype=k.dtype,
            device=k.device,
        )
        self.q_chunk = torch.empty(
            (self.local_tokens, self.q_heads_per_chunk, q.size(2)),
            dtype=q.dtype,
            device=q.device,
        )

        front_begin = self.rank * self.chunk
        back_block = 2 * self.world_size - 1 - self.rank
        back_begin = back_block * self.chunk
        self.blocks = [
            _llama3_block_metadata(
                self.global_seqlens,
                front_begin,
                front_begin + self.chunk,
                0,
                causal,
                q.device,
            ),
            _llama3_block_metadata(
                self.global_seqlens,
                back_begin,
                back_begin + self.chunk,
                self.chunk,
                causal,
                q.device,
            ),
        ]
        self.forward_out = [self.out[block.local_slice] for block in self.blocks]
        self.forward_lse = [
            torch.empty(
                (q.size(1), self.chunk), dtype=torch.float32, device=q.device
            )
            for _ in self.blocks
        ]
        self._forward_ready = False

        if enable_backward:
            self.dq = torch.empty_like(q)
            self.local_dk_fp32 = torch.empty_like(k, dtype=torch.float32)
            self.local_dv_fp32 = torch.empty_like(v, dtype=torch.float32)
            self.local_dk = torch.empty_like(k)
            self.local_dv = torch.empty_like(v)
            self.backward_dout_chunk = torch.empty_like(self.q_chunk)
            self.backward_out_chunk = torch.empty_like(self.q_chunk)
            self.backward_dq_chunk = torch.empty_like(self.q_chunk)
            self.backward_lse = [
                torch.empty(
                    (self.q_heads_per_chunk, self.chunk),
                    dtype=torch.float32,
                    device=q.device,
                )
                for _ in self.blocks
            ]
            max_block_k_tokens = max(
                block.global_k_slice.stop - block.global_k_slice.start
                for block in self.blocks
            )
            self.backward_block_dkv = torch.empty(
                (2, max_block_k_tokens, heads_k_stride, k.size(2)),
                dtype=k.dtype,
                device=k.device,
            )
            self.backward_ordered_dkv = torch.empty(
                (2, self.total_tokens, heads_k_stride, k.size(2)),
                dtype=torch.float32,
                device=k.device,
            )
            self.backward_rank_major_dkv = torch.empty_like(self.backward_ordered_dkv)
            self.backward_local_dk_fp32 = torch.empty(
                (self.local_tokens, heads_k_stride, k.size(2)),
                dtype=torch.float32,
                device=k.device,
            )
            self.backward_local_dv_fp32 = torch.empty_like(self.backward_local_dk_fp32)

    @property
    def note(self) -> str:
        backend = (
            "external FA3"
            if self.backend == "external_fa3"
            else "in-repo min_fa3 fallback"
        )
        mode = "causal" if self.causal else "noncausal"
        return (
            "whole-packed zigzag KV-head-sharded all-gather "
            f"({self.heads_k_stride} KVH/chunk, comm/compute overlap); {backend}; {mode}"
        )

    def _q_head_slice(self, kv_head_start: int) -> slice:
        q_head_start = kv_head_start * self.q_heads_per_kv_head
        return slice(q_head_start, q_head_start + self.q_heads_per_chunk)

    def _start_kv_all_gather(
        self, buffer_idx: int, kv_head_start: int
    ) -> tuple[object, object]:
        kv_head_slice = slice(kv_head_start, kv_head_start + self.heads_k_stride)
        self.kv_send[0].copy_(self.k[:, kv_head_slice])
        self.kv_send[1].copy_(self.v[:, kv_head_slice])
        k_work = dist.all_gather_into_tensor(
            self.kv_gather[buffer_idx, 0],
            self.kv_send[0],
            group=self.process_group,
            async_op=True,
        )
        v_work = dist.all_gather_into_tensor(
            self.kv_gather[buffer_idx, 1],
            self.kv_send[1],
            group=self.process_group,
            async_op=True,
        )
        if k_work is None or v_work is None:
            raise RuntimeError("asynchronous K/V all-gather returned no Work handle")
        return k_work, v_work

    @staticmethod
    def _wait_kv_all_gather(work: tuple[object, object]) -> None:
        for item in work:
            item.wait()  # type: ignore[attr-defined]

    def _order_kv_chunk(self, buffer_idx: int) -> None:
        torch.index_select(
            self.kv_gather[buffer_idx, 0],
            0,
            self.global_order,
            out=self.kv_ordered[buffer_idx, 0],
        )
        torch.index_select(
            self.kv_gather[buffer_idx, 1],
            0,
            self.global_order,
            out=self.kv_ordered[buffer_idx, 1],
        )

    def _run_forward_block(
        self,
        block: _Llama3BlockMetadata,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "external_fa3":
            return _external_forward(
                q, k, v, block.cu_q, block.cu_k, block.max_q, block.max_k, self.causal
            )
        return _local_forward(
            q,
            k,
            v,
            block.cu_q,
            block.cu_k,
            block.cu_q_host,
            block.cu_k_host,
            block.max_q,
            block.max_k,
            self.causal,
        )

    def forward(self) -> torch.Tensor:
        self._forward_ready = False
        current_buffer = 0
        current_work = self._start_kv_all_gather(current_buffer, 0)
        for kv_head_start in range(0, self.k.size(1), self.heads_k_stride):
            self._wait_kv_all_gather(current_work)
            self._order_kv_chunk(current_buffer)

            q_head_slice = self._q_head_slice(kv_head_start)
            self.q_chunk.copy_(self.q[:, q_head_slice])
            next_kv_head_start = kv_head_start + self.heads_k_stride
            if next_kv_head_start < self.k.size(1):
                next_buffer = 1 - current_buffer
                current_work = self._start_kv_all_gather(
                    next_buffer, next_kv_head_start
                )

            for block_idx, block in enumerate(self.blocks):
                out, lse = self._run_forward_block(
                    block,
                    self.q_chunk[block.local_slice],
                    self.kv_ordered[current_buffer, 0][block.global_k_slice],
                    self.kv_ordered[current_buffer, 1][block.global_k_slice],
                )
                self.out[block.local_slice, q_head_slice].copy_(out)
                target_lse = self.forward_lse[block_idx][q_head_slice]
                if lse.shape != target_lse.shape:
                    raise RuntimeError(
                        "FlashAttention returned an unexpected LSE shape for a KV-head slice: "
                        f"got {tuple(lse.shape)}, expected {tuple(target_lse.shape)}"
                    )
                target_lse.copy_(lse)

            current_buffer = 1 - current_buffer
        self._forward_ready = True
        return self.out

    def _run_backward_block(
        self,
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        block: _Llama3BlockMetadata,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.backend == "external_fa3":
            return _external_backward(
                dout,
                q,
                k,
                v,
                out,
                lse,
                block.cu_q,
                block.cu_k,
                block.max_q,
                block.max_k,
                self.causal,
                dq,
                dk,
                dv,
            )
        return _local_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            block.cu_q,
            block.cu_k,
            block.max_q,
            block.max_k,
            self.causal,
            dq,
            dk,
            dv,
        )

    def backward(self, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enable_backward:
            raise RuntimeError("Llama3 all-gather runner was created without backward workspaces")
        if not self._forward_ready:
            raise RuntimeError("Llama3 all-gather backward requires a prepared forward")
        if dout.shape != self.q.shape:
            raise ValueError("dout must match the local Q shape")

        self.local_dk_fp32.zero_()
        self.local_dv_fp32.zero_()
        current_buffer = 0
        current_work = self._start_kv_all_gather(current_buffer, 0)
        for kv_head_start in range(0, self.k.size(1), self.heads_k_stride):
            self._wait_kv_all_gather(current_work)
            self._order_kv_chunk(current_buffer)

            q_head_slice = self._q_head_slice(kv_head_start)
            self.q_chunk.copy_(self.q[:, q_head_slice])
            self.backward_dout_chunk.copy_(dout[:, q_head_slice])
            self.backward_out_chunk.copy_(self.out[:, q_head_slice])
            for block_idx, block_lse in enumerate(self.forward_lse):
                self.backward_lse[block_idx].copy_(block_lse[q_head_slice])

            next_kv_head_start = kv_head_start + self.heads_k_stride
            if next_kv_head_start < self.k.size(1):
                next_buffer = 1 - current_buffer
                current_work = self._start_kv_all_gather(
                    next_buffer, next_kv_head_start
                )

            self.backward_ordered_dkv.zero_()
            for block_idx, block in enumerate(self.blocks):
                k_tokens = block.global_k_slice.stop - block.global_k_slice.start
                block_dk = self.backward_block_dkv[0, :k_tokens]
                block_dv = self.backward_block_dkv[1, :k_tokens]
                block_dk.zero_()
                block_dv.zero_()
                dq, dk, dv = self._run_backward_block(
                    self.backward_dout_chunk[block.local_slice],
                    self.q_chunk[block.local_slice],
                    self.kv_ordered[current_buffer, 0][block.global_k_slice],
                    self.kv_ordered[current_buffer, 1][block.global_k_slice],
                    self.backward_out_chunk[block.local_slice],
                    self.backward_lse[block_idx],
                    block,
                    self.backward_dq_chunk[block.local_slice],
                    block_dk,
                    block_dv,
                )
                self.backward_ordered_dkv[0, block.global_k_slice].add_(dk)
                self.backward_ordered_dkv[1, block.global_k_slice].add_(dv)
                self.backward_dq_chunk[block.local_slice].copy_(dq)

            self.dq[:, q_head_slice].copy_(self.backward_dq_chunk)
            torch.index_select(
                self.backward_ordered_dkv[0],
                0,
                self.inverse_global_order,
                out=self.backward_rank_major_dkv[0],
            )
            torch.index_select(
                self.backward_ordered_dkv[1],
                0,
                self.inverse_global_order,
                out=self.backward_rank_major_dkv[1],
            )
            dist.reduce_scatter_tensor(
                self.backward_local_dk_fp32,
                self.backward_rank_major_dkv[0],
                group=self.process_group,
            )
            dist.reduce_scatter_tensor(
                self.backward_local_dv_fp32,
                self.backward_rank_major_dkv[1],
                group=self.process_group,
            )
            kv_head_slice = slice(
                kv_head_start, kv_head_start + self.heads_k_stride
            )
            self.local_dk_fp32[:, kv_head_slice].copy_(self.backward_local_dk_fp32)
            self.local_dv_fp32[:, kv_head_slice].copy_(self.backward_local_dv_fp32)
            current_buffer = 1 - current_buffer

        self.local_dk.copy_(self.local_dk_fp32)
        self.local_dv.copy_(self.local_dv_fp32)
        return self.dq, self.local_dk, self.local_dv


__all__ = [
    "AllGatherAttention",
    "EXTERNAL_FA3_AVAILABLE",
    "EXTERNAL_FA3_BACKWARD_AVAILABLE",
    "EXTERNAL_FA3_FORWARD_AVAILABLE",
    "Llama3AllGatherAttention",
    "backend_note",
    "llama3_rank_local_global_indices",
    "llama3_rank_major_to_global_order",
    "repartition_sequence_shards_to_llama3",
    "select_allgather_backend",
    "select_fa3_backend",
    "sequence_shards_to_global_order",
]
