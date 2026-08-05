"""Copied-and-trimmed serving-runtime baselines for DCP tests.

The vLLM paths follow commit a89015c6df8eeb37a843b717c97a5be1355de83d.
The packed A2A combine includes the behavior from vLLM PRs #41160, #45487,
and #47801. The SGLang path follows commit
8d6549bc4039d33635844495d86684677a4f0df8.

These runners deliberately use the local minimal FA3 kernels so benchmark
results isolate orchestration and collective differences. Importing this
module does not require vLLM or SGLang to be installed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import min_fa3_op
from dcp_test.utils import BenchmarkTimingMixin
from min_fa3_dcp import (
    DCPAttentionRunner,
    _DCPResult,
    _merge_attn_states_kernel,
    _nvtx_range,
)


# Copied and trimmed from vLLM commit
# a89015c6df8eeb37a843b717c97a5be1355de83d,
# vllm/v1/attention/ops/dcp_alltoall.py. The packed A2A combine arrived in
# vLLM PR #41160; PR #45487 made per-call allocations CUDA-Graph safe and
# PR #47801 restored the required FP32 LSE bit-cast contract.
def _dcp_a2a_lse_weighted_combine_reference(
    outputs: torch.Tensor,
    lses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Torch base-e reference for the copied A2A combine."""
    if outputs.ndim != 4 or lses.shape != outputs.shape[:3]:
        raise ValueError("outputs must be [N, T, H, D] and lses must be [N, T, H]")
    valid_lses = torch.where(
        torch.isnan(lses) | torch.isinf(lses),
        torch.full_like(lses, -float("inf")),
        lses,
    )
    lse_max = valid_lses.max(dim=0).values
    finite_max = torch.where(
        lse_max == -float("inf"), torch.zeros_like(lse_max), lse_max
    )
    weights = torch.exp(valid_lses - finite_max.unsqueeze(0))
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
    weight_sum = weights.sum(dim=0)
    normalized = weights / weight_sum.clamp(min=1.0e-10).unsqueeze(0)
    combined = (outputs * normalized.unsqueeze(-1)).sum(dim=0)
    global_lse = torch.log(weight_sum) + finite_max
    return combined, global_lse


def _dcp_a2a_lse_pack_dim(output_dtype: torch.dtype) -> int:
    bits = torch.finfo(output_dtype).bits
    if bits == 16:
        return 2
    if bits == 32:
        return 1
    raise ValueError(f"Cannot pack fp32 LSE into output dtype {output_dtype}.")


def _dcp_a2a_head_owner_ranges(
    h_group: int, world_size: int
) -> tuple[tuple[int, int], ...]:
    if h_group <= 0 or world_size <= 0 or h_group % world_size:
        raise ValueError(
            f"H_group={h_group} must be positive and divisible by DCP={world_size}"
        )
    h_local = h_group // world_size
    return tuple(
        (rank * h_local, (rank + 1) * h_local) for rank in range(world_size)
    )


def _dcp_a2a_payload_bytes(
    total_tokens: int,
    h_local: int,
    head_dim: int,
    world_size: int,
    element_size: int = 2,
) -> tuple[int, int]:
    values = (total_tokens, h_local, head_dim, world_size, element_size)
    if any(value <= 0 for value in values):
        raise ValueError("A2A payload dimensions and element_size must be positive")
    per_owner = total_tokens * h_local * (head_dim + 2) * element_size
    return world_size * per_owner, (world_size - 1) * per_owner


@triton.jit
def _dcp_a2a_pack_send_kernel(
    out_ptr,
    lse_ptr,
    send_ptr,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    lse_stride_B,
    lse_stride_H,
    send_stride_N,
    send_stride_B,
    send_stride_H,
    send_stride_D,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H_PER_RANK: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    local_head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    for rank_idx in tl.static_range(N):
        src_head_idx = rank_idx * H_PER_RANK + local_head_idx
        send_base = (
            rank_idx * send_stride_N
            + batch_idx * send_stride_B
            + local_head_idx * send_stride_H
        )
        out_offsets = (
            batch_idx * out_stride_B
            + src_head_idx * out_stride_H
            + d_offsets * out_stride_D
        )
        tl.store(
            send_ptr + send_base + d_offsets * send_stride_D,
            tl.load(out_ptr + out_offsets),
        )

        lse_val = tl.load(
            lse_ptr + batch_idx * lse_stride_B + src_head_idx * lse_stride_H
        )
        if LSE_PACK_DIM == 1:
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lse_val.to(send_ptr.dtype.element_ty),
            )
        else:
            lse_bits = lse_val.to(tl.uint32, bitcast=True)
            lo = (lse_bits & 0xFFFF).to(tl.uint16)
            hi = ((lse_bits >> 16) & 0xFFFF).to(tl.uint16)
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lo.to(send_ptr.dtype.element_ty, bitcast=True),
            )
            tl.store(
                send_ptr + send_base + (HEAD_DIM + 1) * send_stride_D,
                hi.to(send_ptr.dtype.element_ty, bitcast=True),
            )


@triton.jit
def _dcp_a2a_unpack_combine_kernel(
    recv_ptr,
    out_ptr,
    out_lse_ptr,
    recv_stride_N,
    recv_stride_B,
    recv_stride_H,
    recv_stride_D,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    out_lse_stride_B,
    out_lse_stride_H,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    lse_max = -float("inf")
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(
                recv_ptr + recv_base + HEAD_DIM * recv_stride_D
            ).to(tl.float32)
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(
                recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D
            )
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        lse_max = tl.maximum(lse_max, lse_val)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    lse_sum = 0.0
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(
                recv_ptr + recv_base + HEAD_DIM * recv_stride_D
            ).to(tl.float32)
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(
                recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D
            )
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            lse_sum += tl.exp(lse_val - lse_max)
        else:
            lse_sum += tl.exp2(lse_val - lse_max)

    if IS_BASE_E:
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(
                recv_ptr + recv_base + HEAD_DIM * recv_stride_D
            ).to(tl.float32)
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(
                recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D
            )
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            weight = tl.exp(lse_val - global_lse)
        else:
            weight = tl.exp2(lse_val - global_lse)
        weight = tl.where(weight != weight, 0.0, weight)
        acc += (
            tl.load(recv_ptr + recv_base + d_offsets * recv_stride_D).to(
                tl.float32
            )
            * weight
        )

    final_offsets = (
        batch_idx * out_stride_B
        + head_idx * out_stride_H
        + d_offsets * out_stride_D
    )
    tl.store(out_ptr + final_offsets, acc)
    if RETURN_LSE:
        out_lse_offset = (
            batch_idx * out_lse_stride_B + head_idx * out_lse_stride_H
        )
        tl.store(out_lse_ptr + out_lse_offset, global_lse)


def _dcp_a2a_pack_send(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    send_buffer: torch.Tensor,
    world_size: int,
    h_per_rank: int,
    head_dim: int,
    lse_pack_dim: int,
) -> None:
    grid = (partial_out.shape[0], h_per_rank, 1)
    _dcp_a2a_pack_send_kernel[grid](
        partial_out,
        partial_lse,
        send_buffer,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        send_buffer.stride(0),
        send_buffer.stride(1),
        send_buffer.stride(2),
        send_buffer.stride(3),
        N=world_size,
        HEAD_DIM=head_dim,
        H_PER_RANK=h_per_rank,
        LSE_PACK_DIM=lse_pack_dim,
    )


def _dcp_a2a_unpack_combine(
    recv_buffer: torch.Tensor,
    head_dim: int,
    lse_pack_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_size, total_tokens, h_per_rank, _ = recv_buffer.shape
    output = torch.empty(
        (total_tokens, h_per_rank, head_dim),
        device=recv_buffer.device,
        dtype=recv_buffer.dtype,
    )
    output_lse = torch.empty(
        (total_tokens, h_per_rank),
        device=recv_buffer.device,
        dtype=torch.float32,
    )
    grid = (total_tokens, h_per_rank, 1)
    _dcp_a2a_unpack_combine_kernel[grid](
        recv_buffer,
        output,
        output_lse,
        recv_buffer.stride(0),
        recv_buffer.stride(1),
        recv_buffer.stride(2),
        recv_buffer.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output_lse.stride(0),
        output_lse.stride(1),
        N=world_size,
        HEAD_DIM=head_dim,
        IS_BASE_E=True,
        RETURN_LSE=True,
        LSE_PACK_DIM=lse_pack_dim,
    )
    return output, output_lse


@triton.jit
def _vllm_correct_attn_cp_out_kernel(
    partial_out_ptr,
    gathered_lse_ptr,
    corrected_out_ptr,
    global_lse_ptr,
    out_stride_t,
    out_stride_h,
    lse_stride_n,
    lse_stride_t,
    lse_stride_h,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    WORLD_SIZE_ROUNDED: tl.constexpr,
    RANK: tl.constexpr,
):
    """Trimmed from vLLM ``_correct_attn_cp_out_kernel`` at the pinned commit."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    rank_offsets = tl.arange(0, WORLD_SIZE_ROUNDED)
    rank_mask = rank_offsets < WORLD_SIZE
    lse_offsets = (
        rank_offsets * lse_stride_n
        + token_idx * lse_stride_t
        + head_idx * lse_stride_h
    )
    partial_lses = tl.load(
        gathered_lse_ptr + lse_offsets, mask=rank_mask, other=-float("inf")
    )
    partial_lses = tl.where(
        (partial_lses != partial_lses) | (partial_lses == float("inf")),
        -float("inf"),
        partial_lses,
    )
    lse_max = tl.max(partial_lses, axis=0)
    finite_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    lse_sum = tl.sum(tl.exp(partial_lses - finite_max), axis=0)
    global_lse = tl.log(lse_sum) + finite_max
    tl.store(global_lse_ptr + token_idx * H + head_idx, global_lse)

    local_lse = tl.load(
        gathered_lse_ptr
        + RANK * lse_stride_n
        + token_idx * lse_stride_t
        + head_idx * lse_stride_h
    )
    log_weight = local_lse - global_lse
    log_weight = tl.where(
        (log_weight != log_weight) | (log_weight == float("inf")),
        -float("inf"),
        log_weight,
    )
    weight = tl.exp(log_weight)
    dim_offsets = tl.arange(0, D)
    out_offsets = token_idx * out_stride_t + head_idx * out_stride_h + dim_offsets
    values = tl.load(partial_out_ptr + out_offsets).to(tl.float32) * weight
    tl.store(corrected_out_ptr + out_offsets, tl.where(weight == 0.0, 0.0, values))


class _SequentialDCPAttentionRunnerBase(DCPAttentionRunner):
    """Shared local-kernel plumbing for pinned production orchestration paths."""

    chunk_before_context = False
    workspace_policy = "framework_style_per_call"
    supports_overlap_q_allgather = False

    def _reject_overlap(self, overlap_q_allgather: bool) -> None:
        if overlap_q_allgather:
            raise ValueError(
                f"{type(self).__name__} only supports single-stream execution; "
                "overlap_q_allgather must be False"
            )

    def _check_packed_runner_topology(self, h_kv: int) -> None:
        if h_kv != 1:
            raise ValueError(
                f"{type(self).__name__} models TP > global Hkv, so each rank must "
                f"hold exactly one KV head; got {h_kv}"
            )

    def _check_reference_common(
        self,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        seqlens_local: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        shape = self._check_common(q_local, k_local, v_local, seqlens_local)
        if k_local.shape[2] != 1:
            raise ValueError(
                f"{type(self).__name__} models TP > global Hkv, so each rank must "
                f"hold exactly one KV head; got {k_local.shape[2]}"
            )
        return shape

    def _check_chunk_inputs(
        self,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        b: int,
        sq: int,
        d: int,
    ) -> None:
        if k_chunk.shape != (b, sq, 1, d) or v_chunk.shape != k_chunk.shape:
            raise ValueError("k_chunk and v_chunk must have shape [B, Sq, 1, 128]")
        for tensor, name in ((k_chunk, "k_chunk"), (v_chunk, "v_chunk")):
            self._check_cuda_bf16_contiguous(tensor, name)

    def _all_gather_q(
        self,
        q_local: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> torch.Tensor:
        if timing:
            self._record_phase("q_ag_start", compute_stream)
        if self.world_size == 1:
            q_group = q_local
        else:
            b, sq, h_local, d = q_local.shape
            q_rank_major = torch.empty(
                (self.world_size * b, sq, h_local, d),
                device=self.device,
                dtype=q_local.dtype,
            )
            with _nvtx_range(f"{self.method_name}_q_allgather"):
                dist.all_gather_into_tensor(
                    q_rank_major, q_local, group=self.process_group
                )
                q_group = (
                    q_rank_major.view(self.world_size, b, sq, h_local, d)
                    .permute(1, 2, 0, 3, 4)
                    .reshape(b, sq, self.world_size * h_local, d)
                    .contiguous()
                )
        if timing:
            self._record_phase("q_ag_end", compute_stream)
        return q_group

    def _all_gather_q_varlen_sequential(
        self,
        q_local: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> torch.Tensor:
        return self._all_gather_q(
            q_local.unsqueeze(0), compute_stream, timing=timing
        ).squeeze(0)

    def _run_history_attention_sequential(
        self,
        q_group: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        seqlens_local: torch.Tensor,
        num_splits: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timing:
            self._record_phase("history_start", compute_stream)
        with _nvtx_range(f"{self.method_name}_local_history_attention"):
            history_out, history_lse = min_fa3_op.forward_kvcache(
                q_group,
                k_local,
                v_local,
                seqlens_local,
                num_splits=num_splits,
                return_lse=True,
                is_causal=False,
            )
        if timing:
            self._record_phase("history_end", compute_stream)
        return history_out, history_lse

    def _run_chunk_attention_sequential(
        self,
        q_local: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        num_splits: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timing:
            self._record_phase("chunk_start", compute_stream)
        chunk_lengths = torch.full(
            (q_local.shape[0],),
            q_local.shape[1],
            device=self.device,
            dtype=torch.int32,
        )
        with _nvtx_range(f"{self.method_name}_local_chunk_attention"):
            chunk_out, chunk_lse = min_fa3_op.forward_kvcache(
                q_local,
                k_chunk,
                v_chunk,
                chunk_lengths,
                num_splits=num_splits,
                return_lse=True,
                is_causal=True,
            )
        if timing:
            self._record_phase("chunk_end", compute_stream)
        return chunk_out, chunk_lse

    def _run_history_attention_varlen_sequential(
        self,
        q_group: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_local: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k_local: int,
        cu_seqlens_q_host: torch.Tensor,
        cu_seqlens_k_local_host: torch.Tensor,
        num_splits: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timing:
            self._record_phase("history_start", compute_stream)
        with _nvtx_range(f"{self.method_name}_varlen_local_history_attention"):
            history_out, history_lse = min_fa3_op.forward_kvcache_varlen(
                q_group,
                k_local,
                v_local,
                cu_seqlens_q,
                cu_seqlens_k_local,
                max_seqlen_q,
                max_seqlen_k_local,
                cu_seqlens_q_host=cu_seqlens_q_host,
                cu_seqlens_k_host=cu_seqlens_k_local_host,
                num_splits=num_splits,
                return_lse=True,
                is_causal=False,
            )
        if timing:
            self._record_phase("history_end", compute_stream)
        return history_out, history_lse

    def _run_chunk_attention_varlen_sequential(
        self,
        q_local: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        cu_seqlens_q_host: torch.Tensor,
        num_splits: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timing:
            self._record_phase("chunk_start", compute_stream)
        with _nvtx_range(f"{self.method_name}_varlen_local_chunk_attention"):
            chunk_out, chunk_lse = min_fa3_op.forward_kvcache_varlen(
                q_local,
                k_chunk,
                v_chunk,
                cu_seqlens_q,
                cu_seqlens_q,
                max_seqlen_q,
                max_seqlen_q,
                cu_seqlens_q_host=cu_seqlens_q_host,
                cu_seqlens_k_host=cu_seqlens_q_host,
                num_splits=num_splits,
                return_lse=True,
                is_causal=True,
            )
        if timing:
            self._record_phase("chunk_end", compute_stream)
        return chunk_out, chunk_lse

    def _combine_context(
        self,
        history_out: torch.Tensor,
        history_lse: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _merge_context_and_chunk(
        self,
        context_out: torch.Tensor,
        context_lse: torch.Tensor,
        chunk_out: torch.Tensor,
        chunk_lse: torch.Tensor,
        return_lse: bool,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> _DCPResult:
        raise NotImplementedError

    def _combine_context_varlen_sequential(
        self,
        history_out: torch.Tensor,
        history_lse: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, lse = self._combine_context(
            history_out.unsqueeze(0),
            history_lse.unsqueeze(0),
            compute_stream,
            timing=timing,
        )
        return output.squeeze(0), lse.squeeze(0)

    def _merge_context_and_chunk_varlen_sequential(
        self,
        context_out: torch.Tensor,
        context_lse: torch.Tensor,
        chunk_out: torch.Tensor,
        chunk_lse: torch.Tensor,
        return_lse: bool,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> _DCPResult:
        result = self._merge_context_and_chunk(
            context_out.unsqueeze(0),
            context_lse.unsqueeze(0),
            chunk_out.unsqueeze(0),
            chunk_lse.unsqueeze(0),
            return_lse,
            compute_stream,
            timing=timing,
        )
        if return_lse:
            output, lse = result
            return output.squeeze(0), lse.squeeze(0)
        return result.squeeze(0)

    def forward_decode(
        self,
        q_local: torch.Tensor,
        k_cache_local: torch.Tensor,
        v_cache_local: torch.Tensor,
        cache_seqlens_local: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
    ) -> _DCPResult:
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError(f"{type(self).__name__} does not support concurrent calls")
        try:
            self._reap_inflight_tensors()
            self._reject_overlap(overlap_q_allgather)
            _, sq, _, _ = self._check_reference_common(
                q_local, k_cache_local, v_cache_local, cache_seqlens_local
            )
            if sq != 1:
                raise ValueError(f"forward_decode requires Sq=1, got {sq}")
            compute_stream = torch.cuda.current_stream(self.device)
            timing = self._phase_recorder is not None
            if timing:
                self._begin_phase_recording("decode", compute_stream)
            q_group = self._all_gather_q(
                q_local, compute_stream, timing=timing
            )
            if timing:
                self._record_phase("ag_chunk_end", compute_stream)
            history_out, history_lse = self._run_history_attention_sequential(
                q_group,
                k_cache_local,
                v_cache_local,
                cache_seqlens_local,
                num_splits,
                compute_stream,
                timing=timing,
            )
            output, lse = self._combine_context(
                history_out,
                history_lse,
                compute_stream,
                timing=timing,
            )
            if timing:
                self._record_phase("attention_end", compute_stream)
            return (output, lse) if return_lse else output
        finally:
            self._enqueue_lock.release()

    def forward_chunk_prefill(
        self,
        q_local: torch.Tensor,
        k_history_local: torch.Tensor,
        v_history_local: torch.Tensor,
        history_seqlens_local: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
    ) -> _DCPResult:
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError(f"{type(self).__name__} does not support concurrent calls")
        try:
            self._reap_inflight_tensors()
            self._reject_overlap(overlap_q_allgather)
            b, sq, _, d = self._check_reference_common(
                q_local,
                k_history_local,
                v_history_local,
                history_seqlens_local,
            )
            self._check_chunk_inputs(k_chunk, v_chunk, b, sq, d)
            compute_stream = torch.cuda.current_stream(self.device)
            timing = self._phase_recorder is not None
            if timing:
                self._begin_phase_recording("chunk", compute_stream)

            chunk_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
            if self.chunk_before_context:
                chunk_state = self._run_chunk_attention_sequential(
                    q_local,
                    k_chunk,
                    v_chunk,
                    num_splits,
                    compute_stream,
                    timing=timing,
                )

            q_group = self._all_gather_q(
                q_local, compute_stream, timing=timing
            )
            if timing and self.chunk_before_context:
                self._record_phase("ag_chunk_end", compute_stream)
            history_out, history_lse = self._run_history_attention_sequential(
                q_group,
                k_history_local,
                v_history_local,
                history_seqlens_local,
                num_splits,
                compute_stream,
                timing=timing,
            )
            context_out, context_lse = self._combine_context(
                history_out,
                history_lse,
                compute_stream,
                timing=timing,
            )

            if chunk_state is None:
                chunk_state = self._run_chunk_attention_sequential(
                    q_local,
                    k_chunk,
                    v_chunk,
                    num_splits,
                    compute_stream,
                    timing=timing,
                )
                if timing:
                    self._record_phase("ag_chunk_end", compute_stream)
            result = self._merge_context_and_chunk(
                context_out,
                context_lse,
                *chunk_state,
                return_lse,
                compute_stream,
                timing=timing,
            )
            if timing:
                self._record_phase("attention_end", compute_stream)
            return result
        finally:
            self._enqueue_lock.release()

    def forward_decode_varlen(
        self,
        q_local: torch.Tensor,
        k_cache_local: torch.Tensor,
        v_cache_local: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_local: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k_local: int,
        *,
        cu_seqlens_q_host: torch.Tensor,
        cu_seqlens_k_local_host: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
    ) -> _DCPResult:
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError(f"{type(self).__name__} does not support concurrent calls")
        try:
            self._reap_inflight_tensors()
            self._reject_overlap(overlap_q_allgather)
            (
                _,
                _,
                _,
                h_kv,
                _,
                q_lengths,
                _,
            ) = self._check_packed_common(
                q_local,
                k_cache_local,
                v_cache_local,
                cu_seqlens_q,
                cu_seqlens_k_local,
                max_seqlen_q,
                max_seqlen_k_local,
                cu_seqlens_q_host,
                cu_seqlens_k_local_host,
                num_splits,
            )
            self._check_packed_runner_topology(h_kv)
            if any(length != 1 for length in q_lengths):
                raise ValueError("forward_decode_varlen requires every q_len == 1")

            compute_stream = torch.cuda.current_stream(self.device)
            timing = self._phase_recorder is not None
            if timing:
                self._begin_phase_recording("decode", compute_stream)
            q_group = self._all_gather_q_varlen_sequential(
                q_local, compute_stream, timing=timing
            )
            if timing:
                self._record_phase("ag_chunk_end", compute_stream)
            history_out, history_lse = self._run_history_attention_varlen_sequential(
                q_group,
                k_cache_local,
                v_cache_local,
                cu_seqlens_q,
                cu_seqlens_k_local,
                max_seqlen_q,
                max_seqlen_k_local,
                cu_seqlens_q_host,
                cu_seqlens_k_local_host,
                num_splits,
                compute_stream,
                timing=timing,
            )
            output, lse = self._combine_context_varlen_sequential(
                history_out,
                history_lse,
                compute_stream,
                timing=timing,
            )
            if timing:
                self._record_phase("attention_end", compute_stream)
            return (output, lse) if return_lse else output
        finally:
            self._enqueue_lock.release()

    def forward_chunk_prefill_varlen(
        self,
        q_local: torch.Tensor,
        k_history_local: torch.Tensor,
        v_history_local: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_history_local: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_history_local: int,
        *,
        cu_seqlens_q_host: torch.Tensor,
        cu_seqlens_history_local_host: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        overlap_q_allgather: bool = False,
    ) -> _DCPResult:
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError(f"{type(self).__name__} does not support concurrent calls")
        try:
            self._reap_inflight_tensors()
            self._reject_overlap(overlap_q_allgather)
            (
                total_q,
                _,
                d,
                h_kv,
                _,
                _,
                _,
            ) = self._check_packed_common(
                q_local,
                k_history_local,
                v_history_local,
                cu_seqlens_q,
                cu_seqlens_history_local,
                max_seqlen_q,
                max_seqlen_history_local,
                cu_seqlens_q_host,
                cu_seqlens_history_local_host,
                num_splits,
            )
            self._check_packed_runner_topology(h_kv)
            self._check_packed_chunk_inputs(k_chunk, v_chunk, total_q, h_kv, d)

            compute_stream = torch.cuda.current_stream(self.device)
            timing = self._phase_recorder is not None
            if timing:
                self._begin_phase_recording("chunk", compute_stream)

            chunk_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
            if self.chunk_before_context:
                chunk_state = self._run_chunk_attention_varlen_sequential(
                    q_local,
                    k_chunk,
                    v_chunk,
                    cu_seqlens_q,
                    max_seqlen_q,
                    cu_seqlens_q_host,
                    num_splits,
                    compute_stream,
                    timing=timing,
                )

            q_group = self._all_gather_q_varlen_sequential(
                q_local, compute_stream, timing=timing
            )
            if timing and self.chunk_before_context:
                self._record_phase("ag_chunk_end", compute_stream)
            history_out, history_lse = self._run_history_attention_varlen_sequential(
                q_group,
                k_history_local,
                v_history_local,
                cu_seqlens_q,
                cu_seqlens_history_local,
                max_seqlen_q,
                max_seqlen_history_local,
                cu_seqlens_q_host,
                cu_seqlens_history_local_host,
                num_splits,
                compute_stream,
                timing=timing,
            )
            context_out, context_lse = self._combine_context_varlen_sequential(
                history_out,
                history_lse,
                compute_stream,
                timing=timing,
            )

            if chunk_state is None:
                chunk_state = self._run_chunk_attention_varlen_sequential(
                    q_local,
                    k_chunk,
                    v_chunk,
                    cu_seqlens_q,
                    max_seqlen_q,
                    cu_seqlens_q_host,
                    num_splits,
                    compute_stream,
                    timing=timing,
                )
                if timing:
                    self._record_phase("ag_chunk_end", compute_stream)
            result = self._merge_context_and_chunk_varlen_sequential(
                context_out,
                context_lse,
                *chunk_state,
                return_lse,
                compute_stream,
                timing=timing,
            )
            if timing:
                self._record_phase("attention_end", compute_stream)
            return result
        finally:
            self._enqueue_lock.release()


class VLLMDCPAttentionRunner(_SequentialDCPAttentionRunnerBase):
    """Pinned vLLM default AG+RS orchestration using the local min FA3 op."""

    method_name = "vllm_ag_rs_min_fa3"
    varlen_method_name = "vllm_ag_rs_min_fa3_varlen"
    output_collective_kind = "bf16_reduce_scatter"

    def _combine_context(
        self,
        history_out: torch.Tensor,
        history_lse: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, sq, h_group, d = history_out.shape
        t = b * sq
        h_local = h_group // self.world_size
        if self.world_size == 1:
            return history_out, history_lse

        if timing:
            self._record_phase("lse_correct_start", compute_stream)
        with _nvtx_range("vllm_lse_allgather_and_triton_correction"):
            lse_flat = history_lse.permute(0, 2, 1).contiguous().view(t, h_group)
            gathered_lse = torch.empty(
                (self.world_size * t, h_group),
                device=self.device,
                dtype=torch.float32,
            )
            dist.all_gather_into_tensor(
                gathered_lse, lse_flat, group=self.process_group
            )
            gathered_lse_view = gathered_lse.view(self.world_size, t, h_group)
            corrected = torch.empty_like(history_out)
            global_lse_flat = torch.empty(
                (t, h_group), device=self.device, dtype=torch.float32
            )
            _vllm_correct_attn_cp_out_kernel[(t, h_group)](
                history_out.view(t, h_group, d),
                gathered_lse_view,
                corrected.view(t, h_group, d),
                global_lse_flat,
                *history_out.view(t, h_group, d).stride()[:2],
                *gathered_lse_view.stride(),
                T=t,
                H=h_group,
                D=d,
                WORLD_SIZE=self.world_size,
                WORLD_SIZE_ROUNDED=triton.next_power_of_2(self.world_size),
                RANK=self.rank,
                num_warps=4,
            )
        if timing:
            self._record_phase("lse_correct_end", compute_stream)
            self._record_phase("reduce_scatter_start", compute_stream)
        with _nvtx_range("vllm_bf16_output_reduce_scatter"):
            packed_head_major = corrected.view(t, h_group, d).movedim(0, 1).contiguous()
            output_head_major = torch.empty(
                (h_local, t, d), device=self.device, dtype=torch.bfloat16
            )
            dist.reduce_scatter_tensor(
                output_head_major,
                packed_head_major,
                op=dist.ReduceOp.SUM,
                group=self.process_group,
            )
            output = (
                output_head_major.movedim(0, 1)
                .contiguous()
                .view(b, sq, h_local, d)
            )
        if timing:
            self._record_phase("reduce_scatter_end", compute_stream)
        head_start = self.rank * h_local
        local_lse = (
            global_lse_flat.view(b, sq, h_group)[:, :, head_start : head_start + h_local]
            .permute(0, 2, 1)
            .contiguous()
        )
        return output, local_lse

    def _merge_context_and_chunk(
        self,
        context_out: torch.Tensor,
        context_lse: torch.Tensor,
        chunk_out: torch.Tensor,
        chunk_lse: torch.Tensor,
        return_lse: bool,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> _DCPResult:
        b, sq, h_local, d = context_out.shape
        merged_out = torch.empty_like(context_out)
        merged_lse = (
            torch.empty_like(context_lse) if return_lse else context_lse
        )
        if timing:
            self._record_phase("merge_start", compute_stream)
        with _nvtx_range("vllm_triton_merge_attn_states"):
            _merge_attn_states_kernel[(b * sq, h_local)](
                context_out,
                context_lse,
                chunk_out,
                chunk_lse,
                merged_out,
                merged_lse,
                *merged_out.stride()[:3],
                *context_lse.stride(),
                B=b,
                SQ=sq,
                H=h_local,
                D=d,
                STORE_LSE=return_lse,
                num_warps=4,
            )
        if timing:
            self._record_phase("merge_end", compute_stream)
        self._retain_merge_inputs(
            compute_stream, context_out, context_lse, chunk_out, chunk_lse
        )
        return (merged_out, merged_lse) if return_lse else merged_out


class VLLMA2ADCPAttentionRunner(VLLMDCPAttentionRunner):
    """Pinned vLLM A2A orchestration using the local min FA3 op.

    The data flow matches the ordinary GQA backend integration at
    ``vllm/v1/attention/backends/flash_attn.py`` in pinned commit
    ``a89015c6df8eeb37a843b717c97a5be1355de83d``.  Only the partial-state
    combine differs from :class:`VLLMDCPAttentionRunner`.
    """

    method_name = "vllm_a2a_min_fa3"
    varlen_method_name = "vllm_a2a_min_fa3_varlen"
    output_collective_kind = "bf16_packed_all_to_all"

    def _combine_context(
        self,
        history_out: torch.Tensor,
        history_lse: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_out.ndim != 4:
            raise ValueError("A2A history output must have shape [B, S, H_group, D]")
        b, sq, h_group, d = history_out.shape
        if history_out.dtype != torch.bfloat16 or d != 128:
            raise ValueError("A2A history output must be BF16 with head_dim 128")
        if history_lse.shape != (b, h_group, sq):
            raise ValueError(
                "A2A history LSE must have shape "
                f"[{b}, {h_group}, {sq}], got {tuple(history_lse.shape)}"
            )
        _dcp_a2a_head_owner_ranges(h_group, self.world_size)
        if history_lse.dtype != torch.float32:
            # vLLM PR #47801: the pack kernel bit-casts one FP32 LSE into two
            # BF16 lanes even when an attention backend returned activation dtype.
            history_lse = history_lse.to(torch.float32)
        if self.world_size == 1:
            return history_out, history_lse

        total_tokens = b * sq
        h_local = h_group // self.world_size
        output_flat = history_out.view(total_tokens, h_group, d)
        lse_flat = (
            history_lse.permute(0, 2, 1)
            .contiguous()
            .view(total_tokens, h_group)
        )
        lse_pack_dim = _dcp_a2a_lse_pack_dim(history_out.dtype)
        buffer_shape = (
            self.world_size,
            total_tokens,
            h_local,
            d + lse_pack_dim,
        )
        # vLLM PR #45487: graph-captured A2A buffers must be per-call tensors
        # owned by the graph private pool, never slices of a growable workspace.
        send_buffer = torch.empty(
            buffer_shape, device=self.device, dtype=history_out.dtype
        )
        recv_buffer = torch.empty_like(send_buffer)
        if self._capture_in_progress:
            self._capture_tensors.extend((send_buffer, recv_buffer))

        if timing:
            self._record_phase("a2a_pack_start", compute_stream)
        with _nvtx_range("vllm_a2a_pack_partial_states"):
            _dcp_a2a_pack_send(
                output_flat,
                lse_flat,
                send_buffer,
                self.world_size,
                h_local,
                d,
                lse_pack_dim,
            )
        if timing:
            self._record_phase("a2a_pack_end", compute_stream)
            self._record_phase("a2a_all_to_all_start", compute_stream)
        with _nvtx_range("vllm_packed_output_lse_all_to_all"):
            work = dist.all_to_all_single(
                recv_buffer.view(-1),
                send_buffer.view(-1),
                group=self.process_group,
                async_op=True,
            )
            work.wait()
        if self._capture_in_progress:
            self._works.append(work)
        if timing:
            self._record_phase("a2a_all_to_all_end", compute_stream)
            self._record_phase("a2a_unpack_combine_start", compute_stream)
        with _nvtx_range("vllm_a2a_unpack_lse_weighted_combine"):
            output, output_lse = _dcp_a2a_unpack_combine(
                recv_buffer, d, lse_pack_dim
            )
        if timing:
            self._record_phase("a2a_unpack_combine_end", compute_stream)
        self._retain_merge_inputs(
            compute_stream,
            history_out,
            history_lse,
            lse_flat,
            send_buffer,
            recv_buffer,
        )
        return (
            output.view(b, sq, h_local, d),
            output_lse.view(b, sq, h_local).permute(0, 2, 1).contiguous(),
        )


class SGLangDCPAttentionRunner(_SequentialDCPAttentionRunnerBase):
    """Pinned SGLang MHA AG+FP32-AR orchestration using the local min FA3 op."""

    method_name = "sglang_mha_ag_ar_min_fa3"
    varlen_method_name = "sglang_mha_ag_ar_min_fa3_varlen"
    output_collective_kind = "fp32_all_reduce"
    chunk_before_context = True

    def _combine_context(
        self,
        history_out: torch.Tensor,
        history_lse: torch.Tensor,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, sq, h_group, d = history_out.shape
        t = b * sq
        h_local = h_group // self.world_size
        if self.world_size == 1:
            return history_out, history_lse

        if timing:
            self._record_phase("lse_correct_start", compute_stream)
        with _nvtx_range("sglang_lse_allgather_torch_correction"):
            local_lse = history_lse.permute(0, 2, 1).contiguous().view(t, h_group)
            gathered_lse = torch.empty(
                (self.world_size * t, h_group),
                device=self.device,
                dtype=torch.float32,
            )
            dist.all_gather_into_tensor(
                gathered_lse, local_lse, group=self.process_group
            )
            global_lse = torch.logsumexp(
                gathered_lse.view(self.world_size, t, h_group), dim=0
            )
            scale = torch.exp(local_lse - global_lse).unsqueeze(-1)
            scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
            partial = torch.nan_to_num(
                history_out.view(t, h_group, d).float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ) * scale
        if timing:
            self._record_phase("lse_correct_end", compute_stream)
            self._record_phase("reduce_scatter_start", compute_stream)
        with _nvtx_range("sglang_fp32_output_allreduce"):
            dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=self.process_group)
            head_start = self.rank * h_local
            output = (
                partial[:, head_start : head_start + h_local]
                .contiguous()
                .to(torch.bfloat16)
                .view(b, sq, h_local, d)
            )
        if timing:
            self._record_phase("reduce_scatter_end", compute_stream)
        local_global_lse = (
            global_lse.view(b, sq, h_group)[:, :, head_start : head_start + h_local]
            .permute(0, 2, 1)
            .contiguous()
        )
        return output, local_global_lse

    def _merge_context_and_chunk(
        self,
        context_out: torch.Tensor,
        context_lse: torch.Tensor,
        chunk_out: torch.Tensor,
        chunk_lse: torch.Tensor,
        return_lse: bool,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> _DCPResult:
        if timing:
            self._record_phase("merge_start", compute_stream)
        with _nvtx_range("sglang_torch_fp32_merge_attn_states"):
            merged_lse = torch.logaddexp(context_lse, chunk_lse)
            context_scale = (
                torch.exp(context_lse - merged_lse).permute(0, 2, 1).unsqueeze(-1)
            )
            chunk_scale = (
                torch.exp(chunk_lse - merged_lse).permute(0, 2, 1).unsqueeze(-1)
            )
            context_scale = torch.nan_to_num(
                context_scale, nan=0.0, posinf=0.0, neginf=0.0
            )
            chunk_scale = torch.nan_to_num(
                chunk_scale, nan=0.0, posinf=0.0, neginf=0.0
            )
            merged_out = (
                context_out.float() * context_scale + chunk_out.float() * chunk_scale
            ).to(torch.bfloat16)
        if timing:
            self._record_phase("merge_end", compute_stream)
        return (merged_out, merged_lse) if return_lse else merged_out


class TimedVLLMDCPAttentionRunner(
    BenchmarkTimingMixin, VLLMDCPAttentionRunner
):
    """Benchmark-only vLLM AG+RS runner with phase events installed."""


class TimedVLLMA2ADCPAttentionRunner(
    BenchmarkTimingMixin, VLLMA2ADCPAttentionRunner
):
    """Benchmark-only vLLM A2A runner with phase events installed."""


class TimedSGLangDCPAttentionRunner(
    BenchmarkTimingMixin, SGLangDCPAttentionRunner
):
    """Benchmark-only SGLang runner with phase events installed."""


def full_kv_reference_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlens: torch.Tensor,
    *,
    num_splits: int = 0,
    return_lse: bool = False,
    is_causal: bool = False,
):
    """Run the local dense kernel against an unsharded full-KV cache."""
    return min_fa3_op.forward_kvcache(
        q,
        k,
        v,
        seqlens,
        num_splits=num_splits,
        return_lse=return_lse,
        is_causal=is_causal,
    )


def full_kv_reference_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    cu_seqlens_q_host: torch.Tensor,
    cu_seqlens_k_host: torch.Tensor,
    num_splits: int = 0,
    return_lse: bool = False,
    is_causal: bool = False,
):
    """Run the local packed kernel against an unsharded full-KV cache."""
    return min_fa3_op.forward_kvcache_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        cu_seqlens_q_host=cu_seqlens_q_host,
        cu_seqlens_k_host=cu_seqlens_k_host,
        num_splits=num_splits,
        return_lse=return_lse,
        is_causal=is_causal,
    )


__all__ = [
    "SGLangDCPAttentionRunner",
    "TimedSGLangDCPAttentionRunner",
    "TimedVLLMA2ADCPAttentionRunner",
    "TimedVLLMDCPAttentionRunner",
    "VLLMA2ADCPAttentionRunner",
    "VLLMDCPAttentionRunner",
    "_dcp_a2a_head_owner_ranges",
    "_dcp_a2a_lse_weighted_combine_reference",
    "_dcp_a2a_payload_bytes",
    "full_kv_reference_dense",
    "full_kv_reference_varlen",
]
