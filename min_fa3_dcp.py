"""Head-sharded decode context parallel attention for the minimal FA3 demo.

The LSE correction is copied and trimmed from vLLM commit
``a89015c6df8eeb37a843b717c97a5be1355de83d``:
``vllm/v1/attention/ops/common.py::cp_lse_ag_out_rs``.  The two-state merge is
copied and trimmed from the same commit's
``vllm/v1/attention/ops/triton_merge_attn_states.py``.  This module has no
runtime dependency on vLLM or SGLang.

The communication-stream/event ordering follows the side-stream design used
by SGLang commit ``8d6549bc4039d33635844495d86684677a4f0df8``.  The runner is
not CUDA-graph capturable and its workspace must not be used concurrently.
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import min_fa3_op


_DCPResult = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


@triton.jit
def _correct_and_pack_kernel(
    partial_out_ptr,
    gathered_lse_ptr,
    packed_out_ptr,
    local_lse_ptr,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    lse_stride_n,
    lse_stride_b,
    lse_stride_h,
    lse_stride_s,
    local_lse_stride_b,
    local_lse_stride_h,
    local_lse_stride_s,
    B: tl.constexpr,
    SQ: tl.constexpr,
    H_GROUP: tl.constexpr,
    H_LOCAL: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    WORLD_SIZE_ROUNDED: tl.constexpr,
    RANK: tl.constexpr,
    STORE_LOCAL_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = token_idx // SQ
    query_idx = token_idx - batch_idx * SQ

    rank_offsets = tl.arange(0, WORLD_SIZE_ROUNDED)
    rank_mask = rank_offsets < WORLD_SIZE
    lse_offsets = (
        rank_offsets * lse_stride_n
        + batch_idx * lse_stride_b
        + head_idx * lse_stride_h
        + query_idx * lse_stride_s
    )
    partial_lses = tl.load(gathered_lse_ptr + lse_offsets, mask=rank_mask, other=-float("inf"))
    partial_lses = tl.where(
        (partial_lses != partial_lses) | (partial_lses == float("inf")),
        -float("inf"),
        partial_lses,
    )
    lse_max = tl.max(partial_lses, axis=0)
    finite_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    lse_sum = tl.sum(tl.exp(partial_lses - finite_max), axis=0)
    global_lse = tl.log(lse_sum) + finite_max

    local_lse = tl.load(
        gathered_lse_ptr
        + RANK * lse_stride_n
        + batch_idx * lse_stride_b
        + head_idx * lse_stride_h
        + query_idx * lse_stride_s
    )
    log_weight = local_lse - global_lse
    log_weight = tl.where(
        (log_weight != log_weight) | (log_weight == float("inf")),
        -float("inf"),
        log_weight,
    )
    weight = tl.exp(log_weight)

    dim_offsets = tl.arange(0, D)
    partial_offsets = (
        batch_idx * out_stride_b
        + query_idx * out_stride_s
        + head_idx * out_stride_h
        + dim_offsets
    )
    values = tl.load(partial_out_ptr + partial_offsets).to(tl.float32) * weight
    values = tl.where(weight == 0.0, 0.0, values)

    heads_per_group_kv = H_GROUP // H_KV
    heads_per_local_kv = H_LOCAL // H_KV
    kv_head = head_idx // heads_per_group_kv
    head_within_group_kv = head_idx - kv_head * heads_per_group_kv
    owner = head_within_group_kv // heads_per_local_kv
    local_head = (
        kv_head * heads_per_local_kv
        + head_within_group_kv
        - owner * heads_per_local_kv
    )
    packed_offsets = (
        ((((owner * B + batch_idx) * SQ + query_idx) * H_LOCAL + local_head) * D)
        + dim_offsets
    )
    tl.store(packed_out_ptr + packed_offsets, values)

    if STORE_LOCAL_LSE:
        if owner == RANK:
            local_lse_offset = (
                batch_idx * local_lse_stride_b
                + local_head * local_lse_stride_h
                + query_idx * local_lse_stride_s
            )
            tl.store(local_lse_ptr + local_lse_offset, global_lse)


@triton.jit
def _merge_attn_states_kernel(
    context_out_ptr,
    context_lse_ptr,
    chunk_out_ptr,
    chunk_lse_ptr,
    merged_out_ptr,
    merged_lse_ptr,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    lse_stride_b,
    lse_stride_h,
    lse_stride_s,
    B: tl.constexpr,
    SQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = token_idx // SQ
    query_idx = token_idx - batch_idx * SQ
    lse_offset = (
        batch_idx * lse_stride_b
        + head_idx * lse_stride_h
        + query_idx * lse_stride_s
    )
    context_lse = tl.load(context_lse_ptr + lse_offset).to(tl.float32)
    chunk_lse = tl.load(chunk_lse_ptr + lse_offset).to(tl.float32)
    context_lse = tl.where(
        (context_lse != context_lse) | (context_lse == float("inf")),
        -float("inf"),
        context_lse,
    )
    chunk_lse = tl.where(
        (chunk_lse != chunk_lse) | (chunk_lse == float("inf")),
        -float("inf"),
        chunk_lse,
    )
    lse_max = tl.maximum(context_lse, chunk_lse)
    finite_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    context_weight = tl.exp(context_lse - finite_max)
    chunk_weight = tl.exp(chunk_lse - finite_max)
    weight_sum = context_weight + chunk_weight
    merged_lse = tl.log(weight_sum) + finite_max
    inverse_weight_sum = tl.where(weight_sum == 0.0, 0.0, 1.0 / weight_sum)

    dim_offsets = tl.arange(0, D)
    out_offset = (
        batch_idx * out_stride_b
        + query_idx * out_stride_s
        + head_idx * out_stride_h
        + dim_offsets
    )
    context_out = tl.load(context_out_ptr + out_offset).to(tl.float32)
    chunk_out = tl.load(chunk_out_ptr + out_offset).to(tl.float32)
    merged_out = (
        context_out * context_weight + chunk_out * chunk_weight
    ) * inverse_weight_sum
    merged_out = tl.where(weight_sum == 0.0, 0.0, merged_out)
    tl.store(merged_out_ptr + out_offset, merged_out)
    if STORE_LSE:
        tl.store(merged_lse_ptr + lse_offset, merged_lse)


def _nvtx_range(name: str):
    return torch.cuda.nvtx.range(name) if hasattr(torch.cuda, "nvtx") else nullcontext()


class DCPAttentionRunner:
    """Reusable head-sharded DCP workspace backed by one NCCL process group.

    Calls may be enqueued back-to-back, including from different current CUDA
    streams.  Calling the same runner concurrently from multiple host threads
    is unsupported and raises ``RuntimeError``.
    """

    def __init__(self, process_group: Optional[dist.ProcessGroup]):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before DCPAttentionRunner")
        if not torch.cuda.is_available():
            raise RuntimeError("DCPAttentionRunner requires CUDA")
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)
        if self.world_size <= 0:
            raise RuntimeError("DCP process group must contain at least one rank")
        if self.world_size > 8:
            raise ValueError(f"This minimal DCP runner supports at most 8 ranks, got {self.world_size}")
        backend = str(dist.get_backend(process_group)).lower()
        if "nccl" not in backend:
            raise RuntimeError(f"DCPAttentionRunner requires an NCCL process group, got {backend}")

        self.device = torch.device("cuda", torch.cuda.current_device())
        self.communication_stream = torch.cuda.Stream(device=self.device)
        self._input_ready = torch.cuda.Event()
        self._q_group_ready = torch.cuda.Event()
        self._history_ready = torch.cuda.Event()
        self._context_ready = torch.cuda.Event()
        self._buffers: dict[str, torch.Tensor] = {}
        self._works: list[dist.Work] = []
        self._inflight_tensors: list[
            tuple[torch.cuda.Event, tuple[torch.Tensor, ...]]
        ] = []
        self._enqueue_lock = threading.Lock()
        self._timing_events = {
            name: torch.cuda.Event(enable_timing=True)
            for name in (
                "attention_start",
                "attention_end",
                "q_ag_start",
                "q_ag_end",
                "chunk_start",
                "chunk_end",
                "ag_chunk_end",
                "history_start",
                "history_end",
                "lse_correct_start",
                "lse_correct_end",
                "reduce_scatter_start",
                "reduce_scatter_end",
                "merge_start",
                "merge_end",
            )
        }
        self._last_timing_kind: Optional[str] = None

    def _buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        numel = 1
        for extent in shape:
            numel *= extent
        storage = self._buffers.get(name)
        if (
            storage is None
            or storage.device != self.device
            or storage.dtype != dtype
            or storage.numel() < numel
        ):
            storage = torch.empty(numel, device=self.device, dtype=dtype)
            self._buffers[name] = storage
        return storage[:numel].view(shape)

    def _record_timing(self, name: str, stream: torch.cuda.Stream) -> None:
        self._timing_events[name].record(stream)

    def _retain_work(self, work: Optional[dist.Work]) -> None:
        if work is not None:
            self._works = [pending for pending in self._works if not pending.is_completed()]
            self._works.append(work)
            # This inserts a completion dependency only on the currently active
            # G stream and is guaranteed to return immediately.  Work.wait()
            # here can delay host submission of local chunk attention on C.
            work.block_current_stream()

    def _reap_inflight_tensors(self) -> None:
        self._inflight_tensors = [
            item for item in self._inflight_tensors if not item[0].query()
        ]

    def _retain_merge_inputs(
        self,
        compute_stream: torch.cuda.Stream,
        *tensors: torch.Tensor,
    ) -> None:
        # Triton launches with raw pointers rather than through the PyTorch
        # dispatcher.  Keep its inputs alive until the merge kernel completes,
        # including when the next call uses a different current stream.
        completion = torch.cuda.Event()
        completion.record(compute_stream)
        self._inflight_tensors.append((completion, tensors))

    def _check_common(
        self,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        seqlens_local: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DCPAttentionRunner does not support CUDA Graph capture")
        if q_local.device != self.device:
            raise ValueError(f"q_local must be on runner device {self.device}, got {q_local.device}")
        for tensor, name in ((q_local, "q_local"), (k_local, "k_local"), (v_local, "v_local")):
            if not tensor.is_cuda or tensor.dtype != torch.bfloat16 or tensor.ndim != 4:
                raise ValueError(f"{name} must be a CUDA BF16 tensor with shape [B, S, H, 128]")
            if tensor.shape[-1] != 128 or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous BSHD with head_dim 128")
            if tensor.device != self.device:
                raise ValueError(f"{name} must be on runner device {self.device}")
        if k_local.shape != v_local.shape:
            raise ValueError("k_local and v_local must have identical shapes")
        b, sq, h_local, d = q_local.shape
        if b <= 0 or sq <= 0 or h_local <= 0:
            raise ValueError("q_local requires positive B, Sq, and Hq_local")
        if k_local.shape[0] != b or k_local.shape[2] <= 0:
            raise ValueError("local K/V batch must match q_local and Hkv_group must be positive")
        h_kv = k_local.shape[2]
        if h_local % h_kv:
            raise ValueError(
                f"Hq_local must be divisible by Hkv_group, got {h_local} and {h_kv}"
            )
        if (
            not seqlens_local.is_cuda
            or seqlens_local.device != self.device
            or seqlens_local.dtype != torch.int32
            or seqlens_local.shape != (b,)
            or not seqlens_local.is_contiguous()
        ):
            raise ValueError("local sequence lengths must be contiguous CUDA int32 with shape [B]")
        return b, sq, h_local, d

    def _start_q_allgather(
        self,
        q_local: torch.Tensor,
        h_kv: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> torch.Tensor:
        b, sq, h_local, d = q_local.shape
        q_rank_major = self._buffer(
            "q_rank_major", (self.world_size * b, sq, h_local, d), q_local.dtype
        )
        q_group = self._buffer(
            "q_group", (b, sq, self.world_size * h_local, d), q_local.dtype
        )
        q_local.record_stream(self.communication_stream)
        q_rank_major.record_stream(self.communication_stream)
        q_group.record_stream(self.communication_stream)
        with torch.cuda.stream(self.communication_stream):
            self.communication_stream.wait_event(self._input_ready)
            if timing:
                self._record_timing("q_ag_start", self.communication_stream)
            with _nvtx_range("dcp_q_allgather"):
                work = dist.all_gather_into_tensor(
                    q_rank_major,
                    q_local,
                    group=self.process_group,
                    async_op=True,
                )
                self._retain_work(work)
            with _nvtx_range("dcp_q_rank_major_to_bshd"):
                heads_per_local_kv = h_local // h_kv
                rank_major_view = q_rank_major.view(
                    self.world_size,
                    b,
                    sq,
                    h_kv,
                    heads_per_local_kv,
                    d,
                ).permute(1, 2, 3, 0, 4, 5)
                q_group.view(
                    b,
                    sq,
                    h_kv,
                    self.world_size,
                    heads_per_local_kv,
                    d,
                ).copy_(rank_major_view)
            if timing:
                self._record_timing("q_ag_end", self.communication_stream)
            self._q_group_ready.record(self.communication_stream)
        q_group.record_stream(compute_stream)
        return q_group

    def _finish_context(
        self,
        history_out: torch.Tensor,
        history_lse: torch.Tensor,
        output: torch.Tensor,
        local_lse: Optional[torch.Tensor],
        h_kv: int,
        compute_stream: torch.cuda.Stream,
        *,
        timing: bool,
    ) -> None:
        b, sq, h_group, d = history_out.shape
        h_local = h_group // self.world_size
        gathered_lse = self._buffer(
            "gathered_lse",
            (self.world_size * b, h_group, sq),
            torch.float32,
        )
        packed_out = self._buffer(
            "packed_corrected_out",
            (self.world_size * b, sq, h_local, d),
            torch.bfloat16,
        )
        for tensor in (history_out, history_lse, output):
            tensor.record_stream(self.communication_stream)
        if local_lse is not None:
            local_lse.record_stream(self.communication_stream)
        gathered_lse.record_stream(self.communication_stream)
        packed_out.record_stream(self.communication_stream)

        with torch.cuda.stream(self.communication_stream):
            self.communication_stream.wait_event(self._history_ready)
            if timing:
                self._record_timing("lse_correct_start", self.communication_stream)
            with _nvtx_range("dcp_lse_allgather"):
                work = dist.all_gather_into_tensor(
                    gathered_lse,
                    history_lse,
                    group=self.process_group,
                    async_op=True,
                )
                self._retain_work(work)
            with _nvtx_range("dcp_lse_correct_and_head_pack"):
                local_lse_arg = local_lse if local_lse is not None else gathered_lse
                _correct_and_pack_kernel[(b * sq, h_group)](
                    history_out,
                    gathered_lse,
                    packed_out,
                    local_lse_arg,
                    *history_out.stride()[:3],
                    *gathered_lse.view(self.world_size, b, h_group, sq).stride(),
                    *(local_lse.stride() if local_lse is not None else (0, 0, 0)),
                    B=b,
                    SQ=sq,
                    H_GROUP=h_group,
                    H_LOCAL=h_local,
                    H_KV=h_kv,
                    D=d,
                    WORLD_SIZE=self.world_size,
                    WORLD_SIZE_ROUNDED=triton.next_power_of_2(self.world_size),
                    RANK=self.rank,
                    STORE_LOCAL_LSE=local_lse is not None,
                    num_warps=4,
                )
            if timing:
                self._record_timing("lse_correct_end", self.communication_stream)
                self._record_timing("reduce_scatter_start", self.communication_stream)
            with _nvtx_range("dcp_output_reduce_scatter"):
                work = dist.reduce_scatter_tensor(
                    output,
                    packed_out,
                    op=dist.ReduceOp.SUM,
                    group=self.process_group,
                    async_op=True,
                )
                self._retain_work(work)
            if timing:
                self._record_timing("reduce_scatter_end", self.communication_stream)
            self._context_ready.record(self.communication_stream)

        output.record_stream(compute_stream)
        if local_lse is not None:
            local_lse.record_stream(compute_stream)

    def _run_context_attention(
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
        compute_stream.wait_event(self._q_group_ready)
        if timing:
            self._record_timing("ag_chunk_end", compute_stream)
            self._record_timing("history_start", compute_stream)
        with _nvtx_range("dcp_local_history_attention"):
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
            self._record_timing("history_end", compute_stream)
        self._history_ready.record(compute_stream)
        return history_out, history_lse

    def forward_decode(
        self,
        q_local: torch.Tensor,
        k_cache_local: torch.Tensor,
        v_cache_local: torch.Tensor,
        cache_seqlens_local: torch.Tensor,
        num_splits: int = 0,
        return_lse: bool = False,
        *,
        _record_timing: bool = False,
    ) -> _DCPResult:
        """Run DCP decode for ``q_local`` with shape ``[B, 1, Hq_local, 128]``."""
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError("DCPAttentionRunner does not support concurrent forward calls")
        try:
            self._reap_inflight_tensors()
            b, sq, h_local, _ = self._check_common(
                q_local, k_cache_local, v_cache_local, cache_seqlens_local
            )
            if sq != 1:
                raise ValueError(f"forward_decode requires Sq=1, got {sq}")
            compute_stream = torch.cuda.current_stream(self.device)
            if _record_timing:
                self._last_timing_kind = "decode"
                self._record_timing("attention_start", compute_stream)
            self._input_ready.record(compute_stream)

            if self.world_size == 1:
                if _record_timing:
                    self._record_timing("q_ag_start", compute_stream)
                    self._record_timing("q_ag_end", compute_stream)
                    self._record_timing("ag_chunk_end", compute_stream)
                    self._record_timing("history_start", compute_stream)
                with _nvtx_range("dcp_local_history_attention"):
                    result = min_fa3_op.forward_kvcache(
                        q_local,
                        k_cache_local,
                        v_cache_local,
                        cache_seqlens_local,
                        num_splits=num_splits,
                        return_lse=return_lse,
                    )
                if _record_timing:
                    self._record_timing("history_end", compute_stream)
                    self._record_timing("attention_end", compute_stream)
                return result

            q_group = self._start_q_allgather(
                q_local,
                k_cache_local.shape[2],
                compute_stream,
                timing=_record_timing,
            )
            history_out, history_lse = self._run_context_attention(
                q_group,
                k_cache_local,
                v_cache_local,
                cache_seqlens_local,
                num_splits,
                compute_stream,
                timing=_record_timing,
            )
            output = torch.empty_like(q_local)
            local_lse = (
                torch.empty((b, h_local, sq), device=self.device, dtype=torch.float32)
                if return_lse
                else None
            )
            self._finish_context(
                history_out,
                history_lse,
                output,
                local_lse,
                k_cache_local.shape[2],
                compute_stream,
                timing=_record_timing,
            )
            compute_stream.wait_event(self._context_ready)
            if _record_timing:
                self._record_timing("attention_end", compute_stream)
            return (output, local_lse) if return_lse else output
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
        overlap_q_allgather: bool = True,
        *,
        _record_timing: bool = False,
    ) -> _DCPResult:
        """Run split context/chunk prefill and stably merge both states."""
        if not self._enqueue_lock.acquire(blocking=False):
            raise RuntimeError("DCPAttentionRunner does not support concurrent forward calls")
        try:
            self._reap_inflight_tensors()
            b, sq, h_local, d = self._check_common(
                q_local, k_history_local, v_history_local, history_seqlens_local
            )
            if k_chunk.shape != (b, sq, k_history_local.shape[2], d):
                raise ValueError(
                    "k_chunk must have shape [B, Sq, Hkv_group, 128] matching local history"
                )
            if v_chunk.shape != k_chunk.shape:
                raise ValueError("v_chunk must have the same shape as k_chunk")
            for tensor, name in ((k_chunk, "k_chunk"), (v_chunk, "v_chunk")):
                if (
                    not tensor.is_cuda
                    or tensor.device != self.device
                    or tensor.dtype != torch.bfloat16
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(f"{name} must be contiguous CUDA BF16 on the runner device")

            compute_stream = torch.cuda.current_stream(self.device)
            if _record_timing:
                self._last_timing_kind = "chunk"
                self._record_timing("attention_start", compute_stream)
            self._input_ready.record(compute_stream)

            if self.world_size > 1:
                q_group = self._start_q_allgather(
                    q_local,
                    k_history_local.shape[2],
                    compute_stream,
                    timing=_record_timing,
                )
                if not overlap_q_allgather:
                    compute_stream.wait_event(self._q_group_ready)
            else:
                q_group = q_local
                self._q_group_ready.record(compute_stream)
                if _record_timing:
                    self._record_timing("q_ag_start", compute_stream)
                    self._record_timing("q_ag_end", compute_stream)

            if _record_timing:
                self._record_timing("chunk_start", compute_stream)
            chunk_lengths = torch.full(
                (b,), sq, device=self.device, dtype=torch.int32
            )
            with _nvtx_range("dcp_local_chunk_attention"):
                chunk_out, chunk_lse = min_fa3_op.forward_kvcache(
                    q_local,
                    k_chunk,
                    v_chunk,
                    chunk_lengths,
                    num_splits=num_splits,
                    return_lse=True,
                    is_causal=True,
                )
            if _record_timing:
                self._record_timing("chunk_end", compute_stream)

            history_out, history_lse = self._run_context_attention(
                q_group,
                k_history_local,
                v_history_local,
                history_seqlens_local,
                num_splits,
                compute_stream,
                timing=_record_timing,
            )

            context_out = torch.empty_like(q_local)
            context_lse = torch.empty(
                (b, h_local, sq), device=self.device, dtype=torch.float32
            )
            if self.world_size == 1:
                context_out.copy_(history_out)
                context_lse.copy_(history_lse)
                self._context_ready.record(compute_stream)
            else:
                self._finish_context(
                    history_out,
                    history_lse,
                    context_out,
                    context_lse,
                    k_history_local.shape[2],
                    compute_stream,
                    timing=_record_timing,
                )
            compute_stream.wait_event(self._context_ready)

            merged_out = torch.empty_like(q_local)
            merged_lse = (
                torch.empty((b, h_local, sq), device=self.device, dtype=torch.float32)
                if return_lse
                else None
            )
            if _record_timing:
                self._record_timing("merge_start", compute_stream)
            with _nvtx_range("dcp_merge_context_chunk_states"):
                merged_lse_arg = merged_lse if merged_lse is not None else context_lse
                _merge_attn_states_kernel[(b * sq, h_local)](
                    context_out,
                    context_lse,
                    chunk_out,
                    chunk_lse,
                    merged_out,
                    merged_lse_arg,
                    *merged_out.stride()[:3],
                    *context_lse.stride(),
                    B=b,
                    SQ=sq,
                    H=h_local,
                    D=d,
                    STORE_LSE=merged_lse is not None,
                    num_warps=4,
                )
            if _record_timing:
                self._record_timing("merge_end", compute_stream)
                self._record_timing("attention_end", compute_stream)
            self._retain_merge_inputs(
                compute_stream,
                context_out,
                context_lse,
                chunk_out,
                chunk_lse,
            )
            return (merged_out, merged_lse) if return_lse else merged_out
        finally:
            self._enqueue_lock.release()

    def last_timing_ms(self, synchronize: bool = True) -> dict[str, float]:
        """Return timings for the most recent call made with ``_record_timing=True``."""
        if self._last_timing_kind is None:
            raise RuntimeError("No timed DCP forward has been recorded")
        if synchronize:
            self._timing_events["attention_end"].synchronize()
        events = self._timing_events
        values = {
            "q_allgather_and_reorder_ms": events["q_ag_start"].elapsed_time(events["q_ag_end"]),
            "local_history_attention_ms": events["history_start"].elapsed_time(events["history_end"]),
            "attention_end_to_end_ms": events["attention_start"].elapsed_time(events["attention_end"]),
        }
        if self._last_timing_kind == "chunk":
            values.update(
                local_chunk_attention_ms=events["chunk_start"].elapsed_time(events["chunk_end"]),
                overlapped_ag_chunk_window_ms=events["q_ag_start"].elapsed_time(
                    events["ag_chunk_end"]
                ),
                state_merge_ms=events["merge_start"].elapsed_time(events["merge_end"]),
            )
        else:
            values.update(
                local_chunk_attention_ms=0.0,
                overlapped_ag_chunk_window_ms=values["q_allgather_and_reorder_ms"],
                state_merge_ms=0.0,
            )
        if self.world_size > 1:
            values.update(
                lse_allgather_correct_ms=events["lse_correct_start"].elapsed_time(
                    events["lse_correct_end"]
                ),
                output_reduce_scatter_ms=events["reduce_scatter_start"].elapsed_time(
                    events["reduce_scatter_end"]
                ),
            )
        else:
            values.update(lse_allgather_correct_ms=0.0, output_reduce_scatter_ms=0.0)
        values["sequential_ag_plus_chunk_ms"] = (
            values["q_allgather_and_reorder_ms"] + values["local_chunk_attention_ms"]
        )
        return values


__all__ = ["DCPAttentionRunner"]
