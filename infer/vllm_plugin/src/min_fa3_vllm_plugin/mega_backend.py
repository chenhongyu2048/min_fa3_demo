"""vLLM adapters for the in-repository DCP attention runners.

This module deliberately uses only vLLM's generic attention interfaces and
core KV-cache operators.  In particular, importing it does not import the
vLLM FlashAttention backend or ``vllm.vllm_flash_attn``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
import min_fa3_op

from dcp_test.baselines import VLLMA2ADCPAttentionRunner, VLLMDCPAttentionRunner
from min_fa3_dcp import DCPMegaAttentionRunner
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_node_count,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)

from .config import (
    MegaRuntimeConfig,
    is_vllm_short_history_warmup,
    validate_service_config,
)

logger = init_logger(__name__)


@dataclass
class LocalDCPAttentionMetadata(AttentionMetadata):
    """Minimal eager metadata needed by the three local DCP runners."""

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool | torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu_upper_bound: torch.Tensor
    history_start_loc_cpu: torch.Tensor
    history_start_loc: torch.Tensor
    max_history_len_local: int
    local_causal_only: bool = False


class LocalDCPMetadataBuilder(
    AttentionMetadataBuilder[LocalDCPAttentionMetadata]
):
    """Build packed local-history offsets without FlashAttention metadata."""

    _cudagraph_support = AttentionCGSupport.NEVER
    supports_update_block_table = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        try:
            dcp = get_dcp_group()
            self.dcp_world_size = dcp.world_size
            self.dcp_rank = dcp.rank_in_group
        except AssertionError:
            # This path is useful for CPU-only builder unit tests.
            self.dcp_world_size = 1
            self.dcp_rank = 0
        # vLLM's V2 warmup consists of a pure prefill followed by a two-token
        # history/one-token decode batch.  Remembering the first batch lets us
        # identify that second synthetic batch without weakening validation for
        # real service requests.
        self._saw_zero_history_warmup = False
        self._short_history_warmup_consumed = False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> LocalDCPAttentionMetadata:
        del fast_build
        if common_prefix_len != 0:
            raise ValueError(
                "the local DCP benchmark requires prefix caching/cascade "
                "attention to be disabled"
            )
        num_reqs = common_attn_metadata.num_reqs
        query_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        if seq_lens_cpu is None:
            raise RuntimeError(
                "local DCP attention requires CPU sequence-length upper bounds"
            )
        seq_lens_cpu = seq_lens_cpu[:num_reqs]
        query_lengths = query_cpu[1:] - query_cpu[:-1]
        history_lengths = seq_lens_cpu - query_lengths
        all_zero_history = bool((history_lengths == 0).all())
        short_history_warmup = is_vllm_short_history_warmup(
            query_lengths.tolist(),
            history_lengths.tolist(),
            self.dcp_world_size,
            saw_zero_history_warmup=(
                self._saw_zero_history_warmup
                and not self._short_history_warmup_consumed
            ),
        )
        # The first V2 warmup step is specifically a uniform two-token
        # prefill.  Do not arm the short-history escape hatch after an
        # arbitrary zero-history batch supplied by a caller.
        if all_zero_history and all(
            int(query) == 2 for query in query_lengths.tolist()
        ):
            self._saw_zero_history_warmup = True
        if short_history_warmup:
            # Consume the one expected V2 synthetic decode batch.  A later
            # real request with q_len=1/history=2 must not silently bypass DCP.
            self._short_history_warmup_consumed = True

        # The Mega/AG+RS/A2A packed runners require strictly positive history
        # for every row on every rank.  Keep that contract for real requests.
        # The only exception is vLLM's synthetic V2 warmup, which is handled by
        # local causal attention in ``forward`` and never enters a DCP runner or
        # collective.  Mixed real batches and too-short positive histories are
        # still rejected.
        local_causal_only = all_zero_history or short_history_warmup
        if (
            bool((history_lengths < self.dcp_world_size).any())
            and not local_causal_only
        ):
            raise ValueError(
                "every decode-side request must have at least one local history "
                "token on every DCP rank"
            )

        # cp_kv_cache_interleave_size is fixed to one by service validation.
        # This is vLLM's round-robin local length formula for one DCP rank.
        local_history_lengths = (
            history_lengths // self.dcp_world_size
            + (self.dcp_rank < history_lengths % self.dcp_world_size)
        ).to(torch.int32)
        history_cpu = torch.empty(num_reqs + 1, dtype=torch.int32)
        history_cpu[0] = 0
        torch.cumsum(local_history_lengths, dim=0, out=history_cpu[1:])
        history_device = history_cpu.to(device=self.device, non_blocking=True)

        causal = common_attn_metadata.causal
        if not isinstance(causal, bool) or not causal:
            raise ValueError("local DCP attention requires causal decoder attention")
        return LocalDCPAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=causal,
            query_start_loc_cpu=query_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            history_start_loc_cpu=history_cpu,
            history_start_loc=history_device,
            max_history_len_local=int(local_history_lengths.max().item()),
            local_causal_only=local_causal_only,
        )


class LocalDCPAttentionBackend(AttentionBackend):
    """Narrow SM90 BF16 backend shared by all benchmark methods."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "bfloat16"]
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name() -> str:
        # The registered enum slot is CUSTOM for all three server processes.
        return "CUSTOM"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_builder_cls() -> type[LocalDCPMetadataBuilder]:
        return LocalDCPMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        if block_size % 16:
            raise ValueError("KV cache block size must be a multiple of 16")
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # Fix the physical cache layout to NHD.  After transpose(1, 2), the
        # cache is [num_blocks, block_size, kv_heads, 2 * head_size], which is
        # exactly the input layout expected by vLLM's cp_gather_cache op.
        if include_num_layers_dimension:
            return (1, 0, 3, 2, 4)
        return (0, 2, 1, 3)

    @classmethod
    def get_required_kv_cache_layout(cls) -> str:
        return "NHD"

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == 128

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == DeviceCapability(9, 0)

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


class LocalDCPAttentionImpl(AttentionImpl[LocalDCPAttentionMetadata]):
    """Adapt vLLM's paged cache to one in-repository packed DCP runner."""

    runner_kind: ClassVar[str]
    # The runner performs the cross-DCP LSE reduction internally before it
    # returns the final rank-local output.  Mark this true so vLLM's DCP
    # compatibility check recognizes that the implementation has an LSE-aware
    # decode path; unlike vLLM's split local-attention/combine design, there is
    # no outer LSE tensor to return from the standard AttentionImpl API.
    can_return_lse_for_decode = True
    supports_dcp = True

    _runner: ClassVar[
        DCPMegaAttentionRunner
        | VLLMDCPAttentionRunner
        | VLLMA2ADCPAttentionRunner
        | None
    ] = None
    _runtime: ClassVar[MegaRuntimeConfig | None] = None
    _history_k: ClassVar[torch.Tensor | None] = None
    _history_v: ClassVar[torch.Tensor | None] = None
    _q: ClassVar[torch.Tensor | None] = None
    _chunk_k: ClassVar[torch.Tensor | None] = None
    _chunk_v: ClassVar[torch.Tensor | None] = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.supports_quant_query_input = False

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            raise RuntimeError("local DCP attention requires an active VllmConfig")
        validate_service_config(vllm_config)
        if attn_type != AttentionType.DECODER:
            raise ValueError("local DCP attention supports decoder self attention only")
        if alibi_slopes is not None or sinks is not None:
            raise ValueError("local DCP attention does not support ALiBi or sinks")
        if logits_soft_cap not in (None, 0, 0.0):
            raise ValueError("local DCP attention does not support softcap")
        if sliding_window is not None:
            raise ValueError("local DCP attention does not support sliding windows")
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("local DCP attention requires a BF16 KV cache")
        if num_heads != 4 or num_kv_heads != 1 or head_size != 128:
            raise ValueError(
                "each TP rank must use QH=4, KVH=1, and head_dim=128"
            )
        expected_scale = 1.0 / math.sqrt(head_size)
        if not math.isclose(self.scale, expected_scale, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError(
                f"local min-FA3 requires softmax scale {expected_scale}, "
                f"got {self.scale}"
            )
        type(self)._ensure_runner(vllm_config)

    @classmethod
    def _ensure_runner(cls, vllm_config) -> None:
        if cls._runner is not None:
            return
        runtime = MegaRuntimeConfig.from_env()
        dcp = get_dcp_group()
        tp = get_tp_group()
        expected_dcp = 8 // vllm_config.model_config.get_total_num_kv_heads()
        if dcp.world_size != expected_dcp or tp.world_size != 8:
            raise ValueError(
                f"local DCP attention requires TP=8 and DCP={expected_dcp}"
            )
        if get_node_count() != 1:
            raise ValueError("local DCP attention supports a single node only")

        if cls.runner_kind == "mega":
            cls._runner = DCPMegaAttentionRunner(
                dcp.device_group,
                tp.device_group,
                max_total_q=runtime.max_total_q,
                max_batch=runtime.max_batch,
                Hq_local=4,
                max_num_splits=runtime.max_num_splits,
                num_comm_sm=runtime.num_comm_sm,
                block_n_override=runtime.block_n,
            )
        elif cls.runner_kind == "vllm-ag-rs":
            cls._runner = VLLMDCPAttentionRunner(dcp.device_group)
        elif cls.runner_kind == "vllm-a2a":
            cls._runner = VLLMA2ADCPAttentionRunner(dcp.device_group)
        else:
            raise AssertionError(f"unknown local DCP runner {cls.runner_kind!r}")
        logger.info_once(
            "Using in-repository %s DCP attention runner", cls.runner_kind
        )

        max_model_len = vllm_config.model_config.max_model_len
        max_history = runtime.max_batch * (
            (max_model_len + dcp.world_size - 1) // dcp.world_size
        )
        device = torch.device("cuda", torch.cuda.current_device())
        cls._history_k = torch.empty(
            (max_history, 1, 128), dtype=torch.bfloat16, device=device
        )
        cls._history_v = torch.empty_like(cls._history_k)
        if isinstance(cls._runner, DCPMegaAttentionRunner):
            cls._q = cls._runner.q_backing
        else:
            cls._q = torch.empty(
                (runtime.max_total_q, 4, 128),
                dtype=torch.bfloat16,
                device=device,
            )
        cls._chunk_k = torch.empty(
            (runtime.max_total_q, 1, 128), dtype=torch.bfloat16, device=device
        )
        cls._chunk_v = torch.empty_like(cls._chunk_k)
        cls._runtime = runtime

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        key_cache, value_cache = kv_cache.transpose(1, 2).split(
            self.head_size, dim=-1
        )
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: LocalDCPAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "local DCP attention does not support output quantization"
            )
        if attn_metadata is None:
            return output.fill_(0)

        runner = type(self)._runner
        runtime = type(self)._runtime
        history_k_store = type(self)._history_k
        history_v_store = type(self)._history_v
        q_store = type(self)._q
        chunk_k_store = type(self)._chunk_k
        chunk_v_store = type(self)._chunk_v
        assert runner is not None and runtime is not None
        assert history_k_store is not None and history_v_store is not None
        assert q_store is not None and chunk_k_store is not None
        assert chunk_v_store is not None

        num_tokens = attn_metadata.num_actual_tokens
        q_host = attn_metadata.query_start_loc_cpu
        num_reqs = q_host.numel() - 1
        if num_reqs > runtime.max_batch or num_tokens > runtime.max_total_q:
            raise ValueError(
                "local DCP batch exceeds configured capacity: "
                f"requests={num_reqs}/{runtime.max_batch}, "
                f"tokens={num_tokens}/{runtime.max_total_q}"
            )

        history_host = attn_metadata.history_start_loc_cpu
        history_device = attn_metadata.history_start_loc
        total_history = int(history_host[-1])
        if total_history > history_k_store.shape[0]:
            raise ValueError("packed local-history workspace is too small")

        q_local = q_store[:num_tokens]
        chunk_k = chunk_k_store[:num_tokens]
        chunk_v = chunk_v_store[:num_tokens]
        q_local.copy_(query[:num_tokens])
        # Llama's fused QKV split can produce strided views.  The copied FA3
        # kernels require contiguous Q/K/V, so all methods use the same
        # process-wide scratch copies.
        chunk_k.copy_(key[:num_tokens])
        chunk_v.copy_(value[:num_tokens])

        if attn_metadata.local_causal_only:
            # vLLM V2 startup warmup only: run the current chunk as local causal
            # attention. Production decode/chunk requests always carry valid
            # external history and take the selected DCP runner path below.
            # Keeping this branch here avoids changing either the CUDA kernels
            # or the service's no-full-prefill workload semantics. The flag is
            # rank-invariant, so no DCP collective can diverge across ranks.
            result = min_fa3_op.forward_kvcache_varlen(
                q_local,
                chunk_k,
                chunk_v,
                attn_metadata.query_start_loc,
                attn_metadata.query_start_loc,
                attn_metadata.max_query_len,
                attn_metadata.max_query_len,
                cu_seqlens_q_host=q_host,
                cu_seqlens_k_host=q_host,
                num_splits=0,
                return_lse=False,
                is_causal=True,
            )
            output[:num_tokens].copy_(result)
            return output

        key_cache, value_cache = kv_cache.transpose(1, 2).split(
            self.head_size, dim=-1
        )
        history_k = history_k_store[:total_history]
        history_v = history_v_store[:total_history]
        ops.cp_gather_cache(
            key_cache,
            history_k,
            attn_metadata.block_table[:num_reqs],
            history_device,
            num_reqs,
        )
        ops.cp_gather_cache(
            value_cache,
            history_v,
            attn_metadata.block_table[:num_reqs],
            history_device,
            num_reqs,
        )

        common_args = (
            q_local,
            history_k,
            history_v,
            chunk_k,
            chunk_v,
            attn_metadata.query_start_loc,
            history_device,
            attn_metadata.max_query_len,
            attn_metadata.max_history_len_local,
        )
        common_kwargs = {
            "cu_seqlens_q_host": q_host,
            "cu_seqlens_history_local_host": history_host,
            "num_splits": 0,
        }
        if isinstance(runner, DCPMegaAttentionRunner):
            result = runner.forward_chunk_prefill_varlen(
                *common_args,
                **common_kwargs,
                scheduler_heuristic=True,
                reorder_history_override=False,
            )
        else:
            result = runner.forward_chunk_prefill_varlen(
                *common_args,
                **common_kwargs,
                overlap_q_allgather=False,
            )
        output[:num_tokens].copy_(result)
        return output


class MegaDCPAttentionImpl(LocalDCPAttentionImpl):
    runner_kind = "mega"


class VLLMAGRSDCPAttentionImpl(LocalDCPAttentionImpl):
    runner_kind = "vllm-ag-rs"


class VLLMA2ADCPAttentionImpl(LocalDCPAttentionImpl):
    runner_kind = "vllm-a2a"


class MegaDCPAttentionBackend(LocalDCPAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[MegaDCPAttentionImpl]:
        return MegaDCPAttentionImpl


class VLLMAGRSDCPAttentionBackend(LocalDCPAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[VLLMAGRSDCPAttentionImpl]:
        return VLLMAGRSDCPAttentionImpl


class VLLMA2ADCPAttentionBackend(LocalDCPAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[VLLMA2ADCPAttentionImpl]:
        return VLLMA2ADCPAttentionImpl


__all__ = [
    "LocalDCPAttentionMetadata",
    "LocalDCPMetadataBuilder",
    "MegaDCPAttentionBackend",
    "VLLMA2ADCPAttentionBackend",
    "VLLMAGRSDCPAttentionBackend",
]
