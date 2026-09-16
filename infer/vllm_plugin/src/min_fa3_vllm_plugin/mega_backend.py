"""vLLM adapters for the in-repository DCP attention runners.

This module deliberately uses only vLLM's generic attention interfaces and
core KV-cache operators.  In particular, importing it does not import the
vLLM FlashAttention backend or ``vllm.vllm_flash_attn``.
"""

from __future__ import annotations

import math
import os
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
    """Packed history metadata and optional capacity-bound graph arguments."""

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
    graph_args: tuple | None = None
    graph_capacity: bool = False


class LocalDCPMetadataBuilder(
    AttentionMetadataBuilder[LocalDCPAttentionMetadata]
):
    """Build packed local-history offsets without FlashAttention metadata."""

    _cudagraph_support = AttentionCGSupport.ALWAYS
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
        self.full_graphs = vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
        self._graph_block_table = None
        if self.full_graphs:
            runtime = MegaRuntimeConfig.from_env()
            self._graph_query_cpu = torch.empty(
                runtime.max_batch + 1, dtype=torch.int32, device="cpu"
            )
            self._graph_history_cpu = torch.empty_like(self._graph_query_cpu)
            self._graph_query = torch.empty_like(self._graph_query_cpu, device=device)
            self._graph_history = torch.empty_like(self._graph_query, device=device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> LocalDCPAttentionMetadata:
        del fast_build
        if self.full_graphs:
            if common_prefix_len != 0:
                raise ValueError("prefix caching/cascade attention must be disabled")
            from vllm.compilation import monitor

            # The GPU runner warms each capture shape through build(), before
            # calling build_for_cudagraph_capture(). Its small synthetic batches
            # can have history shorter than DCP. vLLM closes this startup window
            # after capture_model(), keeping real-request validation unchanged.
            return self._build_graph(
                common_attn_metadata,
                capture=monitor.cudagraph_capturing_enabled,
            )
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


    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self._build_graph(common_attn_metadata, capture=True)

    def _build_graph(self, common, capture=False):
        backend = os.environ.get("MIN_FA3_DCP_BACKEND", "mega")
        impl = {
            "mega": MegaDCPAttentionImpl,
            "mega-fa3-native": MegaDCPAttentionImpl,
            "vllm-ag-rs": VLLMAGRSDCPAttentionImpl,
            "vllm-a2a": VLLMA2ADCPAttentionImpl,
        }[backend]
        impl._ensure_runner(self.vllm_config)
        runtime, runner = impl._runtime, impl._runner
        query = common.query_start_loc_cpu[:common.num_reqs + 1]
        lengths = query[1:] - query[:-1]
        num_reqs = int((lengths > 0).sum())
        query = query[:num_reqs + 1]
        lengths = lengths[:num_reqs]
        history = common.seq_lens_cpu_upper_bound[:num_reqs] - lengths
        all_zero_history = bool((history == 0).all())
        short_warmup = is_vllm_short_history_warmup(
            lengths.tolist(), history.tolist(), self.dcp_world_size,
            saw_zero_history_warmup=(
                self._saw_zero_history_warmup
                and not self._short_history_warmup_consumed
            ),
        )
        if all_zero_history and bool((lengths == 2).all()):
            self._saw_zero_history_warmup = True
        if short_warmup:
            self._short_history_warmup_consumed = True
        # Synthetic capture/warmup uses block zero and one local history token.
        # Real decode-side batches retain the positive local-history contract.
        if capture or all_zero_history or short_warmup:
            history = history.clamp(min=self.dcp_world_size)
        elif bool((history < self.dcp_world_size).any()):
            raise ValueError("every request must have local history on every DCP rank")
        local = (history // self.dcp_world_size
                 + (self.dcp_rank < history % self.dcp_world_size)).to(torch.int32)
        history_cpu = torch.cat((torch.zeros(1, dtype=torch.int32, device="cpu"), local.cumsum(0).int()))
        total_q = common.num_actual_tokens
        if num_reqs > runtime.max_batch or total_q > runtime.max_total_q:
            raise ValueError("local DCP batch exceeds configured graph capacity")
        if int(history_cpu[-1]) > impl._history_k.shape[0]:
            raise ValueError("packed local-history workspace is too small")
        self._graph_query_cpu.fill_(int(query[-1]))
        self._graph_query_cpu[:num_reqs + 1].copy_(query)
        self._graph_history_cpu.fill_(int(history_cpu[-1]))
        self._graph_history_cpu[:num_reqs + 1].copy_(history_cpu)
        # Each copy owns its pinned source until DMA finishes. The next CPU
        # batch may prepare these reusable mirrors before graph replay ends.
        self._graph_query.copy_(self._graph_query_cpu.pin_memory(), non_blocking=True)
        self._graph_history.copy_(self._graph_history_cpu.pin_memory(), non_blocking=True)
        if self._graph_block_table is None:
            self._graph_block_table = torch.zeros(
                (runtime.max_batch, common.block_table_tensor.shape[1]),
                dtype=common.block_table_tensor.dtype, device=self.device,
            )
        self._graph_block_table[:num_reqs].copy_(common.block_table_tensor[:num_reqs])
        graph_args = None
        if isinstance(runner, DCPMegaAttentionRunner):
            graph_args = runner.prepare_graph_forward(
                impl._q[:total_q], impl._history_k, impl._history_v,
                impl._chunk_k[:total_q], impl._chunk_v[:total_q],
                self._graph_query, self._graph_history,
                cu_seqlens_q_host=query,
                cu_seqlens_history_local_host=history_cpu,
                scheduler_heuristic=runtime.scheduler_heuristic,
                reorder_history_override=(None if runtime.scheduler_heuristic is None else False),
            )
        max_history = (self.vllm_config.model_config.max_model_len
                       + self.dcp_world_size - 1) // self.dcp_world_size
        if common.causal is not True:
            raise ValueError("local DCP attention requires causal decoder attention")
        return LocalDCPAttentionMetadata(
            num_actual_tokens=total_q, max_query_len=total_q,
            query_start_loc=self._graph_query, max_seq_len=common.max_seq_len,
            seq_lens=common.seq_lens, block_table=self._graph_block_table,
            slot_mapping=common.slot_mapping, causal=True,
            query_start_loc_cpu=self._graph_query_cpu,
            seq_lens_cpu_upper_bound=common.seq_lens_cpu_upper_bound,
            history_start_loc_cpu=self._graph_history_cpu,
            history_start_loc=self._graph_history,
            max_history_len_local=max_history,
            graph_args=graph_args, graph_capacity=True,
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
    """Supply packed history, either synthetic or gathered from vLLM's cache."""

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
    _synthetic_history: ClassVar[bool] = False
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
        if num_heads not in (4, 8) or num_kv_heads != 1 or head_size != 128:
            raise ValueError(
                "each TP rank must use QH=4/8, KVH=1, and head_dim=128"
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
        expected_dcp = tp.world_size // vllm_config.model_config.get_total_num_kv_heads()
        hq_local = 32 // tp.world_size
        if dcp.world_size != expected_dcp or tp.world_size not in (4, 8):
            raise ValueError(
                f"local DCP attention requires TP=4/8 and DCP={expected_dcp}"
            )
        if get_node_count() != 1:
            raise ValueError("local DCP attention supports a single node only")

        if cls.runner_kind == "mega":
            cls._runner = DCPMegaAttentionRunner(
                dcp.device_group,
                tp.device_group,
                max_total_q=runtime.max_total_q,
                max_batch=runtime.max_batch,
                Hq_local=hq_local,
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
        if isinstance(cls._runner, DCPMegaAttentionRunner):
            split_policy = {
                None: "auto",
                True: "critical_wave",
                False: "fa3_native",
            }[runtime.scheduler_heuristic]
            logger.info_once(
                "Using in-repository Mega DCP attention runner "
                "(scheduler_mode=%s)",
                split_policy,
            )
        else:
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
        transfer = vllm_config.kv_transfer_config
        cls._synthetic_history = transfer.get_from_extra_config(
            "history_kv_mode", "paged"
        ) == "synthetic"
        if cls._synthetic_history:
            # Benchmark-only, immutable history shared by all layers. Lengths
            # and DCP shards still come from the real scheduled batch.
            fill_mean = transfer.get_from_extra_config("fill_mean", 0.015)
            cls._history_k.fill_(fill_mean)
            cls._history_v.fill_(fill_mean)
            logger.info_once(
                "Using fixed synthetic contiguous history KV (fill=%s); "
                "paged cache fill, updates, and gather are disabled",
                fill_mean,
            )
        if isinstance(cls._runner, DCPMegaAttentionRunner):
            cls._q = cls._runner.q_backing
        else:
            cls._q = torch.empty(
                (runtime.max_total_q, hq_local, 128),
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
        if self._synthetic_history or kv_cache.numel() == 0:
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

        history_k = history_k_store if attn_metadata.graph_capacity else history_k_store[:total_history]
        history_v = history_v_store if attn_metadata.graph_capacity else history_v_store[:total_history]
        if not self._synthetic_history:
            key_cache, value_cache = kv_cache.transpose(1, 2).split(
                self.head_size, dim=-1
            )
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
        if attn_metadata.graph_capacity:
            if isinstance(runner, DCPMegaAttentionRunner):
                result = runner.forward_prepared_graph(attn_metadata.graph_args)
            else:
                result = runner.forward_chunk_prefill_varlen_graph(
                    *common_args,
                    cu_seqlens_q_host=q_host,
                    cu_seqlens_history_local_host=history_host,
                )
        elif isinstance(runner, DCPMegaAttentionRunner):
            result = runner.forward_chunk_prefill_varlen(
                *common_args,
                **common_kwargs,
                scheduler_heuristic=runtime.scheduler_heuristic,
                reorder_history_override=(
                    None if runtime.scheduler_heuristic is None else False
                ),
            )
            dispatch = runner.last_queue_counts
            assert dispatch is not None
            logger.info_once(
                "Mega DCP batch dispatch: scheduler_mode=%s, "
                "split_policy=%s, history_order=%s, block_n=%s",
                dispatch["scheduler_mode"],
                dispatch["split_policy"],
                dispatch["history_order_policy"],
                dispatch["effective_block_n"],
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
