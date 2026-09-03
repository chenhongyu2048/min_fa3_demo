"""Configuration and validation shared by the plugin and launch tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

SUPPORTED_KV_HEADS = (1, 2, 4)


def is_vllm_short_history_warmup(
    query_lengths: list[int] | tuple[int, ...],
    history_lengths: list[int] | tuple[int, ...],
    dcp_world_size: int,
    *,
    saw_zero_history_warmup: bool,
) -> bool:
    """Recognize vLLM V2's second generic JIT-warmup batch.

    ``warmup_kernels`` first runs a two-token pure prefill and then schedules
    one decode token for each request.  The latter therefore has q_len=1 and
    exactly two history tokens.  With DCP > 2 those two tokens do not reach
    every rank.  The zero-history prefill is observed immediately before this
    batch in each worker process; requiring that observation keeps this
    compatibility escape hatch scoped to startup warmup rather than relaxing
    the real decode-side history contract.
    """

    if not saw_zero_history_warmup or not query_lengths:
        return False
    if len(query_lengths) != len(history_lengths) or dcp_world_size <= 2:
        return False
    return all(
        query == 1 and history == 2
        for query, history in zip(query_lengths, history_lengths)
    )


def _positive_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class MegaRuntimeConfig:
    max_total_q: int = 4096
    max_batch: int = 64
    max_num_splits: int = 8
    num_comm_sm: int = 8
    block_n: int | None = None

    @classmethod
    def from_env(cls) -> "MegaRuntimeConfig":
        raw_block_n = os.environ.get("MEGA_DCP_BLOCK_N", "auto").lower()
        if raw_block_n == "auto":
            block_n = None
        else:
            try:
                block_n = int(raw_block_n)
            except ValueError as exc:
                raise ValueError(
                    "MEGA_DCP_BLOCK_N must be auto, 128, or 176"
                ) from exc
            if block_n not in (128, 176):
                raise ValueError("MEGA_DCP_BLOCK_N must be auto, 128, or 176")
        config = cls(
            max_total_q=_positive_env("MEGA_DCP_MAX_TOTAL_Q", 4096),
            max_batch=_positive_env("MEGA_DCP_MAX_BATCH", 64),
            max_num_splits=_positive_env("MEGA_DCP_MAX_NUM_SPLITS", 8),
            num_comm_sm=_positive_env("MEGA_DCP_NUM_COMM_SM", 8),
            block_n=block_n,
        )
        if config.max_batch > config.max_total_q:
            raise ValueError("MEGA_DCP_MAX_BATCH cannot exceed MEGA_DCP_MAX_TOTAL_Q")
        if config.max_num_splits > 128:
            raise ValueError("MEGA_DCP_MAX_NUM_SPLITS cannot exceed 128")
        return config


def validate_history_tokens(history_tokens: Any, num_tokens: int) -> int:
    if isinstance(history_tokens, bool) or not isinstance(history_tokens, int):
        raise ValueError("kv_transfer_params.history_tokens must be an integer")
    if not 0 <= history_tokens < num_tokens:
        raise ValueError(
            "kv_transfer_params.history_tokens must satisfy "
            f"0 <= history_tokens < request.num_tokens ({num_tokens})"
        )
    return history_tokens


def validate_service_config(vllm_config: Any) -> None:
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    compilation = vllm_config.compilation_config

    q_heads = (
        model.get_num_attention_heads(parallel) * parallel.tensor_parallel_size
    )
    kv_heads = model.get_total_num_kv_heads()
    head_size = model.get_head_size()
    dtype = model.dtype
    errors: list[str] = []
    if q_heads != 32:
        errors.append(f"global Q heads must be 32, got {q_heads}")
    if kv_heads not in SUPPORTED_KV_HEADS:
        errors.append(
            f"global KV heads must be one of {SUPPORTED_KV_HEADS}, got {kv_heads}"
        )
    if head_size != 128:
        errors.append(f"head size must be 128, got {head_size}")
    if str(dtype) != "torch.bfloat16":
        errors.append(f"model dtype must be torch.bfloat16, got {dtype}")
    if parallel.tensor_parallel_size != 8:
        errors.append(
            f"tensor parallel size must be 8, got {parallel.tensor_parallel_size}"
        )
    expected_dcp = 8 // kv_heads if kv_heads in SUPPORTED_KV_HEADS else 0
    if parallel.decode_context_parallel_size != expected_dcp:
        errors.append(
            "decode context parallel size must equal TP / global KV heads "
            f"({expected_dcp}), got {parallel.decode_context_parallel_size}"
        )
    if parallel.cp_kv_cache_interleave_size != 1:
        errors.append("cp_kv_cache_interleave_size must be 1")
    if parallel.use_ubatching:
        errors.append("dual-batch overlap and microbatching are unsupported")
    if model.get_num_attention_heads(parallel) != 4:
        errors.append("each rank must own exactly 4 query heads")
    if model.get_num_kv_heads(parallel) != 1:
        errors.append("each rank must own exactly one KV head")
    if scheduler.max_num_seqs > 64:
        errors.append("max_num_seqs cannot exceed 64")
    if scheduler.max_num_batched_tokens > 4096:
        errors.append("max_num_batched_tokens cannot exceed 4096")
    if not scheduler.enable_chunked_prefill:
        errors.append("chunked prefill must be enabled")
    if cache.enable_prefix_caching:
        errors.append("prefix caching must be disabled")
    if compilation.cudagraph_mode.has_full_cudagraphs():
        errors.append("Mega DCP must run in eager mode")
    if vllm_config.speculative_config is not None:
        errors.append("speculative decoding is unsupported")
    if errors:
        raise ValueError(
            "unsupported Mega DCP vLLM configuration: " + "; ".join(errors)
        )
