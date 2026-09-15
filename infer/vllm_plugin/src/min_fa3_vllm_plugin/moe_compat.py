"""Adapt Qwen3 routing to the benchmark's pinned precompiled vLLM wheel."""

from __future__ import annotations


def install_moe_compat() -> None:
    import torch
    import vllm._custom_ops as ops

    # The pinned ancestor wheel predates the optional is_padding argument.
    # Keep the newer wrapper untouched when a matching wheel is installed.
    native = torch.ops._moe_C.topk_softmax.default
    if len(native._schema.arguments) != 6:
        return

    def topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize=False,
        e_score_correction_bias=None,
        is_padding=None,
    ) -> None:
        if is_padding is not None:
            raise ValueError(
                "The pinned MoE wheel requires VLLM_MOE_SKIP_PADDING=0"
            )
        native(
            topk_weights, topk_ids, token_expert_indices, gating_output,
            renormalize, e_score_correction_bias,
        )

    ops.topk_softmax = topk_softmax
