"""Lazy, performance-only adapter for the optional MagiAttention baseline."""

from .attention import (
    MagiAttentionBaseline,
    MagiAttentionConfig,
    probe_magi_attention,
    probe_magi_attention_all_ranks,
)

__all__ = [
    "MagiAttentionBaseline",
    "MagiAttentionConfig",
    "probe_magi_attention",
    "probe_magi_attention_all_ranks",
]
