"""Lazy, performance-only adapter for the optional MagiAttention baseline."""

from .attention import (
    MagiAttentionBaseline,
    MagiAttentionConfig,
    MagiAttentionMetadata,
    build_magi_attention_metadata,
    probe_magi_attention,
    probe_magi_attention_all_ranks,
)

__all__ = [
    "MagiAttentionBaseline",
    "MagiAttentionConfig",
    "MagiAttentionMetadata",
    "build_magi_attention_metadata",
    "probe_magi_attention",
    "probe_magi_attention_all_ranks",
]
