"""Python wrapper for the minimal Hopper forward-only FlashAttention demo."""

from _min_fa3_op import forward, forward_varlen

__all__ = ["forward", "forward_varlen"]
