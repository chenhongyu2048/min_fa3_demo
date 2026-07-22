"""Performance-only MagiAttention varlen baseline adapter.

The optional dependency is imported only by the probe or runner constructor so
that importing the regular ring benchmarks does not require MagiAttention.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from itertools import accumulate
from types import ModuleType
from typing import Sequence

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class MagiAttentionConfig:
    overlap_degree: int = 2
    seed: int = 0


def _load_magi_api() -> ModuleType:
    return importlib.import_module("magi_attention.api")


def probe_magi_attention() -> tuple[bool, str | None]:
    """Import the public API and both required extension modules."""

    try:
        for extension in (
            "magi_attention.magi_attn_ext",
            "magi_attention.magi_attn_comm",
        ):
            if importlib.util.find_spec(extension) is None:
                return False, f"{extension} is not installed"

        api = _load_magi_api()
        for name in (
            "magi_attn_varlen_key",
            "dispatch",
            "calc_attn",
            "DistAttnConfig",
            "DispatchConfig",
            "MinHeapDispatchAlg",
            "OverlapConfig",
            "UniformOverlapAlg",
            "AttnOverlapMode",
        ):
            getattr(api, name)
        importlib.import_module("magi_attention.magi_attn_ext")
        importlib.import_module("magi_attention.magi_attn_comm")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def probe_magi_attention_all_ranks(
    process_group: dist.ProcessGroup,
) -> tuple[bool, str | None]:
    """Return one availability decision with rank-specific import failures."""

    local_result = probe_magi_attention()
    world_size = dist.get_world_size(process_group)
    gathered: list[tuple[bool, str | None] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_result, group=process_group)
    failures = [
        f"rank {rank}: {reason or 'unknown import failure'}"
        for rank, result in enumerate(gathered)
        if result is not None
        for available, reason in (result,)
        if not available
    ]
    if failures:
        return False, "; ".join(failures)
    return True, None


class MagiAttentionBaseline:
    def __init__(
        self,
        process_group: dist.ProcessGroup,
        global_lengths: Sequence[int],
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        is_causal: bool,
        device: torch.device,
        *,
        config: MagiAttentionConfig = MagiAttentionConfig(),
        enable_backward: bool = False,
    ) -> None:
        lengths = tuple(int(length) for length in global_lengths)
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("MagiAttention requires non-empty positive global lengths")
        if head_dim != 128:
            raise ValueError(f"MagiAttention requires head_dim=128, got {head_dim}")
        if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
            raise ValueError(
                "MagiAttention requires positive heads and q_heads % kv_heads == 0"
            )
        if not 1 <= config.overlap_degree <= 8:
            raise ValueError("MagiAttention overlap degree must be in [1, 8]")
        if dist.get_world_size(process_group) != dist.get_world_size():
            raise ValueError(
                "MagiAttention process group must cover all benchmark ranks"
            )

        api = _load_magi_api()
        dist_attn_config = api.DistAttnConfig(
            dispatch_config=api.DispatchConfig(
                chunk_size=None,
                alg=api.MinHeapDispatchAlg(),
            ),
            overlap_config=api.OverlapConfig(
                mode=api.AttnOverlapMode.STATIC,
                degree=config.overlap_degree,
                min_chunk_size=512,
                max_num_chunks=4096,
                alg=api.UniformOverlapAlg(
                    random_costs=True,
                    random_seed=config.seed,
                ),
            ),
        )
        cu_seqlens = torch.tensor(
            [0, *accumulate(lengths)],
            dtype=torch.int32,
            device="cpu",
        )
        self._key = api.magi_attn_varlen_key(
            cu_seqlens,
            cu_seqlens,
            num_heads_q=q_heads,
            num_heads_kv=kv_heads,
            head_dim=head_dim,
            pad_size=0,
            cp_group_or_mesh=process_group,
            causal=is_causal,
            dist_attn_config=dist_attn_config,
        )
        self._calc_attn = api.calc_attn

        original_tokens = sum(lengths)
        global_stub = torch.empty(
            (original_tokens, 1), dtype=torch.bfloat16, device=device
        )
        local_stub = api.dispatch(global_stub, self._key)
        local_tokens = local_stub.size(0)
        del local_stub, global_stub

        generator = torch.Generator(device=device)
        generator.manual_seed(config.seed + dist.get_rank(process_group))
        self._q = torch.randn(
            (local_tokens, q_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        self._k = torch.randn(
            (local_tokens, kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        self._v = torch.randn(
            (local_tokens, kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        self._enable_backward = enable_backward
        self._out: torch.Tensor | None = None
        self._dout: torch.Tensor | None = None
        if enable_backward:
            self._q.requires_grad_(True)
            self._k.requires_grad_(True)
            self._v.requires_grad_(True)
            self._dout = torch.randn(
                self._q.shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )

        padded_tokens = int(self._key.total_seqlen_q)
        self._note = (
            f"chunk_size={int(self._key.chunk_size)}, "
            f"tokens(original/padded)={original_tokens}/{padded_tokens}, "
            f"overlap_degree={config.overlap_degree}; dispatch excluded; "
            "performance-only"
        )

    @property
    def note(self) -> str:
        return self._note

    def forward(self) -> torch.Tensor:
        out, _meta = self._calc_attn(self._q, self._k, self._v, self._key)
        return out

    def prepare_backward(self) -> None:
        if not self._enable_backward:
            raise RuntimeError("MagiAttention baseline was not initialized for backward")
        self._q.grad = None
        self._k.grad = None
        self._v.grad = None
        self._out = self.forward()

    def backward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._out is None or self._dout is None:
            raise RuntimeError("prepare_backward() must run before backward()")
        self._out.backward(self._dout)
        dq, dk, dv = self._q.grad, self._k.grad, self._v.grad
        if dq is None or dk is None or dv is None:
            raise RuntimeError("MagiAttention backward did not produce all Q/K/V gradients")
        self._out = None
        return dq, dk, dv
