"""Portable packed-causal UltraAttn allocation-plan format.

This module deliberately depends only on NumPy and the Python standard library.
The offline planner may require Gurobi and the wider UltraAttn source tree, but
the ring benchmark only imports this file and consumes a validated ``.npz``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
BLOCK_TOKENS = 8192
SUPPORTED_BLOCK_TOKENS = (BLOCK_TOKENS,)
PLANNER_KIND = "ultraattn_ilp_allocation"
RUNTIME_KIND = "ultraattn_graph_torch_distributed_min_fa3_varlen"
LEGACY_RUNTIME_KINDS = ("staged_torch_distributed_min_fa3_varlen",)
DEFAULT_PLANNER_SOURCE_REVISION = "ultraattn-packed-ilp-gqa-v1"


class BlockType(IntEnum):
    EMPTY = 0
    FULL = 1
    CAUSAL = 2


@dataclass(frozen=True)
class PackedCausalPlan:
    metadata: dict[str, Any]
    cmap: np.ndarray
    block_types: np.ndarray
    allocation: np.ndarray

    @property
    def cache_key(self) -> str:
        return str(self.metadata["cache_key"])

    @property
    def world_size(self) -> int:
        return int(self.metadata["world_size"])

    @property
    def global_seqlens(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.metadata["global_seqlens"])

    @property
    def block_tokens(self) -> int:
        return int(self.metadata["block_tokens"])

    @property
    def content_sha256(self) -> str:
        return plan_content_sha256(self)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def plan_content_sha256(plan: PackedCausalPlan) -> str:
    """Hash metadata and all allocation arrays for cross-rank consistency."""
    digest = hashlib.sha256()
    digest.update(_canonical_json(plan.metadata).encode("utf-8"))
    for name, values, dtype in (
        ("cmap", plan.cmap, np.int32),
        ("block_types", plan.block_types, np.int8),
        ("allocation", plan.allocation, np.int16),
    ):
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(name.encode("ascii"))
        digest.update(_canonical_json({"shape": list(array.shape)}).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def normalize_block_tokens(block_tokens: int) -> int:
    block_tokens = int(block_tokens)
    if block_tokens not in SUPPORTED_BLOCK_TOKENS:
        raise ValueError(
            f"block_tokens must be one of {SUPPORTED_BLOCK_TOKENS}, got {block_tokens}"
        )
    return block_tokens


def _normalized_lengths(
    global_seqlens: Sequence[int], block_tokens: int = BLOCK_TOKENS
) -> tuple[int, ...]:
    block_tokens = normalize_block_tokens(block_tokens)
    lengths = tuple(int(length) for length in global_seqlens)
    if not lengths:
        raise ValueError("global_seqlens must not be empty")
    for index, length in enumerate(lengths):
        if length <= 0:
            raise ValueError(f"global_seqlens[{index}] must be positive, got {length}")
        if length % block_tokens:
            raise ValueError(
                f"global_seqlens[{index}]={length} must be divisible by {block_tokens}"
            )
    return lengths


def build_packed_causal_mask(
    global_seqlens: Sequence[int], block_tokens: int = BLOCK_TOKENS
) -> np.ndarray:
    """Build an exact fixed-granularity document-block-diagonal causal mask."""
    block_tokens = normalize_block_tokens(block_tokens)
    lengths = _normalized_lengths(global_seqlens, block_tokens)
    tile_counts = [length // block_tokens for length in lengths]
    par_d = sum(tile_counts)
    block_types = np.full(
        (par_d, par_d), int(BlockType.EMPTY), dtype=np.int8
    )
    offset = 0
    for tile_count in tile_counts:
        end = offset + tile_count
        rows, cols = np.tril_indices(tile_count, k=-1)
        block_types[offset + rows, offset + cols] = int(BlockType.FULL)
        diagonal = np.arange(offset, end)
        block_types[diagonal, diagonal] = int(BlockType.CAUSAL)
        offset = end
    return block_types


def build_default_cmap(par_d: int, world_size: int) -> np.ndarray:
    """Return UltraAttn's default contiguous equal-tile context placement."""
    par_d = int(par_d)
    world_size = int(world_size)
    if par_d <= 0:
        raise ValueError(f"par_d must be positive, got {par_d}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if par_d % world_size:
        raise ValueError(
            f"packed tile count {par_d} must be divisible by world_size {world_size}"
        )
    return (np.arange(par_d, dtype=np.int32) // (par_d // world_size)).astype(
        np.int32, copy=False
    )


def gqa_communication_costs(
    qhead: int,
    kvhead: int,
    headdim: int,
    block_tokens: int = BLOCK_TOKENS,
) -> dict[str, int]:
    """Return per-tile runtime byte costs used by the adapted UltraAttn ILP."""
    qhead = int(qhead)
    kvhead = int(kvhead)
    headdim = int(headdim)
    block_tokens = normalize_block_tokens(block_tokens)
    if qhead <= 0 or kvhead <= 0 or headdim <= 0:
        raise ValueError("qhead, kvhead, and headdim must be positive")
    if qhead % kvhead:
        raise ValueError("qhead must be divisible by kvhead")
    return {
        "q": block_tokens * qhead * headdim * 2,
        "kv": 2 * block_tokens * kvhead * headdim * 2,
        "partial": block_tokens * qhead * (headdim + 1) * 4,
    }


def _cache_identity(
    *,
    global_seqlens: Sequence[int],
    world_size: int,
    qhead: int,
    kvhead: int,
    headdim: int,
    block_tokens: int,
    planner_source_revision: str,
) -> dict[str, Any]:
    block_tokens = normalize_block_tokens(block_tokens)
    return {
        "schema_version": SCHEMA_VERSION,
        "planner_kind": PLANNER_KIND,
        "planner_source_revision": str(planner_source_revision),
        "global_seqlens": list(_normalized_lengths(global_seqlens, block_tokens)),
        "world_size": int(world_size),
        "qhead": int(qhead),
        "kvhead": int(kvhead),
        "headdim": int(headdim),
        "dtype": "bfloat16",
        "direction": "forward",
        "causal": True,
        "block_tokens": block_tokens,
    }


def make_plan_metadata(
    *,
    global_seqlens: Sequence[int],
    world_size: int,
    qhead: int,
    kvhead: int,
    headdim: int,
    planner_source_revision: str,
    solver_metadata: Mapping[str, Any],
    block_tokens: int = BLOCK_TOKENS,
) -> dict[str, Any]:
    block_tokens = normalize_block_tokens(block_tokens)
    lengths = _normalized_lengths(global_seqlens, block_tokens)
    if int(world_size) not in (2, 4, 8):
        raise ValueError(f"world_size must be one of 2, 4, or 8, got {world_size}")
    if int(qhead) <= 0 or int(kvhead) <= 0 or int(qhead) % int(kvhead):
        raise ValueError("qhead and kvhead must be positive and qhead must be divisible by kvhead")
    if int(headdim) != 128:
        raise ValueError(f"headdim must be 128, got {headdim}")
    par_d = sum(lengths) // block_tokens
    build_default_cmap(par_d, int(world_size))

    identity = _cache_identity(
        global_seqlens=lengths,
        world_size=world_size,
        qhead=qhead,
        kvhead=kvhead,
        headdim=headdim,
        block_tokens=block_tokens,
        planner_source_revision=planner_source_revision,
    )
    metadata: dict[str, Any] = {
        **identity,
        "runtime_kind": RUNTIME_KIND,
        "par_d": par_d,
        "cache_key": _sha256_json(identity),
        "solver": dict(solver_metadata),
    }
    metadata["metadata_sha256"] = _sha256_json(metadata)
    return metadata


def expected_plan_path(
    plan_dir: str | os.PathLike[str],
    *,
    global_seqlens: Sequence[int],
    world_size: int,
    qhead: int,
    kvhead: int,
    headdim: int,
    planner_source_revision: str,
    block_tokens: int = BLOCK_TOKENS,
) -> Path:
    identity = _cache_identity(
        global_seqlens=global_seqlens,
        world_size=world_size,
        qhead=qhead,
        kvhead=kvhead,
        headdim=headdim,
        block_tokens=block_tokens,
        planner_source_revision=planner_source_revision,
    )
    return Path(plan_dir) / f"{_sha256_json(identity)}.npz"


def validate_plan(plan: PackedCausalPlan) -> None:
    metadata = plan.metadata
    metadata_without_hash = dict(metadata)
    stored_metadata_hash = metadata_without_hash.pop("metadata_sha256", None)
    if stored_metadata_hash != _sha256_json(metadata_without_hash):
        raise ValueError("plan metadata_sha256 does not match canonical metadata")
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported plan schema_version {metadata.get('schema_version')!r}"
        )
    if metadata.get("planner_kind") != PLANNER_KIND:
        raise ValueError(f"unsupported planner_kind {metadata.get('planner_kind')!r}")
    if metadata.get("runtime_kind") not in (RUNTIME_KIND, *LEGACY_RUNTIME_KINDS):
        raise ValueError(f"unsupported runtime_kind {metadata.get('runtime_kind')!r}")
    if metadata.get("dtype") != "bfloat16" or metadata.get("direction") != "forward":
        raise ValueError("packed UltraAttn plans must be BF16 forward plans")
    if metadata.get("causal") is not True:
        raise ValueError("packed UltraAttn plans must be causal")
    block_tokens = normalize_block_tokens(int(metadata.get("block_tokens", -1)))

    world_size = int(metadata["world_size"])
    qhead = int(metadata["qhead"])
    kvhead = int(metadata["kvhead"])
    headdim = int(metadata["headdim"])
    if world_size not in (2, 4, 8):
        raise ValueError(f"plan world_size must be one of 2, 4, or 8, got {world_size}")
    if qhead <= 0 or kvhead <= 0 or qhead % kvhead:
        raise ValueError("plan qhead must be divisible by positive kvhead")
    if headdim != 128:
        raise ValueError(f"plan headdim must be 128, got {headdim}")

    lengths = _normalized_lengths(metadata["global_seqlens"], block_tokens)
    identity = _cache_identity(
        global_seqlens=lengths,
        world_size=world_size,
        qhead=qhead,
        kvhead=kvhead,
        headdim=headdim,
        block_tokens=block_tokens,
        planner_source_revision=metadata["planner_source_revision"],
    )
    if metadata.get("cache_key") != _sha256_json(identity):
        raise ValueError("plan cache_key does not match its workload identity")
    expected_blocks = build_packed_causal_mask(lengths, block_tokens)
    par_d = expected_blocks.shape[0]
    if int(metadata.get("par_d", -1)) != par_d:
        raise ValueError("plan par_d does not match global_seqlens")
    expected_cmap = build_default_cmap(par_d, world_size)

    cmap = np.asarray(plan.cmap)
    block_types = np.asarray(plan.block_types)
    allocation = np.asarray(plan.allocation)
    if cmap.shape != (par_d,):
        raise ValueError(f"cmap shape must be {(par_d,)}, got {cmap.shape}")
    if block_types.shape != (par_d, par_d):
        raise ValueError(
            f"block_types shape must be {(par_d, par_d)}, got {block_types.shape}"
        )
    if allocation.shape != (par_d, par_d):
        raise ValueError(
            f"allocation shape must be {(par_d, par_d)}, got {allocation.shape}"
        )
    if not np.array_equal(cmap.astype(np.int32, copy=False), expected_cmap):
        raise ValueError("plan cmap is not UltraAttn's default contiguous placement")
    if not np.array_equal(block_types.astype(np.int8, copy=False), expected_blocks):
        raise ValueError("plan block_types do not match the packed document-causal mask")

    empty = expected_blocks == int(BlockType.EMPTY)
    nonempty = ~empty
    if np.any(allocation[empty] != -1):
        raise ValueError("EMPTY attention blocks must have allocation -1")
    if np.any(allocation[nonempty] < 0) or np.any(allocation[nonempty] >= world_size):
        raise ValueError("non-empty attention blocks must be assigned to a valid rank")
    diagonal = np.arange(par_d)
    if not np.array_equal(allocation[diagonal, diagonal], expected_cmap):
        raise ValueError("diagonal causal blocks must be assigned to their cmap owner")


def save_plan(path: str | os.PathLike[str], plan: PackedCausalPlan) -> Path:
    validate_plan(plan)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        metadata_json=np.asarray(_canonical_json(plan.metadata)),
        cmap=np.asarray(plan.cmap, dtype=np.int32),
        block_types=np.asarray(plan.block_types, dtype=np.int8),
        allocation=np.asarray(plan.allocation, dtype=np.int16),
    )
    os.replace(temporary, output)
    return output


def load_plan(path: str | os.PathLike[str]) -> PackedCausalPlan:
    input_path = Path(path)
    try:
        with np.load(input_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            plan = PackedCausalPlan(
                metadata=metadata,
                cmap=np.asarray(archive["cmap"], dtype=np.int32).copy(),
                block_types=np.asarray(archive["block_types"], dtype=np.int8).copy(),
                allocation=np.asarray(archive["allocation"], dtype=np.int16).copy(),
            )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load packed UltraAttn plan {input_path}: {exc}") from exc
    try:
        validate_plan(plan)
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"failed to validate packed UltraAttn plan {input_path}: {exc}"
        ) from exc
    return plan


def make_round_robin_fixture_allocation(
    block_types: np.ndarray, cmap: np.ndarray, world_size: int
) -> np.ndarray:
    """Create a deterministic test-only allocation; never a benchmark fallback."""
    blocks = np.asarray(block_types)
    cmap_array = np.asarray(cmap)
    if blocks.ndim != 2 or blocks.shape[0] != blocks.shape[1]:
        raise ValueError("block_types must be square")
    if cmap_array.shape != (blocks.shape[0],):
        raise ValueError("cmap shape does not match block_types")
    allocation = np.full(blocks.shape, -1, dtype=np.int16)
    for row, col in zip(*np.nonzero(blocks != int(BlockType.EMPTY))):
        allocation[row, col] = (
            int(cmap_array[row]) if row == col else int((row + col) % world_size)
        )
    return allocation
