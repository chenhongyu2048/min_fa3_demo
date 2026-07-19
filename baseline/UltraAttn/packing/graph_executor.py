"""Runtime-only UltraAttn dependent-graph lowering for the fixed 8K suite.

The kernel types and per-rank execution-plan structure are trimmed from
``search_algo/dependent_graph.py`` and ``search_algo/execute_plan.py``.  This
runtime copy deliberately removes profile maps, online Gurobi scheduling,
external FlashAttention, and PyNCCL imports.  Backend launch functions remain
owned by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .plan_format import BlockType


class KernelKind(str, Enum):
    INPUT_Q = "input_q"
    INPUT_KV = "input_kv"
    COMPUTE_FULL = "compute_full"
    COMPUTE_CAUSAL = "compute_causal"
    RETURN_PARTIAL = "return_partial"
    MERGE = "merge"


@dataclass(frozen=True)
class Cuda_Kernel:
    """One local runtime kernel and its explicit precursor IDs."""

    kernel_id: int
    kind: KernelKind
    precursors: tuple[int, ...]


@dataclass(frozen=True)
class Comm_Kernel(Cuda_Kernel):
    pass


@dataclass(frozen=True)
class Comp_Kernel(Cuda_Kernel):
    q_tile: int
    kv_tiles: tuple[int, ...]

    @property
    def causal(self) -> bool:
        return self.kind is KernelKind.COMPUTE_CAUSAL


@dataclass(frozen=True)
class Execution_Plan:
    """Per-rank UltraAttn graph grouped by input-readiness wave."""

    input_q: Comm_Kernel
    input_kv: Comm_Kernel
    local_kernels: tuple[Comp_Kernel, ...]
    q_ready_kernels: tuple[Comp_Kernel, ...]
    kv_ready_kernels: tuple[Comp_Kernel, ...]
    return_partial: Comm_Kernel
    merge_kernels: tuple[Cuda_Kernel, ...]

    @property
    def compute_kernels(self) -> tuple[Comp_Kernel, ...]:
        return (
            self.local_kernels
            + self.q_ready_kernels
            + self.kv_ready_kernels
        )


def build_execution_plan(
    allocation: np.ndarray,
    block_types: np.ndarray,
    cmap: np.ndarray,
    rank: int,
) -> Execution_Plan:
    """Lower one UltraAttn allocation into its local dependent kernel graph."""
    allocation = np.asarray(allocation)
    block_types = np.asarray(block_types)
    cmap = np.asarray(cmap)
    if allocation.ndim != 2 or allocation.shape[0] != allocation.shape[1]:
        raise ValueError("allocation must be square")
    if block_types.shape != allocation.shape:
        raise ValueError("block_types must match allocation")
    if cmap.shape != (allocation.shape[0],):
        raise ValueError("cmap must match the allocation tile count")

    input_q = Comm_Kernel(0, KernelKind.INPUT_Q, ())
    input_kv = Comm_Kernel(1, KernelKind.INPUT_KV, ())
    next_id = 2
    compute: list[Comp_Kernel] = []
    compute_q_tiles = np.unique(np.nonzero(allocation == int(rank))[0])
    for q_tile_value in compute_q_tiles:
        q_tile = int(q_tile_value)
        full_kv = tuple(
            int(kv_tile)
            for kv_tile in range(allocation.shape[1])
            if int(allocation[q_tile, kv_tile]) == int(rank)
            and int(block_types[q_tile, kv_tile]) == int(BlockType.FULL)
        )
        if full_kv:
            precursors: list[int] = []
            if int(cmap[q_tile]) != int(rank):
                precursors.append(input_q.kernel_id)
            if any(int(cmap[kv_tile]) != int(rank) for kv_tile in full_kv):
                precursors.append(input_kv.kernel_id)
            compute.append(
                Comp_Kernel(
                    next_id,
                    KernelKind.COMPUTE_FULL,
                    tuple(precursors),
                    q_tile,
                    full_kv,
                )
            )
            next_id += 1
        if int(allocation[q_tile, q_tile]) == int(rank):
            compute.append(
                Comp_Kernel(
                    next_id,
                    KernelKind.COMPUTE_CAUSAL,
                    (),
                    q_tile,
                    (q_tile,),
                )
            )
            next_id += 1

    local = tuple(kernel for kernel in compute if not kernel.precursors)
    q_ready = tuple(
        kernel
        for kernel in compute
        if kernel.precursors == (input_q.kernel_id,)
    )
    kv_ready = tuple(
        kernel
        for kernel in compute
        if input_kv.kernel_id in kernel.precursors
    )
    if len(local) + len(q_ready) + len(kv_ready) != len(compute):
        raise RuntimeError("UltraAttn graph lowering left compute kernels unscheduled")

    return_partial = Comm_Kernel(
        next_id,
        KernelKind.RETURN_PARTIAL,
        tuple(kernel.kernel_id for kernel in compute),
    )
    next_id += 1
    owned_tiles = tuple(int(tile) for tile in np.nonzero(cmap == int(rank))[0])
    merge: list[Cuda_Kernel] = []
    for tile in owned_tiles:
        local_producers = tuple(
            kernel.kernel_id for kernel in compute if kernel.q_tile == tile
        )
        merge.append(
            Cuda_Kernel(
                next_id,
                KernelKind.MERGE,
                local_producers + (return_partial.kernel_id,),
            )
        )
        next_id += 1

    return Execution_Plan(
        input_q=input_q,
        input_kv=input_kv,
        local_kernels=local,
        q_ready_kernels=q_ready,
        kv_ready_kernels=kv_ready,
        return_partial=return_partial,
        merge_kernels=tuple(merge),
    )


__all__ = [
    "Comm_Kernel",
    "Comp_Kernel",
    "Cuda_Kernel",
    "Execution_Plan",
    "KernelKind",
    "build_execution_plan",
]
