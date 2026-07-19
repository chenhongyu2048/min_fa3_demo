"""UltraAttn graph-executed forward adapter for the fixed 8K suite.

The offline UltraAttn ILP owns the QxK block allocation.  At runtime that
allocation is compiled into an UltraAttn-style dependency graph with input
communication, compute, partial-return, and owner-merge nodes.  Communication
uses asynchronous ``torch.distributed`` NCCL collectives and computation uses
the in-repo min-FA3 varlen kernel.

This adapter intentionally supports only the five fixed 128K-token workloads
used by ``ultraattn/benchmark_hybrid_fixed_forward.py``.  It has no staged-executor or
256-token packing fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist

import min_fa3_op
from baseline.UltraAttn.packing.graph_executor import (
    Comp_Kernel,
    Execution_Plan,
    build_execution_plan,
)
from baseline.UltraAttn.packing.plan_format import (
    DEFAULT_PLANNER_SOURCE_REVISION,
    PackedCausalPlan,
    expected_plan_path,
    load_plan,
)


BLOCK_TOKENS = 8192
FIXED_WORLD_SIZE = 8
FIXED_QHEAD = 32
FIXED_KVHEAD = 8
FIXED_HEADDIM = 128
FIXED_GLOBAL_SEQLENS = (
    (131072,),
    (65536, 65536),
    (32768,) * 4,
    (16384,) * 8,
    (8192,) * 16,
)


@dataclass(frozen=True)
class _ExchangeSchedule:
    send_tiles: tuple[tuple[int, ...], ...]
    recv_tiles: tuple[tuple[int, ...], ...]
    send_splits: tuple[int, ...]
    recv_splits: tuple[int, ...]
    send_token_indices: torch.Tensor
    recv_offsets: dict[int, int]

    @property
    def send_tokens(self) -> int:
        return sum(self.send_splits)

    @property
    def recv_tokens(self) -> int:
        return sum(self.recv_splits)


@dataclass(frozen=True)
class _TaskBatch:
    nodes: tuple[Comp_Kernel, ...]
    q_tokens: int
    k_tokens: int
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    cu_q_host: torch.Tensor
    cu_k_host: torch.Tensor
    max_k: int

def _is_fixed_workload(global_seqlens: Sequence[int]) -> bool:
    normalized = tuple(int(value) for value in global_seqlens)
    return normalized in FIXED_GLOBAL_SEQLENS


def packed_plan_path(
    plan_dir: str | Path,
    global_seqlens: Sequence[int],
    world_size: int,
    qhead: int,
    kvhead: int,
    headdim: int,
    block_tokens: int = BLOCK_TOKENS,
) -> Path:
    return expected_plan_path(
        plan_dir,
        global_seqlens=global_seqlens,
        world_size=world_size,
        qhead=qhead,
        kvhead=kvhead,
        headdim=headdim,
        planner_source_revision=DEFAULT_PLANNER_SOURCE_REVISION,
        block_tokens=block_tokens,
    )


def plan_incompatibility(
    plan_dir: str | Path,
    global_seqlens: Sequence[int],
    world_size: int,
    qhead: int,
    kvhead: int,
    headdim: int,
    is_causal: bool,
    block_tokens: int = BLOCK_TOKENS,
) -> str | None:
    if not is_causal:
        return "UltraAttn graph baseline supports causal mode only"
    if int(world_size) != FIXED_WORLD_SIZE:
        return f"UltraAttn graph baseline requires world_size={FIXED_WORLD_SIZE}"
    if int(block_tokens) != BLOCK_TOKENS:
        return f"UltraAttn graph baseline requires block_tokens={BLOCK_TOKENS}"
    if (int(qhead), int(kvhead), int(headdim)) != (
        FIXED_QHEAD,
        FIXED_KVHEAD,
        FIXED_HEADDIM,
    ):
        return (
            "UltraAttn graph baseline requires "
            f"QH/KVH/D={FIXED_QHEAD}/{FIXED_KVHEAD}/{FIXED_HEADDIM}"
        )
    if not _is_fixed_workload(global_seqlens):
        return (
            "UltraAttn graph baseline supports only 1x128K, 2x64K, 4x32K, "
            "8x16K, and 16x8K"
        )
    path = packed_plan_path(
        plan_dir,
        global_seqlens,
        world_size,
        qhead,
        kvhead,
        headdim,
        block_tokens,
    )
    if not path.is_file():
        return f"cached UltraAttn allocation plan is missing: {path}"
    try:
        plan = load_plan(path)
    except ValueError as exc:
        return str(exc)
    expected = {
        "world_size": int(world_size),
        "qhead": int(qhead),
        "kvhead": int(kvhead),
        "headdim": int(headdim),
        "global_seqlens": [int(value) for value in global_seqlens],
        "block_tokens": BLOCK_TOKENS,
    }
    for key, value in expected.items():
        if plan.metadata.get(key) != value:
            return (
                f"UltraAttn plan metadata mismatch for {key}: "
                f"expected {value!r}, got {plan.metadata.get(key)!r}"
            )
    return None


def missing_plan_command(
    plan_dir: str | Path,
    global_seqlens: Sequence[int],
    world_size: int,
    qhead: int,
    kvhead: int,
    headdim: int,
    block_tokens: int = BLOCK_TOKENS,
) -> str:
    lengths = ",".join(str(int(value)) for value in global_seqlens)
    return (
        "python baseline/UltraAttn/packing/export_packed_causal_plan.py "
        f"--global-seqlens {lengths} --world-size {world_size} "
        f"--qhead {qhead} --kvhead {kvhead} --headdim {headdim} "
        f"--block-tokens {block_tokens} --output-dir {Path(plan_dir)}"
    )


def _merge_partial(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> None:
    """Merge [S,H,D]/[H,S] partials into caller-owned FP32 buffers."""
    old_lse = lse.transpose(0, 1).unsqueeze(-1)
    new_lse = block_lse.transpose(0, 1).unsqueeze(-1)
    merged_lse = torch.logaddexp(old_lse, new_lse)
    out.mul_(torch.exp(old_lse - merged_lse))
    out.add_(block_out.float() * torch.exp(new_lse - merged_lse))
    lse.copy_(merged_lse.squeeze(-1).transpose(0, 1))


class UltraAttnPackedCausalForward:
    """Execute one fixed-suite UltraAttn allocation through a dependency graph."""

    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_seqlens: Sequence[int],
        plan_path: str | Path,
        *,
        workspace_mib: int = 2048,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        if workspace_mib <= 0:
            raise ValueError("workspace_mib must be positive")
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("UltraAttn graph runtime requires BF16 Q/K/V")
        if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
            raise ValueError("UltraAttn graph runtime requires Q/K/V on one CUDA device")
        if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
            raise ValueError("UltraAttn graph runtime requires contiguous Q/K/V")
        if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
            raise ValueError("expected Q [T,QH,D] and K/V [T,KVH,D]")

        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.q = q
        self.k = k
        self.v = v
        self.global_seqlens = tuple(int(value) for value in global_seqlens)
        self.plan_path = Path(plan_path)
        self.plan = load_plan(self.plan_path)
        self.block_tokens = self.plan.block_tokens
        self.workspace_mib = int(workspace_mib)
        self._validate_runtime()

        self.par_d = self.plan.block_types.shape[0]
        self.tiles_per_rank = self.par_d // self.world_size
        self.local_tokens = self.tiles_per_rank * self.block_tokens
        if q.size(0) != self.local_tokens or k.size(0) != self.local_tokens:
            raise ValueError(
                f"local Q/K/V rows must be {self.local_tokens}, "
                f"got Q={q.size(0)}, K={k.size(0)}"
            )

        device = q.device
        self.q_exchange = self._make_input_exchange("q", device)
        self.kv_exchange = self._make_input_exchange("kv", device)
        self.remote_q = torch.empty(
            (self.q_exchange.recv_tokens, q.size(1), q.size(2)),
            dtype=q.dtype,
            device=device,
        )
        self.remote_k = torch.empty(
            (self.kv_exchange.recv_tokens, k.size(1), k.size(2)),
            dtype=k.dtype,
            device=device,
        )
        self.remote_v = torch.empty_like(self.remote_k)
        self.send_q = torch.empty(
            (self.q_exchange.send_tokens, q.size(1), q.size(2)),
            dtype=q.dtype,
            device=device,
        )
        self.send_k = torch.empty(
            (self.kv_exchange.send_tokens, k.size(1), k.size(2)),
            dtype=k.dtype,
            device=device,
        )
        self.send_v = torch.empty_like(self.send_k)

        self.compute_q_tiles = tuple(
            int(tile)
            for tile in np.unique(np.nonzero(self.plan.allocation == self.rank)[0])
        )
        self.compute_slot = {
            tile: slot for slot, tile in enumerate(self.compute_q_tiles)
        }
        self.partial_out = torch.empty(
            (len(self.compute_q_tiles), self.block_tokens, q.size(1), q.size(2)),
            dtype=torch.float32,
            device=device,
        )
        self.partial_lse = torch.empty(
            (len(self.compute_q_tiles), q.size(1), self.block_tokens),
            dtype=torch.float32,
            device=device,
        )

        self.execution_plan: Execution_Plan = build_execution_plan(
            self.plan.allocation,
            self.plan.block_types,
            self.plan.cmap,
            self.rank,
        )
        if len(self.execution_plan.merge_kernels) != self.tiles_per_rank:
            raise RuntimeError("UltraAttn execution plan has invalid merge-node count")
        workspace_bytes = self.workspace_mib * 1024 * 1024
        self.local_batches = self._make_task_batches(
            self.execution_plan.local_kernels, workspace_bytes, device
        )
        self.q_ready_batches = self._make_task_batches(
            self.execution_plan.q_ready_kernels, workspace_bytes, device
        )
        self.kv_ready_batches = self._make_task_batches(
            self.execution_plan.kv_ready_kernels, workspace_bytes, device
        )
        all_batches = (
            self.local_batches + self.q_ready_batches + self.kv_ready_batches
        )
        max_batch_q = max((batch.q_tokens for batch in all_batches), default=0)
        max_batch_k = max((batch.k_tokens for batch in all_batches), default=0)
        self.task_q = torch.empty(
            (max_batch_q, q.size(1), q.size(2)), dtype=q.dtype, device=device
        )
        self.task_k = torch.empty(
            (max_batch_k, k.size(1), k.size(2)), dtype=k.dtype, device=device
        )
        self.task_v = torch.empty_like(self.task_k)

        self.partial_exchange = self._make_partial_exchange(device)
        self.send_partial_out = torch.empty(
            (self.partial_exchange.send_tokens, q.size(1), q.size(2)),
            dtype=torch.float32,
            device=device,
        )
        self.recv_partial_out = torch.empty(
            (self.partial_exchange.recv_tokens, q.size(1), q.size(2)),
            dtype=torch.float32,
            device=device,
        )
        self.send_partial_lse = torch.empty(
            (self.partial_exchange.send_tokens, q.size(1)),
            dtype=torch.float32,
            device=device,
        )
        self.recv_partial_lse = torch.empty(
            (self.partial_exchange.recv_tokens, q.size(1)),
            dtype=torch.float32,
            device=device,
        )
        self.final_out_fp32 = torch.empty_like(q, dtype=torch.float32)
        self.final_out = torch.empty_like(q)
        self.final_lse = torch.empty(
            (q.size(1), q.size(0)), dtype=torch.float32, device=device
        )
        self._verify_plan_hash_across_ranks()

    @property
    def note(self) -> str:
        solver = self.plan.metadata.get("solver", {})
        status = solver.get("status", "unknown")
        gap = solver.get("mip_gap")
        solver_note = f"solver={status}"
        if gap is not None:
            solver_note += f", gap={float(gap):.4g}"
        return (
            "UltraAttn ILP graph executor; async torch.distributed NCCL; "
            "min_fa3 varlen; FP32 O/LSE merge; "
            f"block={self.block_tokens}; "
            f"nodes={len(self.execution_plan.compute_kernels)}; "
            f"workspace={self.workspace_mib}MiB; {solver_note}; "
            f"plan={self.plan.content_sha256[:12]}"
        )

    def _validate_runtime(self) -> None:
        if self.world_size != FIXED_WORLD_SIZE:
            raise ValueError(
                f"UltraAttn graph runtime requires world_size={FIXED_WORLD_SIZE}"
            )
        if self.block_tokens != BLOCK_TOKENS:
            raise ValueError(
                f"UltraAttn graph runtime requires block_tokens={BLOCK_TOKENS}"
            )
        if not _is_fixed_workload(self.global_seqlens):
            raise ValueError(
                "UltraAttn graph runtime supports only the fixed five 128K workloads"
            )
        expected_shape = (FIXED_QHEAD, FIXED_KVHEAD, FIXED_HEADDIM)
        actual_shape = (self.q.size(1), self.k.size(1), self.q.size(2))
        if actual_shape != expected_shape or self.k.size(2) != FIXED_HEADDIM:
            raise ValueError(
                f"UltraAttn graph runtime requires QH/KVH/D={expected_shape}, "
                f"got {actual_shape}"
            )
        expected = {
            "world_size": self.world_size,
            "global_seqlens": list(self.global_seqlens),
            "qhead": self.q.size(1),
            "kvhead": self.k.size(1),
            "headdim": self.q.size(2),
            "block_tokens": BLOCK_TOKENS,
        }
        for key, value in expected.items():
            if self.plan.metadata.get(key) != value:
                raise ValueError(
                    f"plan metadata mismatch for {key}: expected {value!r}, "
                    f"got {self.plan.metadata.get(key)!r}"
                )

    def _verify_plan_hash_across_ranks(self) -> None:
        raw = bytes.fromhex(self.plan.content_sha256)
        value = torch.tensor(list(raw), dtype=torch.uint8, device=self.q.device)
        low = value.clone()
        high = value.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN, group=self.process_group)
        dist.all_reduce(high, op=dist.ReduceOp.MAX, group=self.process_group)
        if not torch.equal(low, high):
            raise RuntimeError("ranks loaded different UltraAttn plan contents")

    def _local_token_indices(
        self, tiles_by_peer: Sequence[Sequence[int]]
    ) -> torch.Tensor:
        first_tile = self.rank * self.tiles_per_rank
        indices: list[int] = []
        for tiles in tiles_by_peer:
            for tile in tiles:
                local_tile = int(tile) - first_tile
                if local_tile < 0 or local_tile >= self.tiles_per_rank:
                    raise RuntimeError(f"rank {self.rank} does not own tile {tile}")
                begin = local_tile * self.block_tokens
                indices.extend(range(begin, begin + self.block_tokens))
        return torch.tensor(indices, dtype=torch.int64, device=self.q.device)

    def _make_input_exchange(
        self, kind: str, device: torch.device
    ) -> _ExchangeSchedule:
        if kind not in ("q", "kv"):
            raise ValueError(f"unsupported exchange kind {kind}")
        allocation = self.plan.allocation
        send: list[tuple[int, ...]] = []
        recv: list[tuple[int, ...]] = []
        for peer in range(self.world_size):
            send_tiles: list[int] = []
            recv_tiles: list[int] = []
            if peer != self.rank:
                for tile in range(self.par_d):
                    owner = int(self.plan.cmap[tile])
                    values = allocation[tile] if kind == "q" else allocation[:, tile]
                    consumers = {int(value) for value in values if int(value) >= 0}
                    if owner == self.rank and peer in consumers:
                        send_tiles.append(tile)
                    if owner == peer and self.rank in consumers:
                        recv_tiles.append(tile)
            send.append(tuple(send_tiles))
            recv.append(tuple(recv_tiles))

        send_splits = tuple(len(tiles) * self.block_tokens for tiles in send)
        recv_splits = tuple(len(tiles) * self.block_tokens for tiles in recv)
        recv_offsets: dict[int, int] = {}
        offset = 0
        for tiles in recv:
            for tile in tiles:
                recv_offsets[tile] = offset
                offset += self.block_tokens
        return _ExchangeSchedule(
            send_tiles=tuple(send),
            recv_tiles=tuple(recv),
            send_splits=send_splits,
            recv_splits=recv_splits,
            send_token_indices=self._local_token_indices(send),
            recv_offsets=recv_offsets,
        )

    def _task_bytes(self, node: Comp_Kernel) -> int:
        q_bytes = self.block_tokens * self.q.size(1) * self.q.size(2) * 2
        kv_tokens = len(node.kv_tiles) * self.block_tokens
        kv_bytes = 2 * kv_tokens * self.k.size(1) * self.k.size(2) * 2
        result_bytes = self.block_tokens * self.q.size(1) * (
            self.q.size(2) + 1
        ) * 4
        return q_bytes + kv_bytes + result_bytes

    def _make_task_batches(
        self,
        nodes: tuple[Comp_Kernel, ...],
        workspace_bytes: int,
        device: torch.device,
    ) -> tuple[_TaskBatch, ...]:
        batches: list[_TaskBatch] = []
        for causal in (False, True):
            selected = [node for node in nodes if node.causal is causal]
            groups: list[list[Comp_Kernel]] = []
            current: list[Comp_Kernel] = []
            current_bytes = 0
            for node in selected:
                task_bytes = self._task_bytes(node)
                if task_bytes > workspace_bytes:
                    required_mib = (task_bytes + 1024 * 1024 - 1) // (1024 * 1024)
                    raise ValueError(
                        "one UltraAttn graph compute node exceeds workspace: "
                        f"q_tile={node.q_tile}, required={required_mib}MiB, "
                        f"configured={self.workspace_mib}MiB"
                    )
                if current and current_bytes + task_bytes > workspace_bytes:
                    groups.append(current)
                    current = []
                    current_bytes = 0
                current.append(node)
                current_bytes += task_bytes
            if current:
                groups.append(current)

            for group in groups:
                q_lengths = [self.block_tokens] * len(group)
                k_lengths = [len(node.kv_tiles) * self.block_tokens for node in group]
                cu_q_host = torch.zeros(len(group) + 1, dtype=torch.int32)
                cu_k_host = torch.zeros(len(group) + 1, dtype=torch.int32)
                cu_q_host[1:] = torch.cumsum(
                    torch.tensor(q_lengths, dtype=torch.int32), 0
                )
                cu_k_host[1:] = torch.cumsum(
                    torch.tensor(k_lengths, dtype=torch.int32), 0
                )
                batches.append(
                    _TaskBatch(
                        nodes=tuple(group),
                        q_tokens=sum(q_lengths),
                        k_tokens=sum(k_lengths),
                        cu_q=cu_q_host.to(device=device),
                        cu_k=cu_k_host.to(device=device),
                        cu_q_host=cu_q_host,
                        cu_k_host=cu_k_host,
                        max_k=max(k_lengths),
                    )
                )
        return tuple(batches)

    def _make_partial_exchange(self, device: torch.device) -> _ExchangeSchedule:
        send: list[tuple[int, ...]] = []
        recv: list[tuple[int, ...]] = []
        allocation = self.plan.allocation
        for peer in range(self.world_size):
            send_tiles: list[int] = []
            recv_tiles: list[int] = []
            if peer != self.rank:
                for q_tile in range(self.par_d):
                    owner = int(self.plan.cmap[q_tile])
                    producers = {
                        int(value)
                        for value in allocation[q_tile]
                        if int(value) >= 0
                    }
                    if owner == peer and self.rank in producers:
                        send_tiles.append(q_tile)
                    if owner == self.rank and peer in producers:
                        recv_tiles.append(q_tile)
            send.append(tuple(send_tiles))
            recv.append(tuple(recv_tiles))

        send_splits = tuple(len(tiles) * self.block_tokens for tiles in send)
        recv_splits = tuple(len(tiles) * self.block_tokens for tiles in recv)
        keyed_offsets: dict[tuple[int, int], int] = {}
        offset = 0
        for source, tiles in enumerate(recv):
            for tile in tiles:
                keyed_offsets[(source, tile)] = offset
                offset += self.block_tokens
        self.partial_recv_offsets = keyed_offsets
        return _ExchangeSchedule(
            send_tiles=tuple(send),
            recv_tiles=tuple(recv),
            send_splits=send_splits,
            recv_splits=recv_splits,
            send_token_indices=torch.empty(0, dtype=torch.int64, device=device),
            recv_offsets={},
        )

    def _local_tile(self, tensor: torch.Tensor, tile: int) -> torch.Tensor:
        first_tile = self.rank * self.tiles_per_rank
        local_tile = tile - first_tile
        if local_tile < 0 or local_tile >= self.tiles_per_rank:
            raise RuntimeError(f"rank {self.rank} does not own tile {tile}")
        begin = local_tile * self.block_tokens
        return tensor[begin : begin + self.block_tokens]

    def _q_tile(self, tile: int) -> torch.Tensor:
        if int(self.plan.cmap[tile]) == self.rank:
            return self._local_tile(self.q, tile)
        begin = self.q_exchange.recv_offsets[tile]
        return self.remote_q[begin : begin + self.block_tokens]

    def _k_tile(self, tile: int) -> torch.Tensor:
        if int(self.plan.cmap[tile]) == self.rank:
            return self._local_tile(self.k, tile)
        begin = self.kv_exchange.recv_offsets[tile]
        return self.remote_k[begin : begin + self.block_tokens]

    def _v_tile(self, tile: int) -> torch.Tensor:
        if int(self.plan.cmap[tile]) == self.rank:
            return self._local_tile(self.v, tile)
        begin = self.kv_exchange.recv_offsets[tile]
        return self.remote_v[begin : begin + self.block_tokens]

    def _launch_input_nodes(self) -> tuple[dist.Work, dist.Work, dist.Work]:
        if self.q_exchange.send_tokens:
            torch.index_select(
                self.q, 0, self.q_exchange.send_token_indices, out=self.send_q
            )
        if self.kv_exchange.send_tokens:
            torch.index_select(
                self.k, 0, self.kv_exchange.send_token_indices, out=self.send_k
            )
            torch.index_select(
                self.v, 0, self.kv_exchange.send_token_indices, out=self.send_v
            )
        q_work = dist.all_to_all_single(
            self.remote_q,
            self.send_q,
            output_split_sizes=list(self.q_exchange.recv_splits),
            input_split_sizes=list(self.q_exchange.send_splits),
            group=self.process_group,
            async_op=True,
        )
        k_work = dist.all_to_all_single(
            self.remote_k,
            self.send_k,
            output_split_sizes=list(self.kv_exchange.recv_splits),
            input_split_sizes=list(self.kv_exchange.send_splits),
            group=self.process_group,
            async_op=True,
        )
        v_work = dist.all_to_all_single(
            self.remote_v,
            self.send_v,
            output_split_sizes=list(self.kv_exchange.recv_splits),
            input_split_sizes=list(self.kv_exchange.send_splits),
            group=self.process_group,
            async_op=True,
        )
        if q_work is None or k_work is None or v_work is None:
            raise RuntimeError("asynchronous Torch NCCL input node returned no Work")
        return q_work, k_work, v_work

    def _run_batches(
        self,
        batches: Sequence[_TaskBatch],
        written: set[int],
    ) -> None:
        for batch in batches:
            causal = batch.nodes[0].causal
            if any(node.causal is not causal for node in batch.nodes):
                raise RuntimeError("UltraAttn graph batch mixed causal node types")
            batch_q = self.task_q[: batch.q_tokens]
            batch_k = self.task_k[: batch.k_tokens]
            batch_v = self.task_v[: batch.k_tokens]
            q_offset = 0
            k_offset = 0
            for node in batch.nodes:
                batch_q[q_offset : q_offset + self.block_tokens].copy_(
                    self._q_tile(node.q_tile)
                )
                q_offset += self.block_tokens
                for kv_tile in node.kv_tiles:
                    batch_k[k_offset : k_offset + self.block_tokens].copy_(
                        self._k_tile(kv_tile)
                    )
                    batch_v[k_offset : k_offset + self.block_tokens].copy_(
                        self._v_tile(kv_tile)
                    )
                    k_offset += self.block_tokens

            block_out, block_lse = min_fa3_op.forward_varlen(
                batch_q,
                batch_k,
                batch_v,
                batch.cu_q,
                batch.cu_k,
                self.block_tokens,
                batch.max_k,
                causal,
                cu_seqlens_q_host=batch.cu_q_host,
                cu_seqlens_k_host=batch.cu_k_host,
                return_lse=True,
            )
            for task_index, node in enumerate(batch.nodes):
                begin = task_index * self.block_tokens
                end = begin + self.block_tokens
                slot = self.compute_slot[node.q_tile]
                if node.q_tile not in written:
                    self.partial_out[slot].copy_(block_out[begin:end])
                    self.partial_lse[slot].copy_(block_lse[:, begin:end])
                    written.add(node.q_tile)
                else:
                    _merge_partial(
                        self.partial_out[slot],
                        self.partial_lse[slot],
                        block_out[begin:end],
                        block_lse[:, begin:end],
                    )

    def _pack_partial_node(self) -> None:
        offset = 0
        for tiles in self.partial_exchange.send_tiles:
            for tile in tiles:
                slot = self.compute_slot[tile]
                end = offset + self.block_tokens
                self.send_partial_out[offset:end].copy_(self.partial_out[slot])
                self.send_partial_lse[offset:end].copy_(
                    self.partial_lse[slot].transpose(0, 1)
                )
                offset = end

    def _launch_partial_node(self) -> tuple[dist.Work, dist.Work]:
        out_work = dist.all_to_all_single(
            self.recv_partial_out,
            self.send_partial_out,
            output_split_sizes=list(self.partial_exchange.recv_splits),
            input_split_sizes=list(self.partial_exchange.send_splits),
            group=self.process_group,
            async_op=True,
        )
        lse_work = dist.all_to_all_single(
            self.recv_partial_lse,
            self.send_partial_lse,
            output_split_sizes=list(self.partial_exchange.recv_splits),
            input_split_sizes=list(self.partial_exchange.send_splits),
            group=self.process_group,
            async_op=True,
        )
        if out_work is None or lse_work is None:
            raise RuntimeError("asynchronous Torch NCCL partial node returned no Work")
        return out_work, lse_work

    def _initialize_merge_nodes(self) -> None:
        first_tile = self.rank * self.tiles_per_rank
        for local_tile in range(self.tiles_per_rank):
            tile = first_tile + local_tile
            slot = self.compute_slot[tile]
            begin = local_tile * self.block_tokens
            end = begin + self.block_tokens
            self.final_out_fp32[begin:end].copy_(self.partial_out[slot])
            self.final_lse[:, begin:end].copy_(self.partial_lse[slot])

    def _finish_merge_nodes(self) -> None:
        first_tile = self.rank * self.tiles_per_rank
        for local_tile in range(self.tiles_per_rank):
            tile = first_tile + local_tile
            begin = local_tile * self.block_tokens
            end = begin + self.block_tokens
            producers = sorted(
                {
                    int(value)
                    for value in self.plan.allocation[tile]
                    if int(value) >= 0 and int(value) != self.rank
                }
            )
            for source in producers:
                remote_begin = self.partial_recv_offsets[(source, tile)]
                remote_end = remote_begin + self.block_tokens
                _merge_partial(
                    self.final_out_fp32[begin:end],
                    self.final_lse[:, begin:end],
                    self.recv_partial_out[remote_begin:remote_end],
                    self.recv_partial_lse[remote_begin:remote_end].transpose(0, 1),
                )
        self.final_out.copy_(self.final_out_fp32)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the compiled UltraAttn dependency graph once."""
        with torch.cuda.nvtx.range("ultraattn_graph/input_nodes"):
            q_work, k_work, v_work = self._launch_input_nodes()

        written: set[int] = set()
        with torch.cuda.nvtx.range("ultraattn_graph/local_compute"):
            self._run_batches(self.local_batches, written)

        # The Q collective is first on the ProcessGroupNCCL stream.  Waiting it
        # exposes graph nodes that need a remote Q but only locally owned K/V,
        # while the K/V collectives continue progressing.
        q_work.wait()
        with torch.cuda.nvtx.range("ultraattn_graph/q_ready_compute"):
            self._run_batches(self.q_ready_batches, written)

        k_work.wait()
        v_work.wait()
        with torch.cuda.nvtx.range("ultraattn_graph/kv_ready_compute"):
            self._run_batches(self.kv_ready_batches, written)

        if len(written) != len(self.compute_q_tiles):
            missing = sorted(set(self.compute_q_tiles) - written)
            raise RuntimeError(
                f"UltraAttn graph produced no partial for Q tiles {missing}"
            )

        with torch.cuda.nvtx.range("ultraattn_graph/return_partial"):
            self._pack_partial_node()
            out_work, lse_work = self._launch_partial_node()

        # Owner-local partials can initialize merge nodes while returned remote
        # partials are moving on the ProcessGroupNCCL stream.
        with torch.cuda.nvtx.range("ultraattn_graph/local_merge"):
            self._initialize_merge_nodes()
        out_work.wait()
        lse_work.wait()
        with torch.cuda.nvtx.range("ultraattn_graph/remote_merge"):
            self._finish_merge_nodes()
        return self.final_out, self.final_lse


def packed_causal_reference(
    process_group: Optional[dist.ProcessGroup],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    global_seqlens: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference the fixed-suite graph result with full-document min-FA3 calls.

    Each local query segment attends the matching document prefix.  FA3's
    bottom-right causal alignment then maps the first local query to its true
    document position.  This validates graph communication and partial merging
    without materializing a quadratic dense score tensor.
    """
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    gathered_k = torch.empty(
        (world_size * k.size(0), k.size(1), k.size(2)),
        dtype=k.dtype,
        device=k.device,
    )
    gathered_v = torch.empty_like(gathered_k)
    dist.all_gather_into_tensor(gathered_k, k, group=process_group)
    dist.all_gather_into_tensor(gathered_v, v, group=process_group)

    local_begin = rank * q.size(0)
    local_end = local_begin + q.size(0)
    q_lengths: list[int] = []
    k_lengths: list[int] = []
    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    document_begin = 0
    for length in global_seqlens:
        document_end = document_begin + int(length)
        query_begin = max(local_begin, document_begin)
        query_end = min(local_end, document_end)
        if query_begin < query_end:
            q_lengths.append(query_end - query_begin)
            k_lengths.append(query_end - document_begin)
            k_parts.append(gathered_k[document_begin:query_end])
            v_parts.append(gathered_v[document_begin:query_end])
        document_begin = document_end
    if not q_lengths:
        return torch.empty_like(q), torch.empty(
            (q.size(1), 0), dtype=torch.float32, device=q.device
        )

    cu_q_host = torch.zeros(len(q_lengths) + 1, dtype=torch.int32)
    cu_k_host = torch.zeros(len(k_lengths) + 1, dtype=torch.int32)
    cu_q_host[1:] = torch.cumsum(torch.tensor(q_lengths, dtype=torch.int32), 0)
    cu_k_host[1:] = torch.cumsum(torch.tensor(k_lengths, dtype=torch.int32), 0)
    cu_q = cu_q_host.to(device=q.device)
    cu_k = cu_k_host.to(device=q.device)
    reference_k = torch.cat(k_parts, dim=0)
    reference_v = torch.cat(v_parts, dim=0)
    return min_fa3_op.forward_varlen(
        q,
        reference_k,
        reference_v,
        cu_q,
        cu_k,
        max(q_lengths),
        max(k_lengths),
        True,
        cu_seqlens_q_host=cu_q_host,
        cu_seqlens_k_host=cu_k_host,
        return_lse=True,
    )


__all__ = [
    "BLOCK_TOKENS",
    "FIXED_GLOBAL_SEQLENS",
    "UltraAttnPackedCausalForward",
    "missing_plan_command",
    "packed_causal_reference",
    "packed_plan_path",
    "plan_incompatibility",
]
