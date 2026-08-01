"""Production-shaped TP/DCP topology helpers for the minimal FA3 demo.

The validation rules are copied and trimmed from the GQA/MQA constraints in
vLLM commit ``a89015c6df8eeb37a843b717c97a5be1355de83d`` and the contiguous
DCP group construction in SGLang commit
``8d6549bc4039d33635844495d86684677a4f0df8``.  This module is deliberately
CPU-only and has no runtime dependency on either project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class TopologyIssue:
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DCPTopology:
    q_heads: int
    kv_heads: int
    tp_size: int
    dcp_size: int

    @property
    def q_heads_local(self) -> int:
        return self.q_heads // self.tp_size

    @property
    def q_heads_per_kv(self) -> int:
        return self.q_heads // self.kv_heads

    @property
    def kv_replicas(self) -> int:
        return self.tp_size // self.kv_heads

    def kv_head_for_rank(self, tp_rank: int) -> int:
        self._check_rank(tp_rank)
        return tp_rank // self.kv_replicas

    def kv_replica_ranks(self, tp_rank: int) -> tuple[int, ...]:
        kv_head = self.kv_head_for_rank(tp_rank)
        start = kv_head * self.kv_replicas
        return tuple(range(start, start + self.kv_replicas))

    def dcp_group_ranks(self, tp_rank: int) -> tuple[int, ...]:
        replica_ranks = self.kv_replica_ranks(tp_rank)
        offset = tp_rank - replica_ranks[0]
        start = replica_ranks[0] + (offset // self.dcp_size) * self.dcp_size
        return tuple(range(start, start + self.dcp_size))

    def dcp_rank(self, tp_rank: int) -> int:
        group = self.dcp_group_ranks(tp_rank)
        return tp_rank - group[0]

    def q_head_range(self, tp_rank: int) -> tuple[int, int]:
        self._check_rank(tp_rank)
        start = tp_rank * self.q_heads_local
        return start, start + self.q_heads_local

    def all_dcp_groups(self) -> tuple[tuple[int, ...], ...]:
        groups: list[tuple[int, ...]] = []
        for rank in range(self.tp_size):
            group = self.dcp_group_ranks(rank)
            if not groups or groups[-1] != group:
                groups.append(group)
        return tuple(groups)

    def to_dict(self) -> dict[str, object]:
        return {
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "tp_size": self.tp_size,
            "dcp_size": self.dcp_size,
            "q_heads_local": self.q_heads_local,
            "q_heads_per_kv": self.q_heads_per_kv,
            "kv_replicas": self.kv_replicas,
            "dcp_groups": [list(group) for group in self.all_dcp_groups()],
        }

    def _check_rank(self, tp_rank: int) -> None:
        if not 0 <= tp_rank < self.tp_size:
            raise ValueError(
                f"tp_rank must be in [0, {self.tp_size}), got {tp_rank}"
            )


def validate_topology(
    q_heads: int,
    kv_heads: int,
    tp_size: int,
    dcp_size: int,
) -> tuple[TopologyIssue, ...]:
    issues: list[TopologyIssue] = []
    values = {
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "tp_size": tp_size,
        "dcp_size": dcp_size,
    }
    for name, value in values.items():
        if value <= 0:
            issues.append(
                TopologyIssue("nonpositive_value", f"{name} must be positive, got {value}")
            )
    if issues:
        return tuple(issues)

    if q_heads % tp_size:
        issues.append(
            TopologyIssue(
                "q_heads_not_divisible_by_tp",
                f"global Q heads {q_heads} must be divisible by TP size {tp_size}",
            )
        )
    if q_heads % kv_heads:
        issues.append(
            TopologyIssue(
                "q_heads_not_divisible_by_kv_heads",
                f"global Q heads {q_heads} must be divisible by global KV heads {kv_heads}",
            )
        )
    if tp_size <= kv_heads:
        issues.append(
            TopologyIssue(
                "tp_not_greater_than_kv_heads",
                f"DCP GQA/MQA requires TP size {tp_size} > global KV heads {kv_heads}",
            )
        )
    if tp_size % kv_heads:
        issues.append(
            TopologyIssue(
                "kv_heads_not_divisible_into_tp",
                f"TP size {tp_size} must be divisible by global KV heads {kv_heads}",
            )
        )
    if tp_size % dcp_size:
        issues.append(
            TopologyIssue(
                "dcp_not_divisible_into_tp",
                f"DCP size {dcp_size} must divide TP size {tp_size}",
            )
        )

    if tp_size % kv_heads == 0:
        replicas = tp_size // kv_heads
        if dcp_size > replicas:
            issues.append(
                TopologyIssue(
                    "dcp_exceeds_kv_replicas",
                    f"DCP size {dcp_size} exceeds KV replica count {replicas}",
                )
            )
        if replicas % dcp_size:
            issues.append(
                TopologyIssue(
                    "kv_replicas_not_divisible_by_dcp",
                    f"KV replica count {replicas} must be divisible by DCP size {dcp_size}",
                )
            )

    if q_heads % kv_heads == 0:
        q_per_kv = q_heads // kv_heads
        if q_per_kv % dcp_size:
            issues.append(
                TopologyIssue(
                    "q_per_kv_not_divisible_by_dcp",
                    f"Q heads per KV head {q_per_kv} must be divisible by DCP size {dcp_size}",
                )
            )
    return tuple(issues)


def make_topology(
    q_heads: int,
    kv_heads: int,
    tp_size: int,
    dcp_size: int,
) -> DCPTopology:
    issues = validate_topology(q_heads, kv_heads, tp_size, dcp_size)
    if issues:
        details = "; ".join(issue.detail for issue in issues)
        raise ValueError(f"invalid DCP topology: {details}")
    return DCPTopology(q_heads, kv_heads, tp_size, dcp_size)


def validate_group_ranks(
    topology: DCPTopology,
    ranks: Iterable[int],
) -> tuple[TopologyIssue, ...]:
    group = tuple(ranks)
    issues: list[TopologyIssue] = []
    if len(group) != topology.dcp_size:
        issues.append(
            TopologyIssue(
                "dcp_group_wrong_size",
                f"DCP group has {len(group)} ranks, expected {topology.dcp_size}",
            )
        )
    if any(rank < 0 or rank >= topology.tp_size for rank in group):
        issues.append(
            TopologyIssue(
                "dcp_group_rank_out_of_range",
                f"DCP group ranks must be in [0, {topology.tp_size}), got {group}",
            )
        )
        return tuple(issues)
    kv_heads = {topology.kv_head_for_rank(rank) for rank in group}
    if len(kv_heads) != 1:
        issues.append(
            TopologyIssue(
                "dcp_group_crosses_kv_replica_boundary",
                f"DCP group {group} spans global KV heads {sorted(kv_heads)}",
            )
        )
    if group and group != tuple(range(group[0], group[0] + len(group))):
        issues.append(
            TopologyIssue(
                "dcp_group_not_contiguous",
                f"DCP group must contain contiguous TP ranks, got {group}",
            )
        )
    return tuple(issues)


__all__ = [
    "DCPTopology",
    "TopologyIssue",
    "make_topology",
    "validate_group_ranks",
    "validate_topology",
]
