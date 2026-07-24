"""Standalone Megatron-style hybrid context-parallel baseline."""

from typing import Any

from .plan import (
    ExecutionGroup,
    HybridCPPlan,
    HybridCPProcessGroups,
    SampleAssignment,
    build_hybrid_cp_plan,
    build_hybrid_cp_plan_for_fa3_ring,
    create_hybrid_cp_process_groups,
    hybrid_cp_incompatibility,
    validate_hybrid_cp_plan,
)
from .scheduler import BalancedCPScheduler

__all__ = [
    "BalancedCPScheduler",
    "ExecutionGroup",
    "HybridCPPlan",
    "HybridCPProcessGroups",
    "MegatronHybridCPAttention",
    "PackedHybridCPInputs",
    "SampleAssignment",
    "build_hybrid_cp_plan",
    "build_hybrid_cp_plan_for_fa3_ring",
    "backward_reference",
    "create_hybrid_cp_process_groups",
    "hybrid_cp_incompatibility",
    "forward_reference",
    "make_packed_hybrid_cp_inputs",
    "validate_hybrid_cp_plan",
]


def __getattr__(name: str) -> Any:
    if name in {
        "MegatronHybridCPAttention",
        "PackedHybridCPInputs",
        "make_packed_hybrid_cp_inputs",
    }:
        from . import attention

        return getattr(attention, name)
    if name in {"backward_reference", "forward_reference"}:
        from . import reference

        return getattr(reference, name)
    raise AttributeError(name)
