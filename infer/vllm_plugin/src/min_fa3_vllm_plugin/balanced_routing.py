"""Deterministic, EP-balanced expert routing for attention benchmarks."""

import torch

from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    RoutingSimulator,
    RoutingStrategy,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _balanced_routes(ids, weights, count, EP: tl.constexpr,
                     LOCAL_EXPERTS: tl.constexpr, TOP_K: tl.constexpr,
                     BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Interleave EP ranks first, then rotate through each rank's experts.
    expert = offsets % EP * LOCAL_EXPERTS + offsets // EP % LOCAL_EXPERTS
    tl.store(ids + offsets, expert, offsets < count)
    tl.store(weights + offsets, 1.0 / TOP_K, offsets < count)


def balanced_routes(num_tokens, num_experts, top_k, ep_size, device, indices_type):
    if num_experts % ep_size or not 0 < top_k <= num_experts:
        raise ValueError("balanced routing requires equal expert shards and valid top_k")
    ids = torch.empty((num_tokens, top_k), device=device, dtype=indices_type)
    weights = torch.empty_like(ids, dtype=torch.float32)
    count = num_tokens * top_k
    if count:
        _balanced_routes[(triton.cdiv(count, 256),)](
            ids, weights, count, ep_size, num_experts // ep_size, top_k, 256,
        )
    return weights, ids


class BalancedRouting(RoutingStrategy):
    def route_tokens(self, hidden_states, router_logits, top_k, indices_type=None):
        return balanced_routes(
            hidden_states.shape[0], router_logits.shape[-1], top_k,
            get_ep_group().world_size, hidden_states.device,
            indices_type if indices_type is not None else torch.int64,
        )


def register_balanced_routing():
    RoutingSimulator.register_strategy("min_fa3_balanced", BalancedRouting())
