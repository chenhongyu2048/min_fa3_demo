"""Python wrapper for the minimal Hopper forward-only FlashAttention demo."""

import os
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist

from _min_fa3_op import (
    TKParallelTensor,
    parallel_remote_load_out as _parallel_remote_load_out_cuda,
    parallel_remote_load_vec_out as _parallel_remote_load_vec_out_cuda,
    forward,
    forward_varlen as _forward_varlen_cuda,
    forward_varlen_mega_ring as _forward_varlen_mega_ring_cuda,
    forward_varlen_ring as _forward_varlen_ring_cuda,
    parallel_remote_load as _parallel_remote_load_cuda,
    parallel_remote_load_vec as _parallel_remote_load_vec_cuda,
)

# Resolve the local rank metadata used by the TK parallel IPC path.
# Args:
#   local_rank: Optional explicit CUDA device index for the current process.
#   local_world_size: Optional explicit number of local ranks on this node.
def _resolve_parallel_context(
    local_rank: Optional[int],
    local_world_size: Optional[int],
) -> Tuple[int, int]:
    if local_rank is None:
        env_local_rank = os.environ.get("LOCAL_RANK")
        if env_local_rank is not None:
            local_rank = int(env_local_rank)
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for parallel_remote_load")
            local_rank = torch.cuda.current_device()

    if local_world_size is None:
        env_local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
        if env_local_world_size is not None:
            local_world_size = int(env_local_world_size)
        elif dist.is_available() and dist.is_initialized():
            local_world_size = dist.get_world_size()
        else:
            raise RuntimeError(
                "local_world_size is required when LOCAL_WORLD_SIZE is not set "
                "and torch.distributed is not initialized"
            )

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        if world_size != local_world_size:
            raise RuntimeError(
                "ThunderKittens parallel IPC path is single-node only: "
                f"world_size={world_size}, local_world_size={local_world_size}"
            )

    return int(local_rank), int(local_world_size)


# Wrap a validated local CUDA tensor in the TKParallelTensor IPC container.
# Args:
#   tensor: Contiguous CUDA tensor owned by the current local rank.
#   local_rank: Optional explicit CUDA device index for the current process.
#   local_world_size: Optional explicit number of local ranks on this node.
def create_parallel_tensor(
    tensor: torch.Tensor,
    *,
    local_rank: Optional[int] = None,
    local_world_size: Optional[int] = None,
) -> TKParallelTensor:
    local_rank, local_world_size = _resolve_parallel_context(local_rank, local_world_size)
    if not tensor.is_cuda:
        raise ValueError("tensor must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
    if tensor.device.index != local_rank:
        raise ValueError(
            f"tensor device index ({tensor.device.index}) must match local_rank ({local_rank})"
        )
    return TKParallelTensor(
        tensor,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=False,
    )


# Load a remote tensor from src_rank, optionally writing into a caller-provided output buffer.
# Args:
#   input_tensor: Source tensor as either a raw CUDA tensor or a TKParallelTensor wrapper.
#   src_rank: Rank that owns the source tensor to be remotely loaded.
#   output: Optional preallocated destination tensor to fill in-place.
#   num_blocks: Optional thread-block count; defaults to the local device SM count.
#   local_rank: Optional explicit CUDA device index for the current process.
#   local_world_size: Optional explicit number of local ranks on this node.
def parallel_remote_load(
    input_tensor: Union[torch.Tensor, TKParallelTensor],
    src_rank: int,
    output: Optional[torch.Tensor] = None,
    num_blocks: Optional[int] = None,
    local_rank: Optional[int] = None,
    local_world_size: Optional[int] = None,
) -> torch.Tensor:
    if isinstance(input_tensor, TKParallelTensor):
        parallel_input = input_tensor
    else:
        parallel_input = create_parallel_tensor(
            input_tensor,
            local_rank=local_rank,
            local_world_size=local_world_size,
        )

    if num_blocks is None:
        num_blocks = torch.cuda.get_device_properties(parallel_input.local_rank_).multi_processor_count

    if output is not None:
        _parallel_remote_load_out_cuda(output, parallel_input, int(src_rank), int(num_blocks))
        return output

    return _parallel_remote_load_cuda(parallel_input, int(src_rank), int(num_blocks))

# Load a remote tensor row-by-row with a vector TMA path, optionally writing into a caller-provided output buffer.
# Args:
#   input_tensor: Source tensor as either a raw CUDA tensor or a TKParallelTensor wrapper.
#   src_rank: Rank that owns the source tensor to be remotely loaded.
#   output: Optional preallocated destination tensor to fill in-place.
#   num_blocks: Optional thread-block count; defaults to the local device SM count.
#   local_rank: Optional explicit CUDA device index for the current process.
#   local_world_size: Optional explicit number of local ranks on this node.
def parallel_remote_load_vec(
    input_tensor: Union[torch.Tensor, TKParallelTensor],
    src_rank: int,
    output: Optional[torch.Tensor] = None,
    num_blocks: Optional[int] = None,
    local_rank: Optional[int] = None,
    local_world_size: Optional[int] = None,
) -> torch.Tensor:
    if isinstance(input_tensor, TKParallelTensor):
        parallel_input = input_tensor
    else:
        parallel_input = create_parallel_tensor(
            input_tensor,
            local_rank=local_rank,
            local_world_size=local_world_size,
        )

    if num_blocks is None:
        num_blocks = torch.cuda.get_device_properties(parallel_input.local_rank_).multi_processor_count

    if output is not None:
        _parallel_remote_load_vec_out_cuda(output, parallel_input, int(src_rank), int(num_blocks))
        return output

    return _parallel_remote_load_vec_cuda(parallel_input, int(src_rank), int(num_blocks))


def forward_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool,
    *,
    cu_seqlens_q_host: torch.Tensor,
    cu_seqlens_k_host: torch.Tensor,
    manual_block_count: Optional[int] = None,
) -> torch.Tensor:
    return _forward_varlen_cuda(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_q_host,
        cu_seqlens_k_host,
        int(max_seqlen_q),
        int(max_seqlen_k),
        bool(is_causal),
        manual_block_count=manual_block_count,
    )


def forward_varlen_ring(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool,
    *,
    cu_seqlens_q_host: torch.Tensor,
    cu_seqlens_k_host: torch.Tensor,
    remote_k: TKParallelTensor,
    remote_v: TKParallelTensor,
    src_rank: int,
    num_comp_sm: int,
    num_comm_sm: int,
    ring_step: int,
    prefetch_k: Union[torch.Tensor, TKParallelTensor],
    prefetch_v: Union[torch.Tensor, TKParallelTensor],
) -> torch.Tensor:
    prefetch_k_tensor = prefetch_k.data_ if isinstance(prefetch_k, TKParallelTensor) else prefetch_k
    prefetch_v_tensor = prefetch_v.data_ if isinstance(prefetch_v, TKParallelTensor) else prefetch_v

    return _forward_varlen_ring_cuda(
        q,
        k,
        v,
        remote_k,
        remote_v,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_q_host,
        cu_seqlens_k_host,
        int(max_seqlen_q),
        int(max_seqlen_k),
        bool(is_causal),
        int(src_rank),
        int(num_comp_sm),
        int(num_comm_sm),
        int(ring_step),
        prefetch_k_tensor,
        prefetch_v_tensor,
    )


def forward_varlen_mega_ring(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool,
    *,
    cu_seqlens_q_host: torch.Tensor,
    cu_seqlens_k_host: torch.Tensor,
    remote_k: TKParallelTensor,
    remote_v: TKParallelTensor,
    num_comp_sm: int,
    num_comm_sm: int,
) -> torch.Tensor:
    # MEGA_RING: explicit opt-in path. K/V are the local concatenated
    # [world_size * local_total_k, kv_heads, 128] buffers used by the fused
    # persistent kernel; default forward_varlen_ring remains single-step.
    return _forward_varlen_mega_ring_cuda(
        q,
        k,
        v,
        remote_k,
        remote_v,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_q_host,
        cu_seqlens_k_host,
        int(max_seqlen_q),
        int(max_seqlen_k),
        bool(is_causal),
        int(num_comp_sm),
        int(num_comm_sm),
    )


__all__ = [
    "TKParallelTensor",
    "create_parallel_tensor",
    "forward",
    "forward_varlen",
    "forward_varlen_mega_ring",
    "forward_varlen_ring",
    "parallel_remote_load",
    "parallel_remote_load_vec",
]
