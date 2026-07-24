# Standalone Megatron Hybrid CP baseline

This directory contains a forward and causal-backward hybrid context-parallel
baseline that does not import Megatron-LM, Transformer Engine, or either Python
package at runtime. It uses only PyTorch CUDA/NCCL, this repository's existing
P2P ring helpers, and an FA3 varlen block backend.

The length scheduler in `scheduler.py` is copied and trimmed from
`megatron/core/pipeline_parallel/hybrid_cp_schedule.py` at Megatron-LM commit
`368fa88e382b274c8fc12af851331cc1d30d69cc`. The retained algorithm sorts
requests by decreasing length, buckets estimated attention work, selects the
next-power-of-two CP size, creates balanced execution groups, and expands a
sample's CP group when necessary to fill otherwise idle ranks. Dataloader
rerouting, model execution, TP/PP, loss, all-to-all, and TE integration were
removed.

## Execution contract

`build_hybrid_cp_plan(global_lengths, world_size,
max_seqlen_per_rank=8192)` returns every execution group's per-rank sample-id
sequence and one assignment per input sample. Assignments contain the actual
CP size after any scheduler expansion, a contiguous aligned rank start, and the
execution-group id. The compiler rejects samples that need more ranks than the
physical world and validates sample uniqueness, aligned members, and identical
relative execution order among each sample's members.

`build_hybrid_cp_plan_for_fa3_ring(global_lengths, world_size, is_causal,
max_seqlen_per_rank=8192)` uses the same copied scheduler algorithm but caps
each sample's initial CP demand to the physical world before constructing
execution groups:

```text
uncapped_required_cp = max(1, ceil_pow2(length / max_seqlen_per_rank))
required_cp = min(uncapped_required_cp, world_size)
```

The 8192 value therefore remains the normal CP-sizing target, but it is not a
hard local-length limit once a sample reaches the largest physical CP group.
For samples whose uncapped requirement fits in the world, scheduling is
identical to `build_hybrid_cp_plan()`. After scheduling, the FA3 builder copies
the plan and minimally rounds only the copied `global_lengths` and
`SampleAssignment.global_length` fields for execution:

```text
CP1:              alignment = 1
CP>1 noncausal:   alignment = actual_cp
CP>1 causal:      alignment = 256 * actual_cp
execution_length = ceil(original_length / alignment) * alignment
```

Padding is deliberately post-schedule. It does not change the capped
scheduler's CP sizes, rank starts, execution groups, or per-rank sample order,
and the base plan API remains unchanged. Causal CP2/CP4/CP8 therefore use
512/1024/2048-token alignment, while noncausal CP>1 only pads enough to divide
the sample across its final CP members.

`create_hybrid_cp_process_groups(dist.group.WORLD)` must run on every rank. It
creates all aligned contiguous CP2 groups, then all CP4 groups, in the same
global order. CP equal to the physical world reuses `dist.group.WORLD`; CP1
uses no communicator. The
baseline targets one-node 2/4/8-rank jobs.

`MegatronHybridCPAttention.forward_all()` executes execution groups and each
rank's samples in forward order. CP1 calls local varlen FA3. CP>1 uses the
existing P2P ring: ordinary ring for noncausal attention and zigzag ring for
causal attention. Output and LSE remain live for a later complete backward
phase. There is one world barrier between adjacent execution groups and no
barrier after the final group.

`backward_all()` is causal-only and deliberately replays groups and samples in
the same forward order. This differs from conventional reverse-order autograd
and from Megatron's current per-sample forward-then-backward memory behavior.
CP1 calls a local FA3 block backward. CP>1 reuses the existing FA3 zigzag
backward with P2P K/V rotation and FP32 dK/dV ring reduction. Returned dQ/dK/dV
use this rank's plan-assignment packing order.

All ranks use external FA3 only when its required entry points are importable
on every rank. Otherwise all ranks fall back consistently to the in-repo
`min_fa3` extension. Supported tensors are CUDA BF16 with D=128 and
`QH % KVH == 0`. The topology and dataset frontends execute the post-schedule
padded plan, so CP divisibility and causal local-half alignment are satisfied
without rescheduling. An uncapped CP demand larger than the physical world is
saturated to the world size. Such a sample can have a local execution length
larger than `max_seqlen_per_rank`; available memory and backend kernel support
remain the practical limits for extreme lengths.

Forward timing covers the complete `forward_all()` phase, including FA3, P2P,
and execution-group barriers. It excludes scheduling, process-group creation,
backend selection, input packing, and the phase-start synchronization.
Backward iterations first run a complete `forward_all()` outside the measured
interval, synchronize, and then time only the complete `backward_all()` phase.
There is no combined forward-plus-backward result.

The baseline is invoked through the existing topology and dataset frontends
with `--methods megatron_hybrid_cp`. Use `--check` only for small workloads
because the dense reference materializes quadratic attention scores.

## Reporting and CP saturation

Input allocation, `cu_seqlens`, correctness references, P2P communication,
forward preparation, and timed forward/backward kernels all use execution
lengths. Padding is benchmarked as real physical work; no additional mask or
non-FA3 fallback restores the original per-token numerical semantics.

The latency table and cross-case summary continue to calculate effective
TFLOPS from the original lengths. Each Megatron result Note reports
`tokens(original/aligned)`, padding tokens, and aligned-length aggregate and
average-per-GPU TFLOPS using the same measured latency. If CP demand was
saturated, the Note also reports `required_cp(original/capped)` and the actual
local execution length. The metadata-only load model similarly assigns
original effective tokens/scores evenly across the scheduled CP members, while
physical tokens/scores, tile work, and communication use the padded execution
plan.

Historical dataset skips caused only by final-CP divisibility or causal
half-sequence alignment are eliminated. For example, a 7168-token sample that
the scheduler expands to CP8 now executes at 8192 tokens rather than being
rejected because its original 448-token local half is not 128-aligned.

The base builder still preserves the original strict behavior:

```text
required_cp = max(1, ceil_pow2(length / max_seqlen_per_rank))
```

With `max_seqlen_per_rank=8192`, the 75776-token sample has an uncapped CP16
demand:

```text
ceil_pow2(75776 / 8192) = ceil_pow2(9.25) = 16
```

`build_hybrid_cp_plan()` therefore rejects it, preserving its existing API
semantics. The FA3 execution builder caps it to CP8 instead:

```text
actual_cp    = 8
local_length = 75776 / 8 = 9472
local_half   = 4736 = 37 * 128
```

The length is already a multiple of the causal CP8 alignment, 2048, so it runs
without padding. A length such as 75777 is capped to CP8 and then padded to
77824 before execution.
