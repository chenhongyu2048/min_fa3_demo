# Hierarchical Hybrid Mega-Ring Backward Design

> **文档状态：backward 专项细节。** Forward/backward 的统一架构、设计动机、TMA
> 粒度与论文实验建议见
> [MegaRing Hybrid Megakernel 设计](./MEGARING_HYBRID_KERNEL_DESIGN.md)。本文继续作为
> backward tensor、readiness 和 completion contract 的专项参考。

## Status And Scope

This document describes the implemented causal varlen hierarchical mega-ring
backward path. The kernel remains a copied-and-trimmed FA3 Hopper backward:
preprocess, the persistent compute CTA, the communication CTA, and dQ/dKV
postprocess keep their existing roles. The hierarchy changes how work and
communication are scheduled; it does not replace the FA3 backward main path.

The supported specialization is intentionally narrow:

- Hopper SM90
- BF16 input and output gradients
- head dimension 128
- causal zigzag self-attention
- non-deterministic backward
- `qhead % kvhead == 0`
- `kvhead * 128 == 1024`
- physical world size 1, 2, 4, or 8
- positive compute-CTA and communication-CTA SM counts

World size 1 supports only G1. World sizes 2 and 4 support the hierarchy levels
that fit in the physical world. The complete G8/G4/G2/G1 hierarchy requires
eight local GPUs. Noncausal backward, deterministic backward, other dtypes,
other head dimensions, and cross-node transport are not implemented here.

## Public Interface

The public Python entry is:

```python
backward_varlen_mega_ring(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *,
    cu_seqlens_q_host,
    cu_seqlens_k_host,
    remote_k,
    remote_v,
    remote_dk_accum,
    remote_dv_accum,
    remote_dkv_completion,
    num_comp_sm,
    num_comm_sm,
    global_seqlens_host,
    ring_sizes_host,
    ring_starts_host,
)
```

`half_cu_seqlens` and `half_cu_seqlens_host` are not public inputs. The binding
derives causal half lengths from the local sequence lengths and ring metadata,
then copies the generated prefix sum to CUDA for the scheduler and mainloop.

All-CP is represented explicitly rather than through a separate API:

```text
ring_size = world_size
ring_start = 0
```

The three topology tensors are CPU, contiguous, int32 tensors with shape
`[B]`. Every rank is expected to pass identical topology metadata.

## Tensor And Arena Contract

Q, O, dO, and the returned gradients are compact rank-local tensors:

```text
q, out, dout, dq: [local_total_q, QH, 128]
dk, dv:            [local_total_k, KVH, 128]
```

K and V are rank-major IPC arenas with one shared row stride for all ranks:

```text
k, v: [world_size * rank_kv_capacity, KVH, 128]
```

The owner-local rows for rank `r` begin at:

```text
owner_base = r * rank_kv_capacity
```

`rank_kv_capacity` must be positive and 128-row aligned. It must cover the
largest compact local K/V total used by any rank. Returned dK/dV contain only
the current rank's `local_total_k` active rows, not the full arena capacity.

Each rank owns one VMM-backed FP32 dK accumulator and one dV accumulator. Their
capacity includes one zero-padded 128-row separator per batch:

```text
padded_rank_capacity = round_up(rank_kv_capacity + B * 128, 128)
accumulator_numel = KVH * padded_rank_capacity * 128
```

Because both `rank_kv_capacity` and `B * 128` are already 128-row aligned, the
round-up does not normally add another row block. The explicit formula remains
the binding contract. `remote_dkv_completion` is one VMM-backed int32 element.

Before every backward call, the caller must zero both owner accumulators and the
completion scalar, synchronize CUDA, and execute a distributed barrier. Owner
K/V initialization requires the same synchronization before forward/backward
uses the arena.

## Ring Metadata Validation

For batch `b`:

```text
ring_size in {1, 2, 4, 8}
ring_start >= 0
ring_start % ring_size == 0
ring_start + ring_size <= world_size
global_length > 0
global_length % ring_size == 0
```

Batches are ordered by non-increasing ring size: G8, then G4, G2, and G1.
Within one level, starts and lengths can be mixed.

The expected local length is:

```text
expected_local_length = global_length / ring_size  for subgroup members
                        0                          for nonmembers
```

Backward is self-attention, so local Q and K lengths must match. Every nonzero
local length and the arena capacity are 128-row aligned. G8/G4/G2 additionally
require a 128-row-aligned local half, which makes their local sequence length a
multiple of 256. A rank may have no local Q/K rows at all.

## Shared Topology Metadata

Forward and backward keep separate parameter structures. They share only the
fixed topology descriptors and section helpers in
`include/min_fa3_mega_ring_hierarchy.h`:

```text
MegaRingLevelDesc
MegaRingHierarchyDesc
```

Each level records its exact-size batch interval, compact full/half row ranges,
full/half KV tile counts, reduction tile prefix, and readiness base. The fixed
level order is G8, G4, G2, G1. Padded dKV row offsets are derived separately
from `cu_seqlens_k` and the batch index during communication-task decoding.

## Exact-Size Scheduler Stream

One scheduler ticket always represents exactly one:

```text
(level, ring_step, KV tile)
```

There is no multi-segment claim fusion and no fragment lifetime spanning ring
steps. The stream is:

```text
step 0: all G8/G4/G2/G1 local full-KV tiles
G8:     steps 1..7
G4:     steps 1..3
G2:     step 1
G1:     no remote replay
```

For one level, let `r` be the current rank within its aligned subgroup. The
rank-local work count is:

```text
base_tiles
+ r * half_tiles
+ (G - 1 - r) * full_tiles
```

Steps `1..r` use front-half KV tiles. Steps `r+1..G-1` use full KV tiles. The
level decoder only scans the batch range belonging to the target ring size, so
overlapping physical subgroups do not alias scheduler batches.

## Owner Addressing And Zigzag Work

For every decoded batch and step:

```text
ring_base = (global_rank / ring_size) * ring_size
ring_local_rank = global_rank - ring_base
owner_rank = ring_base
           + (ring_local_rank - step + ring_size) % ring_size
```

Aligned starts guarantee that this arithmetic selects the batch's subgroup.
K/V addressing is:

```text
owner_rank * rank_kv_capacity + batch_local_offset
```

The attention regions are:

```text
step 0:       local causal attention
steps 1..r:   full local Q against owner front-half KV
later steps:  local back-half Q against owner full KV
```

This preserves the causal zigzag schedule used by the previous all-CP backward
while allowing several exact-size subrings to overlap on the same GPUs.

## K/V Ingress Readiness

Remote K/V ingress has 11 readiness sections:

```text
G8 steps 1..7: base 0,  7 sections
G4 steps 1..3: base 7,  3 sections
G2 step 1:     base 10, 1 section
G1:                     0 sections
```

Each logical communication task covers 128 K/V token rows. It is transferred
as fixed 16-row by 1024-BF16 TMA subtiles. There is no tail fallback. The
compute mainloop waits on the readiness counter associated with its exact
`(level, step)` section before consuming remotely loaded K/V.

## dKV Step Buffers And Readiness

The local FP32 intermediate buffers retain the world-size step dimension:

```text
dk_steps, dv_steps: [world_size, step_stride]
step_stride = KVH * padded_rank_capacity * 128
```

Different hierarchy levels reuse the same step buffers. Their batch ranges are
made disjoint by adding 128 padding rows after every batch. The epilogue writes
one step buffer and increments one of 15 dKV readiness sections:

```text
G8 steps 0..7: base 0,  8 sections
G4 steps 0..3: base 8,  4 sections
G2 steps 0..1: base 12, 2 sections
G1 step 0:     base 14, 1 section
```

Each section waits for its exact full-tile or half-tile work count. Empty level
sections have no communication work and do not contribute completion signals.

## Padded dKV Reduction

For a level covering batches `[batch_begin, batch_end)`, the communication CTA
uses this padded row interval:

```text
padded_begin = cu_seqlens_k[batch_begin] + batch_begin * 128
padded_end   = cu_seqlens_k[batch_end]   + batch_end * 128
```

The interval is decoded by KV head and 128-token padded block. One logical dK
or dV task is one fixed `16 x 1024` FP32 TMA transaction. The communication CTA
loads that transaction from the selected local step buffer and executes remote
TMA reduce-add into the target owner's FP32 accumulator.

The padding rows start at zero in `dk_steps`/`dv_steps`, remain zero, and are
reduced like ordinary blocks. This avoids a tail path and keeps every remote
transaction aligned without changing the compact returned dK/dV layout.

The FP32 accumulator uses FA3's internal MMA/postprocess layout. It is not a
natural `[KVH, token, D]` gradient tensor. Numerical validation belongs on the
postprocessed BF16 dK/dV outputs; accumulator validation checks active values,
finite state, and zero padding.

## Completion Protocol

After all communication CTAs finish one valid `(level, step)` section, only the
last CTA performs a system-scope increment on that section's target owner.

An owner waits for:

```text
expected_completion = sum(
    ring_size for each hierarchy level with owner-local data
)
```

For example, an owner with data in G8, G4, G2, and G1 waits for
`8 + 4 + 2 + 1 = 15`. This prevents one smaller overlapping subring from
starting dK/dV postprocess before a larger subring has delivered all of its
contributions. Postprocess reads the owner accumulator only after the target is
reached.

## Empty-Rank Behavior

A rank with no local Q/K rows still participates in IPC setup and launches the
fused kernel. The binding provides one-row dummy Q/O/dO and one-column LSE
descriptor backing because zero-sized TMA descriptors are invalid. Device
`cu_seqlens` remain zero-length by batch, so scheduler, preprocess, and
postprocess perform no logical tensor access. K/V arena capacity must still be
positive and 128-row aligned.

## Verification Entry Points

All-CP compatibility:

```bash
torchrun --standalone --nproc_per_node=2 \
  mega_ring_test_min_fa3_varlen_backward_multi_rank.py \
  --b 1 --seqlen 256 --qhead 16 --kvhead 8 \
  --num-comp-sm 64 --num-comm-sm 8
```

Eight-GPU overlapping hierarchy:

```bash
torchrun --standalone --nproc_per_node=8 \
  mega_ring_test_min_fa3_varlen_backward_hybrid_multi_rank.py \
  --global-seqlens 2048,1024,512,256 \
  --ring-sizes 8,4,2,1 --ring-starts 0,4,2,7 \
  --qhead 16 --kvhead 8 --repeat 2 \
  --num-comp-sm 100 --num-comm-sm 16
```

Binding validation failures:

```bash
torchrun --standalone --nproc_per_node=8 \
  mega_ring_test_min_fa3_varlen_backward_validation_multi_rank.py
```

Dataset-shaped performance sweep:

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_dataset_backward.py \
  --dataset arxiv --target-tokens 131072 --seed 0 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --methods all \
  --sm-configs 128:4,124:8,120:12,116:16 \
  --warmup-iters 10 --num-iters 40 --no-check
```

The dataset frontend reuses the forward `balancer` placement and calls the
programmatic backward benchmark with explicit global lengths, ring sizes, and
ring starts. Forward preparation, owner accumulator reset, and the distributed
barrier remain outside the timed region. The result reports each rank's
average time, the average of per-iteration max-rank wall times, and
aggregate/average-per-GPU causal backward TFLOP/s.
The same table includes per-sequence all-gather, Llama3 whole-packed
all-gather, and external-FA3/NCCL zigzag ring baselines; external FA3 falls back
consistently to local min-FA3 when unavailable. Dense `--check` validation is
intended only for small token budgets.

The validation test deliberately supplies a later invalid accumulator capacity
as a safeguard. If a target metadata validation is accidentally removed, the
call fails on the safeguard instead of entering the fused kernel, and the test
reports that the observed error did not match the expected validation.

## Explicit Non-Goals

This path does not implement:

- multi-segment scheduler claim fusion
- fragments retained across ring steps
- deterministic mega-ring backward
- noncausal mega-ring backward
- FP16, FP8, or non-128 head dimensions
- unaligned communication tails
- cross-node IPC or metadata collectives in the hot path

Any future multi-segment optimization should remain a separate specialization
and be accepted only after resource, spill, occupancy, and end-to-end H100 A/B
measurements.
