# Archived Mega-Ring Forward/Backward Remote Load Optimization Plan

## Status

Archived and superseded by the implemented mega-ring tile-copy path. The final
implementation uses 128/176-row logical communication tasks decomposed into
fixed 16-row 2D TMA subtiles, assumes 128-row-aligned target ranges, and has no
unaligned-tail fallback. See the [root README](../README.md) for the current
contract.

Everything below is retained only as the pre-implementation design draft. In
particular, its proposed 8-row chunks, row-granular hybrid fallback, readiness
layout, communication-CTA responsibilities, resource estimates, and
verification plan do not describe the final implementation.

## Pre-Optimization Path

At the time of this draft, forward and backward used the same
communication-CTA ingress:

```text
peer global memory
    -> communication CTA shared memory
    -> local concatenated K/V staging buffer
    -> compute CTA FA3 TMA load
    -> compute CTA shared memory
```

The compute-side FA3 loads are already tile-sized. For BF16, head dim 128, and
block N 128, one compute K or V tile is 32 KiB:

```text
128 tokens * 128 dimensions * 2 bytes = 32 KiB
```

Only the peer-to-staging leg is row-granular. With the supported KVH=8 and
head dim 128, one communication row is:

```text
1 token * 8 KV heads * 128 dimensions * 2 bytes = 2 KiB
```

For 128 tokens, K and V therefore require 256 remote-load tasks and 256 local
store tasks. Every task also carries semaphore, commit, wait, and ready-counter
overhead.

The pre-optimization ready counter was rank-granular. A compute CTA waited until
all K/V rows required for that remote rank or causal half-step were resident,
even when the CTA only needed one 128-token block.

Relevant implementation points at the time of the draft were:

- `include/mega_ring_min_fa3_varlen_ring_launch.h`: communication CTA loop.
- `include/min_fa3_mainloop.h`: forward remote-ready wait and local FA3 TMA.
- `include/backward/min_fa3_bwd_mainloop.h`: backward remote-ready wait and
  local FA3 TMA.

## Historical Selected Implementation

Keep communication CTAs and the local K/V staging buffers. Increase only the
peer ingress granularity and readiness granularity.

Use eight consecutive token rows per communication chunk:

```text
kCommRows = 8
kRowElements = 8 * 128 = 1024 BF16
kChunkElements = 8 * 1024 = 8192 BF16
kChunkBytes = 16 KiB
```

This layout is contiguous in both peer and local staging memory because the
buffers use `[token, kv_head, dim]` layout. It does not require a new strided
tensor map or a layout conversion.

Preserve the then-existing number of communication pipeline slots:

- The draft assumed four backward slots, using 64 KiB for four 16 KiB chunks.
- Forward would keep its then-existing `kNumCommChunks`, using 16 KiB per slot.
- Load and store warp roles remain paired per slot.

Do not directly load peer K/V from compute CTAs. Direct peer-to-compute TMA
would remove staging but would repeat NVLink traffic across Q blocks and GQA
Q-head CTAs. The staging buffer ensures each remote K/V element crosses NVLink
once and can then be reused from local HBM.

Do not attempt to move all 128 tokens and all eight KV heads in one shared
tile. That object is 256 KiB for K alone and is not a practical communication
pipeline slot.

## Historical Communication Task Mapping

Replace the row task with a contiguous-run task:

```text
(ring_step, is_v, batch, first_row, row_count, owner_rank)
```

Rules:

1. A task never crosses a batch boundary.
2. A task never crosses a causal front-half boundary.
3. `row_count` is `min(8, rows_remaining_in_segment)`.
4. K and V remain separate tasks because they have different base addresses.
5. Source and destination byte offsets are computed from the same packed row.
6. Full eight-row tasks use one 16 KiB non-tensor bulk transaction.
7. Tail tasks use `row_count * 2048` bytes; the byte count remains 16-byte
   aligned for every positive row count.
8. Pure noncausal and causal mega ring would use the chunk path. Hybrid CP would
   retain the then-existing row path until sparse runs were explicitly
   implemented and tested.

Each slot continues to use a load-complete semaphore and a store-read-complete
semaphore:

```text
load warp:
    wait until slot is reusable
    expect chunk bytes
    issue peer global -> shared bulk load

store warp:
    wait for peer load completion
    issue shared -> local staging bulk store
    wait_group.read before returning the shared slot
    wait for global store completion before publishing readiness
```

The communication CTA must not signal readiness after only
`wait_group.read`: that wait protects shared-memory reuse but does not guarantee
that a compute CTA can observe the local staging data.

## Historical Tile-Level Readiness

Replace the rank-level counter with a tile-level counter:

```text
kv_tile_ready[owner_rank][batch][n_block]
```

For a block containing `valid_rows`:

```text
chunks_per_tensor = ceil_div(valid_rows, 8)
expected_ready = 2 * chunks_per_tensor  # K chunks plus V chunks
```

Each completed local staging store performs a device-scope release increment
on its tile counter. The elected compute producer performs an acquire load and
waits only for the tile it will consume.

The local owner would not need to pass through the communication counters;
compute code would keep the then-existing owner-local fast path. Counter
storage would be zero-initialized per invocation and sized from:

```text
[world_size, batch_size, ceil_div(max_seqlen_k, 128)]
```

Forward waits immediately before its K/V pipeline issues the corresponding
local-staging TMA. Backward waits before acquiring `KVEmpty` and issuing the
single K/V load for its work tile. Waiting before `KVEmpty` avoids holding the
shared K/V resource while peer ingress is incomplete.

For causal zigzag:

- Step 0 uses owner-local K/V and does not wait on a remote counter.
- A front-half step publishes counters only for front-half blocks.
- A full step publishes counters for all blocks in each batch.
- The then-existing requirement that each causal half be 128-row aligned would
  remain.

## Historical Forward and Backward Integration

The proposal would keep the communication implementation shared by forward and
backward. Only the compute-side consumption would differ:

```text
Forward:
    wait for one remote n_block
    feed the then-existing multi-stage K/V FA3 pipeline
    preserve the then-existing pipeline acquire/release semantics

Backward:
    wait for one remote n_block
    wait for KVEmpty
    load one resident K/V tile
    preserve the Q/dO pipeline and dKV epilogue protocol
```

The proposal intended to keep the following behavior unchanged:

- BF16, head dim 128, KVH=8, and QH divisible by 8 restrictions.
- Varlen packed K/V layout and causal zigzag position mapping.
- Source-local dK/dV aggregation and owner-directed remote FP32 reduce-add.
- Communication CTAs perform ingress only; they do not perform dKV egress.
- Regular forward/backward and non-mega ring APIs and kernels.

## Historical Resource Expectations

Compared with the pre-optimization 2 KiB row path:

- Remote-load task count should decrease by approximately 8x for aligned
  lengths.
- Local-store task count should decrease by approximately 8x.
- Ready-counter atomics should decrease by approximately 8x.
- Backward communication shared memory becomes 64 KiB for four slots.
- Forward communication shared memory becomes
  `16 KiB * kNumCommChunks`.
- Compute-kernel register and shared-memory usage must not change.

Before implementation, add compile-time assertions that the communication
scratch fits inside the dynamic shared-memory allocation used by the fused
kernel wrapper for every causal/noncausal forward and backward instance.

## Historical Implementation Sequence

1. Add a reusable chunk decoder for pure mega-ring communication tasks while
   retaining the pre-optimization row decoder as the hybrid fallback.
2. Change the communication shared tile from 1024 BF16 elements to
   `8 * 1024` BF16 elements per slot.
3. Replace row tensor loads/stores with contiguous bulk loads/stores whose byte
   count is 16 KiB for full chunks and dynamic for batch tails.
4. Replace `kv_ready_counts[world_size]` with the tile-level counter layout in
   the bindings, launch params, forward mainloop, and backward mainloop.
5. Move forward/backward waits to the individual `n_block` consumption point.
6. Preserve the backward ordering `remote ready -> KVEmpty -> K/V TMA`.
7. Keep a compile-time or internal test-only row implementation long enough to
   benchmark the old and new communication paths under identical inputs, then
   remove it after the chunk path is validated.

## Historical Verification Plan

Build and static checks:

- Rebuild every forward, backward, ring, and mega-ring instance with nvcc.
- Confirm regular kernel SASS is unchanged outside shared headers.
- Check communication dynamic shared-memory size for world sizes 1 through 8.
- Check ptxas registers and spills for compute kernels against the
  pre-optimization baseline.

Correctness tests on Hopper, initially limited to one or two GPUs:

- World size 1 owner-local fast path for forward and backward.
- World size 2 noncausal MHA and GQA.
- World size 2 causal zigzag.
- Uniform lengths 128, 256, 512, and 2048.
- Noncausal tail lengths such as 129, 257, and 513.
- Heterogeneous varlen batches such as `[129, 257]` and `[256, 512]`.
- One compute SM to force persistent work-tile reuse.
- Repeated execution for at least 100 iterations.
- `compute-sanitizer --tool synccheck` for communication semaphores and the
  backward `KVEmpty` protocol.

Add a debug primitive that copies peer K/V chunks into a local staging tensor
and checks the result byte-for-byte before running attention correctness tests.

Performance measurements:

- Compare 1-row and 8-row ingress with identical compute/communication SM
  allocation.
- Report communication-only effective peer bandwidth.
- Report forward and backward end-to-end latency and TFLOP/s.
- Sweep `num_comm_sm` and sequence lengths 512 through 4096.
- Confirm tile-level readiness advances compute before the complete remote rank
  is resident.
- Profile TMA instruction count, semaphore/atomic overhead, NVLink throughput,
  and compute CTA stalls.

Acceptance criteria:

- Staged K/V is bitwise identical to the peer source.
- No compute CTA observes a partially stored K/V tile.
- Forward outputs and backward dQ/dK/dV pass existing tolerances.
- No deadlock or barrier error in 100 repeated launches.
- Compute-kernel registers, spills, and shared memory do not regress.
- Eight-row chunks reduce communication task and ready-atomic counts by the
  expected factor for aligned lengths.
- End-to-end performance does not regress on tested one- and two-GPU Hopper
  configurations.

## Historical Deferred Alternatives

After the eight-row implementation is validated, benchmark these separately:

1. Sixteen-row, 32 KiB contiguous chunks. This halves task count again but
   increases communication shared-memory pressure.
2. One 128x128, 32 KiB tile per KV head using a strided tensor map. This aligns
   communication and compute tile coordinates but requires peer tensor-map
   management and more complex varlen tails.
3. Direct peer-to-compute TMA. Keep this deferred unless profiling shows that
   staging HBM traffic dominates; it risks multiplying NVLink traffic through
   repeated K/V consumption by Q blocks and GQA heads.
