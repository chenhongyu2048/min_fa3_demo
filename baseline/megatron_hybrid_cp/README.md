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
`QH % KVH == 0`. A sample length must be divisible by its actual CP size.
Causal CP>1 additionally requires each local half-sequence to be 128-token
aligned. A sample for which `ceil_pow2(length / max_seqlen_per_rank)` exceeds
world size is incompatible.

Forward timing covers the complete `forward_all()` phase, including FA3, P2P,
and execution-group barriers. It excludes scheduling, process-group creation,
backend selection, input packing, and the phase-start synchronization.
Backward iterations first run a complete `forward_all()` outside the measured
interval, synchronize, and then time only the complete `backward_all()` phase.
There is no combined forward-plus-backward result.

The baseline is invoked through the existing topology and dataset frontends
with `--methods megatron_hybrid_cp`. Use `--check` only for small workloads
because the dense reference materializes quadratic attention scores.

## Dataset incompatibility cases

The skipped cases in
`benchmark_logs/20260721-215504/benchmark_dataset.log` are compatibility
rejections performed before the runner or an attention kernel is launched.
They are not kernel failures, NCCL deadlocks, or correctness failures. That log
contains no traceback, assertion failure, runtime error, CUDA error, or
`check failed` message, and it was produced with `check=False`. Consequently,
`Check=skip` on a result row means that the dense correctness check was
disabled; only entries under `Skipped methods (causal)` were excluded.

The cross-case summaries show the following coverage:

| Dataset | Executed cases | Skipped cases |
| --- | ---: | --- |
| Arxiv | 18/20 | 9, 17 |
| FreeLaw | 17/20 | 9, 12, 19 |
| GitHub | 19/20 | 17 |
| Pile | 20/20 | none |
| ProLong | 9/20 | 2, 6, 7, 8, 9, 10, 14, 15, 17, 18, 19 |

There are 17 skipped cases in total. One is caused by a sample requiring more
than eight ranks; the other 16 are caused by causal half-sequence alignment
after the scheduler expands a short sample to a larger actual CP group.

### Required CP exceeds the physical world

The initial CP requirement is:

```text
required_cp = max(1, ceil_pow2(length / max_seqlen_per_rank))
```

With `max_seqlen_per_rank=8192` and `world_size=8`, the largest directly
representable sample is `8192 * 8 = 65536` tokens. Arxiv case 17 contains a
75776-token sample:

```text
ceil_pow2(75776 / 8192) = ceil_pow2(9.25) = 16
```

It is therefore rejected with:

```text
sample 1 length 75776 requires CP16, which exceeds world_size=8
```

### Actual CP can exceed the initial CP

The CP derived from the length threshold is only the initial requirement. The
copied Megatron scheduler's `fill_empty_gpus()` pass recursively expands the
smallest existing group when ranks would otherwise be idle:

```text
CP1 -> CP2 -> CP4 -> CP8
```

The plan records the final member count as the sample's actual CP size. Thus a
sample shorter than 8192 tokens can start as CP1 and still execute as CP4 or
CP8. This behavior is intentional and matches the copied Megatron scheduling
algorithm.

For causal CP>1, the current zigzag FA3 path requires:

```text
local_half = global_length / actual_cp / 2
local_half % 128 == 0
```

Equivalently, the global length must satisfy:

```text
global_length % (actual_cp * 256) == 0
```

The resulting alignment requirements are 512 tokens for CP2, 1024 tokens for
CP4, and 2048 tokens for CP8.

Arxiv case 9 demonstrates the mismatch. Its 7168-token sample initially needs
only CP1, but the scheduler expands it to CP8:

```text
global_length = 7168
actual_cp     = 8
local_length  = 7168 / 8 = 896
local_half    = 896 / 2 = 448
448 % 128     = 64
```

The frontend therefore reports:

```text
sample 0 causal CP8 local half length 448 is not 128-aligned
```

Other representative failures from the same log are:

| Dataset/case | Sample length | Actual CP | Local half | Rejection |
| --- | ---: | ---: | ---: | --- |
| Arxiv 9 | 7168 | 8 | 448 | not 128-aligned |
| FreeLaw 9 | 1536 | 8 | 96 | not 128-aligned |
| FreeLaw 12 | 256 | 8 | 16 | not 128-aligned |
| FreeLaw 19 | 1792 | 4 | 224 | not 128-aligned |
| GitHub 17 | 256 | 8 | 16 | not 128-aligned |

The same mechanism accounts for the eleven skipped ProLong cases: short
256/512/768/1024-token filler samples are expanded to CP4 or CP8, producing
16/32/64/96-token local halves.

### Why dataset alignment does not guarantee compatibility

The dataset sampler aligns lengths for the BR-PBS topology that it generates:

```text
length < 2K:  256-token alignment
length < 4K:  512-token alignment
length < 8K:  1024-token alignment
length >= 8K: 2048-token alignment
```

`megatron_hybrid_cp` intentionally ignores the BR-PBS ring placement and
builds a new schedule from the same global lengths. A sample that is valid as
BR-PBS CP1 or CP4 can therefore become Megatron CP4 or CP8 after idle-rank
expansion and violate the stronger alignment required by that final CP size.
For example, BR-PBS assigns the 7168-token sample in Arxiv case 9 to G4, where
its local half is 896 and is 128-aligned. Megatron expands the same sample to
CP8, where its local half is 448 and is not aligned.

In short, the failures are caused by a contract mismatch between dataset
alignment based on the BR-PBS placement and validation based on Megatron's
independently generated final CP assignment. `--methods all` handles this by
reporting and skipping the incompatible method before timed execution.
