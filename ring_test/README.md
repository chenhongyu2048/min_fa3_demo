# Multi-rank ring attention tests and benchmarks

This directory contains torchrun entry points for multi-rank forward and
backward ring attention over the existing varlen demo layout.

The `fa3` method runs a real Python-side ring:

- each rank starts from its local `[B * S, H, D]` K/V block
- K/V are passed around the ring with `batch_isend_irecv`
- each block attention call returns `(out, lse)`
- per-step outputs are merged with the usual online LSE update

`allgather_attention` is the all-CP baseline. It gathers every rank's K/V,
reorders the gathered tensors into one batch-major global sequence, and runs
varlen FlashAttention for the local Q. The timed forward includes K/V
all-gather, reordering, and attention. The baseline uses the external FA3
varlen forward/backward implementation when it is importable on every rank;
otherwise every rank consistently falls back to the local
`min_fa3_op.forward_varlen` / `backward_varlen` implementation. The result Note
column reports the backend that was selected.

`llama3_allgather_attention` is the whole-packed all-CP baseline. Instead of
zigzag-partitioning every sequence independently, it concatenates the global
varlen batch into one packed token stream, divides that stream into `2 * W`
equal blocks, and gives rank `r` blocks `r` and `2 * W - 1 - r`. Its two local
blocks may cross sequence boundaries. The timed path includes full K/V
all-gather, restoration of global packed order, and two varlen attention calls.
It uses the same external-FA3/local-min-FA3 backend selection as
`allgather_attention`.

For causal attention, each rank keeps the same zigzag `[front | back]` layout
used by the mega-ring path. Gathered K/V are ordered as:

```text
rank0.front, rank1.front, ..., rankN.front,
rankN.back, ..., rank1.back, rank0.back
```

The two local Q halves are non-adjacent in that global sequence, so the
baseline runs two bottom-right-aligned causal varlen calls. For rank `r`, local
half length `H`, and world size `W`, the front call uses `Sk=(r+1)*H` and the
back call uses `Sk=(2*W-r)*H`. Their total KV work is identical on every rank,
which preserves zigzag load balancing. This is the causal
`allgather_attention` layout in both `benchmark_ring_forward.py` and
`benchmark_hybrid_forward.py`.

If the FA3 Python package import fails, the `fa3` method falls back to the
local `min_fa3_op.forward_varlen(..., return_lse=True)` block backend while
keeping the same Python-side ring and online LSE merge. Result tables mark this
case with `fallback: min_fa3_varlen block`.

The minimal FA3 methods follow the current local demo APIs:

- `min_varlen` is a timing-only local step loop. It neither exchanges remote
  K/V nor performs the online O/LSE merge required for a complete ring.
- `min_varlen_ring` launches the existing single-step ring kernel for multiple
  steps. The kernel does its own remote load and per-launch reduction. The timed
  path passes running `out` and `lse` buffers through each launch, so correctness
  checks use the kernel-produced reduction state directly without Python
  communication or Python reduction.
- `min_varlen_mega_ring` launches the fused mega-ring kernel once. This is the
  complete multi-step ring kernel path and is checked by default.

Example:

```bash
# Run from the min_fa3_demo repository root.
make
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_forward.py \
  --b 1 --seqlen 128 --qhead 8 --kvhead 8 --headdim 128 \
  --mode both --methods all --num-comp-sm 1 --num-comm-sm 1 \
  --warmup-iters 5 --num-iters 20
```

Use `--sm-configs comp:comm,comp:comm,...` to run multiple SM allocations in one
invocation, for example `--sm-configs 128:4,124:8,116:16`.
`--b` and `--seqlen` are comma-separated lists of equal length; each benchmark
shape pairs `b[i]` with `seqlen[i]`.
The forward and backward all-CP benchmarks run non-mega baselines only for the
first SM configuration. `min_varlen_mega_ring` is the only method swept over
the full configuration list.

For a faster smoke run:

```bash
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_forward.py \
  --b 1 --seqlen 128 --qhead 8 --kvhead 8 --mode noncausal \
  --methods fa3,min_varlen_mega_ring --num-comp-sm 1 --num-comm-sm 1 \
  --warmup-iters 1 --num-iters 3
```

To compare the per-sequence and whole-packed all-gather baselines:

```bash
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_forward.py \
  --b 3 --seqlen 256 --qhead 8 --kvhead 8 --headdim 128 \
  --mode both --methods allgather_attention,llama3_allgather_attention \
  --warmup-iters 1 --num-iters 3 --check
```

To run only the single-step min ring path with correctness checks:

```bash
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_forward.py \
  --b 1 --seqlen 128 --qhead 8 --kvhead 8 --mode both \
  --methods min_varlen_ring --num-comp-sm 1 --num-comm-sm 1 \
  --warmup-iters 1 --num-iters 3
```

## Mega-ring backward benchmark

`benchmark_ring_backward.py` benchmarks the causal varlen backward paths:

- `allgather_attention` prepares its all-gather zigzag forward outside the
  timed interval, then times both varlen backward calls, global dK/dV gradient
  merging and inverse reordering, and the FP32 dK/dV reduce-scatter
- `llama3_allgather_attention` uses the same timing boundary but partitions the
  complete packed batch into two zigzag blocks per rank; its dK/dV contributions
  are accumulated in global packed order before inverse reordering and FP32
  reduce-scatter.
- `min_varlen_python_ring` is a complete zigzag ring baseline using local
  min_fa3 varlen backward block kernels plus NCCL K/V and FP32 dK/dV P2P.
- `min_varlen_mega_ring` is the fused persistent compute/communication kernel.

The fused backward communication CTAs use TMA for remote K/V load, local K/V
store, local FP32 dK/dV load, and remote FP32 reduce-add. K/V ingress uses
128-row logical tasks split into fixed 16-row by 1024-BF16 TMA subtiles. dKV
egress decodes each level range by KV head and 128-token padded block; every dK
or dV task is one fixed `16 x 1024` FP32 TMA transaction. Tasks are spread
across all communication CTAs instead of assigning one CTA to an entire ring
step.

Forward preparation, tensor allocation, and the fused path's required remote
dK/dV accumulator and completion-counter reset are outside the CUDA-event
timing interval. Correctness checks compare fused dQ/dK/dV against the Python
ring baseline and are enabled by default. Both all-gather baselines are checked
against the same logical dQ/dK/dV reference after layout conversion.

Compare both all-gather backward baselines with correctness checking:

```bash
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_backward.py \
  --b 3 --seqlen 256 --qhead 8 --kvhead 8 --headdim 128 \
  --methods allgather_attention,llama3_allgather_attention,min_varlen_python_ring \
  --warmup-iters 1 --num-iters 3 --check
```

```bash
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_backward.py \
  --b 4,4,4 --seqlen 256,512,1024 --qhead 32 --kvhead 8 --headdim 128 \
  --methods all --sm-configs 64:8,70:8 \
  --warmup-iters 5 --num-iters 20 --check
```

Use `--no-check` for timing-only sweeps. Local sequence lengths must be
divisible by 256, and the current fused backward requires causal mode,
`D=128`, `qhead % kvhead == 0`, and `kvhead * D == 1024`.

### Hierarchical hybrid backward

`benchmark_hybrid_backward.py` compares the same causal global workload with:

- `allgather_attention`: per-sequence zigzag all-gather plus batched varlen backward
- `llama3_allgather_attention`: whole-packed two-block all-gather backward
- `fa3_ring`: NCCL zigzag K/V and FP32 dKV ring using FA3 block backward
- `zepplin`: short-sequence G1 attention plus long-sequence all-rank FA3 ring
- `mega_ring_all_cp`: fused backward with every sequence split across all ranks
- `mega_ring_hybrid`: fused G8/G4/G2/G1 hierarchical backward

The four block baselines prefer external FA3 consistently across all ranks
and fall back to this repository's min-FA3 varlen forward/backward ops when it
is unavailable. `mega_ring_all_cp` and `mega_ring_hybrid` sweep every requested
SM configuration; the four block baselines run once. Results
include the average of the per-iteration maximum end-to-end wall times and
each rank's average time, aggregate/average-per-GPU causal backward TFLOP/s,
and the fused compute/communication SM split. Forward preparation is outside
every method's timed interval. The fused owner-accumulator reset and
distributed barrier are also outside its timed interval. Zeppelin's timed
backward runs one all-rank phase barrier first, followed by the all-rank ring
backward and then the rank-local G1 backward. That internal phase barrier is
included in its result;
there is no barrier after either work phase.

`--b` and `--seqlen` are comma-separated integer lists with equal length. Each
`(b[i], seqlen[i])` pair is one case; `seqlen[i]` is the local sequence length
on a member rank. The `--ring-sizes`/`--ring-starts` pattern is repeated until
the case has `b[i]` batches, then sorted by non-increasing ring size. The global
length for each generated batch is `seqlen[i] * ring_size`.

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_hybrid_backward.py \
  --b 1,4 --seqlen 256,256 \
  --ring-sizes 8,4,2,1 --ring-starts 0,4,2,7 \
  --qhead 16 --kvhead 8 --headdim 128 \
  --methods all --zepplin-threshold 4096 \
  --num-comp-sm 100 --num-comm-sm 16 \
  --warmup-iters 5 --num-iters 20
```

The same entry point accepts one planner-generated explicit topology and an SM
configuration sweep:

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_hybrid_backward.py \
  --global-seqlens 2048,1024,512,256 \
  --ring-sizes 8,4,2,1 --ring-starts 0,4,2,7 \
  --qhead 16 --kvhead 8 --headdim 128 \
  --methods all \
  --sm-configs 128:4,124:8,120:12,116:16 \
  --warmup-iters 5 --num-iters 20 --check
```

`--check` builds subgroup-aware FP32 autograd references for the all-CP and
hybrid layouts, validates hybrid forward O/LSE preparation, and checks every
method's dQ/dK/dV. Llama3 reference gradients are repartitioned into its
whole-packed local layout before comparison. Default tolerances are
`dq_atol=1.0`, `dkv_atol=0.5`, and `rtol=0.2`.

The fused backward API receives compact local Q/O/dO and returns compact local
dQ/dK/dV. K/V remain in a rank-major
`[world_size * rank_kv_capacity, KVH, 128]` IPC arena. Each VMM-backed FP32
owner accumulator contains:

```text
KVH * round_up(rank_kv_capacity + B * 128, 128) * 128
```

The public topology tensors are CPU int32 contiguous `[B]` tensors named
`global_seqlens_host`, `ring_sizes_host`, and `ring_starts_host`. Causal half
prefix sums are generated by the C++ binding. Accumulators and the one-element
int32 completion tensor must be zeroed and globally synchronized before every
call.

Correctness and binding-validation entry points are:

```bash
torchrun --standalone --nproc_per_node=8 \
  mega_ring_test_min_fa3_varlen_backward_hybrid_multi_rank.py \
  --global-seqlens 2048,1024,512,256 \
  --ring-sizes 8,4,2,1 --ring-starts 0,4,2,7 \
  --qhead 16 --kvhead 8 --repeat 2 \
  --num-comp-sm 100 --num-comm-sm 16

torchrun --standalone --nproc_per_node=8 \
  mega_ring_test_min_fa3_varlen_backward_validation_multi_rank.py
```

The full scheduler/readiness/completion contract is recorded in
`../docs/HIERARCHICAL_HYBRID_MEGA_RING_BACKWARD_DESIGN.md`.

### Dataset-shaped hybrid backward

`benchmark_hybrid_dataset_backward.py` uses the same `balancer` workload as the
forward dataset frontend. It samples Arxiv, Github, or Pile-CC lengths, packs
the target token budget, prints the same placement/cap/load report, and calls
`benchmark_hybrid_backward.main(...)` in the same process with explicit ring
metadata. Backward is causal-only and benchmarks the all-CP and hierarchical
fused kernels together with the per-sequence all-gather, Llama3 all-gather,
FA3/NCCL ring, and Zeppelin baselines. The dataset planner's hierarchical ring
metadata is forwarded unchanged; Zeppelin independently rebuilds its G1/Gworld
placement from the global sequence lengths.

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_hybrid_dataset_backward.py \
  --dataset arxiv --target-tokens 131072 --seed 0 --num-cases 4 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --methods all --zepplin-threshold 4096 \
  --sm-configs 128:4,124:8,120:12,116:16 \
  --warmup-iters 10 --num-iters 40 --no-check

DATASETS="arxiv github pile" GPU_COUNTS="2 4 8" NUM_CASES=4 DIRECTION=backward \
  ZEPPLIN_THRESHOLD=4096 ./benchmark_hybrid_dataset.sh
```

Planner-only inspection does not import or initialize CUDA:

```bash
python ring_test/benchmark_hybrid_dataset_backward.py \
  --dataset github --target-tokens 131072 \
  --world-size 8 --print-workload
```

Use `--check` only for small token budgets because the dense backward reference
has quadratic score memory. The shell wrapper shares the forward wrapper's
`DATASETS`, `GPU_COUNTS`, balancing controls, `SM_CONFIGS`, logging, `CHECK`,
and `DRY_RUN` environment variables. `ZEPPLIN_THRESHOLD` controls the shared
forward/backward threshold and defaults to `4096`.

## Hybrid mega-ring benchmark

`benchmark_hybrid_forward.py` compares the same global varlen batch with six
methods:

- `allgather_attention`: all-CP K/V all-gather followed by batched varlen attention
- `llama3_allgather_attention`: all-CP K/V all-gather with whole-packed zigzag partitioning
- `fa3_ring`: all-CP Python ring using FA3 blocks plus NCCL P2P
- `zepplin`: LPT-placed rank-local attention for short sequences and an all-rank ring for long sequences
- `mega_ring_all_cp`: fused mega-ring with every sequence split across all ranks
- `mega_ring_hybrid`: fused mega-ring using the requested per-batch ring hierarchy

The hybrid benchmark entry points are self-contained within `ring_test/` plus
the built demo extension. Their shared workload and reference helpers live in
`ring_test/utils.py`; running them does not require the scripts under
`scripts/test_mega_ring/` or an additional `PYTHONPATH`.

The first three methods are all-CP baselines. Every global sequence is divided
evenly over all physical ranks. `mega_ring_hybrid` instead uses `--ring-sizes` and
`--ring-starts`; rank-local length is `global_len / ring_size` for members of
that batch's ring and zero for other ranks. External FA3 is used by all three
all-CP block baselines and Zeppelin when available; all four consistently fall
back to the local min-FA3 varlen block when it is unavailable.

Zeppelin uses `--zepplin-threshold` (default `4096`) independently of the
hierarchical metadata. A sequence with `global_length < threshold` is placed
whole on one rank (G1); equality belongs to the long side, so length `4096`
uses Gworld at the default threshold. Short sequences are assigned by
deterministic longest-processing-time-first placement. Causal weight is
`L * (L + 1) / 2`, noncausal weight is `L * L`; equal weights retain original
batch-index order, and equal rank loads choose the smaller rank id. Long
sequences add the same distributed load to every rank and do not affect the
G1 choice.

Each rank packs its owned short sequences first and every long sequence's
local shard second; each part retains original batch order. Forward timing
includes one all-rank phase barrier, the long `fa3_ring`, and then local varlen
attention. There is no barrier after the ring or local phase. Ranks with no
short work and workloads with no long work still participate in the initial
barrier, while empty kernel launches are skipped.

Forward selection requires the external FA3 varlen forward entry point on
every rank. Backward selection requires both its forward and backward entry
points; otherwise all four block baselines use the current repository's matching
min-FA3 operators for the entire run.

For an `--sm-configs` list, the four block baselines run once using the first
configuration. `mega_ring_all_cp` and `mega_ring_hybrid` run once for every
configuration in both hybrid forward and backward.

The three block-based all-CP baseline lengths must be divisible by the physical
world size. The total global token count for `llama3_allgather_attention` must
also be divisible by `2 * world_size`. `mega_ring_all_cp` is benchmarked on a
separate workload where every global sequence is rounded upward to a multiple
of 2048, ensuring that each causal rank-local half is 128-aligned on eight GPUs.
Its table TFLOPS use the original unaligned lengths; its Note reports aggregate
and average-per-GPU TFLOPS using the physically executed aligned lengths and
the same measured latency. Use `--methods` to select a subset or
`--methods all` for all six. Zeppelin only requires sequences at or above its
threshold to be divisible by `world_size`; causal long shards must also be even
for the zigzag ring. Short G1 sequences have no all-CP divisibility constraint,
and the threshold must be a positive integer.

Example:

```bash
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_hybrid_forward.py \
  --global-seqlens 8192,1024,1024 \
  --ring-sizes 2,1,1 --ring-starts 0,0,1 \
  --qhead 16 --kvhead 8 --headdim 128 --methods all \
  --zepplin-threshold 4096 \
  --sm-configs 128:4,116:16 --mode both \
  --warmup-iters 5 --num-iters 20
```

### Dataset-shaped hybrid workload

`benchmark_hybrid_dataset_forward.py` uses the standalone `balancer` package
to sample an Arxiv, Github, or Pile-CC length distribution and pack global
batches to 128K tokens by default. `--seed` initializes one RNG stream;
`--num-cases` consumes that stream continuously to build multiple complete
batches before assigning each sequence to a G8/G4/G2/G1 subgroup.
Compute and local tokens are hard placement constraints; normalized token
balance is the main objective, while compute variance and estimated ring
token-hops are soft costs. The frontend then calls
`benchmark_hybrid_forward.main(...)` in the same process with the generated
ring metadata.

All generated cases run under one process group. The fused mega-ring methods
allocate K/V IPC arenas once using the largest per-rank capacity across the
case set and refill them between cases. Backward also reuses its remote dK/dV
accumulators and completion tensors while retaining a case-local logical dKV
stride. Each case keeps the existing result table, followed by cross-case
latency, arithmetic-mean TFLOPS, workload-weighted aggregate TFLOPS, and
workload-weighted per-GPU TFLOPS summaries.

The three empirical distributions live in
`../dataset/sequence_length_buckets.json`. Each one contains 512 counts for
256-token `(lower, upper]` buckets up to 128K. Samples longer than 128K are
merged into the final bucket, and a sampled bucket contributes its upper bound
before the existing ring-aware alignment is applied. Regenerate the file from
the sampled `*_doc_lengths.npy` arrays with
`python dataset/build_length_bucket_stats.py` from the demo root.

Every sampled length, including the final packing residual, is padded upward.
Lengths below 4K use `256 * 2` alignment, lengths from 4K to 8K use
`256 * 4`, and lengths from 8K upward use `256 * 8` (including lengths at or
above 16K). This makes each causal sequence eligible for at least its target
G2/G4/G8 ring. The physical padded tokens participate in the benchmark, so
`actual_tokens` can exceed `target_tokens` by fewer than 2048 tokens.

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_hybrid_dataset_forward.py \
  --dataset arxiv --target-tokens 131072 --seed 0 --num-cases 4 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --mode causal --methods all --zepplin-threshold 4096 --no-check

DATASETS="arxiv github pile" GPU_COUNTS="2 4 8" NUM_CASES=4 ZEPPLIN_THRESHOLD=4096 \
  ./benchmark_hybrid_dataset.sh
```

The requested compute and token tolerances default to 5% and 10%. The planner
searches compute caps up to 20% and token caps up to 50%, stopping at the first
token cap that has a feasible placement. The relevant controls are:

```text
--balance-tolerance 0.05
--token-balance-tolerance 0.10
--max-compute-balance-tolerance 0.20
--max-token-balance-tolerance 0.50
--communication-weight 0.05
--local-search-passes 4
```

The shell wrapper exposes the same settings as `BALANCE_TOLERANCE`,
`TOKEN_BALANCE_TOLERANCE`, `MAX_COMPUTE_BALANCE_TOLERANCE`,
`MAX_TOKEN_BALANCE_TOLERANCE`, `COMMUNICATION_WEIGHT`, and
`LOCAL_SEARCH_PASSES`. `ZEPPLIN_THRESHOLD` defaults to `4096` and is forwarded
to both dataset directions. `--print-workload` reports the requested, topology, and
final caps; estimated per-rank communication; cap-relaxation flags; emergency
fallback; and the number of accepted local-repair moves. A topology-limited
workload can exceed the configured imbalance tolerance when a sequence cannot
use a larger legal ring. Emergency fallback is used only when no placement is
feasible within either configured maximum cap.

`--methods` defaults to `all`. Dataset mode runs every compatible method for
each causal/noncausal mode and prints a skip reason for an incompatible method.
The `mega_ring_all_cp` baseline is not skipped for an unaligned generated
length because it uses its separate 2048-aligned workload in both directions.
Explicitly requesting another incompatible method remains an error. Use
`--print-workload --world-size 8` to inspect generated lengths and ring
assignments without CUDA. Keep `--no-check` for the full 128K workload because
the current correctness reference materializes quadratic attention scores.

## 1/2/4/8-GPU causal sweep

`benchmark_ring_1_2_4_8.sh` runs the all-CP forward and backward varlen
benchmarks on 1, 2, 4, and 8 GPUs. Hierarchical hybrid forward runs on 2, 4,
and 8 GPUs; the script explicitly skips it for world size 1. It fixes QH/KVH
at 32/8, head dim at 128, warmup iterations at 10, and measured iterations at
40. Run it inside a single-node allocation exposing eight SM90 GPUs:

```bash
./benchmark_ring_1_2_4_8.sh
```

The all-CP forward and backward defaults pair batch sizes `16,8,4,2,1` with
local sequence lengths `1K,2K,4K,8K,16K`. This keeps each setting at 16K local
tokens per rank. The corresponding global token totals are 16K, 32K, 64K, and
128K for 1, 2, 4, and 8 GPUs. The hybrid workload keeps one fixed global batch
and balances its local-only sequences across the supported 2/4/8-GPU hybrid
runs. All runs are appended to one timestamped
`benchmark_logs/<timestamp>/benchmark_ring_1_2_4_8.log` file, with a separator,
run label, GPU count, and full command before each result section.

Environment variables can override the workload without editing the script.
For example:

```bash
GPU_COUNTS="2 4 8" \
ALL_CP_GLOBAL_SEQLENS_2="2048,4096,8192,16384,32768" \
ALL_CP_GLOBAL_SEQLENS_4="4096,8192,16384,32768,65536" \
ALL_CP_GLOBAL_SEQLENS_8="8192,16384,32768,65536,131072" \
BACKWARD_GLOBAL_SEQLENS_2="2048,4096,8192,16384,32768" \
BACKWARD_GLOBAL_SEQLENS_4="4096,8192,16384,32768,65536" \
BACKWARD_GLOBAL_SEQLENS_8="8192,16384,32768,65536,131072" \
LOG_DIR=benchmark_logs/selected \
./benchmark_ring_1_2_4_8.sh
```

Set `DRY_RUN=1` to print the complete commands without launching `torchrun`,
or `CHECK=1` to enable the benchmark entry points' correctness checks.

## Result columns

`benchmark_ring_forward.py` prints one row per selected method. `Time ms`
contains each rank's average measured time and the average of the
per-iteration maximum times across ranks. The latter is used to calculate
`Agg TFLOPS`, which sums the visible attention work across ranks; `Avg/GPU`
divides that value by the physical world size. `Check` records reference
validation or marks a timing-only method; `Note` records the selected block
backend and other method details. The exact method rows follow `--methods` and
the current CLI method list rather than a hard-coded result snapshot.
