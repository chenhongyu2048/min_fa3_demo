# Multi-rank ring attention tests and benchmarks

This directory contains torchrun entry points for multi-rank forward and
backward ring attention over the existing varlen demo layout.

The `fa3` method runs a real Python-side ring:

- each rank starts from its local `[B * S, H, D]` K/V block
- K/V are passed around the ring with `batch_isend_irecv`
- each block attention call returns `(out, lse)`
- per-step outputs are merged with the usual online LSE update

`allgather_attention` is the per-sequence all-CP baseline. It preserves the
existing per-sequence zigzag ordering, but gathers K/V one contiguous KV-head
slice at a time. Its two receive buffers ping-pong: after slice `i` is ready,
the all-gather for slice `i + 1` starts before attention runs for slice `i`.
The timed forward includes K/V all-gather, reordering, and attention. The
baseline uses the external FA3 varlen forward/backward implementation when it
is importable on every rank; otherwise every rank consistently falls back to
the local `min_fa3_op.forward_varlen` / `backward_varlen` implementation. The
result Note column reports the backend that was selected.

`megatron_hybrid_cp` is the standalone Megatron-scheduled hybrid baseline in
`../baseline/megatron_hybrid_cp/`. It derives CP1/2/4/8 assignments from the
input global lengths rather than consuming the benchmark's Buddy-ring
placement. Its scheduler is copied and trimmed from Megatron-LM commit
`368fa88e382b274c8fc12af851331cc1d30d69cc`; there is no runtime Megatron-LM or
Transformer Engine import. CP1 uses local FA3, while CP>1 uses the existing
ordinary/zigzag P2P FA3 rings. External FA3 is selected only when every rank
can import it, otherwise all ranks use the local min-FA3 fallback.

The method executes every execution group and per-rank sample list in forward
order, with exactly one world barrier between groups. Backward is causal-only
and intentionally uses the same forward order after a complete retained
forward phase. Forward timing includes all group compute, P2P, and internal
barriers. Backward preparation runs the complete forward outside timing, then
the timed interval covers the complete backward phase. Use
`--megatron-max-seqlen-per-rank` (default 8192) to control the initial CP-size
threshold. `--methods all` skips workloads whose required power-of-two CP size
exceeds the world or whose actual CP shard violates divisibility/causal
alignment; explicitly requesting the method reports the incompatibility as an
error.

`llama3_allgather_attention` is the whole-packed all-CP baseline. Instead of
zigzag-partitioning every sequence independently, it concatenates the global
varlen batch into one packed token stream, divides that stream into `2 * W`
equal blocks, and gives rank `r` blocks `r` and `2 * W - 1 - r`. Its two local
blocks may cross sequence boundaries. Rather than materializing one full K/V
all-gather, it processes a contiguous KV-head slice at a time. Two receive
buffers ping-pong: after slice `i` is gathered and restored to global packed
order, the all-gather for slice `i + 1` is launched before the two varlen
attention calls for slice `i`. This overlaps the next K/V all-gather with the
current attention work while preserving the GQA mapping from each KV slice to
its corresponding Q-head range. It uses the same external-FA3/local-min-FA3
backend selection as `allgather_attention`.

`--allgather-overlapping-heads-k-stride` controls the number of KV heads in
one pipeline slice for both `allgather_attention` and
`llama3_allgather_attention`. It must be a positive divisor of `--kvhead`.
Smaller values create more communication/computation overlap but launch more
all-gathers and attention calls; larger values reduce launch overhead. The
same head slicing is used by both all-gather backward paths before each FP32
dK/dV reduce-scatter. The ordinary ring and retained UltraAttn entry points
default to `1`; the explicit and dataset-shaped hybrid entry points default to
`4`. The checked-in shell and Slurm scripts pass the option explicitly through
`ALLGATHER_OVERLAPPING_HEADS_K_STRIDE`.

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
`benchmark_topology_forward.py`.

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
  --allgather-overlapping-heads-k-stride 1 \
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

- `allgather_attention` prepares its zigzag forward outside the timed interval,
  then pipelines KV-head-sliced all-gathers with the two varlen backward calls,
  global dK/dV gradient merging, inverse reordering, and FP32 reduce-scatter
- `llama3_allgather_attention` uses the same timing boundary but partitions the
  complete packed batch into two zigzag blocks per rank. It pipelines
  KV-head-sliced all-gathers with the block backward calls; each slice's dK/dV
  contributions are accumulated in global packed order before inverse
  reordering and FP32 reduce-scatter.
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

### Explicit-topology backward benchmark

`benchmark_topology_backward.py` compares the same causal global workload with:

- `allgather_attention`: overlapped KV-head-sliced per-sequence zigzag all-gather backward
- `llama3_allgather_attention`: overlapped KV-head-sliced whole-packed two-block all-gather backward
- `fa3_ring`: NCCL zigzag K/V and FP32 dKV ring using FA3 block backward
- `megatron_hybrid_cp`: Megatron length schedule with CP1/2/4/8 FA3 P2P phases
- `magi_attention`: full-WORLD MagiAttention dynamic packing/dispatch baseline
- `zepplin`: short-sequence G1 attention plus long-sequence all-rank FA3 ring
- `mega_ring_all_cp`: fused backward with every sequence split across all ranks
- `mega_ring_hybrid`: fused G8/G4/G2/G1 hierarchical backward

The five block baselines prefer external FA3 consistently across all ranks
and fall back to this repository's min-FA3 varlen forward/backward ops when it
is unavailable. `mega_ring_all_cp` and `mega_ring_hybrid` sweep every requested
SM configuration; the five block baselines run once. Results
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
  ring_test/benchmark_topology_backward.py \
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
  ring_test/benchmark_topology_backward.py \
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

### Dataset-shaped topology backward

`benchmark_dataset_backward.py` uses the same `balancer` workload as the
forward dataset frontend. It samples Arxiv, Github, or Pile-CC lengths, packs
the target token budget, prints the same placement/cap/load report, and calls
`benchmark_topology_backward.main(...)` in the same process with explicit ring
metadata. Backward is causal-only and benchmarks the all-CP and hierarchical
fused kernels together with the per-sequence all-gather, Llama3 all-gather,
FA3/NCCL ring, MagiAttention, and Zeppelin baselines. The dataset planner's
hierarchical ring metadata is forwarded unchanged; MagiAttention ignores that
metadata and dynamically dispatches the same global lengths over WORLD, while
Zeppelin independently rebuilds its G1/Gworld placement.

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_dataset_backward.py \
  --dataset arxiv --target-tokens 131072 --seed 0 --num-cases 4 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --methods all --zepplin-threshold 4096 \
  --sm-configs 128:4,124:8,120:12,116:16 \
  --warmup-iters 10 --num-iters 40 --no-check

DATASETS="arxiv github pile freelaw prolong" GPU_COUNTS="2 4 8" NUM_CASES=4 DIRECTION=backward \
  ZEPPLIN_THRESHOLD=4096 ./benchmark_dataset.sh
```

Planner-only inspection does not import or initialize CUDA:

```bash
python ring_test/benchmark_dataset_backward.py \
  --dataset github --target-tokens 131072 \
  --world-size 8 --print-workload
```

Use `--check` only for small token budgets because the dense backward reference
has quadratic score memory. The shell wrapper shares the forward wrapper's
`DATASETS`, `GPU_COUNTS`, balancing controls, `SM_CONFIGS`, logging, `CHECK`,
and `DRY_RUN` environment variables. `ZEPPLIN_THRESHOLD` controls the shared
forward/backward threshold and defaults to `4096`.

## Forward/backward load-balance metadata benchmark

`benchmark_load_balance.py` is the unified static analysis entry point for all
eight baselines in `benchmark_topology_forward.py` and
`benchmark_topology_backward.py`:

- `allgather_attention`
- `llama3_allgather_attention`
- `fa3_ring`
- `megatron_hybrid_cp`
- `magi_attention`
- `zepplin`
- `mega_ring_all_cp`
- `mega_ring_hybrid`

`--direction` accepts `forward` or `backward` and defaults to `forward`.
Backward accepts only `--mode causal`; `noncausal` and `both` are errors. Both
directions are BF16, D=128, and world size 2, 4, or 8. The old
`benchmark_load_balance_forward.py` and `../benchmark_load_balance_forward.sh`
names were removed rather than retained as aliases.

The tool does not run attention kernels, measure latency, perform warmup, sweep
SM allocations, check numerical output, or construct an autograd graph. Regular
adapters consume the latency runners' placement and task boundaries. The Magi
adapter calls the same metadata key builder and reads `DispatchMeta`, `CalcMeta`,
backward `AttnArg`, collective splits, precision flags, backend, and tail-stage
policy. It never calls `dispatch`, `calc_attn`, or `backward()`.

Explicit topology:

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_load_balance.py \
  --global-seqlens 8192,4096,2048 \
  --ring-sizes 8,4,2 --ring-starts 0,0,4 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --mode causal --methods all
```

Dataset workload with the existing BR-PBS controls:

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_load_balance.py --direction backward \
  --dataset arxiv --target-tokens 131072 --seed 0 --num-cases 4 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --mode causal --methods all
```

Run a dataset/GPU matrix with the same environment-variable style as
`benchmark_dataset.sh`:

```bash
DIRECTION=backward GPU_COUNTS="2 4 8" \
  DATASETS="arxiv freelaw github pile prolong" \
  NUM_CASES=4 ./benchmark_load_balance.sh

DRY_RUN=1 DIRECTION=backward GPU_COUNTS="2 4 8" DATASETS=arxiv \
  ./benchmark_load_balance.sh
```

`DIRECTION`, `TARGET_TOKENS`, both balance tolerances, `BEAM_WIDTH`,
`FINALIST_COUNT`, `STRUCTURE_THRESHOLD`, `MAX_REPAIR_ITERATIONS`, `SEED`,
`NUM_CASES`, `MODE`, `METHODS`, head settings, baseline settings,
`CUDA_VISIBLE_DEVICES`, `TORCHRUN`, `LOG_DIR`, and `LOG_FILE` can be overridden.
The default log is
`benchmark_logs/<timestamp>/benchmark_load_balance_<direction>.log`. The
wrapper does not expose explicit topology, timing, SM-sweep, or correctness
settings; explicit topology remains a direct Python invocation.

The non-Magi methods also support CPU-only static analysis:

```bash
python ring_test/benchmark_load_balance.py --direction backward \
  --global-seqlens 2048,1024 --ring-sizes 2,1 --ring-starts 0,1 \
  --world-size 2 --qhead 32 --kvhead 8 --headdim 128 \
  --mode causal --methods allgather_attention,fa3_ring,mega_ring_hybrid
```

Selecting `magi_attention` explicitly requires `torchrun`, an SM90 CUDA device,
and both Magi extensions. Under ordinary Python, `--methods all` reports a Magi
skip reason and continues with the CPU adapters. As in the latency frontends,
`all` prints a reason for any method that cannot represent a case, while an
explicit incompatible selection is an error.

Each method prints load, efficiency, and min/avg/max/max-to-avg summary tables.
Dataset runs also sum every rank field over cases and recompute ratios from the
cumulative counters. Shared fields use these definitions:

- Effective tokens/scores describe the original visible full or causal
  workload. `mega_ring_all_cp` uses the original world-normalized share here.
- Physical tokens/scores describe the baseline's executed workload, including
  the all-CP mega-ring's 2048-token alignment and Magi padding/slices.
- Forward FLOPs are `4 * score_count * QH * D` for QK and PV only. Backward
  FLOPs are `10 * score_count * QH * D`, matching the backward latency
  benchmark. Neither adds mask, softmax, scheduler, atomic, or address FLOPs.
- `Sent` counts logical payload inside the corresponding latency benchmark's
  measured forward/backward boundary. Communication load and `Send/avg` use
  only each rank's sent bytes for every method. Received bytes are retained
  internally only to validate `sum(Sent) == sum(Received)` and are not reported
  as load. Consequently the global transmitted payload is `sum(Sent)`. Fabric
  routing, NCCL protocols, barriers, atomics, semaphores, and completion
  counters are excluded.
- BF16 Q/K/V/O/dO elements use 2 bytes. FP32 LSE and high-precision gradient
  payloads use 4 bytes.
- Q and KV tiles are algorithm-level 128-token tiles for every backend. A Q/O
  visit is one attention task or segment reading a Q tile and writing its
  partial/final O tile. A KV tile read means that tile intersects the task's
  full, causal, inverse-causal, or bi-causal mask.
- `KV tiles / QO visit` is recomputed from aggregate counters. Larger values
  mean more KV work per Q read/O write visit; it is not a hardware counter.

For causal fused mega-ring, `[lower,upper]` captures dynamic segment formation.
The numerator is fixed. The lower bound uses the worst case where every remote
step is a separate Q/O segment; the upper bound uses the best case where all
consecutive ready steps allowed by the scheduler merge. G1 and non-dynamic
paths have equal bounds. Dataset multi-case output prints every case and then
sums rank fields across cases; its ratio is recomputed from cumulative tile
counters rather than averaging per-case ratios.

Backward uses a separate mirrored efficiency metric. A `Q tile read` is one
logical 128-token Q tile intersecting the causal mask for the current K/dKV
task. A `K/dKV visit` is one task visiting a logical 128-token K tile for K/V
read and dK/dV update. Both counters are expanded by Q heads, matching backward
scheduler work. `Q tiles / K-dKV` is their quotient. Backward has no
multi-step segment fusion, so every method reports one value rather than a
lower/upper interval.

Backward communication follows each runner's actual boundary:

- All-gather and Llama3 repeat the BF16 K/V all-gather and perform FP32 dK/dV
  reduce-scatter. Setup repartition and untimed forward preparation are
  excluded.
- FA3 ring, Megatron CP2/4/8, and Zeppelin Gworld use `p-1` BF16 K/V ring
  steps and `p` FP32 dK/dV owner-return steps. Megatron CP1 and Zeppelin's LPT
  G1 sequences are local with zero payload. Inter-group and phase barriers are
  excluded.
- Fused mega-ring fetches remote BF16 K/V with its causal half/full row rule.
  Every remote step store-adds FP32 dK/dV directly to its final owner using the
  real accumulator layout, including one extra 128-row gap for every batch in
  that level. As for every other method, communication load counts each
  transfer once at its sending endpoint. Step 0's local owner store, resets,
  counters, semaphores, and barriers are excluded. `mega_ring_all_cp`
  additionally uses 2048-token aligned physical work while retaining the
  original world-normalized effective share.
- Magi reads physical tasks from `CalcMeta` and backward `AttnArg`. Backward KV
  fetch, dKV reduction, and optional Q/O/dO/LSE fetch plus dQ reduction are
  included. Q/K/V/O/dO use BF16, LSE uses FP32, and dQ/dK/dV use the runtime's
  native/high-precision/backend dtype branch. A forward-cached tail stage is
  not fetched again in backward, but its gradient reduction remains included.
  Input dispatch and untimed forward graph preparation are excluded.

## Explicit-topology mega-ring benchmark

`benchmark_topology_forward.py` compares the same global varlen batch with eight
methods:

- `allgather_attention`: KV-head-sliced, overlapped all-CP K/V all-gather with per-sequence zigzag partitioning
- `llama3_allgather_attention`: KV-head-sliced, overlapped all-CP K/V all-gather with whole-packed zigzag partitioning
- `fa3_ring`: all-CP Python ring using FA3 blocks plus NCCL P2P
- `megatron_hybrid_cp`: independently scheduled Megatron CP1/2/4/8 groups
- `magi_attention`: full-WORLD dynamic packing/dispatch through MagiAttention
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

`magi_attention` is a performance-only baseline using the same global lengths
but not the explicit `ring_sizes`/`ring_starts` Hybrid-CP placement. It lets
MagiAttention pad, pack, and dynamically dispatch tokens over the full WORLD
group. It runs once per case rather than participating in the mega-ring SM
sweep; `--magi-overlap-degree` defaults to 2 and accepts 1 through 8. Forward
timing includes only `calc_attn`; backward preparation builds a fresh forward
graph outside timing and the measured callable is only `out.backward(dout)`.
TFLOPS count the original effective full/causal mask area, not padding work.
MagiAttention always reports `Check=skip`. Installation, availability behavior,
automatic chunking, padding notes, and CUDA 12.8 caveats are documented in
[`../baseline/magi_attention/README.md`](../baseline/magi_attention/README.md).

### UltraAttn fixed-8K graph baseline

`ultraattn` has been removed in main branch, but keeped in the `ultraattn_baseline` branch.

`ultraattn` is restricted to eight GPUs, BF16 causal forward,
`QH/KVH/D=32/8/128`, `block_tokens=8192`, and the five fixed 128K-token
workloads. It does not use the benchmark's Buddy-ring placement or all-CP
placement. The portable `.npz` contains UltraAttn's QxK allocation and default
contiguous `cmap`.

At runtime the allocation is compiled into input-Q, input-KV, fused compute,
partial-return, and owner-merge graph nodes. Input collectives are launched
asynchronously; local-only nodes run first, remote-Q nodes run when Q arrives,
and the remaining nodes run when K/V arrives. Attention computation uses
`min_fa3_op.forward_varlen`, and partial O/LSE is returned and merged in FP32.
There is no staged or 256-token packing fallback.

The timed callable includes graph input packing/communication, min-FA3 task
launches, distributed partial return, and final O/LSE merge. Plan loading,
graph compilation, buffer allocation, and outer benchmark synchronization are
outside it. The normal `.venv` does not import Gurobi, external FlashAttention,
or UltraAttn PyNCCL.

UltraAttn uses its dedicated copy of the explicit-topology frontend at
`ring_test/ultraattn/benchmark_hybrid_forward.py`. The root-level
`ring_test/benchmark_topology_forward.py` remains the original six-method
benchmark and does not import the UltraAttn runtime.

The fixed suite uses `1xG8`, `2xG4`, `4xG2`, `8xG1`, and two G1 sequences per
rank for Mega Ring Hybrid; every rank owns 16K tokens. UltraAttn ignores these
ring topologies:

```bash
PLANNER_PY=/path/to/planner-venv/bin/python \
QHEAD=32 KVHEAD=8 HEADDIM=128 BLOCK_TOKENS=8192 TIME_LIMIT=1800 \
baseline/UltraAttn/packing/generate_fixed_128k_plans.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc_per_node=8 \
  ring_test/ultraattn/benchmark_hybrid_fixed_forward.py \
  --qhead 32 --kvhead 8 --headdim 128 \
  --methods ultraattn,mega_ring_hybrid \
  --ultraattn-plan-dir baseline/UltraAttn/packing_plans \
  --ultraattn-block-tokens 8192 \
  --sm-configs 128:4 --warmup-iters 10 --num-iters 40 --no-check
```

Explicit selection fails for any non-fixed workload or missing plan. Selection
through `--methods all` reports the incompatibility and skips UltraAttn. See
`baseline/UltraAttnREADME.md` for planner installation, correctness commands,
solver status, and formal five-case results.

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
`--methods all` for all eight. Zeppelin only requires sequences at or above its
threshold to be divisible by `world_size`; causal long shards must also be even
for the zigzag ring. Short G1 sequences have no all-CP divisibility constraint,
and the threshold must be a positive integer.

Example:

```bash
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_topology_forward.py \
  --global-seqlens 8192,1024,1024 \
  --ring-sizes 2,1,1 --ring-starts 0,0,1 \
  --qhead 16 --kvhead 8 --headdim 128 --methods all \
  --zepplin-threshold 4096 \
  --sm-configs 128:4,116:16 --mode both \
  --warmup-iters 5 --num-iters 20
```

### Dataset-shaped topology workload

`benchmark_dataset_forward.py` uses the standalone `balancer` package
to sample an Arxiv, Github, or Pile-CC length distribution and pack global
batches to 128K tokens by default. `--seed` initializes one RNG stream;
`--num-cases` consumes that stream continuously to build multiple complete
batches before assigning each sequence to a G8/G4/G2/G1 subgroup. The BR-PBS
planner separately constrains maximum absolute token and attention-compute
deviation. Within the feasible set it lexicographically protects short
sequences from splitting before reducing communication, active groups, and
residual imbalance. The frontend then calls
`benchmark_topology_forward.main(...)` in the same process with the generated
ring metadata.

All generated cases run under one process group. The fused mega-ring methods
allocate K/V IPC arenas once using the largest per-rank capacity across the
case set and refill them between cases. Backward also reuses its remote dK/dV
accumulators and completion tensors while retaining a case-local logical dKV
stride. Each case keeps the existing result table, followed by cross-case
latency, arithmetic-mean TFLOPS, workload-weighted aggregate TFLOPS, and
workload-weighted per-GPU TFLOPS summaries.

The five empirical distributions live in
`../dataset/sequence_length_buckets.json`. Each one contains 512 counts for
256-token `(lower, upper]` buckets up to 128K. Samples longer than 128K are
merged into the final bucket, and a sampled bucket contributes its upper bound
before the existing ring-aware alignment is applied. Regenerate the file from
the sampled `*_doc_lengths.npy` arrays with
`python dataset/build_length_bucket_stats.py` from the demo root.

Every sampled length, including the final packing residual, is padded upward.
Lengths below 2K use `256` alignment, lengths from 2K to 4K use `256 * 2`,
lengths from 4K to 8K use `256 * 4`, and lengths from 8K upward use `256 * 8`.
The physical padded tokens participate in the benchmark, so `actual_tokens`
can exceed `target_tokens` by fewer than 2048 tokens.

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_dataset_forward.py \
  --dataset arxiv --target-tokens 131072 --seed 0 --num-cases 4 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --mode causal --methods all --zepplin-threshold 4096 --no-check

DATASETS="arxiv github pile freelaw prolong" GPU_COUNTS="2 4 8" NUM_CASES=4 ZEPPLIN_THRESHOLD=4096 \
  ./benchmark_dataset.sh
```

The requested compute and token tolerances default to 5% and 10%. BR-PBS first
places sequences whose normalized token or compute size is at least the
structure threshold, or whose minimum useful ring is larger than one. It keeps
a quantized Pareto beam, greedily fills the remaining singleton sequences, and
repairs the best complete candidates with local Buddy-tree moves. When needed,
the five length buckets `<=2K`, `2K-4K`, `4K-8K`, `8K-16K`, and `>16K` are
unlocked from longest to shortest, with G2 before G4 before G8. The relevant
controls are:

```text
--compute-balance-tolerance 0.05
--token-balance-tolerance 0.10
--beam-width 64
--finalist-count 8
--structure-threshold 0.5
--max-repair-iterations 32
```

The shell wrapper exposes the same settings as `COMPUTE_BALANCE_TOLERANCE`,
`TOKEN_BALANCE_TOLERANCE`, `BEAM_WIDTH`, `FINALIST_COUNT`,
`STRUCTURE_THRESHOLD`, and `MAX_REPAIR_ITERATIONS`. The load quantization step
is fixed at 2% of the corresponding average and the residual-fill smooth-max
lambda is fixed at 8. `ZEPPLIN_THRESHOLD` defaults to `4096`, and
`MAGI_OVERLAP_DEGREE` defaults to `2`; both are forwarded to both dataset
directions. `--print-workload` reports absolute deviations,
feasibility and violation, the relaxation level, split counts and penalties,
the communication proxy, active groups, and accepted repair moves. If no legal
placement satisfies both tolerances, the planner returns the lowest-violation
plan and reports `feasible=False`.

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
