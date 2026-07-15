# Multi-rank ring attention forward test

This directory contains a torchrun entry point for forward-only multi-rank
ring attention over the existing varlen demo layout.

The standard PyTorch / FA2 / FA3 methods run a real Python-side ring:

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
which preserves zigzag load balancing. This path is all-CP only and is not
used by `benchmark_hybrid_forward.py`.

If the FA3 Python package import fails, the `fa3` method falls back to the
local `min_fa3_op.forward_varlen(..., return_lse=True)` block backend while
keeping the same Python-side ring and online LSE merge. Result tables mark this
case with `fallback: min_fa3_varlen block`. FA2 remains optional and is skipped
when unavailable.

The minimal FA3 methods follow the current local demo APIs:

- `min_varlen` is a timing-only local step loop. The current binding returns
  only `out`, so it does not expose enough state for a strict Python-side
  cross-step ring merge.
- `min_varlen_ring` launches the existing single-step ring kernel for multiple
  steps. The kernel does its own remote load and per-launch reduction. The timed
  path passes running `out` and `lse` buffers through each launch, so correctness
  checks use the kernel-produced reduction state directly without Python
  communication or Python reduction.
- `min_varlen_mega_ring` launches the fused mega-ring kernel once. This is the
  complete multi-step ring kernel path and is checked by default.

Example:

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo
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

For a faster smoke run:

```bash
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_forward.py \
  --b 1 --seqlen 128 --qhead 8 --kvhead 8 --mode noncausal \
  --methods pytorch,min_varlen_mega_ring --num-comp-sm 1 --num-comm-sm 1 \
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
store, local FP32 dK/dV load, and remote FP32 reduce-add. Row tasks are spread
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

## Hybrid mega-ring benchmark

`benchmark_hybrid_forward.py` compares the same global varlen batch with five
methods:

- `allgather_attention`: all-CP K/V all-gather followed by batched varlen attention
- `llama3_allgather_attention`: all-CP K/V all-gather with whole-packed zigzag partitioning
- `fa3_ring`: all-CP Python ring using FA3 blocks plus NCCL P2P
- `mega_ring_all_cp`: fused mega-ring with every sequence split across all ranks
- `mega_ring_hybrid`: fused mega-ring using the requested per-batch ring hierarchy

The first three methods are baselines. Every global sequence is divided evenly
over all physical ranks. `mega_ring_hybrid` instead uses `--ring-sizes` and
`--ring-starts`; rank-local length is `global_len / ring_size` for members of
that batch's ring and zero for other ranks. External FA3 is used by the first
two baselines when available, otherwise they fall back to the local min-FA3
varlen block.

All-CP baseline lengths must be divisible by the physical world size. The total
global token count for `llama3_allgather_attention` must also be divisible by
`2 * world_size`. Causal all-CP mega-ring additionally requires each rank-local
half length to be 128-aligned. Use `--methods` to select a subset or
`--methods all` for all five.

Example:

```bash
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_hybrid_forward.py \
  --global-seqlens 8192,1024,1024 \
  --ring-sizes 2,1,1 --ring-starts 0,0,1 \
  --qhead 16 --kvhead 8 --headdim 128 --methods all \
  --sm-configs 128:4,116:16 --mode both \
  --warmup-iters 5 --num-iters 20
```

### Dataset-shaped hybrid workload

`benchmark_hybrid_dataset_forward.py` uses the standalone `balancer` package
to sample an Arxiv or Github length distribution, pack one global batch to
128K tokens by default, and assign each sequence to a G8/G4/G2/G1 subgroup.
Compute and local tokens are hard placement constraints; normalized token
balance is the main objective, while compute variance and estimated ring
token-hops are soft costs. The frontend then calls
`benchmark_hybrid_forward.main(...)` in the same process with the generated
ring metadata.

Every sampled length, including the final packing residual, is padded upward.
Lengths below 4K use `256 * 2` alignment, lengths from 4K to 8K use
`256 * 4`, and lengths from 8K upward use `256 * 8` (including lengths at or
above 16K). This makes each causal sequence eligible for at least its target
G2/G4/G8 ring. The physical padded tokens participate in the benchmark, so
`actual_tokens` can exceed `target_tokens` by fewer than 2048 tokens.

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_hybrid_dataset_forward.py \
  --dataset arxiv --target-tokens 131072 --seed 0 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --mode causal --methods all --no-check

DATASETS="arxiv github" GPU_COUNTS="2 4 8" \
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
`LOCAL_SEARCH_PASSES`. `--print-workload` reports the requested, topology, and
final caps; estimated per-rank communication; cap-relaxation flags; emergency
fallback; and the number of accepted local-repair moves. A topology-limited
workload can exceed the configured imbalance tolerance when a sequence cannot
use a larger legal ring. Emergency fallback is used only when no placement is
feasible within either configured maximum cap.

`--methods` defaults to `all`. Dataset mode runs every compatible method for
each causal/noncausal mode and prints a skip reason for an incompatible method.
In particular, short causal sequences cannot run `mega_ring_all_cp` because
its rank-local half length must be 128-aligned. Explicitly requesting an
incompatible method remains an error. Use `--print-workload --world-size 8`
to inspect generated lengths and ring assignments without CUDA. Keep
`--no-check` for the full 128K workload because the current correctness
reference materializes quadratic attention scores.

## 1/2/4/8-GPU causal sweep

`benchmark_ring_1_2_4_8.sh` runs the all-CP forward, hybrid forward, and
backward varlen benchmarks on 1, 2, 4, and 8 GPUs. It fixes QH/KVH at 32/8,
head dim at 128, warmup iterations at 10, and measured iterations at 40. Run
it inside a single-node allocation exposing eight SM90 GPUs:

```bash
./benchmark_ring_1_2_4_8.sh
```

The all-CP forward and backward defaults pair batch sizes `16,8,4,2,1` with
local sequence lengths `1K,2K,4K,8K,16K`. This keeps each setting at 16K local
tokens per rank. The corresponding global token totals are 16K, 32K, 64K, and
128K for 1, 2, 4, and 8 GPUs. The hybrid workload keeps one fixed global batch
and balances its local-only sequences across all tested world sizes. All runs
are appended to one timestamped
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

# Result Example

```
Config: world_size=2, methods=['pytorch', 'fa2', 'fa3', 'min_varlen', 'min_varlen_ring', 'min_varlen_mega_ring'], B=16, seqlen=4096, qhead=32, kvhead=8, D=128, mode=both, num_comp_sm=74, num_comm_sm=4, warmup=20, iters=30, check=True
Checks compare each rank output against a full-rank PyTorch reference.

Running local_S=4096, causal=False

B=16, local_S=4096, QH=32, KVH=8, D=128, mode=noncausal
Method                        Time ms       TFLOPS          Check  Note
pytorch                       114.027        154.3             ok  
fa2                           100.139        175.7             ok  
fa3                            69.020        254.9             ok  
min_varlen                     68.260        257.7    timing-only  timing-only local step loop
min_varlen_ring                69.536        253.0             ok
min_varlen_mega_ring           69.726        252.3             ok  

Running local_S=4096, causal=True

B=16, local_S=4096, QH=32, KVH=8, D=128, mode=causal
Method                        Time ms       TFLOPS          Check  Note
pytorch                        89.154         98.7             ok  
fa2                            77.318        113.8             ok  
fa3                            52.941        166.2             ok  
min_varlen                     51.464        170.9    timing-only  timing-only local step loop
min_varlen_ring                52.665        167.0             ok
min_varlen_mega_ring           51.806        169.8             ok
```
