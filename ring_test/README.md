# Multi-rank ring attention forward test

This directory contains a torchrun entry point for forward-only multi-rank
ring attention over the existing varlen demo layout.

The standard PyTorch / FA2 / FA3 methods run a real Python-side ring:

- each rank starts from its local `[B * S, H, D]` K/V block
- K/V are passed around the ring with `batch_isend_irecv`
- each block attention call returns `(out, lse)`
- per-step outputs are merged with the usual online LSE update

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

For a faster smoke run:

```bash
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_forward.py \
  --b 1 --seqlen 128 --qhead 8 --kvhead 8 --mode noncausal \
  --methods pytorch,min_varlen_mega_ring --num-comp-sm 1 --num-comm-sm 1 \
  --warmup-iters 1 --num-iters 3
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
ring baseline and are enabled by default.

```bash
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_backward.py \
  --b 4 --seqlen 256,512,1024 --qhead 32 --kvhead 8 --headdim 128 \
  --methods all --sm-configs 64:8,70:8 \
  --warmup-iters 5 --num-iters 20 --check
```

Use `--no-check` for timing-only sweeps. Local sequence lengths must be
divisible by 256, and the current fused backward requires causal mode,
`D=128`, `qhead % kvhead == 0`, and `kvhead * D == 1024`.

## Hybrid mega-ring benchmark

`benchmark_hybrid_forward.py` compares the same global batch under three
execution strategies:

- `fa3_all_cp`: Python-side all-CP batched varlen ring using FA3 block kernels
- `fa3_hybrid`: Python-side hybrid FA3; CP sequences use batched varlen ring, local-only short sequences use one batched local FA call
- `mega_ring_all_cp`: legacy fused mega-ring CP where every sequence is split across all ranks
- `mega_ring_hybrid`: fused mega-ring hybrid mode selected by `--cp-threshold`

`--global-seqlens` gives the global sequence lengths. Entries above
`--cp-threshold` are CP sequences in hybrid mode. Entries at or below the
threshold are local-only in hybrid mode and are assigned whole to one rank. In
the all-CP baseline, those shorter sequences are still split across all ranks.

For `mega_ring_hybrid`, CP sequences must appear before local-only short
sequences in `--global-seqlens`. CP batch offsets must be identical across
ranks, while local-only batches may be full length on one rank and zero length
on another. The local-only assignment also needs equal total local-only tokens
per rank for the current TKParallelTensor layout.

Example:

```bash
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_hybrid_forward.py \
  --global-seqlens 4096,1024,1024 \
  --qhead 16 --kvhead 8 --headdim 128 \
  --num-comp-sm 1 --num-comm-sm 1 \
  --cp-threshold 2048 --mode both \
  --warmup-iters 5 --num-iters 20
```

## 1/2/4/8-GPU causal sweep

`benchmark_ring_1_2_4_8.sh` runs the all-CP forward, hybrid forward, and
backward varlen benchmarks on 1, 2, 4, and 8 GPUs. It fixes QH/KVH at 32/8,
head dim at 128, warmup iterations at 10, and measured iterations at 40. Run
it inside a single-node allocation exposing eight SM90 GPUs:

```bash
./benchmark_ring_1_2_4_8.sh
```

The all-CP forward and backward defaults are strong-scaling workloads: their
configured global sequence lengths are divided by the GPU count before being
passed to the Python entry points. The hybrid workload keeps one fixed global
batch and balances its eight local-only sequences across all tested world
sizes. All runs are appended to one timestamped
`benchmark_logs/<timestamp>/benchmark_ring_1_2_4_8.log` file, with a separator,
run label, GPU count, and full command before each result section.

Environment variables can override the workload without editing the script.
For example:

```bash
GPU_COUNTS="2 4 8" \
ALL_CP_GLOBAL_SEQLENS="8192,16384,32768" \
BACKWARD_GLOBAL_SEQLENS="4096,8192" \
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
