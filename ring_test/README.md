# Multi-rank ring attention forward test

This directory contains a torchrun entry point for forward-only multi-rank
ring attention over the existing varlen demo layout.

The standard PyTorch / FA2 / FA3 methods run a real Python-side ring:

- each rank starts from its local `[B * S, H, D]` K/V block
- K/V are passed around the ring with `batch_isend_irecv`
- each block attention call returns `(out, lse)`
- per-step outputs are merged with the usual online LSE update

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
