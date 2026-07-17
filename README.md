# min_fa3_demo

This directory contains a minimal Hopper FlashAttention forward/backward demo copied and trimmed from the existing `hopper/` implementation in this repository.

## Source provenance

The demo is built by copying Hopper forward and backward sources into `hopper/min_fa3_demo/` and trimming them down to a fixed configuration.

The params structures are copied from the original Hopper forward/backward params paths and trimmed, not rewritten from scratch.

## Main copied sources

- `hopper/flash.h`
- `hopper/flash_api.cpp`
- `hopper/flash_fwd_launch_template.h`
- `hopper/flash_fwd_kernel_sm90.h`
- `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`
- `hopper/epilogue_fwd.hpp`
- `hopper/tile_scheduler.hpp`
- `hopper/tile_size.h`
- `hopper/named_barrier.hpp`
- `hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu`
- `hopper/flash_bwd_launch_template.h`
- `hopper/flash_bwd_preprocess_kernel.h`
- `hopper/flash_bwd_postprocess_kernel.h`
- `hopper/flash_bwd_kernel_sm90.h`
- `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp`
- `hopper/epilogue_bwd.hpp`
- `hopper/instantiations/flash_bwd_hdim128_bf16_sm90.cu`

## Mapping to original Hopper code

- `hopper/flash.h` -> `hopper/min_fa3_demo/include/min_fa3_params.h`
- `hopper/flash_fwd_launch_template.h` -> `hopper/min_fa3_demo/include/min_fa3_launch.h`
- `hopper/flash_fwd_kernel_sm90.h` -> `hopper/min_fa3_demo/include/min_fa3_kernel.h`
- `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` -> `hopper/min_fa3_demo/include/min_fa3_mainloop.h`
- `hopper/epilogue_fwd.hpp` -> `hopper/min_fa3_demo/include/min_fa3_epilogue.h`
- `hopper/tile_scheduler.hpp` -> `hopper/min_fa3_demo/include/min_fa3_scheduler.h`
- `hopper/tile_size.h` -> `hopper/min_fa3_demo/include/min_fa3_traits.h`
- `hopper/named_barrier.hpp` -> `hopper/min_fa3_demo/include/min_fa3_named_barrier.h`
- `hopper/flash_fwd_kernel_sm90.h` -> `hopper/min_fa3_demo/include/min_fa3_prologue.h`
- `hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu` -> `hopper/min_fa3_demo/csrc/min_fa3_kernel.cu`
- `hopper/flash.h` -> `hopper/min_fa3_demo/include/min_fa3_varlen_params.h`
- `hopper/tile_scheduler.hpp` -> `hopper/min_fa3_demo/include/min_fa3_varlen_scheduler.h`
- `hopper/flash_fwd_launch_template.h` -> `hopper/min_fa3_demo/include/min_fa3_varlen_launch.h`
- `hopper/flash_prepare_scheduler.cu` -> `hopper/min_fa3_demo/csrc/min_fa3_varlen_prepare_scheduler.cu`
- `hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu` -> `hopper/min_fa3_demo/csrc/min_fa3_varlen_kernel.cu`
- Hopper backward params and launch layers -> `hopper/min_fa3_demo/include/backward/`
- Hopper backward instantiation and host bindings -> `hopper/min_fa3_demo/csrc/backward/`

## Fixed supported configuration

- Architecture: Hopper / SM90 only
- Direction: forward and backward
- Dtype: `torch.bfloat16`
- Head dim: `128`
- Layout: external API is fixed to `BSHD`
- Q/K/V/O shapes:
  - `q: [B, Sq, H, 128]`
  - `k: [B, Sk, H, 128]`
  - `v: [B, Sk, H, 128]`
  - `o: [B, Sq, H, 128]`
- Modes: `is_causal=False` and `is_causal=True`

## Varlen sibling kernel

Alongside the fixed-layout BSHD kernel, this demo directory contains copied-and-trimmed varlen forward and backward paths.

Varlen public API:

- `q: [total_q, qhead, 128]`
- `k: [total_k, kvhead, 128]`
- `v: [total_k, kvhead, 128]`
- `cu_seqlens_q: [B + 1]` with `cu_seqlens_q[-1] == total_q`
- `cu_seqlens_k: [B + 1]` with `cu_seqlens_k[-1] == total_k`
- `max_seqlen_q`
- `max_seqlen_k`
- `is_causal`

Varlen fixed configuration:

- Architecture: Hopper / SM90 only
- Direction: forward and backward
- Dtype: `torch.bfloat16`
- Head dim: `128`
- Layout: flattened varlen tensors with per-batch `cu_seqlens`
- GQA/MQA: supported when `qhead % kvhead == 0`

## Retained Hopper features

- SM90 WGMMA / GMMA path
- TMA for Q, K, and V
- Warp-specialized producer/consumer structure
- Online softmax state in the copied mainloop
- Scheduler barrier logic from the copied SM90 mainloop
- Separate copied prologue, mainloop, epilogue, kernel wrapper, and launch layers

## What was trimmed away

- Paged KV
- Append KV / KV cache growth
- Rotary
- Qv path
- FP8
- Split-KV
- PackGQA
- Softcap
- Local attention
- Non-128 head dims
- Non-bf16 dtypes
- Non-SM90 architectures

## BSHD mapping

The public API accepts BSHD tensors directly. The demo does not require the Python caller to permute inputs.

BSHD is adapted using the copied Hopper stride-based interface:

- `row_stride = stride(-3)`
- `head_stride = stride(-2)`
- `batch_stride = stride(0)`

These strides are then fed into the copied Hopper launch path to build the internal CuTe tensor descriptors.

## Build

```bash
make
```

## Test

```bash
python test_min_fa3.py
```

Varlen test:

```bash
python test_min_fa3_varlen.py --b 1 --seqlen 128 --qhead 8 --kvhead 8 --headdim 128 --mode both
python test_min_fa3_varlen.py --b 2 --seqlen 256 --qhead 16 --kvhead 8 --headdim 128 --mode causal
```

Backward tests:

```bash
python test_min_fa3_backward.py --b 2 --seqlen 128,129 --qhead 8 --kvhead 8 --headdim 128 --mode both
python test_min_fa3_backward.py --b 2 --seqlen 128,129 --qhead 8 --kvhead 2 --headdim 128 --mode both --deterministic
python test_min_fa3_varlen_backward.py --b 3 --seqlen 128,129 --qhead 8 --kvhead 8 --headdim 128 --mode both
python test_min_fa3_varlen_backward.py --b 3 --seqlen 128,129 --qhead 8 --kvhead 2 --headdim 128 --mode both --deterministic
```

Remote load test:

```bash
torchrun --nproc_per_node=2 test_parallel_remote_load.py --shape 256x384 --src-rank 0
torchrun --nproc_per_node=4 test_parallel_remote_load.py --shape 256x384,512x512 --src-rank 1
torchrun --nproc_per_node=2 test_parallel_remote_load.py --shape 512x512 --src-rank 0 --num-blocks 64
```

Ring-attention varlen tests:

```bash
python test_min_fa3_varlen_ring_local.py --b 2 --seqlen 128 --qhead 16 --kvhead 8 --num-comp-sm 1 --num-comm-sm 1 --mode both
python test_min_fa3_varlen_ring_local.py --b 3 --seqlen 128,256 --qhead 16 --kvhead 8 --num-comp-sm 2 --num-comm-sm 2 --mode both
torchrun --nproc_per_node=2 test_min_fa3_varlen_ring_multi_rank.py --b 2 --seqlen 128,256 --qhead 16 --kvhead 8 --src-rank 0 --num-comp-sm 1 --num-comm-sm 1 --mode both
```

Hierarchical hybrid mega-ring forward test:

```bash
sbatch mega_ring_test_hybrid.slurm

torchrun --standalone --nproc_per_node=8 \
  mega_ring_test_min_fa3_varlen_hybrid_multi_rank.py \
  --global-seqlens 8192,4096,2048,2048 \
  --ring-sizes 8,4,2,1 \
  --ring-starts 0,4,2,7 \
  --qhead 16 --kvhead 8 --headdim 128 \
  --num-comp-sm 116 --num-comm-sm 16 \
  --mode both --check-arena --repeat 20
```

Hierarchical mega-ring backward tests:

```bash
# Explicit all-CP metadata on two GPUs.
torchrun --standalone --nproc_per_node=2 \
  mega_ring_test_min_fa3_varlen_backward_multi_rank.py \
  --b 1 --seqlen 256 --qhead 16 --kvhead 8 \
  --num-comp-sm 64 --num-comm-sm 8

# Overlapping G8/G4/G2/G1 subrings, including repeated backward execution.
torchrun --standalone --nproc_per_node=8 \
  mega_ring_test_min_fa3_varlen_backward_hybrid_multi_rank.py \
  --global-seqlens 2048,1024,512,256 \
  --ring-sizes 8,4,2,1 --ring-starts 0,4,2,7 \
  --qhead 16 --kvhead 8 --repeat 2 \
  --num-comp-sm 100 --num-comm-sm 16

# C++ binding validation failures; every case is guarded against kernel launch.
torchrun --standalone --nproc_per_node=8 \
  mega_ring_test_min_fa3_varlen_backward_validation_multi_rank.py
```

Hierarchical mega-ring notes:

- Forward supports one node with 2, 4, or 8 SM90 GPUs. Backward supports physical world size 1, 2, 4, or 8; world size 1 permits only G1. A logical ring cannot exceed the physical world size.
- The 8-GPU path uses one fused persistent launch for G8/G4/G2/G1 sequences; the 2-GPU path similarly fuses G2/G1.
- Batches are ordered by non-increasing ring size and explicitly pass global lengths, ring sizes, and aligned ring starts.
- K/V use a shared rank-major capacity arena. Communication scheduling and readiness are tracked per logical KV tile: 128 rows for causal forward/backward and 176 rows for noncausal forward. Each logical task is physically transferred through 16-row 2D TMA subtiles spanning `KVH * D = 1024` values.
- Every local Q/K sequence length and the per-rank K/V arena capacity must be 128-row aligned. Causal G8/G4/G2 additionally requires each local half to be 128-row aligned. There is no single-row or unaligned-tail communication fallback.
- A full 128/176-row logical tile is not staged in shared memory at once: communication CTAs reuse a small number of 16-row slots so the fused launch stays below Hopper's shared-memory limit.
- Causal G8/G4/G2 uses the zigzag `[front half | back half]` layout.
- The caller must synchronize owner-local K/V initialization across ranks before entering the op.
- Ranks with no local sequence still enter the fused kernel and exit with an empty scheduler work stream.
- Mega-ring backward is causal and non-deterministic only. Its public topology inputs are the CPU int32 contiguous `[B]` tensors `global_seqlens_host`, `ring_sizes_host`, and `ring_starts_host`; causal half prefix sums are generated inside the binding.
- All-CP backward uses `ring_size=world_size, ring_start=0`. The public `half_cu_seqlens` and `half_cu_seqlens_host` arguments no longer exist.
- K/V are `[world_size * rank_kv_capacity, KVH, 128]` rank-major IPC arenas. `rank_kv_capacity` is positive and 128-row aligned. Each FP32 owner accumulator contains `KVH * padded_rank_capacity * 128` elements, where `padded_rank_capacity = round_up(rank_kv_capacity + B * 128, 128)`.
- The VMM-backed FP32 dK/dV owner accumulators and one-element int32 completion counter must be zeroed on every rank, followed by CUDA synchronization and a distributed barrier, before every `backward_varlen_mega_ring` call.
- Backward K/V ingress is `remote gmem -> local smem -> local gmem`. dK/dV egress decodes work by KV head and 128-token padded block, then uses one fixed `16 x 1024` FP32 TMA transaction for each remote reduce-add task. Padding stays zero and there is no unaligned tail path.
- The full scheduler, readiness, owner-completion, and zero-rank contracts are documented in `docs/HIERARCHICAL_HYBRID_MEGA_RING_BACKWARD_DESIGN.md`.

Parameterized test examples:

```bash
python test_min_fa3.py --b 1 --seqlen 128 --qhead 8 --kvhead 8 --headdim 128 --mode both
python test_min_fa3.py --b 2 --seqlen 256 --qhead 16 --kvhead 8 --headdim 128 --mode causal
python test_min_fa3.py --b 1 --seqlen 128 --qhead 8 --kvhead 8 --headdim 128 --mode causal --manual-block-count 132
```

## Benchmark

```bash
python benchmark.py
```

Varlen benchmark:

```bash
python benchmark_varlen.py
python benchmark_varlen.py --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 --headdim 128 --mode both
python benchmark_varlen.py --b 4 --seqlen 256 --qhead 16 --kvhead 8 --headdim 128 --mode causal
```

Backward benchmarks against the installed FA3 implementation:

```bash
python benchmark_backward.py --b 4 --seqlen 512,1024,2048,4096 --qhead 32 --kvhead 32 --headdim 128 --mode both
python benchmark_backward.py --b 4 --seqlen 512,1024,2048,4096 --qhead 32 --kvhead 8 --headdim 128 --mode both --deterministic
python benchmark_varlen_backward.py --b 4 --seqlen 512,1024,2048,4096 --qhead 32 --kvhead 32 --headdim 128 --mode both
python benchmark_varlen_backward.py --b 4 --seqlen 512,1024,2048,4096 --qhead 32 --kvhead 8 --headdim 128 --mode both --deterministic
```

Both backward implementations receive preallocated `dq`, `dk`, and `dv`. Timing includes internal FP32 workspaces, semaphore initialization, preprocess, the main backward kernel, and postprocess, but excludes allocation of the final gradient tensors. `vs FA3` is `FA3 time / min_fa3 time`, so values above `1.0x` favor the minimal demo.

Ring-local varlen benchmark:

```bash
python benchmark_varlen_ring_local.py
python benchmark_varlen_ring_local.py --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 128 --num-comm-sm 0 --mode both
python benchmark_varlen_ring_local.py --b 4 --seqlen 1024 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 128 --num-comm-sm 4 --mode causal
```

Hierarchical mega-ring forward benchmark:

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_hybrid_forward.py \
  --global-seqlens 8192,4096,2048,1024 \
  --ring-sizes 8,4,2,1 --ring-starts 0,4,2,7 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --methods all --zepplin-threshold 4096 \
  --sm-configs 128:4,124:8,120:12,116:16 \
  --mode both --warmup-iters 10 --num-iters 40 --no-check

GPU_COUNTS="2 4 8" ./benchmark_ring_1_2_4_8.sh
```

Dataset-shaped hierarchical hybrid forward and backward benchmarks:

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_hybrid_dataset_forward.py \
  --dataset arxiv --target-tokens 131072 --seed 0 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --mode causal --methods all --zepplin-threshold 4096 --no-check

DATASETS="arxiv github pile" GPU_COUNTS=8 ZEPPLIN_THRESHOLD=4096 \
  ./benchmark_hybrid_dataset.sh

torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_hybrid_dataset_backward.py \
  --dataset arxiv --target-tokens 131072 --seed 0 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --methods all --zepplin-threshold 4096 \
  --sm-configs 128:4,124:8,120:12,116:16 --no-check

DATASETS="arxiv github pile" GPU_COUNTS=8 DIRECTION=backward \
  ZEPPLIN_THRESHOLD=4096 ./benchmark_hybrid_dataset.sh
```

The dataset frontends call the standalone `balancer` package to generate global
lengths and token/compute-constrained G8/G4/G2/G1 metadata. The forward frontend
calls `benchmark_hybrid_forward.main(...)`; the causal backward frontend calls
`benchmark_hybrid_backward.main(...)` with the same explicit topology. The
Arxiv, Github, and Pile-CC distributions are loaded from
`dataset/sequence_length_buckets.json`. Each distribution contains 256-token
`(lower, upper]` bucket counts derived from the sampled document lengths;
samples above 128K are counted in the final bucket. Run
`python dataset/build_length_bucket_stats.py` to regenerate the JSON from the
three `*_doc_lengths.npy` files. The planner searches the strictest feasible
token cap, permits compute relaxation only up to its
configured cap unless topology or emergency fallback requires more, and uses
estimated ring token-hops as a tunable soft cost. Use `--communication-weight`
to change that tradeoff and `--print-workload --world-size 8` to inspect the
final caps and per-rank loads. Every sampled length is padded upward: lengths
below 4K use `256 * 2` alignment, lengths from 4K to 8K use `256 * 4`, and
lengths from 8K upward use `256 * 8`. The physical padded tokens participate
in attention, so `actual_tokens` can exceed the target by fewer than 2048
tokens.
With the default `--methods all`, methods that cannot represent a generated
length are reported as skipped; an explicitly requested incompatible method
remains an error. Backward is causal-only and compares per-sequence all-gather,
Llama3 whole-packed all-gather, FA3/NCCL zigzag ring, Zeppelin, and hierarchical
fused mega-ring. Zeppelin independently places lengths below
`--zepplin-threshold` (default `4096`) whole on one rank by deterministic LPT
and splits lengths at or above the threshold across all ranks. Full 128K
forward and backward runs should use `--no-check` because
their correctness references have quadratic memory use.

`ring_test/benchmark_hybrid_forward.py` compares per-sequence and whole-packed
Llama3-style all-CP all-gather attention, FA3+NCCL ring attention, all-CP fused
mega-ring, Zeppelin, and hierarchical hybrid mega-ring. The all-CP methods use
all physical ranks; the hybrid method follows `--ring-sizes` and
`--ring-starts`. Zeppelin uses G1 for lengths strictly below its threshold and
Gworld for equality and above.
It measures the complete Python op with CUDA events, reports maximum elapsed
time across ranks, and prints aggregate and average-per-GPU TFLOPS. `--check`
adds output reference validation after timing.

Distributed ring benchmark:

```bash
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_forward.py --b 16,16,16 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 --headdim 128 --mode both --methods all --num-comp-sm 128 --num-comm-sm 4 --check
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_forward.py --b 16,16,16 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 --headdim 128 --mode both --methods all --num-comp-sm 116 --num-comm-sm 16 --no-check
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_backward.py --b 4,4,4 --seqlen 256,512,1024 --qhead 32 --kvhead 8 --headdim 128 --methods all --num-comp-sm 64 --num-comm-sm 8 --check
```

Hierarchical hybrid backward benchmark:

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

`benchmark_hybrid_backward.py` treats `--b` and `--seqlen` as equal-length
comma-separated integer lists. Each pair `(b[i], seqlen[i])` is one benchmark
case, and `seqlen[i]` is the member-rank local length. The ring-size/start
pattern is repeated to fill that case's batch and sorted by decreasing ring
size; each generated global length is `seqlen[i] * ring_size`. Timing excludes
forward preparation, owner-accumulator reset, and the pre-launch distributed
barrier; method-internal phase barriers such as Zeppelin's are included. It
reports maximum wall time across ranks, and derives aggregate causal backward
TFLOP/s. Alternatively,
`--global-seqlens`, `--ring-sizes`, and `--ring-starts` pass one explicit
topology, while `--sm-configs` sweeps several compute/communication allocations
without rebuilding the workload. `--check` enables subgroup-aware FP32 autograd
dQ/dK/dV validation for small workloads. `--methods all` includes
`allgather_attention`, `llama3_allgather_attention`, `fa3_ring`, `zepplin`,
`mega_ring_all_cp`, and `mega_ring_hybrid`; the four Python block baselines are
measured once, while both fused methods are repeated for every SM configuration.

Distributed ring benchmark notes:

- This path is single-node only because `TKParallelTensor` uses local IPC.
- Causal checks use the zigzag reference layout by default.
- Output reports both aggregate visible-work `Agg TFLOPS` and per-GPU `Avg/GPU`.

Remote load benchmark:

```bash
torchrun --nproc_per_node=2 benchmark_parallel_remote_load.py
torchrun --nproc_per_node=2 benchmark_parallel_remote_load.py --shape 4096x4096,8192x4096 --src-rank 0
torchrun --nproc_per_node=4 benchmark_parallel_remote_load.py --shape 4096x4096 --src-rank 1 --num-blocks 64
```

Parameterized benchmark examples:

```bash
python benchmark.py --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 32 --headdim 128 --mode both
python benchmark.py --b 4 --seqlen 256 --qhead 16 --kvhead 8 --headdim 128 --mode causal
python benchmark.py --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 32 --headdim 128 --mode noncausal
python benchmark.py --b 4 --seqlen 1024 --qhead 32 --kvhead 32 --headdim 128 --mode causal --manual-block-count 132
```

## Slurm

Default test submission:

```bash
sbatch run.slurm
```

Backward correctness, FA3 performance comparison, all-CP ring backward, and
eight-GPU hierarchical backward validation/benchmarking:

```bash
sbatch run_backward.slurm
```

Current `run.slurm` notes:

- The checked-in script currently requests `4` GPUs and `16` CPUs on one node.
- Its active commands run the distributed `ring_test/benchmark_ring_forward.py`
  sweep followed by three `ring_test/benchmark_hybrid_forward.py` workloads, all
  with `torchrun --nproc_per_node=4`.
- `run_backward.slurm` requests `8` GPUs and `32` CPUs. After the local backward
  checks and benchmarks it runs two-GPU all-CP backward, eight-GPU hierarchical
  correctness, binding validation failures, and the hybrid backward benchmark.

## Python usage

```python
import torch
import min_fa3_op

q = torch.randn(1, 128, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 128, 8, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 128, 8, 128, device="cuda", dtype=torch.bfloat16)

o = min_fa3_op.forward(q, k, v, False)
print(o.shape)

o, lse = min_fa3_op.forward(q, k, v, False, return_lse=True)
dout = torch.randn_like(o)
dq, dk, dv = min_fa3_op.backward(dout, q, k, v, o, lse, False)

# Optional preallocated outputs are used by the steady-state benchmark.
dq_buf = torch.empty_like(q)
dk_buf = torch.empty_like(k)
dv_buf = torch.empty_like(v)
dq, dk, dv = min_fa3_op.backward(
    dout, q, k, v, o, lse, False, dq=dq_buf, dk=dk_buf, dv=dv_buf
)

# Optional: override the automatically computed grid.x thread-block count.
o_manual = min_fa3_op.forward(q, k, v, False, manual_block_count=132)
print(o_manual.shape)
```

Varlen usage:

```python
import torch
import min_fa3_op

batch_size = 2
seqlen = 128
cu_seqlens_q_host = torch.tensor([0, 128, 256], dtype=torch.int32)
cu_seqlens_k_host = torch.tensor([0, 128, 256], dtype=torch.int32)
cu_seqlens_q = cu_seqlens_q_host.to(device="cuda")
cu_seqlens_k = cu_seqlens_k_host.to(device="cuda")

q = torch.randn(batch_size * seqlen, 16, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(batch_size * seqlen, 8, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(batch_size * seqlen, 8, 128, device="cuda", dtype=torch.bfloat16)

o = min_fa3_op.forward_varlen(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    seqlen,
    seqlen,
    False,
    cu_seqlens_q_host=cu_seqlens_q_host,
    cu_seqlens_k_host=cu_seqlens_k_host,
)
print(o.shape)
```

Ring varlen usage:

```python
import torch
import min_fa3_op

cu_seqlens_q_host = torch.tensor([0, 128, 256], dtype=torch.int32)
cu_seqlens_k_host = torch.tensor([0, 128, 256], dtype=torch.int32)
cu_seqlens_q = cu_seqlens_q_host.to(device="cuda")
cu_seqlens_k = cu_seqlens_k_host.to(device="cuda")

q = torch.randn(256, 16, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(256, 8, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(256, 8, 128, device="cuda", dtype=torch.bfloat16)
remote_k = min_fa3_op.create_parallel_tensor(k, local_rank=0, local_world_size=1)
remote_v = min_fa3_op.create_parallel_tensor(v, local_rank=0, local_world_size=1)
next_k = torch.empty_like(k)
next_v = torch.empty_like(v)

o = min_fa3_op.forward_varlen_ring(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    128,
    128,
    False,
    cu_seqlens_q_host=cu_seqlens_q_host,
    cu_seqlens_k_host=cu_seqlens_k_host,
    remote_k=remote_k,
    remote_v=remote_v,
    src_rank=0,
    num_comp_sm=1,
    num_comm_sm=1,
    ring_step=0,
    prefetch_k=next_k,
    prefetch_v=next_v,
)
print(o.shape)
print(next_k.shape, next_v.shape)
```

## Manual launch override

Both `min_fa3_op.forward(...)` and `min_fa3_op.forward_varlen(...)` accept an optional keyword argument:

- `manual_block_count`

Behavior:

- default: use the original automatic launch grid from `get_grid_shape(...)`
- override: when provided, replace the current 1D persistent `grid.x` thread-block count
- units: this is a thread-block count / grid dimension override, not a thread count
- validation: the value must be a positive integer

## Current limitations

- The demo currently requires contiguous BSHD tensors.
- The varlen demo currently requires contiguous flattened `[total_tokens, H, D]` tensors, CUDA `int32` `cu_seqlens`, and matching CPU `int32` host copies of `cu_seqlens`.
- The demo fixes cluster size to `1` to keep the standalone launch path small while preserving the original SM90 forward mainloop and kernel structure.
