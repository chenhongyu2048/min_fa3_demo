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

Mega-ring varlen tests:

```bash
python mega_ring_test_min_fa3_varlen_ring_local.py --b 1 --seqlen 256 --qhead 8 --kvhead 8 --headdim 128 --num-comp-sm 1 --num-comm-sm 1 --mode both
python mega_ring_test_min_fa3_varlen_hybrid_local.py --seqlens 1152,4096,1408 --qhead 16 --kvhead 8 --headdim 128 --num-comp-sm 1 --num-comm-sm 0 --mode both
torchrun --standalone --nproc_per_node=2 mega_ring_test_min_fa3_varlen_ring_multi_rank.py --b 1 --seqlen 256 --qhead 8 --kvhead 8 --headdim 128 --num-comp-sm 1 --num-comm-sm 1 --mode both
torchrun --standalone --nproc_per_node=1 mega_ring_test_min_fa3_varlen_backward_multi_rank.py --b 1 --seqlen 256,512 --qhead 8 --kvhead 8 --headdim 128 --num-comp-sm 64 --num-comm-sm 8
torchrun --standalone --nproc_per_node=2 mega_ring_test_min_fa3_varlen_backward_multi_rank.py --b 1 --seqlen 256,512 --qhead 8 --kvhead 8 --headdim 128 --num-comp-sm 64 --num-comm-sm 8
```

Mega-ring causal note:

- Causal mega-ring uses the zigzag `[front half | back half]` sequence layout.
- The causal path requires `--seqlen` to be divisible by `256` so each half remains `128`-aligned.
- If this constraint is not met, the causal mega-ring test/benchmark case is skipped or marked unavailable.
- Mega-ring backward is currently causal and non-deterministic only. Its VMM-backed FP32 dK/dV accumulators and owner completion counter must be zeroed on every rank and globally synchronized before calling `backward_varlen_mega_ring`.
- Mega-ring backward communication uses row-granular TMA pipelines for remote K/V load/store and FP32 dK/dV load/remote reduce-add. K/V rows contain `KVH * D = 1024` bf16 values; dKV communication rows are contiguous groups of 1024 floats in the existing padded accumulator layout.

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

Mega-ring local varlen benchmark:

```bash
python benchmark_varlen_mega_ring_local.py
python benchmark_varlen_mega_ring_local.py --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 128 --num-comm-sm 0 --mode both
python benchmark_varlen_mega_ring_local.py --b 4 --seqlen 1024 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 128 --num-comm-sm 4 --mode causal
nsys profile -t cuda,nvtx,osrt -o my_report --stats=true python benchmark_varlen_mega_ring_local.py --profile --b 16 --seqlen 1024 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 116 --num-comm-sm 16 --mode noncausal
```

Distributed ring benchmark:

```bash
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_forward.py --b 16 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 --headdim 128 --mode both --methods all --num-comp-sm 128 --num-comm-sm 4 --check
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_forward.py --b 16 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 --headdim 128 --mode both --methods all --num-comp-sm 116 --num-comm-sm 16 --no-check
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_backward.py --b 4 --seqlen 256,512,1024 --qhead 32 --kvhead 8 --headdim 128 --methods all --num-comp-sm 64 --num-comm-sm 8 --check
```

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

Backward correctness and FA3 performance comparison:

```bash
sbatch run_backward.slurm
```

Current `run.slurm` notes:

- The checked-in script currently requests `4` GPUs and `16` CPUs on one node.
- Its active commands run the distributed `ring_test/benchmark_ring_forward.py` sweep with `torchrun --nproc_per_node=2`.

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

cu_seqlens_q = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
cu_seqlens_k = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)

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
- The output LSE is allocated internally and not exposed.
- The demo fixes cluster size to `1` to keep the standalone launch path small while preserving the original SM90 forward mainloop and kernel structure.
