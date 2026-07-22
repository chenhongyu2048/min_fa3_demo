# min_fa3_demo

This standalone directory contains a minimal Hopper FlashAttention
forward/backward demo copied and trimmed from the original `hopper/`
implementation.

## Source provenance

The local sources preserve the structure of the Hopper forward and backward
paths while trimming them to the fixed configuration documented below.

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

- `hopper/flash.h` -> `include/min_fa3_params.h`
- `hopper/flash_fwd_launch_template.h` -> `include/min_fa3_launch.h`
- `hopper/flash_fwd_kernel_sm90.h` -> `include/min_fa3_kernel.h`
- `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` -> `include/min_fa3_mainloop.h`
- `hopper/epilogue_fwd.hpp` -> `include/min_fa3_epilogue.h`
- `hopper/tile_scheduler.hpp` -> `include/min_fa3_scheduler.h`
- `hopper/tile_size.h` -> `include/min_fa3_traits.h`
- `hopper/named_barrier.hpp` -> `include/min_fa3_named_barrier.h`
- `hopper/flash_fwd_kernel_sm90.h` -> `include/min_fa3_prologue.h`
- `hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu` -> `csrc/min_fa3_kernel.cu`
- `hopper/flash.h` -> `include/min_fa3_varlen_params.h`
- `hopper/tile_scheduler.hpp` -> `include/min_fa3_varlen_scheduler.h`
- `hopper/flash_fwd_launch_template.h` -> `include/min_fa3_varlen_launch.h`
- `hopper/flash_prepare_scheduler.cu` -> `csrc/min_fa3_varlen_prepare_scheduler.cu`
- `hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu` -> `csrc/min_fa3_varlen_kernel.cu`
- Hopper backward params and launch layers -> `include/backward/`
- Hopper backward instantiation and host bindings -> `csrc/backward/`

## Fixed supported configuration

- Architecture: Hopper / SM90 only
- Direction: forward and backward
- Dtype: `torch.bfloat16`
- Head dim: `128`
- Layout: external API is fixed to `BSHD`
- Q/K/V/O shapes:
  - `q: [B, S, QH, 128]`
  - `k: [B, S, KVH, 128]`
  - `v: [B, S, KVH, 128]`
  - `o: [B, S, QH, 128]`
- GQA/MQA: supported when `QH % KVH == 0`
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

The extension requires PyTorch with CUDA extension support, a CUDA toolkit and
driver library, and an SM90 GPU at runtime. CUTLASS is taken from
`third_party/cutlass` by default; `CUTLASS_DIR` may point to another CUTLASS
root or directly to its `include/` directory.

```bash
make

# Optional external CUTLASS checkout.
CUTLASS_DIR=/path/to/cutlass make
```

`make clean` removes the extension and local build products.

## Available entry points

Run the commands below from this directory. Python files below `scripts/` are
invoked as modules so that `min_fa3_op.py` and the in-place extension remain on
the import path.

| Entry point | Purpose |
| --- | --- |
| `benchmark_dataset.sh` | Recommended dataset-shaped forward/backward benchmark wrapper for 2, 4, or 8 GPUs |
| `benchmark_load_balance.sh` | Dataset/GPU matrix wrapper for the metadata-only forward/backward load-balance benchmark |
| `ring_test/benchmark_dataset_{forward,backward}.py` | Dataset sampling, BR-PBS placement, and topology benchmark frontend |
| `ring_test/benchmark_topology_{forward,backward}.py` | Explicit global-length and Buddy-ring topology benchmark |
| `ring_test/benchmark_load_balance.py` | Metadata-only forward/backward token, FLOP, communication, and logical-tile load analysis |
| `ring_test/benchmark_ring_{forward,backward}.py` | Ordinary all-CP distributed ring benchmark |
| `baseline/UltraAttn/packing/export_packed_causal_plan.py` | Offline Gurobi exporter for one fixed-8K UltraAttn allocation plan |
| `baseline/UltraAttn/packing/generate_fixed_128k_plans.sh` | Offline UltraAttn plans for the fixed 1x128K through 16x8K suite |
| `ring_test/ultraattn/benchmark_hybrid_fixed_forward.py` | Five-case UltraAttn versus Mega Ring Hybrid comparison without dataset sampling |
| `balancer/test_balancer.py` | CPU-only sampler and BR-PBS tests |
| `scripts/test_min_fa3/` | Fixed, varlen, backward, remote-load, and ordinary ring tests |
| `scripts/test_mega_ring/` | Hierarchical mega-ring forward/backward and validation tests |
| `scripts/legacy_benchmark/` | Direct single-kernel and remote-load microbenchmarks |
| `dataset/build_length_bucket_stats.py` | Rebuild checked-in 256-token dataset bucket statistics |
| `dataset/plot_sequence_length_buckets.py` | Plot the checked-in dataset length distributions |
| `benchmark_logs/plot_weighted_flops.py` | Plot weighted throughput summaries from dataset benchmark logs |

## Test

CPU-only sampler and BR-PBS tests do not require CUDA:

```bash
python -m unittest balancer.test_balancer
```

Fixed-layout and varlen kernel tests:

```bash
python -m scripts.test_min_fa3.test_min_fa3 \
  --b 1 --seqlen 128 --qhead 8 --kvhead 8 --headdim 128 --mode both
python -m scripts.test_min_fa3.test_min_fa3_varlen \
  --b 2 --seqlen 128,256 --qhead 16 --kvhead 8 --headdim 128 --mode both
```

Backward tests:

```bash
python -m scripts.test_min_fa3.test_min_fa3_backward \
  --b 2 --seqlen 128,129 --qhead 8 --kvhead 2 --headdim 128 \
  --mode both --deterministic
python -m scripts.test_min_fa3.test_min_fa3_varlen_backward \
  --b 3 --seqlen 128,129 --qhead 8 --kvhead 2 --headdim 128 \
  --mode both --deterministic
```

Remote load test:

```bash
torchrun --standalone --nproc_per_node=2 --module \
  scripts.test_min_fa3.test_parallel_remote_load \
  --shape 256x384,512x512 --src-rank 0 --num-blocks 64
```

Ordinary ring-attention varlen tests:

```bash
python -m scripts.test_min_fa3.test_min_fa3_varlen_ring_local \
  --b 3 --seqlen 128,256 --qhead 16 --kvhead 8 \
  --num-comp-sm 2 --num-comm-sm 2 --mode both
torchrun --standalone --nproc_per_node=2 --module \
  scripts.test_min_fa3.test_min_fa3_varlen_ring_multi_rank \
  --b 2 --seqlen 128,256 --qhead 16 --kvhead 8 --src-rank 0 \
  --num-comp-sm 1 --num-comm-sm 1 --mode both
```

Hierarchical hybrid mega-ring forward test:

```bash
torchrun --standalone --nproc_per_node=8 --module \
  scripts.test_mega_ring.mega_ring_test_min_fa3_varlen_hybrid_multi_rank \
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
torchrun --standalone --nproc_per_node=2 --module \
  scripts.test_mega_ring.mega_ring_test_min_fa3_varlen_backward_multi_rank \
  --b 1 --seqlen 256 --qhead 16 --kvhead 8 \
  --num-comp-sm 64 --num-comm-sm 8

# Overlapping G8/G4/G2/G1 subrings, including repeated backward execution.
torchrun --standalone --nproc_per_node=8 --module \
  scripts.test_mega_ring.mega_ring_test_min_fa3_varlen_backward_hybrid_multi_rank \
  --global-seqlens 2048,1024,512,256 \
  --ring-sizes 8,4,2,1 --ring-starts 0,4,2,7 \
  --qhead 16 --kvhead 8 --repeat 2 \
  --num-comp-sm 100 --num-comm-sm 16

# C++ binding validation failures; every case is guarded against kernel launch.
torchrun --standalone --nproc_per_node=8 --module \
  scripts.test_mega_ring.mega_ring_test_min_fa3_varlen_backward_validation_multi_rank
```

Hierarchical mega-ring notes:

- The canonical forward/backward architecture, scheduling, SM-role, TMA-tile,
  reduction, and paper-oriented design notes are documented in
  [docs/MEGARING_HYBRID_KERNEL_DESIGN.md](docs/MEGARING_HYBRID_KERNEL_DESIGN.md).
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

## Benchmark

### Dataset-shaped topology benchmark

The root wrapper is the recommended entry point for current end-to-end
experiments. It runs forward by default; set `DIRECTION=backward` for causal
backward. `DRY_RUN=1` prints commands without launching CUDA work.

```bash
DATASETS="arxiv github pile freelaw prolong" GPU_COUNTS=8 NUM_CASES=4 ZEPPLIN_THRESHOLD=4096 \
  ./benchmark_dataset.sh

DATASETS="arxiv github pile freelaw prolong" GPU_COUNTS=8 NUM_CASES=4 DIRECTION=backward \
  ZEPPLIN_THRESHOLD=4096 ./benchmark_dataset.sh

DRY_RUN=1 GPU_COUNTS="2 4 8" DATASETS=arxiv ./benchmark_dataset.sh
```

The frontends sample ArXiv, GitHub, Pile-CC, FreeLaw, or ProLong lengths from
`dataset/sequence_length_buckets.json`, then use BR-PBS to produce G8/G4/G2/G1
metadata. The main planner controls are:

```text
--compute-balance-tolerance 0.05
--token-balance-tolerance 0.10
--beam-width 64
--finalist-count 8
--structure-threshold 0.5
--max-repair-iterations 32
```

The shell equivalents are `COMPUTE_BALANCE_TOLERANCE`,
`TOKEN_BALANCE_TOLERANCE`, `BEAM_WIDTH`, `FINALIST_COUNT`,
`STRUCTURE_THRESHOLD`, and `MAX_REPAIR_ITERATIONS`. Use the CPU-only planner
view before a distributed run when inspecting a workload:

```bash
python ring_test/benchmark_dataset_forward.py \
  --dataset arxiv --target-tokens 131072 --seed 0 \
  --world-size 8 --print-workload
```

Full 128K runs should use `--no-check`; the correctness reference materializes
quadratic attention scores. With `--methods all`, methods that cannot represent
a generated workload are reported as skipped.

`megatron_hybrid_cp` is an independent baseline copied and trimmed from
Megatron-LM commit `368fa88e382b274c8fc12af851331cc1d30d69cc`. It ignores the
BR-PBS ring placement and compiles its own CP1/2/4/8 execution groups from the
same global lengths. Set `MEGATRON_MAX_SEQLEN_PER_RANK` in the shell wrapper or
pass `--megatron-max-seqlen-per-rank` to either dataset/topology frontend; the
default is 8192. Oversized or alignment-incompatible samples are skipped by
`--methods all` and are errors when the method is requested explicitly. See
`baseline/megatron_hybrid_cp/README.md` for schedule semantics, frontend
integration, backend fallback, and the separate forward/backward timing bounds.

`magi_attention` is an optional performance-only baseline for the topology and
dataset frontends. It consumes the same global sequence lengths but ignores the
BR-PBS `ring_sizes`/`ring_starts`, using the full WORLD group and MagiAttention's
own padding, packing, and dynamic dispatch. Set `MAGI_OVERLAP_DEGREE` in the
shell wrapper or pass `--magi-overlap-degree` (default 2, valid range 1-8).
Forward times only `calc_attn`; backward rebuilds its forward graph outside the
timed region and times only autograd backward. Useful FLOPS use original lengths
and exclude padding work, while the result Note reports original/padded tokens.
See [`baseline/magi_attention/README.md`](baseline/magi_attention/README.md) for
the `uv pip` CUDA 12.8 installation, recursive CUTLASS initialization, optional
dependency probe behavior, timing details, and the upstream CUDA 13 performance
recommendation.

### UltraAttn 8K graph baseline

`ultraattn` has been removed in main branch, but keeped in the `ultraattn_baseline` branch.

The forward benchmark accepts `--methods ultraattn` only for the fixed
eight-GPU `1x128K`, `2x64K`, `4x32K`, `8x16K`, and `16x8K` suite. It consumes
an offline UltraAttn QxK allocation and compiles it into input-communication,
compute, partial-return, and owner-merge dependency nodes. Communication uses
asynchronous `torch.distributed` NCCL; compute nodes call this demo's
`min_fa3_op.forward_varlen`; partial O/LSE is merged in FP32.

The normal `.venv` needs no UltraAttn runtime install, external FlashAttention,
PyNCCL, or Gurobi. Generate the five plans in the isolated planner environment
and run the comparison with:

```bash
PLANNER_PY=/home/hychen/.venvs/ultraattn-planner/bin/python \
BLOCK_TOKENS=8192 WORLD_SIZE=8 QHEAD=32 KVHEAD=8 HEADDIM=128 \
TIME_LIMIT=1800 GUROBI_NUM_THREADS=32 \
baseline/UltraAttn/packing/generate_fixed_128k_plans.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc_per_node=8 \
  ring_test/ultraattn/benchmark_hybrid_fixed_forward.py \
  --qhead 32 --kvhead 8 --headdim 128 \
  --methods ultraattn,mega_ring_hybrid \
  --ultraattn-plan-dir baseline/UltraAttn/packing_plans \
  --ultraattn-block-tokens 8192 \
  --ultraattn-workspace-mib 2048 \
  --sm-configs 128:4 --warmup-iters 10 --num-iters 40 --no-check
```

There is no staged, 256-token packing, dataset-sampler, all-CP, round-robin, or
Buddy-ring fallback for this method. See `baseline/UltraAttnREADME.md` for the
planner environment, graph execution boundary, correctness commands, and
measured five-case results.

### Forward/backward load-balance metadata benchmark

`ring_test/benchmark_load_balance.py` statically analyzes the same eight
baselines registered by the explicit-topology forward and backward latency
benchmarks. It does not time or launch an attention kernel, dispatch tensors,
or build an autograd graph. `--direction` defaults to `forward`; backward is
causal-only BF16 with head dimension 128 and world size 2, 4, or 8.

This is a breaking rename: the old `benchmark_load_balance_forward.py` and
`benchmark_load_balance_forward.sh` entry points are not retained.

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_load_balance.py \
  --global-seqlens 8192,4096,2048 \
  --ring-sizes 8,4,2 --ring-starts 0,0,4 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --mode causal --methods all

torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_load_balance.py --direction backward \
  --global-seqlens 8192,4096,2048 \
  --ring-sizes 8,4,2 --ring-starts 0,0,4 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --mode causal --methods all
```

For a dataset/GPU matrix with timestamped terminal logs, use the shell wrapper:

```bash
DIRECTION=backward GPU_COUNTS="2 4 8" \
  DATASETS="arxiv freelaw github pile prolong" \
  NUM_CASES=4 ./benchmark_load_balance.sh

DRY_RUN=1 DIRECTION=backward GPU_COUNTS="2 4 8" DATASETS=arxiv \
  ./benchmark_load_balance.sh
```

The wrapper uses `DIRECTION=forward|backward`, defaulting to `forward`, and
writes `benchmark_load_balance_<direction>.log`. It retains the dataset/GPU
matrix and existing sampler, head, method, baseline, device, and logging
overrides. It has no warmup, iteration, SM-sweep, correctness, or explicit
topology variables.

MagiAttention metadata construction requires `torchrun`, CUDA, and the Magi
extensions. Without Magi, the other methods can be analyzed on CPU with ordinary
Python and `--world-size 2|4|8`; `--methods all` prints a Magi skip reason in
that mode. Effective fields retain the original workload; physical fields use
the baseline's actual task area, including all-CP 2048-token alignment and Magi
metadata. Forward FLOPs remain `4 * visible_scores * QH * D`. Backward FLOPs
match the latency benchmark at `10 * visible_scores * QH * D`.

Forward keeps the `KV tiles / QO visit` lower/upper metric. Backward reports the
single mirrored `Q tiles / K-dKV` ratio: logical 128-token Q tiles read per
logical K tile visit and dK/dV update, with both counters expanded by Q heads.
Backward communication includes only work inside its measured boundary: BF16
K/V movement and gradient return/reduction payloads. Communication load uses
sent bytes only for every method, so each transfer contributes once rather than
once at each endpoint. Setup repartition, untimed forward preparation, barriers,
semaphores, and Magi input dispatch are excluded. See `ring_test/README.md` for
per-baseline accounting details.

### Explicit topology and ordinary ring benchmarks

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_topology_forward.py \
  --global-seqlens 8192,4096,2048,1024 \
  --ring-sizes 8,4,2,1 --ring-starts 0,4,2,7 \
  --qhead 32 --kvhead 8 --headdim 128 --mode both \
  --methods all --sm-configs 128:4,124:8,120:12,116:16 --no-check

torchrun --standalone --nproc_per_node=2 \
  ring_test/benchmark_ring_forward.py \
  --b 16,8,4 --seqlen 512,1024,2048 \
  --qhead 32 --kvhead 8 --headdim 128 --mode both \
  --methods all --allgather-overlapping-heads-k-stride 1 \
  --sm-configs 128:4,116:16 --no-check

torchrun --standalone --nproc_per_node=2 \
  ring_test/benchmark_ring_backward.py \
  --b 4,4,4 --seqlen 256,512,1024 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --methods all --allgather-overlapping-heads-k-stride 1 \
  --sm-configs 128:4,116:16 --no-check
```

These distributed paths are single-node because `TKParallelTensor` uses local
CUDA IPC. The hybrid benchmark consumes global lengths and explicit Buddy-ring
metadata; the ordinary ring benchmarks consume per-rank local lengths.
`--allgather-overlapping-heads-k-stride` is shared by the per-sequence and
Llama3 all-gather baselines and must divide `--kvhead`.

### Direct kernel microbenchmarks

The older direct benchmarks remain available under `scripts/legacy_benchmark`
and are useful for isolated kernel comparisons:

```bash
python -m scripts.legacy_benchmark.benchmark \
  --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 \
  --headdim 128 --mode both
python -m scripts.legacy_benchmark.benchmark_varlen \
  --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 \
  --headdim 128 --mode both
python -m scripts.legacy_benchmark.benchmark_backward \
  --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 \
  --headdim 128 --mode both --deterministic
python -m scripts.legacy_benchmark.benchmark_varlen_ring_local \
  --b 4 --seqlen 512,1024 --qhead 32 --kvhead 8 --headdim 128 \
  --num-comp-sm 116 --num-comm-sm 16 --mode causal
```

Remote-load microbenchmark:

```bash
torchrun --standalone --nproc_per_node=2 --module \
  scripts.legacy_benchmark.benchmark_parallel_remote_load \
  --shape 4096x4096,8192x4096 --src-rank 0 --num-blocks 64
```

### Dataset maintenance and plotting

```bash
python dataset/build_length_bucket_stats.py
python dataset/plot_sequence_length_buckets.py
python benchmark_logs/plot_weighted_flops.py --world-size 8
```

`dataset/sample_length.py` is the manual raw-data collection utility. It
requires `datasets` and `transformers`, and its `DATASET_CHOICE` constant selects
which source distribution to sample before rebuilding the shared JSON.

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

- All kernels require Hopper SM90, `torch.bfloat16`, head dimension `128`, and
  contiguous tensors.
- BSHD uses `[B, S, H, 128]`; varlen uses flattened
  `[total_tokens, H, 128]` tensors, CUDA `int32` `cu_seqlens`, and matching CPU
  `int32` host copies.
- Distributed ring and mega-ring paths are single-node because their parallel
  tensors use local CUDA IPC. Hierarchical BR-PBS placement supports physical
  world sizes `2`, `4`, and `8`.
- Mega-ring backward is causal and non-deterministic only.
- Cluster size is fixed to `1` to keep the standalone launch path small while
  preserving the copied SM90 mainloop and kernel structure.
