# min_fa3_demo

This standalone directory contains a minimal Hopper FlashAttention
forward/backward demo copied and trimmed from the original `hopper/`
implementation.

## Install

The checked-in dependency metadata targets Python 3.12, PyTorch 2.10.0 with
its CUDA 12.8 runtime, and Triton 3.6.0. Building the extension additionally
requires a CUDA toolkit with `nvcc`, a linkable CUDA driver library, and the
vendored CUTLASS and ThunderKittens submodules. Runtime execution requires an
SM90 Hopper GPU.

Create the repository environment, install the locked core and build
dependencies, and build the extension in place with:

```bash
git submodule update --init third_party/cutlass third_party/ThunderKittens
uv venv --python 3.12
uv sync --frozen --no-install-project --group build
make PYTHON=.venv/bin/python
```

`--no-install-project` intentionally leaves the CUDA extension to the existing
in-place `make` workflow. The commands above target a fresh core environment;
when updating an existing environment that contains manually installed
optional packages, add `--inexact` to preserve those undeclared packages. To
use the plotting and dataset-maintenance scripts, sync their optional
dependency groups before building:

```bash
# Plotting only.
uv sync --frozen --no-install-project --group build --group plot

# Dataset tools include the plotting dependencies.
uv sync --frozen --no-install-project --group build --group dataset
```

The `uv.lock` file pins all transitive Python packages. CUDA toolkit, driver,
and GPU requirements remain system prerequisites; in particular, do not
replace the active toolkit with the separately packaged CUDA runtime libraries
that PyTorch installs into the virtual environment. `CUTLASS_DIR` can still
select an external CUTLASS checkout as described in the Build section below.

MagiAttention remains an optional performance-only baseline with a separate
CUDA-aware installation procedure. Follow
[`baseline/magi_attention/README.md`](baseline/magi_attention/README.md) only
when that benchmark method is needed; it is not part of the default dependency
sync.

## Source provenance

The local sources preserve the structure of the Hopper forward and backward
paths while trimming them to the fixed configuration documented below.

The params structures are copied from the original Hopper forward/backward params paths and trimmed, not rewritten from scratch.

The dense and packed-varlen KV-cache decode/chunk-prefill siblings are pinned to:

- FlashAttention commit `c75d019dea9d910312974417bc28f190dfdda6d9`
- CUTLASS commit `7127592069c2fe01b041e174ba4345ef9b279671`

The head-sharded DCP runner copies and trims its LSE correction and
attention-state merge from vLLM commit
`a89015c6df8eeb37a843b717c97a5be1355de83d`. Its stream/event design was
cross-checked against SGLang commit
`8d6549bc4039d33635844495d86684677a4f0df8`. The runner has no runtime
dependency on either project.

The vendored `third_party/cutlass` checkout matches the CUTLASS commit above.

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
- `hopper/instantiations/flash_fwd_hdim128_bf16_packgqa_sm90.cu`
- `hopper/instantiations/flash_fwd_hdim128_bf16_split_sm90.cu`
- `hopper/flash_fwd_combine_launch_template.h`
- `hopper/flash_fwd_combine_kernel.h`
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
- Hopper PackGQA/Split instantiations -> `csrc/min_fa3_kvcache_*_kernel.cu`
- `hopper/flash_fwd_combine_launch_template.h` -> `include/min_fa3_kvcache_combine_launch.h`
- Hopper Split combine path -> `include/hopper_compat/min_fa3_fwd_combine_kernel.h`
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

## Dense KV-cache decode / chunk-prefill sibling

`min_fa3_op.forward_kvcache` is a read-only dense KV-cache path:

- H100-class Hopper, compiled for `sm_90a`
- `q: [B, Sq, QH, 128]`
- `k_cache/v_cache: [B, Sk_capacity, KVH, 128]`
- `cache_seqlens: [B]`, CUDA contiguous `torch.int32`
- `o: [B, Sq, QH, 128]`; optional FP32 `lse: [B, QH, Sq]`
- `num_splits=0` uses the official automatic heuristic, `1` forces NoSplit,
  and `2..128` forces Split with the official FP32 partial/combine path

The caller must provide `Sq <= cache_seqlens[b] <= Sk_capacity`; the cache
already contains the K/V rows corresponding to the current decode token or
chunk. This API does not append or mutate the cache. Dense `Sq=1` causal decode
uses the official equivalent noncausal `128x176` instance; larger chunks use
bottom-right causal `128x128`. NoSplit retains both PackGQA variants and Split
uses PackGQA, matching the official Hopper dispatch.

The optional keyword `is_causal` defaults to `None`, preserving the behavior
above. Explicit `is_causal=False` selects noncausal context attention and
permits `Sq > Sk_capacity`; explicit `True` selects bottom-right causal
attention.

## Packed-varlen KV-cache decode / chunk-prefill sibling

`min_fa3_op.forward_kvcache_varlen` is the packed-Q/K/V sibling of the dense
KV-cache entry point:

- `q: [total_q, QH, 128]`
- `k_cache/v_cache: [total_k, KVH, 128]`
- CUDA contiguous `torch.int32` `cu_seqlens_q/cu_seqlens_k: [B + 1]`
- matching CPU contiguous `torch.int32` `cu_seqlens_q_host/cu_seqlens_k_host`
- `o: [total_q, QH, 128]`; optional FP32 natural-log
  `lse: [QH, total_q]`
- `num_splits=0` uses the upstream automatic heuristic, `1` forces NoSplit,
  and `2..128` forces Split

Every sequence length must be positive. The two maximum sequence length
arguments must exactly match the largest adjacent difference in the respective
CPU cumulative-length mirror. The host mirrors allow shape, boundary, and
causal checks without synchronizing CUDA cumulative lengths back to the host;
the caller is responsible for keeping each CUDA tensor equal to its CPU mirror.

`is_causal=None` selects noncausal attention when `max_seqlen_q == 1` and
bottom-right causal attention otherwise. Explicit `True` requires
`q_len <= k_len` for every sequence. Explicit `False` supports noncausal
context attention, including sequences where `q_len > k_len`.

K/V are tightly packed effective rows: `cu_seqlens_k[-1] == total_k`; there is
no per-sequence spare capacity. For causal decode or chunk prefill the packed
K/V input must already include the current token or current chunk. This entry
point only reads K/V and never appends to or modifies it. Split uses the copied
dynamic varlen scheduler and allocates FP32 partials as
`[num_splits, QH, total_q, 128]` and `[num_splits, QH, total_q]`, so ragged Q
does not allocate a dense `B * max_seqlen_q` workspace.

## Head-sharded decode context parallel attention

`min_fa3_dcp` provides the local `DCPAttentionRunner`, its CUDA Graph wrapper,
topology validation, and the packed-varlen `DCPMegaAttentionRunner`. The
ordinary runner uses single-stream execution by default and exposes an
explicit Q-all-gather overlap mode. Same-kernel vLLM/SGLang comparison runners
live in the repository-only `dcp_test.baselines` module and are documented in
`dcp_test/README.md`; they are not exported by the runtime module.

```python
from min_fa3_dcp import DCPAttentionRunner

runner = DCPAttentionRunner(process_group)
out = runner.forward_decode(
    q_local, k_cache_local, v_cache_local, cache_seqlens_local,
    num_splits=0, return_lse=False,
)
out = runner.forward_chunk_prefill(
    q_local, k_history_local, v_history_local, history_seqlens_local,
    k_chunk, v_chunk, num_splits=0, return_lse=False,
    overlap_q_allgather=False,
)

out = runner.forward_decode_varlen(
    q_local, k_cache_local, v_cache_local,
    cu_seqlens_q, cu_seqlens_k_local,
    max_seqlen_q, max_seqlen_k_local,
    cu_seqlens_q_host=cu_seqlens_q_host,
    cu_seqlens_k_local_host=cu_seqlens_k_local_host,
    num_splits=0, return_lse=False,
)
out = runner.forward_chunk_prefill_varlen(
    q_local, k_history_local, v_history_local, k_chunk, v_chunk,
    cu_seqlens_q, cu_seqlens_history_local,
    max_seqlen_q, max_seqlen_history_local,
    cu_seqlens_q_host=cu_seqlens_q_host,
    cu_seqlens_history_local_host=cu_seqlens_history_local_host,
    num_splits=0, return_lse=False, overlap_q_allgather=False,
)

# Capture one fixed decode bucket, update bound tensors in place, then replay.
with runner.capture_decode(
    q_local, k_cache_local, v_cache_local, cache_seqlens_local,
    num_splits=0, return_lse=False, overlap_q_allgather=False,
    capture_warmup=3,
) as graph:
    q_local.copy_(next_q)
    cache_seqlens_local.copy_(next_cache_seqlens)
    out = graph.replay()
```

Decode inputs use `q_local: [B, 1, Hq_local, 128]`. Chunk inputs use
`q_local: [B, Sq, Hq_local, 128]` and replicated
`k_chunk/v_chunk: [B, Sq, Hkv_group, 128]`. Both methods optionally return
natural-log FP32 LSE with shape `[B, Hq_local, Sq]`.

Decode assumes that the current token's K/V have already been written to the
last valid position of `k_cache_local/v_cache_local` before any runner is
called. Consequently, the local cache attention includes the current token's
self-attention term; there is no separate current-token attention or cache
append inside the runner. The repository comparison baselines use the same
contract. They compare DCP communication, LSE correction, output
reduction, layout, and scheduling after the cache update, not KV-cache
insertion strategy. Accordingly, benchmark decode `Sk` is the inclusive cache
length containing the current token, while chunk history length excludes the
separately supplied current chunk.

The runners are intentionally limited to SM90, contiguous CUDA BF16 tensors,
head dimension 128, and NCCL group sizes 1 through 8. The production-shaped
comparison topology uses global model head counts with `TP > Hkv`, so every TP
rank owns `Hq/TP` query heads and one replicated global KV head. A DCP group
must remain inside one KV-head replica group. Replicated current chunk K/V
must contain identical values on all ranks in that group; cache insertion
remains outside this API.

The runner all-gathers Q heads and runs min FA3 over the local history shard.
It fuses LSE correction with head packing and performs
a BF16 reduce-scatter. Decode and chunk calls default to non-overlap mode,
which runs the full forward sequentially on the caller's compute stream
without intra-forward dependency events. Passing `overlap_q_allgather=True`
uses its persistent communication stream and CUDA dependency events; for chunk
it overlaps Q all-gather with local chunk attention. Calls return BF16
`[B, Sq, Hq_local, 128]`
and optional FP32 LSE `[B, Hq_local, Sq]`.

The packed siblings use `q_local: [total_q, Hq_local, 128]`, tightly packed
local K/V `[total_k_local, Hkv_group, 128]`, and CUDA plus CPU-mirror
`int32` cumulative lengths. They return BF16 `[total_q, Hq_local, 128]` and,
when requested, natural-log FP32 LSE `[Hq_local, total_q]`. Every sequence
must have positive Q and local-K length, both max-length arguments must equal
the exact maximum adjacent difference in the corresponding host mirror, and
`num_splits` supports `0`, `1`, and every value in `[2, 128]` independently
for each local packed attention call. CUDA cumulative lengths must equal their
CPU mirrors.

Packed decode requires every `q_len == 1`; its local cache already contains
the current token on the position-owner DCP rank. Packed chunk history excludes
the current chunk, while replicated `k_chunk/v_chunk` has exactly `total_q`
rows and uses the Q cumulative lengths. Q lengths and chunk values must agree
inside a DCP group. Local history lengths may differ by rank but may not be
empty. The packed Q all-gather is rank-major
`[DCP, total_q, Hq_local, 128]` before
head reordering; no `B * max_seqlen_q` padding is introduced.

The local `DCPAttentionRunner` owns one persistent communication stream,
reusable CUDA events, grow-only collective/layout buffers, and outstanding
NCCL work handles for overlap. Non-overlap decode/chunk calls reuse the same
buffers on one compute stream and retain only the completion dependency needed
to make back-to-back calls from different current streams safe. Calls may be
enqueued back-to-back, but one runner must not be called concurrently by
multiple host threads.

The four formal capture entry points are `capture_decode`,
`capture_chunk_prefill`, `capture_decode_varlen`, and
`capture_chunk_prefill_varlen`. Capture performs three eager warmup calls by
default, synchronizes the compute and communication streams, barriers the DCP
subgroup, and captures min FA3, Torch/Triton post-processing, NCCL collectives,
and the optional two-stream fork/join in one graph. `replay()` asynchronously
submits the graph and returns its bound static output; every replay overwrites
that output. Copy it explicitly when a result must survive another replay.

A graph fixes tensor addresses, shapes, strides, dtypes, `num_splits`, LSE
return mode, overlap mode, and scalar max lengths. Dense Q/K/V data and CUDA
`cache_seqlens` contents may be changed in place within the captured capacity.
Packed Q/K/V data may be changed in place, but CPU/CUDA cumulative-length
arrays, their contents, batch size, total token counts, and max lengths remain
fixed for the graph lifetime. The capture APIs do not copy a large KV cache.
One runner owns at most one active graph: eager calls and a second capture are
rejected until `close()`. Always close the graph, preferably with its context
manager, before `destroy_process_group()`; `close()` synchronizes in-flight
replay, resets the graph, releases NCCL/tensor references, and makes the runner
reusable. Wrapping a forward directly in `torch.cuda.graph` is rejected with a
message directing callers to these APIs. Current chunk
insertion into sharded KV cache is intentionally outside this attention-only
API.

## What was trimmed away

- Paged KV
- Append KV / KV cache growth
- Rotary
- Qv path
- FP8
- Split-KV and PackGQA outside the dense KV-cache sibling
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
| `benchmark_uniform.sh` | Fixed-total-token uniform matrix with one reusable `torchrun` per selected direction |
| `benchmark_dataset.sh` | Recommended dataset-shaped forward/backward benchmark wrapper for 2, 4, or 8 GPUs |
| `benchmark_dataset_kvh_matrix.sh` | Causal 128K five-dataset, KVH 1/2/4, eight-method forward/backward matrix with one `torchrun` per direction/KVH/dataset |
| `scripts/test_dcp/benchmark_dcp_mega_trace.sh` | Generate `NUM_CASES` trace snapshots, then run eager Mega/baselines and graph baselines in separate 8-rank launches |
| `benchmark_dcp_mega_arrival_matrix.sh` | Run the 3-arrival x 3-DCP trace matrix with one eager Mega comm-SM sweep plus eager/graph baselines per combination |
| `dcp_test/benchmark_dcp_mega_batch.py` | Reuse one 8-rank process group across a filtered packed-varlen Mega DCP case matrix |
| `dcp_test/summarize_dcp_mega_matrix.py` | Validate matrix manifests and flatten workload-weighted summaries to JSON and CSV |
| `benchmark_load_balance.sh` | Dataset/GPU matrix wrapper for the metadata-only forward/backward load-balance benchmark |
| `ring_test/load_balance_bench/run.sh` | Dataset/GPU wrapper for the fixed five-method runtime load-balance suite |
| `ring_test/benchmark_dataset_{forward,backward}.py` | Dataset sampling, BR-PBS placement, and topology benchmark frontend |
| `ring_test/load_balance_bench/benchmark_{forward,backward}.py` | Native Megatron/Zeppelin versus three placement-mapped fused Mega Ring runtime frontends |
| `ring_test/benchmark_forward_ablation.py` | Strict causal W8 six-level forward accumulation ablation with preallocated plans |
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

CPU-only sampler, BR-PBS, load-balance topology-adapter, and DCP topology tests do not
require CUDA:

```bash
python -m unittest balancer.test_balancer \
  ring_test.load_balance_bench.test_topology \
  scripts.test_min_fa3.test_dcp_topology \
  scripts.test_min_fa3.test_dcp_mega_batch
```

Fixed-layout and varlen kernel tests:

```bash
python -m scripts.test_min_fa3.test_min_fa3 \
  --b 1 --seqlen 128 --qhead 8 --kvhead 8 --headdim 128 --mode both
python -m scripts.test_min_fa3.test_min_fa3_varlen \
  --b 2 --seqlen 128,256 --qhead 16 --kvhead 8 --headdim 128 --mode both
python -m scripts.test_min_fa3.test_min_fa3_kvcache \
  --b 3 --seqlen 129,1024,3131 --sq 1,8,32,120,128 \
  --qhead 8 --kvhead 2 --headdim 128 --mode all
python -m scripts.test_min_fa3.test_min_fa3_kvcache_varlen \
  --b 3 --sq 1,8,32 --seqlen 129,1024,3131 \
  --qhead 8 --kvhead 2 --headdim 128 --mode all
```

The DCP correctness suite models global TP rank, global KV-head ownership, KV
replicas, and DCP rank. The default 8-GPU launch covers the six legal
`Hq=32/64`, `Hkv=2/4`, `DCP=2/4` GQA topologies, ragged decode/chunk history,
chunk sizes 2/8/32/128, `num_splits=0/1/2`, repeated forwards, both local
overlap modes, and the pinned vLLM/SGLang runners. It compares every method
with the complete-KV min FA3 reference and records global max/mean output and
LSE absolute error. The vLLM coverage includes both AG+RS and packed A2A.
CUDA Graph capture/replay is the default; each method is first checked once in
eager mode in the same process group. The suite additionally checks in-place
input updates, dynamic dense effective lengths, A2A graph-private buffer
lifetime, overlap fork/join, active-graph exclusion, and close-then-recapture. Use
`--no-cuda-graph` for the eager fallback smoke:

```bash
torchrun --standalone --nproc_per_node=8 --module \
  scripts.test_min_fa3.test_min_fa3_dcp \
  --qhead 32,64 --kvhead 2,4 --tp-size 8 \
  --dcp-sizes 2,4,8 --sq 2,8,32,128 --num-splits 0,1,2
```

The packed DCP sibling suite uses a full-KV
`forward_kvcache_varlen` call as each TP rank's reference. Its default matrix
covers decode, mixed chunk lengths `[1,8,32]`, ragged history, GQA and MQA,
DCP `2/4/8`, `num_splits=0/1/2/8`, both ours overlap modes, pinned
vLLM AG+RS/A2A and SGLang orchestration, returned LSE, repeated calls on alternating caller
streams, grow-only workspace reuse, uniform packed-vs-dense parity, and input
contract failures. Its default graph checks also freeze and report the packed
cumulative-length metadata while allowing Q/K/V contents to change in place:

```bash
torchrun --standalone --nproc_per_node=8 --module \
  scripts.test_min_fa3.test_min_fa3_dcp_varlen \
  --b 3 --sq 1,8,32 --seqlen 129,258,515 \
  --qhead 32 --kvhead 1,2 --headdim 128 --tp-size 8 \
  --dcp-sizes 2,4,8 --num-splits 0,1,2,8
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

Strict six-level forward ablation on eight H100s:

```bash
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_forward_ablation.py \
  --dataset arxiv --target-tokens 131072 --num-cases 20 \
  --sm-configs 128:4,124:8,120:12,116:16 \
  --qhead 32 --kvhead 8 --headdim 128 --mode causal
```

The driver accepts the same comma-separated `--sm-configs COMP:COMM` sweep
style and default `128:4,124:8,120:12,116:16` sweep as the normal Mega Ring
benchmark. This ablation path retains its narrower head constraint:
`D=128`, `KVH * D = 1024`, and `QH % KVH == 0`; the default is
`QH=32, KVH=8, D=128`. It samples deterministic ArXiv cases with seed `0` and
uses the dataset wrapper's `0.05` token-balance tolerance. L1-L5 run on each
case after independent 2K all-CP alignment. L6 receives the BR-PBS workload
lengths and hybrid metadata, including only the padding required by its
selected ring groups.

Timing matches the normal dataset benchmark: each profile runs 10 warmups and
40 measured iterations, each measured iteration first takes the maximum
latency across ranks, and the case latency is the arithmetic mean of those 40
maxima. The primary `Agg TFLOPS` and `Avg/GPU` columns use the BR-PBS workload
lengths for all six levels. For L1-L5, the Note also reports original/aligned
token counts, padding, and the same-latency TFLOPS calculated from the 2K
aligned execution lengths. A final cross-case table groups results by level
and SM config and reports minimum/mean/P50/maximum latency, arithmetic-mean
TFLOPS, and workload-weighted aggregate/per-GPU TFLOPS.

Correctness is controlled by `--check`/`--no-check` and defaults to disabled,
as in the normal benchmark. When enabled, it runs the representative L1-L5
all-CP and L6 mixed-hierarchy output checks; the ablation driver does not run a
stats probe. `--b --seqlen ...` remains available as an explicit one-case
debugging override.

`run_experiments_when_idle.sh` reserves its experiment-queue lock immediately
but, by default, waits `START_DELAY_SECONDS=28800` (eight hours) before it
starts polling GPU availability and launches the matrix. Its ablation queue
runs 20 ArXiv cases at 64K, 128K, and 256K with the same seed, token tolerance,
head configuration, SM sweep, warmup/iteration counts, and correctness flag as
the normal dataset runs. `DRY_RUN=1` skips the delay so it can still print the
queue immediately; set `START_DELAY_SECONDS=0` only when an immediate real
launch is intentional.

Hierarchical mega-ring notes:

- The canonical forward/backward architecture, scheduling, SM-role, TMA-tile,
  reduction, and paper-oriented design notes are documented in
  [docs/MEGARING_HYBRID_KERNEL_DESIGN.md](docs/MEGARING_HYBRID_KERNEL_DESIGN.md).
- Forward supports one node with 2, 4, or 8 SM90 GPUs. Backward supports physical world size 1, 2, 4, or 8; world size 1 permits only G1. A logical ring cannot exceed the physical world size.
- The 8-GPU path uses one fused persistent launch for G8/G4/G2/G1 sequences; the 2-GPU path similarly fuses G2/G1.
- Batches are ordered by non-increasing ring size and explicitly pass global lengths, ring sizes, and aligned ring starts.
- K/V use a shared rank-major capacity arena. Production all-CP and hybrid forward/backward support `KVH` in `{1, 2, 4, 8}` with `D=128` and `QH % KVH == 0`. Communication scheduling and readiness remain aligned to 128-row causal or 176-row noncausal attention tasks. Causal physical transfers use `(128/KVH) x (KVH*128)` BF16/FP32 TMA tiles, preserving a fixed byte size; noncausal forward uses `16 x (KVH*128)` BF16 tiles so its 176-row task stays evenly divisible.
- Every local Q/K sequence length and the per-rank K/V arena capacity must be 128-row aligned. Causal G8/G4/G2 additionally requires each local half to be 128-row aligned. There is no single-row or unaligned-tail communication fallback.
- A full logical tile is not staged at once: communication CTAs pipeline physical TMA subtiles while signaling readiness once each logical K or V task completes.
- Causal G8/G4/G2 uses the zigzag `[front half | back half]` layout.
- The caller must synchronize owner-local K/V initialization across ranks before entering the op.
- Ranks with no local sequence still enter the fused kernel and exit with an empty scheduler work stream.
- Mega-ring backward is causal and non-deterministic only. Its public topology inputs are the CPU int32 contiguous `[B]` tensors `global_seqlens_host`, `ring_sizes_host`, and `ring_starts_host`; causal half prefix sums are generated inside the binding.
- All-CP backward uses `ring_size=world_size, ring_start=0`. The public `half_cu_seqlens` and `half_cu_seqlens_host` arguments no longer exist.
- K/V are `[world_size * rank_kv_capacity, KVH, 128]` rank-major IPC arenas. `rank_kv_capacity` is positive and 128-row aligned. Each FP32 owner accumulator contains `KVH * padded_rank_capacity * 128` elements, where `padded_rank_capacity = round_up(rank_kv_capacity + B * 128, 128)`.
- The VMM-backed FP32 dK/dV owner accumulators and one-element int32 completion counter must be zeroed on every rank, followed by CUDA synchronization and a distributed barrier, before every `backward_varlen_mega_ring` call.
- Backward K/V ingress is `remote gmem -> local smem -> local gmem`. dK/dV egress decodes work by KV head and 128-token padded block. Both use `(128/KVH) x (KVH*128)` physical tiles: 32 KiB for BF16 ingress and 64 KiB for each FP32 remote reduce-add. Padding stays zero and there is no unaligned tail path.
- The full scheduler, readiness, owner-completion, and zero-rank contracts are documented in `docs/HIERARCHICAL_HYBRID_MEGA_RING_BACKWARD_DESIGN.md`.

## Benchmark

### Dataset-shaped topology benchmark

The root wrapper is the recommended entry point for current end-to-end
experiments. It runs forward by default; set `DIRECTION=backward` for causal
backward. `DRY_RUN=1` prints commands without launching CUDA work.

For the full causal 128K KV-head matrix, use the dedicated wrapper. It runs one
`torchrun` per direction, KV-head count, and dataset. The defaults cover five
datasets and KVH 1/2/4, producing 30 independent launches. Each launch keeps
all `NUM_CASES` for that dataset in one process group; `NUM_CASES` defaults to
20. Mega Ring all-CP and hybrid methods traverse every entry in `SM_CONFIGS`;
other methods run only at the first entry. Logs are isolated under
`<LOG_DIR>/<direction>/kvh<KVH>/<dataset>.log`.

Use `DIRECTIONS=forward` or `DIRECTIONS=backward` to run only one direction.

```bash
./benchmark_dataset_kvh_matrix.sh

DRY_RUN=1 WARMUP_ITERS=1 NUM_ITERS=2 \
  DATASETS=arxiv KVHEADS=1,2 ./benchmark_dataset_kvh_matrix.sh
```

```bash
DATASETS="arxiv github pile freelaw prolong" GPU_COUNTS=8 NUM_CASES=4 ZEPPELIN_THRESHOLD=4096 \
  ./benchmark_dataset.sh

GPU_COUNTS=8 DATASETS=arxiv NUM_CASES=1 METHODS=mega_ring_all_cp,mega_ring_hybrid \
  COLLECT_MEGA_RING_STATS=1 CHECK=0 ./benchmark_dataset.sh

DATASETS="arxiv github pile freelaw prolong" GPU_COUNTS=8 NUM_CASES=4 DIRECTION=backward \
  ZEPPELIN_THRESHOLD=4096 ./benchmark_dataset.sh

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

torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_dataset_forward.py \
  --dataset arxiv --target-tokens 131072 --seed 0 --num-cases 1 \
  --qhead 32 --kvhead 8 --headdim 128 --mode causal \
  --methods mega_ring_all_cp,mega_ring_hybrid \
  --collect-mega-ring-stats --no-check
```

Full 128K runs should use `--no-check`; the correctness reference materializes
quadratic attention scores. With `--methods all`, methods that cannot represent
a generated workload are reported as skipped.

`megatron_hybrid_cp` is an independent baseline copied and trimmed from
Megatron-LM commit `368fa88e382b274c8fc12af851331cc1d30d69cc`. It ignores the
BR-PBS ring placement and compiles its own CP1/2/4/8 execution groups from the
same global lengths. Set `MEGATRON_MAX_SEQLEN_PER_RANK` in the shell wrapper or
pass `--megatron-max-seqlen-per-rank` to either dataset/topology frontend; the
default is 8192. After scheduling on original lengths, each sample is minimally
padded for its final CP group: CP1 is unchanged, noncausal CP>1 aligns to CP,
and causal CP>1 aligns to `256 * CP`. This removes divisibility/alignment skips
without changing the post-schedule topology. If the initial CP demand exceeds
the physical world, the FA3 plan caps it to `world_size`; 8K remains the normal
CP-sizing target rather than a hard local-length limit at maximum CP. For
example, 75776 tokens run as CP8 with 9472 tokens per rank. Table and cross-case
TFLOPS use original lengths; each Note reports original/aligned tokens,
padding, aligned-length TFLOPS, and any CP saturation from the same measured
latency. See
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
the baseline's actual task area, including Megatron's final-CP-dependent
padding, all-CP mega-ring's 2048-token alignment, and Magi metadata. Forward
FLOPs remain `4 * visible_scores * QH * D`. Backward FLOPs match the latency
benchmark at `10 * visible_scores * QH * D`.

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
# The uniform wrapper waits for all selected GPUs, then launches once per
# direction and runs every context-length/batch-size point in that process group.
DIRECTION=both CONTEXT_LENGTHS="65536 131072 262144" \
  BATCH_SIZES="1 2 4 8 16" ./benchmark_uniform.sh

# The topology frontends expose the same uniform multi-case mode directly.
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_topology_forward.py \
  --context-lengths 65536,131072,262144 --batch-sizes 1,2,4,8,16 \
  --qhead 32 --kvhead 8 --headdim 128 --mode causal \
  --methods mega_ring_all_cp,mega_ring_hybrid \
  --sm-configs 128:4,124:8,120:12,116:16 --no-check

torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_topology_forward.py \
  --global-seqlens 8192,4096,2048,1024 \
  --ring-sizes 8,4,2,1 --ring-starts 0,4,2,7 \
  --qhead 32 --kvhead 8 --headdim 128 --mode both \
  --methods all --sm-configs 128:4,124:8,120:12,116:16 \
  --collect-mega-ring-stats --no-check

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
metadata. Passing `--context-lengths` together with `--batch-sizes` selects the
uniform multi-case mode; the frontend reuses one process group and its IPC pools
across the full Cartesian product. The ordinary ring benchmarks consume per-rank
local lengths.
`--allgather-overlapping-heads-k-stride` is shared by the per-sequence and
Llama3 all-gather baselines and must divide `--kvhead`.

For `mega_ring_all_cp` and `mega_ring_hybrid`, add
`--collect-mega-ring-stats` to either the topology or dataset-shaped forward
frontend. After each measured fused configuration it runs one separate,
single post-timing probe and prints device-side `qo_visits`, `kv_tile_reads`,
and `kv_tile_reads / qo_visits` for every rank plus `sum(KV) / sum(QO)`.
`qo_visits` counts actual scheduler work tiles or causal merged segments with
positive KV work; `kv_tile_reads` counts attention KV tiles in the mainloop and
does not count the 16-row communication TMA subtransfers. The probe, counter
reset, synchronization, and distributed collection are outside latency and
TFLOPS timing. In causal mode the dynamic ready-segment merge changes observed
Q/O visits, so its ratio can fall within the static lower/upper range reported
by `ring_test/benchmark_load_balance.py`; the KV-tile total is stable.

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
python -m scripts.legacy_benchmark.benchmark_kvcache \
  --b 4 --seqlen 1024,4096,16384 --sq 1,32,128 \
  --qhead 32 --kvhead 8 --headdim 128 --mode all --profile-kernels
python -m scripts.legacy_benchmark.benchmark_kvcache_varlen \
  --b 3 --sq 1,8,32 --seqlen 129,1024,3131 \
  --qhead 32 --kvhead 8 --headdim 128 --mode all --profile-kernels
python -m scripts.legacy_benchmark.benchmark_backward \
  --b 4 --seqlen 512,1024,2048 --qhead 32 --kvhead 8 \
  --headdim 128 --mode both --deterministic
python -m scripts.legacy_benchmark.benchmark_varlen_ring_local \
  --b 4 --seqlen 512,1024 --qhead 32 --kvhead 8 --headdim 128 \
  --num-comp-sm 116 --num-comm-sm 16 --mode causal
```

The KV-cache benchmark reports causal QK+PV model FLOPs and minimum logical
tensor I/O in addition to latency. For bottom-right causal attention it uses
`valid_pairs = Sq * Sk - Sq * (Sq - 1) / 2` and
`FLOPs = 2 * B * QH * valid_pairs * (D + Dv)`. Logical I/O is the BF16
`Q + K + V + O` traffic plus the FP32 LSE write. `TFLOP/s` and
`effective_GB/s` use the end-to-end CUDA-event median and decimal units.
The bandwidth is an algorithmic effective rate, not a hardware DRAM counter;
scheduler metadata and Split's internal FP32 partial O/LSE traffic are not
included.

The packed-varlen KV-cache benchmark accepts either one broadcast value or
exactly `B` comma-separated values for both `--sq` and `--seqlen`. It reports
one fused packed call against a loop of `B` calls to the dense
`forward_kvcache`, using the same split and mask selection. Optional profiling
reports the packed call's prepare, attention, and combine CUDA kernel time.

The DCP attention-only benchmark compares six method labels while holding the
local attention kernel fixed: default `ours_no_overlap`, explicit
`ours_overlap`, `vllm_ag_rs_min_fa3`, `vllm_a2a_min_fa3`,
`sglang_mha_ag_ar_min_fa3`, and `full_kv_min_fa3`. The `vllm` implementation
category expands to both vLLM labels without an additional CLI switch. It uses
global model head counts and defaults to `TP=8`,
`Hq=32/64`, `Hkv=2/4`, and candidate `DCP=2/4/8`. Production topology checks
select exactly six legal GQA combinations; every rejected combination is
written as `skipped_topology` with structured reasons.

```bash
torchrun --standalone --nproc_per_node=8 --module \
  dcp_test.benchmark_dcp \
  --qhead 32,64 --kvhead 2,4 --tp-size 8 --dcp-sizes 2,4,8 \
  --implementations ours,vllm,sglang --workload both \
  --decode-b 1,8,32 --chunk-b 1,4,16 \
  --seqlen 4096,16384,65536 --sq 8,32,128 \
  --headdim 128 --num-splits 0 --warmup 5 --iters 20
```

The main matrix has 36 workload shapes per legal topology: decode
`B=1/8/32, Sk=4K/16K/64K` and chunk
`B=1/4/16, Sq=8/32/128, history=4K/16K/64K`, for 216 GQA cases. Two fixed
`Hq=64,Hkv=1,TP=8,DCP=8` MQA controls run separately by default: decode
`B=8,Sk=16K` and chunk `B=4,Sq=128,history=16K`. Use `--no-mqa-control` to
omit them. MQA controls are excluded from GQA summaries.

Timing starts after Q/K/V are ready and ends when BF16 local-head output is
ready. It excludes projections, input/cache construction, and chunk cache
insertion. The full-KV method runs each TP rank's distinct `Hq/TP` query shard
against a complete cache for that rank's global KV head. For chunk prefill it
concatenates full history and current chunk and runs bottom-right causal min
FA3. This isolates the latency exchanged for KV-memory reduction.

CUDA Graph is the benchmark default for all six methods, including full-KV.
Each fixed case performs three eager capture warmups before capture, then
`--warmup` unmeasured graph replays, followed by timed replays. Use
`--no-cuda-graph` to run the same methods eagerly; in that mode `--warmup`
counts eager calls. The local no-overlap, vLLM, SGLang, and full-KV paths use
one stream. `ours_overlap` alone captures compute plus communication streams.

The A2A path is copied and trimmed from vLLM commit
`a89015c6df8eeb37a843b717c97a5be1355de83d`, including the ordinary GQA/MQA
backend integration rather than only MLA. Its packed combine originates in PR
#41160; the per-call `torch.empty` graph-private buffer policy follows PR
#45487, and LSE is forced to FP32 before bit packing as in PR #47801. It issues
one Q all-gather and one packed `all_to_all_single`; it does not issue the
AG+RS path's LSE all-gather or output reduce-scatter.

All DCP subgroups for a topology run concurrently. Every sample is reduced to
the maximum over all eight global ranks before p50/p90 aggregation, and only
global rank 0 emits a case. The JSON records full parameters, environment and
pinned commits, topology decisions, raw samples, stage p50/p90, effective
global-model TFLOP/s, full/local KV bytes, speedups, and method-specific
collective payload. SGLang's output collective is modeled as an FP32 ring
all-reduce; current/vLLM AG+RS output collectives are BF16 reduce-scatter.
The A2A report separates pack, pure all-to-all, and unpack/combine time;
`output_collective_ms` is the pure all-to-all interval. Its payload records
both the full send/receive buffer size and remote bytes excluding self-copy:
`DCP*T*H_local*(D+2)*2` and `(DCP-1)*T*H_local*(D+2)*2`, respectively.
Per-method execution metadata records the mode, capture warmup, stream policy,
overlap flag, and graph-static tensor/scalar signature. Primary comparison
fields use `ours_no_overlap_speedup_vs_method`; overlap latency and hidden-time
metrics remain separate.

Useful FLOPs and chunk KV-memory reduction use:

```text
decode useful FLOPs = 4 * B * Hq_global * D * Sk
chunk useful FLOPs = 4 * B * Hq_global * D
                     * (Sq * Sk_history + Sq * (Sq + 1) / 2)
effective TFLOP/s = useful FLOPs / global-rank-max latency
KV reduction = (Sk_history + Sq)
               / (ceil(Sk_history / DCP_size) + Sq)
```

Results default to the ignored timestamped path
`benchmarks/results/dcp_gqa_compare_h100_8gpu_<timestamp>.json`; use
`--output-json PATH` to select an explicit file. Summaries are grouped by
`Hq/Hkv/DCP`, workload, batch, and context length. This is an attention-only
DCP orchestration comparison under one min FA3 kernel, not an end-to-end
serving-engine or native backend benchmark for vLLM or SGLang.

A short vLLM dual-baseline smoke run can use:

```bash
torchrun --standalone --nproc_per_node=8 --module \
  dcp_test.benchmark_dcp \
  --qhead 32 --kvhead 2 --tp-size 8 --dcp-sizes 2 \
  --implementations vllm \
  --workload both --decode-b 1 --chunk-b 1 \
  --seqlen 4096 --sq 8 --warmup 1 --iters 3 --no-mqa-control
```

The independent packed-varlen DCP benchmark uses the six default sibling method labels
`ours_no_overlap_varlen`, explicit `ours_overlap_varlen`,
`vllm_ag_rs_min_fa3_varlen`, `vllm_a2a_min_fa3_varlen`,
`sglang_mha_ag_ar_min_fa3_varlen`, and `full_kv_min_fa3_varlen`. `--sq` and `--seqlen` each accept one broadcast
value or exactly `B` comma-separated values. Decode requires `--sq 1`, and
its cache lengths include the current token. Chunk `--seqlen` values are
history lengths and exclude the supplied chunk. Use `--implementations
vllm_a2a` to run only the packed-varlen vLLM A2A baseline; `vllm` continues to
select both vLLM baselines.

Packing, host cumulative-length construction, and interleaved DCP sharding
occur before timing. Each latency sample is the maximum across every TP rank.
Useful FLOPs, KV bytes, collective payload, and throughput use `sum(q_len)`,
actual packed local K tokens, and per-sequence effective decode/causal pairs.
Dense JSON schema version 4 and packed-varlen schema version 3 preserve the
older method fields while adding A2A stages and payload details. JSON records
global and rank-local lengths, packed token counts, stage p50/p90,
speedup, effective TFLOP/s, memory reduction, and both pinned source commits.
It uses the same default CUDA Graph policy, fixed three-call capture warmup,
post-capture `--warmup` semantics, full-KV graph baseline, execution metadata,
and `--no-cuda-graph` eager fallback as the dense benchmark.
Packed-varlen accepts `--no-check` to skip the eager full-KV correctness
precheck for performance-only runs. Its non-Mega orchestration runners record
the full CUDA-event phase breakdown by default. Pass
`--no-baseline-phase-timing` to allocate and record only the start/end events
needed for end-to-end latency; the omitted schema-version-3 phase fields remain
present as zero.

For multi-case Mega DCP measurements, the checked-in
`dcp_test/configs/dcp_mega_six_loads.json` defines six workloads and three
`DCP/Hkv` topologies. The batch frontend expands their 18-case Cartesian
product and reuses one initialized 8-rank world plus cached DCP subgroups:

```bash
torchrun --standalone --nproc_per_node=8 --module \
  dcp_test.benchmark_dcp_mega_batch \
  --workloads small1,medium1 --dcp-sizes 2,8 \
  --implementations mega,ours,vllm,sglang --no-cuda-graph \
  --warmup 5 --iters 20 --output-dir benchmarks/results/dcp_batch_eager
```

`scripts/test_dcp/benchmark_dcp_mega_trace.sh` applies the same one-launch-per-mode and
eager-only Mega grouping to the Mooncake-derived scheduler replay under
`dcp_test/trace`. For example:

```bash
NUM_CASES=20 MODES=eager,graph ./scripts/test_dcp/benchmark_dcp_mega_trace.sh
```

To materialize the sampled workload in the current directory first and then
run the batch wrapper without regenerating it:

```bash
cd /home/hychen/min_fa3_demo

NUM_CASES=20
TRACE_CASES=./mega_dcp_trace_cases.jsonl

python -m dcp_test.trace.generate \
  --config dcp_test/trace/example_config.json \
  --output "$TRACE_CASES" --num-cases "$NUM_CASES"

GENERATE_TRACE=0 TRACE_CASES="$TRACE_CASES" NUM_CASES="$NUM_CASES" \
  MODES=eager,graph ./scripts/test_dcp/benchmark_dcp_mega_trace.sh
```

The second command requires the existing JSONL and validates its case count
and effective config SHA before benchmarking. The default
`GENERATE_TRACE=1` keeps the one-command generate-and-run behavior.

`NUM_CASES` overrides the replay config before reservoir sampling and is
included in the effective config SHA; it is not a prefix truncation of a fixed
case file. The trace config's DCP selects the matching `DCP/Hkv` topology. Each
completed mode manifest contains trace provenance, per-case results, and a
`weighted_summary` grouped by topology and method. Workload-weighted TFLOPS is
computed as `sum(global_effective_flops) / sum(case_p50_seconds)`, equivalent
to the p50-latency-weighted mean used by the ring dataset benchmark.

The larger arrival/DCP matrix has a separate wrapper:

```bash
./benchmark_dcp_mega_arrival_matrix.sh
```

Its defaults independently replay and reservoir-sample 20 cases for every
`arrival_time_scale=1,2,4` and `DCP=2,4,8` combination. Each combination uses
three 8-rank launches: one eager Mega launch sweeps
`num_comm_sms=4,8,12,16,20` inside the same process group, one launch runs the
six eager baseline method labels (including full-KV), and one runs the same
six baselines under CUDA Graph. Thus the default run has 9 trace generations,
27 launches, and 153 aggregate summary rows. Different DCP values deliberately
use independently sampled workloads; the three launches within one
arrival/DCP combination reuse exactly the same JSONL.

Each run is written below `benchmark_logs/bench_dcp/<timestamp>/`. The wrapper
writes incremental per-case and per-launch manifests plus
`matrix_manifest.json` and `matrix_summary.csv`. An abnormal exit still scans
the completed manifests and marks missing, failed, or invalid launches in the
top-level JSON.

Runtime depends on the sampled trace shape, GPU availability, CUDA/NCCL
initialization, and host scheduling state. A two-case, one-combination Hopper
smoke can be launched with:

```bash
ARRIVAL_TIME_SCALES=4 DCP_SIZES=2 MEGA_NUM_COMM_SMS=4,8 \
  NUM_CASES=2 WARMUP=1 ITERS=2 CHECK=1 \
  ./benchmark_dcp_mega_arrival_matrix.sh
```

Use `DRY_RUN=1` to print the expanded generator, launcher, and summarizer
commands without creating result files or initializing CUDA.

Plot the completed matrix as 1-by-3 grouped bar charts with one panel per DCP
size:

```bash
./benchmark_logs/bench_dcp/plot_dcp_mega_latency.py \
  benchmark_logs/bench_dcp/<timestamp>
```

One invocation writes three figures: latency in microseconds, workload-weighted
effective TFLOPS/GPU, and workload-weighted effective KV GB/s/GPU. Each
arrival-scale group compares eager and CUDA Graph versions of vLLM AG+RS, vLLM
A2A, and SGLang, plus eager Mega DCP. Mega defaults to the lowest-latency
measured comm-SM value for each arrival/DCP workload, and all three figures use
that same selection. The Mega annotation reports its improvement over the best
of the six baseline bars: `baseline / Mega` for latency and `Mega / baseline`
for TFLOPS and bandwidth. Use `--mega-num-comm-sm 8` to select one fixed value,
or `--latency-stat p50` to plot the median instead of the mean of the per-case
p50 latency distribution. The script is self-contained apart from its
Matplotlib dependency and can be run from any working directory. With no input
argument, it selects the newest run below its own `benchmark_logs/bench_dcp`
directory and discovers the available arrival scales and DCP sizes from that
run. Incomplete matrices are rejected unless `--allow-incomplete` is passed;
missing bars are then marked `N/A`.

To split the same three metrics into a 2-by-3 figure, with decode-only batches
in the first row, mixed chunk-prefill batches in the second row, and DCP sizes
2, 4, and 8 in the columns, run:

```bash
./benchmark_logs/bench_dcp/plot_dcp_mega_latency_by_batch_type.py \
  benchmark_logs/bench_dcp/<timestamp>
```

Decode-only means every sequence in the case has `q_len <= 16`; cases with a
larger Q length are classified as mixed chunk prefill. Because
`matrix_summary.csv` only stores whole-run aggregates, this mode recomputes the
latency distribution, workload-weighted TFLOPS/GPU, and workload-weighted KV
GB/s/GPU from the per-case JSON files. It selects the lowest-latency Mega
comm-SM setting independently for each batch type, arrival scale, and DCP size.

For runs collected with `MEGA_PHASE_TIMESTAMPS=1` and
`BASELINE_PHASE_TIMING=1`, plot the paired Mega kernel phase milestones and the
vLLM A2A CUDA Graph phase breakdown for DCP sizes 2, 4, and 8 in one 4-by-3
overview:

```bash
./benchmark_logs/bench_dcp/plot_dcp_mega_phase_timestamps.py \
  benchmark_logs/bench_dcp/<timestamp>
```

The first two rows separate decode-only and mixed chunk-prefill batches and
plot Mega against global effective FLOPs. Each Mega point independently uses
the comm-SM configuration with the lowest E2E latency for that paired workload
case; ties select the smaller comm-SM count. The selected configuration's full
set of completion milestones contributes the per-case points and solid
rolling-median curves. The same panels overlay dashed rolling-median curves for
the vLLM A2A CUDA Graph baseline. For each case, its measured phase durations
are accumulated in runner order to reconstruct completion timestamps: Q
all-gather/reorder, local history attention, A2A pack, pure all-to-all,
unpack/combine, local chunk attention, and state merge. The independently
measured Graph E2E trend is a thicker black dashed line. Phase-summary
quantiles, unmeasured gaps, or overlap mean that the final reconstructed
timestamp need not equal E2E; runner Graph E2E also has a broader boundary than
Mega's in-kernel timestamps. The third row compares all measured fixed Mega
comm-SM settings with paired kernel-done ECDFs using the aggregate-best fixed
setting as its reference. The fourth row shows medians of the explicitly
recorded Mega phase-tail durations for every fixed setting. The script
validates that Mega and baseline workloads are paired across every DCP and
comm-SM setting, and writes both a high-resolution PNG and a vector PDF by
default. Use `--stat p90`
for iteration-tail milestones and phase durations, or `--no-pdf` to skip the
PDF. To omit a known unrepresentative paired workload from every panel and
aggregate, pass a repeatable option such as `--exclude-case-id case_000199`.
The exclusion is recorded in the figure subtitle and console summary.

The bandwidth metric is `sum(average logical BF16 K+V bytes per GPU) /
sum(case p50 latency)`. It is an effective payload rate under the benchmark's
logical traffic model, not NCU-measured HBM transaction bandwidth. New batch
manifests store the required byte totals directly; the matrix summarizer can
also recover them from retained per-case JSON files produced by older runs.

The optional `--implementations mega` category adds the experimental
`dcp_mega_varlen` path for chunk prefill only. It supports the default CUDA
Graph mode and the explicit `--no-cuda-graph` eager fallback;
`--mega-block-n auto|128|176` selects its BlockN policy and
`--mega-num-comm-sm N` selects the explicit communication-CTA budget. BlockN
defaults to `auto`: critical-wave always makes its split decision with the
canonical BlockN=128 model, then dispatches NoSplit with BlockN=176 or a
selected split plan with BlockN=128. Explicit 128 or 176 keeps that BlockN for
both the model and dispatch. The communication-CTA budget defaults to 8 and
has no `0/auto` mode. Mega uses fixed `[16,Hq_local,128]` Q/O
communication tiles and requires PackGQA with `Hq_local` 4 or 8. History
combine publishes each completed remote tile directly with a monotonic phase
signal; there is no separate communication-granularity mode or publish pass.
For Mega with `--num-splits 0|1`, critical-wave split selection is enabled by
default. `--no-mega-scheduler-heuristic` independently selects the FA3 native
split policy. `--mega-history-order auto|fifo|release-lpt` controls Q unlock
ordering plus history descriptor order: FIFO and release-LPT are explicit
overrides, including for NoSplit, while `auto` uses FIFO for critical-wave
NoSplit, release-LPT for a decode-only split, and FIFO for a mixed-batch split.
Critical-wave candidate scoring uses that resolved order. BlockN auto dispatches
the three cases as 176, 128, and 128 respectively; FA3 native and fixed split
paths retain the BlockN=128 fallback. Explicit fixed split values 2 through 128
do not enable critical-wave implicitly. The design is documented in
[`DCP_MEGA_CRITICAL_WAVE_SCHEDULER.md`](DCP_MEGA_CRITICAL_WAVE_SCHEDULER.md).
Fixed-shape benchmark replay builds and
uploads metadata once. Eager internal CUDA events measure only the persistent
mega kernel. CUDA Graph replay captures workspace reset, a device-side
monotonic phase increment, both IPC barriers, and the persistent kernel, and
external CUDA events time that complete graph. Existing default benchmark methods
are unchanged. `--mega-phase-timestamps` records optional in-kernel
`%globaltimer` milestones; fused `history_combine_done` and `publish_done`
share the same timestamp, with the latter meaning all remote ready releases
have been issued.

```bash
torchrun --standalone --nproc_per_node=8 --module \
  dcp_test.benchmark_dcp_varlen \
  --b 3 --sq 1,8,32 --seqlen 129,1024,3131 \
  --qhead 32 --kvhead 1 --headdim 128 --tp-size 8 --dcp-size 8 \
  --workload chunk --implementations mega,vllm,full --cuda-graph \
  --mega-block-n auto --mega-num-comm-sm 8 \
  --num-splits 0 --warmup 5 --iters 20
```

Run the one-process-group correctness matrix with:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
  scripts/test_min_fa3/test_dcp_mega_varlen_multi_rank.py --matrix
```

Mixed chunk and ragged decode smoke runs:

```bash
torchrun --standalone --nproc_per_node=8 --module \
  dcp_test.benchmark_dcp_varlen \
  --b 3 --sq 1,8,32 --seqlen 129,1024,3131 \
  --qhead 32 --kvhead 1 --headdim 128 --tp-size 8 --dcp-size 8 \
  --workload chunk --implementations ours,vllm,sglang,full \
  --num-splits 0 --warmup 2 --iters 5

torchrun --standalone --nproc_per_node=8 --module \
  dcp_test.benchmark_dcp_varlen \
  --b 3 --sq 1 --seqlen 129,1024,3131 \
  --qhead 32 --kvhead 1 --headdim 128 --tp-size 8 --dcp-size 8 \
  --workload decode --implementations ours,vllm,sglang,full \
  --num-splits 2 --warmup 2 --iters 5
```

An NVTX/kernel-stage smoke can wrap the first command with `nsys profile -t
cuda,nvtx`. The trace exposes Q all-gather/reorder, packed prepare and attention,
optional Split combine, LSE correction/collective, output collective, and final
state merge. These vLLM and SGLang entries are copied-and-trimmed same-kernel
orchestration baselines; they do not measure native serving runtimes or backend
integration.

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

### Five-method runtime load-balance suite

`ring_test/load_balance_bench/` is a separate fixed comparison suite. It uses
the same un-reordered dataset sample lengths for every result and always emits:
`native_megatron_hybrid_cp`, `native_zeppelin`,
`mega_ring_hybrid_br_pbs`, `mega_ring_hybrid_megatron_cp`, and
`mega_ring_hybrid_zeppelin`. The forward entry point accepts `noncausal`,
`causal`, or `both`; backward is causal-only. It does not modify the CUDA
kernel or public `min_fa3_op` API.

```bash
# CPU-only placement inspection: no CUDA or process group is initialized.
python ring_test/load_balance_bench/benchmark_forward.py \
  --dataset arxiv --target-tokens 131072 --world-size 8 \
  --mode both --print-workload

# One two-GPU forward smoke run of the five fixed results.
torchrun --standalone --nproc_per_node=2 \
  ring_test/load_balance_bench/benchmark_forward.py \
  --dataset arxiv --target-tokens 16385 --num-cases 1 \
  --qhead 32 --kvhead 8 --headdim 128 --mode causal \
  --sm-configs 100:4 --warmup-iters 1 --num-iters 2 --no-check

# Dataset/GPU matrix wrapper. Set DIRECTION=backward for causal backward.
GPU_COUNTS=8 DATASETS="arxiv freelaw github pile prolong" \
  ring_test/load_balance_bench/run.sh
```

Megatron's mapped fused result uses its final padded FA3-ring CP placement,
but it does not replay Megatron execution groups. Native Zeppelin implements
single-node Algorithm 2 with arbitrary `G=1..P` and explicit ordered group
members. G is computed from raw lengths; native execution then minimally aligns
noncausal samples to G and causal samples to `2*G`, without replanning. Its
mapped fused row preserves each native G, rounds G
up to a power of two, aligns execution length to `256 * mapped_G`, greedily
places the aligned job on a legal Buddy group, and finally orders metadata by
descending G. The primary suite TFLOPS metric uses raw work for fairness; mapped
rows additionally report aligned physical work, tokens, and padding. Megatron
retains the padding already explicit in its FA3-ring planner. Planner
construction, input packing, IPC pool allocation,
scheduler preparation, and optional statistics probes are reported separately
or run outside the existing CUDA-event timing boundaries.

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

Dense KV-cache decode / chunk-prefill usage:

```python
import torch
import min_fa3_op

q = torch.randn(3, 32, 16, 128, device="cuda", dtype=torch.bfloat16)
k_cache = torch.randn(3, 4096, 8, 128, device="cuda", dtype=torch.bfloat16)
v_cache = torch.randn_like(k_cache)
cache_seqlens = torch.tensor([1024, 2048, 4096], device="cuda", dtype=torch.int32)

o, lse = min_fa3_op.forward_kvcache(
    q, k_cache, v_cache, cache_seqlens, num_splits=0, return_lse=True
)
```

Packed-varlen KV-cache decode / chunk-prefill usage:

```python
import torch
import min_fa3_op

cu_seqlens_q_host = torch.tensor([0, 1, 9, 41], dtype=torch.int32)
cu_seqlens_k_host = torch.tensor([0, 129, 1153, 4284], dtype=torch.int32)
cu_seqlens_q = cu_seqlens_q_host.cuda()
cu_seqlens_k = cu_seqlens_k_host.cuda()

q = torch.randn(41, 16, 128, device="cuda", dtype=torch.bfloat16)
k_cache = torch.randn(4284, 8, 128, device="cuda", dtype=torch.bfloat16)
v_cache = torch.randn_like(k_cache)

o, lse = min_fa3_op.forward_kvcache_varlen(
    q,
    k_cache,
    v_cache,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=32,
    max_seqlen_k=3131,
    cu_seqlens_q_host=cu_seqlens_q_host,
    cu_seqlens_k_host=cu_seqlens_k_host,
    num_splits=0,
    return_lse=True,
)
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
- Packed-varlen KV-cache K/V has no spare per-sequence capacity and is read-only;
  paged KV, append KV, rotary, local attention, and softcap are not supported.
- Distributed ring and mega-ring paths are single-node because their parallel
  tensors use local CUDA IPC. Hierarchical BR-PBS placement supports physical
  world sizes `2`, `4`, and `8`.
- Mega-ring backward is causal and non-deterministic only.
- Cluster size is fixed to `1` to keep the standalone launch path small while
  preserving the copied SM90 mainloop and kernel structure.
