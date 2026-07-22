# MagiAttention topology baseline

This optional, performance-only baseline integrates MagiAttention's public
varlen API with the topology and dataset forward/backward benchmarks. It does
not use `ring_sizes` or `ring_starts`: every case uses the full WORLD context
parallel group, and MagiAttention dynamically packs and dispatches the same
`global_seqlens`. It is therefore an independent dynamic-dispatch baseline,
not an implementation of the benchmark's explicit Buddy-ring Hybrid-CP plan.

The adapter supports BF16, head dimension 128, GQA/MQA where
`qhead % kvhead == 0`, causal and noncausal forward, and causal backward through
the current topology backward frontend. It intentionally has no correctness
interface. `--check` reports `Check=skip` for `magi_attention` and does not build
or compare dense output, LSE, or gradient references.

## Installation

The tested environment contains `magi_attention==1.1.1.post16+g872717e1` with
PyTorch CUDA 12.8. Initialize MagiAttention and its recursive CUTLASS submodule
before building, then install the two additional build-time packages needed by
this revision:

```bash
git submodule update --init --recursive third_party/MagiAttention
uv pip install --python .venv/bin/python debugpy wheel

MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12=1 \
  uv pip install --python .venv/bin/python --no-build-isolation \
  ./third_party/MagiAttention
```

Upstream recommends CUDA 13 or newer. On Hopper, CUDA 12.8 may make some WGMMA
instructions synchronous and can be materially slower; the environment flag
above acknowledges that CUDA 12 build-time warning rather than removing it.

The topology modules import the adapter lazily. Selecting another method does
not require MagiAttention. After NCCL initialization, every rank probes the
public `magi_attn_varlen_key`, `dispatch`, and `calc_attn` APIs and both
`magi_attn_ext` and `magi_attn_comm`. An explicit `--methods magi_attention`
fails with rank-specific import reasons if any rank is unavailable;
`--methods all` prints the reason on rank 0, drops MagiAttention on every rank,
and continues with the remaining methods.

## Dispatch and timing

The adapter passes `DispatchConfig(chunk_size=None, alg=MinHeapDispatchAlg())`.
MagiAttention resolves the chunk size as:

```text
ceil(total_packed_tokens / (world_size * min_chunks_per_rank))
```

`MAGI_ATTENTION_MIN_CHUNKS_PER_RANK` controls the final term and defaults to 8.
Default padding remains enabled; `uneven_shard` is not enabled. A temporary
global `[total_tokens, 1]` BF16 stub is dispatched during runner construction
to discover the exact padded local token count, then released before local
Q/K/V allocation. Full global Q/K/V tensors are never retained per rank.

Overlap uses STATIC mode with `min_chunk_size=512`, `max_num_chunks=4096`, and
the benchmark seed passed to the upstream uniform overlap solver. Set
`--magi-overlap-degree` or `MAGI_OVERLAP_DEGREE` to an integer from 1 through 8;
the default is 2. The current runtime shares one overlap configuration between
forward and backward.

Forward runner construction, key/solver creation, padding, stub dispatch, and
local input allocation are outside timing. The timed callable is only
`calc_attn(q, k, v, key)`. For backward, each untimed `prepare_backward()`
clears leaf gradients and runs a fresh `calc_attn()` to build a new autograd
graph. The timed callable is only `out.backward(dout)` and returns local
dQ/dK/dV.

Reported TFLOPS always use the original `global_seqlens` and their effective
full or causal attention area. Work introduced by MagiAttention padding stays
in measured latency but is not counted as useful FLOPS. The result Note reports
the resolved chunk size and original/padded token counts so padding overhead is
visible.

## Metadata-only load accounting

`ring_test/benchmark_load_balance.py` uses `build_magi_attention_metadata()` to
inspect the same runtime plan without allocating or dispatching Q/K/V and
without calling `calc_attn` or autograd. The metadata object exposes dispatch,
calculation, and communication metadata together with `enable_qo_comm`, native
group-collective selection, forward/backward high-precision reduction flags,
kernel backend, and `save_tail_stage`.

Forward load accounting includes runtime KV/Q fetch and partial O/LSE reduction
inside the measured forward boundary. Causal backward accounting reads the
backward Q/K ranges and masks from each `AttnArg`, then includes runtime KV
fetch and dKV reduce. When Q/O communication is enabled, Q/O/dO/LSE fetch and
dQ reduce are also included. K/V/Q/O/dO payloads are BF16 and LSE is FP32;
dQ/dK/dV use BF16 or FP32 according to the runtime's backend, native collective,
and backward high-precision reduction branch. If `save_tail_stage` retained the
last remote K/V stage during untimed forward preparation, backward omits that
stage's repeated fetch but still counts its gradient reduction.

The metadata benchmark excludes input dispatch, untimed forward/autograd graph
preparation, barriers, and protocol overhead. It requires `torchrun`, CUDA,
SM90, and the Magi extensions; ordinary CPU analysis remains available for the
other seven methods.
