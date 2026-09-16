# vLLM Mega DCP end-to-end benchmark

This benchmark package lives under `infer/vllm_bench/`; run its module
commands from the repository root. The companion plugin source is under
`infer/vllm_plugin/`.

This directory integrates the existing Mega DCP kernel with the pinned vLLM
submodule and compares it with this repository's vLLM-style AG+RS and A2A DCP
runners. It is a decode-side benchmark for disaggregated prefill/decode:
history KV defaults to fixed synthetic contiguous buffers, a conversation's
new suffix executes as real chunk prefill, and subsequent tokens use the same
backend with `q_len=1`. The connector reports external-history lengths to
skip historical prefill.

The benchmark uses local Qwen3-30B-A3B MoE configurations and vLLM's dummy loader.
The dummy loader does not download or read a checkpoint: it constructs the
configured model and initializes its weight tensors with random values for
performance evaluation. No tokenizer is loaded; requests use token IDs
directly.

Configurations retain Qwen3's 48 layers (override with `--num-hidden-layers 1`
for smoke), hidden size 2048, explicit head dimension 128, 128 experts, and
8 experts per token. KVH=1/2 variants modify the original KVH=4 for the DCP
topology sweep. All use the [official Qwen3 YaRN configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B#processing-long-texts)
(factor 4, original context 32768) with `--max-model-len 131072`, preserving
the existing long-context trace. These are synthetic performance models,
not evaluations of the pretrained checkpoint.

The pinned ancestor vLLM wheel has the six-argument MoE `topk_softmax` API.
The plugin adapts calls without a padding mask to this API; serving sets
`VLLM_MOE_SKIP_PADDING=0` consistently for all four backends. A newer wheel
with the padding argument keeps its original wrapper. This does not change
expert selection for real tokens; graph padding tokens also execute MoE.
The metadata builder also recognizes the GPU runner's graph-warmup window
through vLLM's capture state, so synthetic short histories can be prepared
before capture while real-request validation remains enabled after startup.

The completed single-layer, four-GPU EP/DCP comparison is recorded in
[the Qwen3 trace report](../../benchmark_logs/vllm_dcp/full_graph_trace_qwen3_tp4_cpp_v5_scale1/README.md).

## Balanced expert routing for attention comparisons

Benchmark serving now defaults to
`VLLM_MOE_ROUTING_SIMULATION_STRATEGY=min_fa3_balanced`, using the pinned
vLLM routing-simulator extension point. The plugin registers a deterministic
strategy; it does not modify the vLLM submodule. Each token selects eight
distinct experts with weights 1/8, independent of hidden states and router
logits. Router projection, expert GEMMs, and EP communication still execute.

With the current contiguous expert placement, EP=8 assigns one expert per
token to each rank. Token 0 selects experts `[0,16,32,48,64,80,96,112]`,
token 1 selects `[1,17,33,49,65,81,97,113]`, and the local expert index wraps
after 16 tokens. EP=4 assigns two experts per token to each rank. Every active
token prefix has exactly equal per-rank assignment counts; counts for the 128
individual experts differ by at most one. Individual expert counts become
exactly equal when the number of assignments is divisible by 128. This also
holds for the real-token prefix of a graph-padded batch.

A single small Triton kernel generates IDs and weights on the GPU and is
captured with the model. The same policy applies to all layers and all four
attention backends. It removes routing-dependent expert-count imbalance;
end-to-end measurements still include MoE work and backend-dependent batch
shapes. This synthetic policy changes model semantics and is intended for
performance comparisons, not model-quality evaluation.

The routing strategy is included in each run's `manifest.json`. Earlier
Qwen3 trace results used the original learned routing and are not results
for this balanced policy. Restore that routing explicitly with an empty
value (the standard shell wrapper preserves it):

```bash
VLLM_MOE_ROUTING_SIMULATION_STRATEGY= ./benchmark_vllm_dcp_matrix.sh
```

## Pinned stack and supported topology

- vLLM source submodule commit: `c6fe94b4d5b418fa213af0e5884eddd304333dcd`
- vLLM stable-ABI core wheel: x86_64 `cu129` from ancestor commit
  `f25953cc59f9b4ba9b04b16228d2b86dcfbcbdb1`
- PyTorch 2.11.0+cu128 and CUDA toolkit 12.8
- one node with eight SM90 GPUs for the formal matrix; four GPUs for smoke
- BF16, head dimension 128, TP=8 (or TP=4 for smoke), global QH=32
- benchmark model configs with global KVH 1, 2, or 4
- EP equals world size via `--enable-expert-parallel`; effective attention
  head partition count is KVH and DCP=`world size / KVH`. vLLM
  `--tensor-parallel-size` still equals world size because DCP reuses its ranks;
  QKV/O projections use that full TP group. Each DCP group shares one KV head.
- formal matrix defaults: Qwen3 MoE, 48 layers, KVH=4, TP=8, DCP=2, EP=8
- DCP groups `[0,1]`, `[2,3]`, `[4,5]`, `[6,7]` share KV heads 0, 1, 2, 3,
  respectively (ranks follow `CUDA_VISIBLE_DEVICES` order)

Backend names are `vllm-ag-rs`, `vllm-a2a`, `mega-fa3-native`, and `mega`. All
four are custom vLLM backends backed by this repository's `min_fa3_op`: the
first two use `VLLMDCPAttentionRunner`/`VLLMA2ADCPAttentionRunner`, while both
Mega variants use `DCPMegaAttentionRunner`. `mega` uses critical-wave automatic
split selection when its full critical-path score wins and otherwise selects
the FA3-native dynamic split plan for that batch. `mega-fa3-native` is the
baseline that always uses FA3-native split selection and FIFO history order. This
fixes the local attention implementation and compares orchestration/collectives. All run
with `cudagraph_mode=FULL`, chunked prefill enabled, and prefix caching disabled.
Capture sizes span 1 through 4096 tokens, including the maximum scheduled batch.
Torch compilation is disabled (`mode=0`) for all methods; CUDA Graphs include
attention and the DCP collectives (plus KV-cache gathering in `paged` mode). FlashInfer
autotuning is also disabled because
the CUSTOM backend does not use FlashInfer; otherwise vLLM runs an unrelated
zero-history full-prefill dummy batch during startup, which is outside this
decode-side backend's contract. The ordinary memory-profiling pass already
uses `skip_attn=True`. Both Mega variants use eight communication SMs and at
most 128 splits by default. `mega` follows the benchmark's automatic history
order (FIFO for NoSplit/mixed plans and release-LPT for decode-only critical-wave
split plans); `mega-fa3-native` always uses FIFO. Serving graphs fix BlockN before capture (128 when `MEGA_DCP_BLOCK_N=auto`),
using a split-capable kernel that reads per-sequence split counts and task
queues from device metadata on every replay. The two Mega variants retain
CPU split planning and ordering outside the graph. Metadata is copied once
per batch and shared by the serial attention layers; IPC phases advance on
the GPU for both warmup and replay. The integration does not require a
separate `vllm-flash-attention` submodule.

The automatic planner keeps a process-local LRU of 128 immutable plans. Its key
includes query lengths, history N-block counts, native split decisions, and all
queue/hardware settings. Native decisions are resolved from exact lengths before
lookup, including the FA L2-size threshold. Candidate scoring uses compact tile
dependencies and groups identical combine work and worker availability times;
it preserves the original candidate search and scores. Cold searches use the
`_dcp_mega_planner` C++ CPU module built by the root `make` target. It releases
the GIL and stops candidates whose work lower bound or partial simulation cannot
beat the incumbent, preserving the original selection and tie-breaking rules.
The Python search remains the reference/fallback when this module is not built.
Check the serving environment with
`python -c 'import _dcp_mega_planner; print(_dcp_mega_planner.__file__)'`.
Both Mega modes cache up to 32 immutable queue images by their selected layout.
The FULL graph preparation path uses the same CPU module's `build_packed_queues`
to construct descriptors, derive dependencies from packed tile indices, and
serialize the int32 payload directly in C++. It caches serialized bytes, so hits
also avoid Python packing. The Python descriptor builder and full validator remain
available as the reference/diagnostic path; byte-for-byte regression tests compare
the direct payload with that validated reference. Environments without the native
queue function fall back to the Python path. FIFO layouts can survive history
growth when their splits remain unchanged. Serialized int32 buffers are staged directly into pinned memory,
and equal payloads reuse the last immutable pinned source. Each batch still copies
its current metadata outside the graph; caches contain no IPC phase state.

The V2 async batch queue allows CPU preparation to overlap preceding GPU work.
Metadata copies and graph replays stay ordered on one CUDA stream, so preparing
the next CPU payload does not overwrite device metadata still being consumed.
The graph regression supports `--scheduler-mode native --pipeline` and
`--scheduler-mode auto --pipeline` to check consecutive submissions without an
intervening host synchronization.

## Installation

The root `setup_fresh_environment.sh` is the single base-environment
orchestrator. The vLLM integration remains an optional second stage in
`third_party/setup_vllm_dcp.sh` so the core Mega-CP environment does not require a
vLLM package installation.

On the networked node:

```bash
./setup_fresh_environment.sh prepare
./third_party/setup_vllm_dcp.sh prepare
```

For the complete Mega-CP environment, on a CUDA 12.8 Hopper node using the
same checkout/shared filesystem:

```bash
CUDA_VISIBLE_DEVICES=0 ./setup_fresh_environment.sh install
./third_party/setup_vllm_dcp.sh verify
```

For a vLLM-only environment that does not need TE, MagiAttention, or the
Transformer-layer CP benchmarks, use this H20 step instead:

```bash
CUDA_VISIBLE_DEVICES=0 ./third_party/setup_vllm_dcp.sh install
```

Both installation entry points require the in-repository `_dcp_mega_planner`
module with `critical_wave_plan` and `build_packed_queues` before skipping the
min-FA3 build. An existing CUDA extension alone is insufficient: missing CPU
modules or APIs trigger `make`, and `verify` reports them as an incomplete install.

The two-stage installer follows vLLM's uv/venv workflow and does not use system
Python or bare pip. `prepare` uses vLLM's `VLLM_USE_PRECOMPILED=1` Python-only
editable install, so this benchmark does not vendor or build a separate
FlashAttention checkout. It downloads the vLLM wheel and Python dependencies
on the networked node. The source revision only publishes a CUDA 13 wheel, so
the installer deliberately pins the core extension to the nearest published
x86_64 `cu129` ancestor above. The two revisions have identical schemas for
the `reshape_and_cache_flash` and `cp_gather_cache` operators used here, and
the extension uses PyTorch's stable-libtorch ABI. For a locked cluster stack,
`VLLM_PRECOMPILED_WHEEL_LOCATION` can instead name an exact compatible wheel.
The vLLM CUDA requirements are intentionally untagged for `torchvision` and
`torchaudio`; after the vLLM editable install, the script explicitly pins
their `+cu128` wheels and verifies their ELF dependencies so a CUDA-13 default
wheel cannot be selected accidentally.
`torchcodec` is intentionally removed after dependency resolution: vLLM treats
it as optional, while the available default wheel is CUDA-13-only and would
fail during text-only CLI import before the model is selected.
The H20-side `install` action performs no vLLM source build; it builds min-FA3
and verifies the prepared environment. Building min-FA3 needs one visible
Hopper GPU; serving needs all eight.

The stock vLLM full source-build path still includes its own FlashAttention
extension as a vLLM build dependency. That is separate from this benchmark's
attention backend. The provided installer intentionally uses precompiled vLLM
extensions to avoid needing that source dependency in this repository.

## Workload semantics

Generate one immutable manifest and reuse it for every backend/load:

```bash
PYTHONPATH=infer .venv/bin/python -m vllm_bench.workload \
  --trace dcp_test/trace/conversation_trace.jsonl \
  --output .cache/vllm_dcp/workload-100-1000.json \
  --warmup-requests 100 \
  --num-requests 1000
```

Each 512-token trace hash maps deterministically to legal non-special token
IDs. At least two matching consecutive hash blocks are required for a lineage.
The longest earlier match becomes external history and the remainder executes
as chunk prefill. Unmatched first turns use
`history_tokens=input_length-1`, avoiding full prefill on the decode service.
The 118 rows whose entire prompt hash sequence already appeared are ambiguous
no-growth turns and are dropped from the fixed manifest.

Arrival scaling divides original relative timestamps. The formal wrapper and
direct matrix command default to scale 1 only. Larger scales increase offered
load while keeping request order/content fixed; scales below 1 reduce the load.

Input and output lengths remain variable and come from each selected trace
row. Every request sets `min_tokens=max_tokens=output_length` and
`ignore_eos=true`, so EOS or another stop token cannot finish generation
before that row's requested output length. The client also treats any response
with a different token count as a failed request.

## Running and results

### History KV storage

All four backends default to `--history-kv-mode synthetic`. Each rank initializes
the existing contiguous K/V history buffers once with `--fill-mean` (default
`0.015`), before CUDA Graph capture. All layers read these immutable buffers
directly, using the scheduled batch's actual packed history lengths and DCP
partitioning. This removes per-layer `cp_gather_cache`, paged-cache updates,
and the connector's paged-cache filling. Current query/chunk K/V computation
and attention still execute; generated KV is not retained as future history.
This mode measures synthetic performance, not autoregressive KV correctness.

The two buffers are shared across layers and keep stable addresses for graph
replay. Their combined capacity is
`max_num_seqs * ceil(max_model_len / DCP) * 2 * 128 * 2` bytes per GPU:
2 GiB for batch 64, DCP=2, and max length 131072. They replace the use of the
existing gather scratch buffers, so no additional history allocation is needed.
Sharing history across layers can also change L2-cache reuse versus per-layer KV.
vLLM still allocates and accounts for its paged cache for scheduler admission;
this change does not remove its memory-capacity limits or block bookkeeping.

For comparison with the previous cache path, select `paged`:

```bash
HISTORY_KV_MODE=paged ./benchmark_vllm_dcp_matrix.sh
```

The wrapper defaults to `HISTORY_KV_MODE=synthetic`; direct matrix/serve commands
accept `--history-kv-mode synthetic|paged`. The mode is recorded in each matrix
run's `manifest.json` and `summary.json`. Use separate result directories for
the two modes; previous results include gather and are not the same workload.

### Launching the matrix

Run short smoke tests, then the formal 100-warmup/1000-measured Qwen3 MoE matrix:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./scripts/smoke_vllm_dcp.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./benchmark_vllm_dcp_matrix.sh
```

Set `ARRIVAL_TIME_SCALES` to select the formal matrix's load points. It accepts
comma- or space-separated values; the default is `1`:

```bash
ARRIVAL_TIME_SCALES=0.5 ./benchmark_vllm_dcp_matrix.sh
ARRIVAL_TIME_SCALES=1,2,4 ./benchmark_vllm_dcp_matrix.sh
# Equivalent: ARRIVAL_TIME_SCALES="1 2 4"
```

Direct matrix invocations accept `--arrival-time-scales 1 2 4`. With the
default five communication-SM values, scale 1 produces 12 service runs
(one each for AG+RS/A2A, five each for Mega FA-native/auto). The short smoke
wrapper retains its explicit scale 4.

The formal wrapper accepts `MAX_NUM_SEQS` (default `64`) to set the batch
request limit for all four backends:

```bash
MAX_NUM_SEQS=128 ./benchmark_vllm_dcp_matrix.sh
```

Direct matrix/serve invocations accept `--max-num-seqs 128`. This single value
sets both vLLM's `--max-num-seqs` and the attention plugin's
`MEGA_DCP_MAX_BATCH`, and is recorded in each run's `manifest.json`.
The per-iteration token budget remains 4096, as do the CUDA Graph capture
sizes. The request limit must be in `[1, 4096]` and fit `--mega-max-total-q`
when that serve option is overridden. Increasing the limit also increases
preallocated attention workspace memory.

The smoke wrapper uses the same default `MEGA_NUM_COMM_SMS=4,8,12,16,20`
communication-SM sweep as the formal matrix. Override it with a smaller set
for a quick check, for example `MEGA_NUM_COMM_SMS=8` or
`MEGA_NUM_COMM_SMS=4,8`.

Both wrappers preallocate capacity for 128 Mega split partials by default so
FA3-native auto selection is not capped at eight. Set `MEGA_MAX_NUM_SPLITS` to
a smaller value when memory is constrained; the host planner applies the same
limit before producing per-sequence split metadata.

Both the smoke and formal wrappers explicitly default and export
`VLLM_USE_FLASHINFER_SAMPLER=0`, because the custom attention backend does not
use FlashInfer's sampling kernel and this avoids an unrelated sampling JIT
during server startup. The smoke wrapper only requires a complete CUDA toolkit
when this is explicitly set to `1`; otherwise an already-built min-FA3
extension can run without sampling JIT. Set
`VLLM_USE_FLASHINFER_SAMPLER=1` only when the CUDA toolkit and headers are
available and that sampler path is intentionally being tested.

The formal wrapper defaults to `KV_HEADS=4`, `NUM_HIDDEN_LAYERS=48`, and
`MEGA_NUM_COMM_SMS=4,8,12,16,20`, matching the Mega comm-SM sweep used by
`benchmark_dcp_mega_arrival_matrix.sh`. It explicitly uses TP=8; the serve
command enables EP=8 and computes DCP=8/KVH=2. Direct matrix/serve invocations
also default to KVH=4. Results are stored under `benchmark_logs/vllm_dcp/kvh4`
by default. The vLLM-style baseline services
run once per arrival scale. Both `mega-fa3-native` and optimized `mega` get
one isolated service run per communication-SM value, with runs recorded as
`{backend}-comm_sm{N}-scale{S}`. To test modified KV-head configurations,
explicitly set `KV_HEADS=1,2,4`; to reduce the communication-SM sweep, use
for example `MEGA_NUM_COMM_SMS="8 16"`.

Both wrappers default to port 8000. Set `PORT` when that endpoint is already
used by another service; the matrix rejects any existing listener before
launching vLLM and also verifies that the new server advertises the expected
model before sending a warmup request:

```bash
PORT=18000 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./scripts/smoke_vllm_dcp.sh
```

The smoke script keeps the production `TP=8`, `KVH=1`, `DCP=8` attention
topology but defaults to a one-layer synthetic Qwen3 MoE and
`--gpu-memory-utilization 0.05`. This lets it exercise all three distributed
attention paths when roughly 10 GiB per H20 is available. It is a functional
integration check, not an 8B performance result. Both values are explicit
overrides:

```bash
GPU_MEMORY_UTILIZATION=0.12 NUM_HIDDEN_LAYERS=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./scripts/smoke_vllm_dcp.sh
```

If other jobs change their allocations while vLLM starts, the smoke script
also supplies an explicit per-GPU KV-cache budget (`KV_CACHE_MEMORY_BYTES`,
default 1 GiB). This makes vLLM skip the free-memory-derived KV-cache sizing
assertion. Increase it only when the co-tenant headroom is stable enough to
hold the larger cache:

```bash
KV_CACHE_MEMORY_BYTES=$((2 * 1024 * 1024 * 1024)) \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./scripts/smoke_vllm_dcp.sh
```

The explicit budget is smoke-only by default; the formal matrix keeps vLLM's
automatic profiling unless `--kv-cache-memory-bytes` is supplied to
`vllm_bench.matrix`.

The formal script continues to default to 48 layers and 0.9 utilization. Its
`KV_HEADS`, `NUM_HIDDEN_LAYERS`, and `GPU_MEMORY_UTILIZATION` environment
variables may be overridden for a deliberately reduced experiment; every run
manifest records the effective values. Only runs with the same KV-head count,
layer count, and memory setting should be compared.

The equivalent Slurm entry is `scripts/vllm_dcp_matrix.slurm`; override
`ROOT_DIR`, module names, or the partition for the target cluster when needed.

For manual inspection:

```bash
PYTHONPATH=infer .venv/bin/python -m vllm_bench.serve \
  --backend mega --kv-heads 1 \
  --num-hidden-layers 1 --gpu-memory-utilization 0.1
```

Replace `mega` with `mega-fa3-native` or either vLLM-style baseline; add
`--dry-run` to print the exact command. Each matrix cell records the service
log, run manifest, raw token timestamps/ITLs, request CSV, and summary. SSE uses
`stream_interval=1` and
`return_token_ids=true`. A delta containing multiple token IDs remains valid
for throughput/E2E but is excluded from TBT rather than receiving fabricated
per-token timestamps. `comparison.json`/CSV report Mega/baseline ratios;
latency ratios below one are lower, while throughput ratios above one are
higher.

## Verification scope

On the current non-CUDA node:

```bash
PYTHONPATH=infer .venv/bin/python -m unittest discover -s infer/vllm_bench/tests -v
```

On H20, smoke each backend with one pure-decode-shaped request and one
continuation/chunk request. Every run must log CUSTOM and the selected
`MIN_FA3_DCP_BACKEND`; AG+RS/A2A runs must instantiate their corresponding
in-repository runner. Existing multi-rank Mega correctness tests cover
`q_len=1,8,32`, so the integration does not repeat a large kernel matrix.

## CUDA Graph regression checks

The focused distributed regression changes request count, query lengths,
history lengths, and input values while alternating two captured graph sizes.
Each graph contains two attention calls sharing scratch. Outputs and LSE are
compared with unsharded causal attention after every replay:

```bash
# Resolve physical nvidia-smi indices to UUIDs on hosts with mixed MIG modes.
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 1,2,5,6 \
  --query-gpu=uuid --format=csv,noheader | paste -sd, -)
for backend in mega vllm-ag-rs vllm-a2a; do
  .venv/bin/torchrun --standalone --nproc-per-node=4 \
    -m scripts.test_min_fa3.test_dcp_mega_serving_graph --hq-local 8 --backend "$backend"
done
```

For a shared four-GPU smoke run, reduce the split workspace for the functional
smoke run. This changes the split capacity and must not be presented as the
formal 128-split benchmark:

```bash
TP_SIZE=4 MEGA_NUM_COMM_SMS=8 MEGA_MAX_NUM_SPLITS=8 \
  RESULT_DIR=benchmark_logs/vllm_dcp/full_graph_smoke \
  ./scripts/smoke_vllm_dcp.sh
```

Choose four non-MIG devices available on the node. TP=4 with KVH=1 uses
DCP=4 and eight query heads per rank. TP=4/KVH=4 is unsupported because
the Mega path requires DCP >= 2.

Every run manifest records `cudagraph_mode`, the fixed graph BlockN, split
capacity, layer count, and full server command. Confirm successful full-graph
capture in `server.log` before comparing results. Shared-GPU smoke timings
are functional evidence, not isolated performance measurements.
