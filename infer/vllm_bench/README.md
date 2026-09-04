# vLLM Mega DCP end-to-end benchmark

This benchmark package lives under `infer/vllm_bench/`; run its module
commands from the repository root. The companion plugin source is under
`infer/vllm_plugin/`.

This directory integrates the existing Mega DCP kernel with the pinned vLLM
submodule and compares it with this repository's vLLM-style AG+RS and A2A DCP
runners. It is a decode-side benchmark for disaggregated prefill/decode:
history KV is filled by a benchmark connector, a conversation's new suffix
executes as real chunk prefill, and subsequent tokens use the same backend with
`q_len=1`.

The benchmark uses local Llama 3.1 8B configurations and vLLM's dummy loader.
The dummy loader does not download or read a checkpoint: it constructs the
configured model and initializes its weight tensors with random values for
performance evaluation. No tokenizer is loaded; requests use token IDs
directly.

## Pinned stack and supported topology

- vLLM source submodule commit: `c6fe94b4d5b418fa213af0e5884eddd304333dcd`
- vLLM stable-ABI core wheel: x86_64 `cu129` from ancestor commit
  `f25953cc59f9b4ba9b04b16228d2b86dcfbcbdb1`
- PyTorch 2.11.0+cu128 and CUDA toolkit 12.8
- one node with eight SM90 GPUs (the target is eight H20s)
- BF16, head dimension 128, TP=8, global QH=32
- benchmark model configs with global KVH 1, 2, or 4
- DCP=`8 / KVH`, so each DCP group is one replicated KV-head group
- formal matrix: KVH=1/2/4, corresponding to DCP=8/4/2

Backend names are `vllm-ag-rs`, `vllm-a2a`, and `mega`. All three are custom
vLLM backends backed by this repository's `min_fa3_op`: the first two use
`VLLMDCPAttentionRunner`/`VLLMA2ADCPAttentionRunner`, while the third uses
`DCPMegaAttentionRunner`. This fixes the local attention implementation and
compares orchestration/collectives. All run eager, with chunked prefill enabled
and prefix caching disabled. FlashInfer autotuning is also disabled because
the CUSTOM backend does not use FlashInfer; otherwise vLLM runs an unrelated
zero-history full-prefill dummy batch during startup, which is outside this
decode-side backend's contract. The ordinary memory-profiling pass already
uses `skip_attn=True`. Mega defaults to FIFO history order, automatic block N,
eight communication SMs, and at most eight splits. The integration does not
change CUDA source and does not require a top-level
`vllm-flash-attention` submodule.

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

Arrival scaling divides original relative timestamps. Scales 1, 2, and 4 keep
request order/content fixed while increasing offered load.

Input and output lengths remain variable and come from each selected trace
row. Every request sets `min_tokens=max_tokens=output_length` and
`ignore_eos=true`, so EOS or another stop token cannot finish generation
before that row's requested output length. The client also treats any response
with a different token count as a failed request.

## Running and results

Run short smoke tests, then the formal 100-warmup/1000-measured KV-head matrix:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./scripts/smoke_vllm_dcp.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./benchmark_vllm_dcp_matrix.sh
```

The smoke wrapper uses the same default `MEGA_NUM_COMM_SMS=4,8,12,16,20`
communication-SM sweep as the formal matrix. Override it with a smaller set
for a quick check, for example `MEGA_NUM_COMM_SMS=8` or
`MEGA_NUM_COMM_SMS=4,8`.

Both the smoke and formal wrappers explicitly default and export
`VLLM_USE_FLASHINFER_SAMPLER=0`, because the custom attention backend does not
use FlashInfer's sampling kernel and this avoids an unrelated sampling JIT
during server startup. The smoke wrapper only requires a complete CUDA toolkit
when this is explicitly set to `1`; otherwise an already-built min-FA3
extension can run without sampling JIT. Set
`VLLM_USE_FLASHINFER_SAMPLER=1` only when the CUDA toolkit and headers are
available and that sampler path is intentionally being tested.

The formal wrapper defaults to `KV_HEADS=1,2,4` and
`MEGA_NUM_COMM_SMS=4,8,12,16,20`, matching the Mega comm-SM sweep used by
`benchmark_dcp_mega_arrival_matrix.sh`. It stores each KV-head configuration
under `benchmark_logs/vllm_dcp/kvh{1,2,4}`. Baseline services run once per
arrival scale; Mega gets one isolated service run per communication-SM value,
with runs recorded as `mega-comm_sm{N}-scale{S}`. Set comma- or
space-separated subsets when needed, for example `KV_HEADS=1,4` or
`MEGA_NUM_COMM_SMS="8 16"`.

Both wrappers default to port 8000. Set `PORT` when that endpoint is already
used by another service; the matrix rejects any existing listener before
launching vLLM and also verifies that the new server advertises the expected
model before sending a warmup request:

```bash
PORT=18000 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./scripts/smoke_vllm_dcp.sh
```

The smoke script keeps the production `TP=8`, `KVH=1`, `DCP=8` attention
topology but defaults to a one-layer synthetic Llama and
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

The formal script continues to default to 32 layers and 0.9 utilization. Its
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

Replace `mega` with either repository baseline; add `--dry-run` to print the
exact command. Each matrix cell records the service log, run manifest, raw token
timestamps/ITLs, request CSV, and summary. SSE uses `stream_interval=1` and
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
