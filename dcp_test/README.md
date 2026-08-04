# DCP attention benchmarks

This directory contains the multi-rank dense and packed-varlen DCP benchmark
entry points. Like `ring_test/`, the commands are run from the repository root
with `torchrun` and keep the local min FA3 attention kernel fixed while
comparing DCP orchestration policies.

The benchmark files are:

- `benchmark_dcp.py`: dense BSHD decode and chunk-prefill comparison
- `benchmark_dcp_varlen.py`: packed-varlen decode or chunk-prefill comparison

The corresponding correctness matrices remain under `scripts/test_min_fa3/`:

- `scripts/test_min_fa3/test_min_fa3_dcp.py`
- `scripts/test_min_fa3/test_min_fa3_dcp_varlen.py`

CUDA Graph is enabled by default. Each fixed shape performs three eager
capture warmups before capture, then `--warmup` unmeasured graph replays and
`--iters` measured replays. Pass `--no-cuda-graph` to use eager execution.
Captured shapes bind tensor addresses and static scalar/varlen metadata; close
the graph before destroying its NCCL process group. The benchmark entry points
handle that close ordering internally.

The six dense labels are `ours_no_overlap`, `ours_overlap`,
`vllm_ag_rs_min_fa3`, `vllm_a2a_min_fa3`, `sglang_mha_ag_ar_min_fa3`, and
`full_kv_min_fa3`; packed-varlen appends `_varlen` to each label. Selecting
`--implementations vllm` runs both vLLM baselines. The A2A code is copied and
trimmed from vLLM commit `a89015c6df8eeb37a843b717c97a5be1355de83d` and its
ordinary GQA/MQA FlashAttention integration. It retains the packed-combine
design from PR #41160, graph-private per-call buffers from PR #45487, and the
FP32 LSE pack contract from PR #47801.

Packed-varlen also accepts the explicit experimental category
`--implementations mega`, reported as `dcp_mega_varlen`. It is chunk-only and
eager-only, so pass `--no-cuda-graph`. `--mega-block-n 128|176` and
`--mega-num-comm-sm N` select the isolated instance and communication-CTA
budget. The communication path is fixed to PackGQA with `Hq_local` 4 or 8 and
`[16,Hq_local,128]` Q/O tiles; there is no runtime communication-layout
selection. History combine performs the remote TMA store and publishes a
monotonic ready phase directly, so communication CTAs proceed from Q
all-gather to receive without a separate publish pass. The existing default method
list remains unchanged. Its first eager
correctness call builds and uploads fixed-shape metadata. Benchmark replays
reuse that device image. Internal CUDA events measure only pre-barrier plus the
persistent mega kernel; workspace reset and post-barrier remain required but
are outside the measured interval. Add `--mega-phase-timestamps` to include
optional `%globaltimer` milestones in JSON. Fused `history_combine_done` and
`publish_done` are intentionally identical; `publish_done` means every remote
ready release has been issued.

A2A stage timing separates pack, `all_to_all_single`, and unpack/FP32 base-e
LSE-weighted combine. `output_collective_ms` covers only the all-to-all. The
payload report distinguishes the full BF16 buffer
`DCP*T*H_local*(D+2)*2` from remote traffic excluding self-copy,
`(DCP-1)*T*H_local*(D+2)*2`; A2A LSE-all-gather and reduce-scatter fields are
zero. Dense JSON uses schema 4 and packed-varlen uses schema 3.

Dense example:

```bash
torchrun --standalone --nproc_per_node=8 --module dcp_test.benchmark_dcp \
  --qhead 32,64 --kvhead 2,4 --tp-size 8 --dcp-sizes 2,4,8 \
  --implementations ours,vllm,sglang --workload both \
  --decode-b 1,8,32 --chunk-b 1,4,16 \
  --seqlen 4096,16384,65536 --sq 8,32,128 \
  --headdim 128 --num-splits 0 --warmup 5 --iters 20
```

To launch the dense benchmark file directly through Python instead of the
`torchrun` executable, run this smaller two-GPU example from `dcp_test/`:

```bash
cd dcp_test
PYTHONPATH=.. python -m torch.distributed.run \
  --standalone --nproc_per_node=2 benchmark_dcp.py \
  --qhead 16 --kvhead 1 --tp-size 2 --dcp-sizes 2 \
  --implementations ours,vllm,sglang --workload both \
  --decode-b 1 --chunk-b 1 --seqlen 256 --sq 8 \
  --headdim 128 --num-splits 0 --warmup 1 --iters 3 \
  --no-mqa-control
```

The distributed Python launcher is required because the benchmark creates an
NCCL process group and validates `TP > Hkv`; plain single-process
`python benchmark_dcp.py` does not provide the required ranks.

Packed-varlen chunk example:

```bash
torchrun --standalone --nproc_per_node=8 --module dcp_test.benchmark_dcp_varlen \
  --b 3 --sq 1,8,32 --seqlen 129,1024,3131 \
  --qhead 32 --kvhead 1 --headdim 128 --tp-size 8 --dcp-size 8 \
  --workload chunk --implementations ours,vllm,sglang,full \
  --num-splits 0 --warmup 2 --iters 5
```

Packed-varlen decode requires every `--sq` value to be `1`. Both benchmarks
also accept `--output-json PATH`. See the repository `README.md` for topology,
timing boundary, method-label, and JSON-schema details. These are
attention-only, same-min-FA3 orchestration baselines, not native vLLM/SGLang
serving-runtime benchmarks.

Packed-varlen performs an eager full-KV correctness precheck by default. Pass
`--no-check` for performance-only runs. Mega still performs one untimed setup
call in that mode to build and upload its reusable metadata, but it does not
compare the output or include that call in the measured interval.

Experimental mega-kernel smoke example:

```bash
torchrun --standalone --nproc_per_node=8 --module dcp_test.benchmark_dcp_varlen \
  --b 3 --sq 1,8,32 --seqlen 129,1024,3131 \
  --qhead 32 --kvhead 1 --headdim 128 --tp-size 8 --dcp-size 8 \
  --workload chunk --implementations mega,vllm,full --no-cuda-graph \
  --mega-block-n 128 --mega-num-comm-sm 8 \
  --num-splits 0 --warmup 5 --iters 20
```

The fixed eight-GPU correctness matrix runs DCP 2/4/8, Hq-local 4/8,
BlockN 128/176, split/nosplit, and auto-split cases in a single process group.
It uses ragged Q/history lengths,
non-16 tails, BF16 O, FP32 LSE, and two forwards per case to cover workspace
reuse:

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/test_min_fa3/test_dcp_mega_varlen_multi_rank.py --matrix
```

## Measured unified-queue mega run (2026-08-03)

This rerun used the unified attention queue and single FA pipeline working tree
on `zkrh-58` with 8 x H100 80GB HBM3, CUDA 12.8, PyTorch 2.10.0, BF16, and
head dimension 128. Every method passed its eager full-KV correctness precheck.
Each latency sample is the maximum over all eight global ranks; the tables show
milliseconds. The final runs use 500 eager warmups and 100 measured calls
because 5-20 warmups were insufficient to stabilize clocks for sub-ms jobs.

Mega timings use CUDA events created inside the C++ binding after argument
validation. The start event immediately precedes the pre-phase barrier and the
end event immediately follows the mega kernel. Host metadata generation,
metadata H2D copy, workspace reset, Python/C++ dispatch delay, and post-phase
barrier are excluded. The other methods retain their existing dcp_test timing
boundaries, so comparisons against A2A/AG+RS are informative but not identical
end-to-end orchestration measurements.

The imbalance workload was `B=3`, `Sq=[1,1,1]`,
`Sk_history=[4097,32769,180225]`, `Hq=32`, `Hkv=1`, and `TP=8`. A representative
command is:

```bash
torchrun --standalone --nproc_per_node=8 --module dcp_test.benchmark_dcp_varlen \
  --b 3 --sq 1,1,1 --seqlen 4097,32769,180225 \
  --qhead 32 --kvhead 1 --headdim 128 --tp-size 8 --dcp-size 8 \
  --workload chunk --implementations mega,vllm,full --no-cuda-graph \
  --mega-block-n 176 --mega-num-comm-sm 8 --num-splits 16 \
  --warmup 500 --iters 100
```

Auto-split selected an effective upper bound of 109. With BlockN 176, its
per-sequence chunk/history splits were `[1,1,1]` and `[3,17,94]`.

| DCP | BlockN | mega p50 / p90 | vLLM A2A p50 | vLLM AG+RS p50 | full-KV p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 176 | 0.1930 / 0.1977 | 0.3341 | 0.3404 | 0.0708 |
| 4 | 176 | 0.1796 / 0.1832 | 0.3368 | 0.3360 | 0.0703 |
| 8 | 176 | 0.1539 / 0.1600 | 0.3452 | 0.3342 | 0.0712 |
| 8 | 128 | 0.1680 / 0.1757 | 0.3498 | 0.3284 | 0.0717 |

The split sweep fixed DCP=8, BlockN=176, and 8 communication CTAs:

| requested splits | attention descriptors | mega p50 / p90 | vLLM A2A p50 | full-KV p50 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 0.2756 / 0.2826 | 0.4066 | 1.7636 |
| 2 | 9 | 0.1853 / 0.1880 | 0.3544 | 1.0160 |
| 8 | 22 | 0.1037 / 0.1067 | 0.3406 | 0.2786 |
| 16 | 38 | 0.1009 / 0.1049 | 0.3443 | 0.1579 |
| 32 | 55 | 0.1136 / 0.1225 | 0.3448 | 0.0938 |
| 64 | 87 | 0.1539 / 0.1567 | 0.3583 | 0.0713 |
| 128 | 117 | 0.1520 / 0.1545 | 0.3413 | 0.0696 |
| auto (109) | 117 | 0.1539 / 0.1600 | 0.3452 | 0.0712 |

For the best measured split value of 16, `num_comm_sm=4/8/16` produced mega
p50 `0.1058/0.1009/0.0983` ms. The 16-CTA point was only 2.6% faster than the
default 8-CTA point, which is too small and workload-specific to justify a
default change. At the default 8 communication CTAs, mega was 3.41x faster
than the same-run vLLM A2A p50 of 0.3443 ms under the differing timing
boundaries described above. The legacy odd/even-role mega binary was not
available, so these numbers compare the unified queue against the existing
dcp_test baselines, not against a reconstructed old mega implementation.

## Measured eight-GPU reference run (2026-08-02)

This section records one complete performance run of all six methods in both
eager and CUDA Graph modes. It is a measured reference for the shapes below,
not a claim about native vLLM or SGLang serving performance. The run used the
working tree containing the A2A implementation documented above; the recorded
repository HEAD was `8432d61df9bbbbf2c04292d12097f8ea4c794971`.

### Environment and timing

- Host: `zkrh-58`
- GPU: 8 x NVIDIA H100 80GB HBM3, compute capability 9.0
- CUDA runtime: 12.8
- PyTorch: `2.10.0+cu128`
- NCCL: 2.27.5
- Python: 3.12.13
- vLLM source commit: `a89015c6df8eeb37a843b717c97a5be1355de83d`
- SGLang source commit: `8d6549bc4039d33635844495d86684677a4f0df8`
- Dtype/head dimension: BF16/128
- Split policy: `--num-splits 0`
- Warmup and samples: `--warmup 5 --iters 20`

Physical GPUs 0-7 were idle immediately before launch and were held by one
repo-local `flock` for a continuous fail-fast batch. The same eight GPUs were
used for all six jobs. Every method performs an eager correctness precheck
before measurement. CUDA Graph mode performs three additional eager capture
warmups, captures a fixed shape, runs five unmeasured graph replays, and then
measures 20 replays. Eager mode runs five unmeasured calls followed by 20
measured calls. Every latency sample is reduced to the maximum across all
eight global ranks before p50/p90 aggregation.

The measured interval starts after Q/K/V are ready and ends when the BF16
local-head output is ready. It excludes projection, input/cache construction,
packing/sharding performed by the benchmark input builder, and serving-engine
overhead. The full-KV baseline has no DCP collective but replicates complete
KV state on each relevant rank; its latency and memory behavior should be
interpreted together.

### Dense matrix

The dense run used the command shown in the earlier dense example, plus
`--output-json`. CUDA Graph used the default `--cuda-graph`; the eager run used
the same command with `--no-cuda-graph`. Dense accepts implementation
categories `ours,vllm,sglang` and always adds `full_kv_min_fa3`, so `full`
must not be passed to its `--implementations` option.

The candidate topology matrix was `Hq=32/64`, `Hkv=2/4`, `TP=8`, and
`DCP=2/4/8`. Production topology checks admitted six GQA topologies and wrote
six rejected combinations as structured `skipped_topology` records. Each
legal topology ran these 36 workload shapes:

- Decode: `B=1/8/32`, `Sq=1`, `Sk=4096/16384/65536`
- Chunk: `B=1/4/16`, `Sq=8/32/128`, `Sk_history=4096/16384/65536`

This produced 216 GQA cases. Two fixed `Hq=64,Hkv=1,TP=8,DCP=8` MQA controls
were added: decode `B=8,Sq=1,Sk=16384` and chunk
`B=4,Sq=128,Sk_history=16384`.

The following values are geometric means of global-rank-max p50 latency over
the 216 GQA cases, in milliseconds. The MQA controls are reported separately
and are not included in these means. `Eager / Graph` greater than one means
CUDA Graph was faster.

| Method | CUDA Graph (ms) | Eager (ms) | Eager / Graph |
| --- | ---: | ---: | ---: |
| `full_kv_min_fa3` | 0.0729 | 0.0634 | 0.87x |
| `ours_no_overlap` | 0.1338 | 0.2815 | 2.10x |
| `ours_overlap` | 0.1280 | 0.3682 | 2.88x |
| `vllm_ag_rs_min_fa3` | 0.1445 | 0.3205 | 2.22x |
| `vllm_a2a_min_fa3` | 0.1421 | 0.3172 | 2.23x |
| `sglang_mha_ag_ar_min_fa3` | 0.1948 | 0.4291 | 2.20x |

CUDA Graph p50 geometric means split by workload were:

| Method | Decode (ms) | Chunk (ms) |
| --- | ---: | ---: |
| `full_kv_min_fa3` | 0.0747 | 0.0722 |
| `ours_no_overlap` | 0.0940 | 0.1505 |
| `ours_overlap` | 0.0968 | 0.1405 |
| `vllm_ag_rs_min_fa3` | 0.0987 | 0.1641 |
| `vllm_a2a_min_fa3` | 0.0982 | 0.1608 |
| `sglang_mha_ag_ar_min_fa3` | 0.1269 | 0.2247 |

The two MQA-control p50 measurements were:

| Method | Graph decode | Eager decode | Graph chunk | Eager chunk |
| --- | ---: | ---: | ---: | ---: |
| `full_kv_min_fa3` | 0.0614 | 0.0532 | 0.0906 | 0.0808 |
| `ours_no_overlap` | 0.0956 | 0.2221 | 0.2761 | 0.3336 |
| `ours_overlap` | 0.0960 | 0.2947 | 0.2595 | 0.4203 |
| `vllm_ag_rs_min_fa3` | 0.1027 | 0.2442 | 0.2971 | 0.3662 |
| `vllm_a2a_min_fa3` | 0.0926 | 0.2880 | 0.2652 | 0.3583 |
| `sglang_mha_ag_ar_min_fa3` | 0.1326 | 0.3523 | 0.4288 | 0.5194 |

For the dense GQA matrix:

- CUDA Graph made the DCP methods 2.10x-2.88x faster than eager. The very
  small, communication-free full-KV calls were instead 13% faster in eager.
- Under CUDA Graph, overlap was 4.6% faster than non-overlap over all GQA
  cases and 7.2% faster for chunk. It was 2.9% slower for decode.
- Under CUDA Graph, A2A was 1.7% faster than AG+RS overall and 2.1% faster for
  chunk. The chunk advantage was 1.1% at DCP=2 and 4.2% at DCP=4.
- In eager mode, multi-stream overlap was 30.8% slower than non-overlap over
  the full matrix because the launch/synchronization cost was not amortized.
- Full-KV won 192 of 216 CUDA Graph GQA cases. A DCP method won the remaining
  24 cases, primarily large-batch, long-context cases. The largest DCP
  latency advantage was 2.26x for
  `Hq=32,Hkv=2,DCP=4,decode,B=32,Sk=65536`: `ours_no_overlap` took
  0.1700 ms versus 0.3841 ms for full-KV.
- In eager mode a DCP method won 6 of 216 cases; the maximum advantage was
  1.61x for the same large decode shape.

### Packed-varlen matrix

Packed-varlen used `Hq=32,Hkv=1,TP=8,DCP=8,B=3`. Decode used `Sq=1,1,1`
and cache lengths `129,1024,3131`, for `total_q=3`. Chunk used query lengths
`1,8,32` and history lengths `129,1024,3131`, for `total_q=41`. Decode cache
lengths include the current token; chunk history lengths exclude the supplied
chunk. Both workloads selected `--implementations ours,vllm,sglang,full` and
were run once with `--cuda-graph` and once with `--no-cuda-graph`.

Global-rank-max p50 latency in milliseconds was:

| Method | Graph decode | Eager decode | Graph chunk | Eager chunk |
| --- | ---: | ---: | ---: | ---: |
| `full_kv_min_fa3_varlen` | 0.0480 | 0.0315 | 0.0356 | 0.0317 |
| `ours_no_overlap_varlen` | 0.0914 | 0.2351 | 0.1259 | 0.3076 |
| `ours_overlap_varlen` | 0.0927 | 0.3046 | 0.1119 | 0.4021 |
| `vllm_ag_rs_min_fa3_varlen` | 0.0990 | 0.2622 | 0.1385 | 0.3565 |
| `vllm_a2a_min_fa3_varlen` | 0.1140 | 0.2982 | 0.1255 | 0.3627 |
| `sglang_mha_ag_ar_min_fa3_varlen` | 0.1538 | 0.3401 | 0.1916 | 0.4868 |

For the packed chunk CUDA Graph case, overlap was 12.5% faster than
non-overlap and A2A was 10.3% faster than AG+RS. The three A2A p50 stages were
0.0077 ms pack, 0.0159 ms `all_to_all_single`, and 0.0059 ms unpack/combine.
For packed decode, A2A was 15.2% slower than AG+RS because only three query
tokens were available to amortize packing and collective fixed costs.

The complete A2A stage p50 measurements were:

| Workload/mode | Pack (ms) | All-to-all (ms) | Unpack/combine (ms) | End-to-end (ms) |
| --- | ---: | ---: | ---: | ---: |
| Decode CUDA Graph | 0.0075 | 0.0127 | 0.0059 | 0.1140 |
| Decode eager | 0.0355 | 0.0911 | 0.0365 | 0.2982 |
| Chunk CUDA Graph | 0.0077 | 0.0159 | 0.0059 | 0.1255 |
| Chunk eager | 0.0356 | 0.0766 | 0.0347 | 0.3627 |

### Result artifacts

The full raw samples, p50/p90 stage timings, communication payloads, topology
records, execution metadata, and environment details are stored under the
gitignored local directory
`benchmarks/results/dcp_six_methods_8gpu_20260802/`:

- `dense_cuda_graph.json` and `dense_eager.json`
- `varlen_decode_cuda_graph.json` and `varlen_decode_eager.json`
- `varlen_chunk_cuda_graph.json` and `varlen_chunk_eager.json`
- Matching `.log` files for all six jobs

All six jobs completed successfully. The selected GPUs returned to zero
reported utilization and zero allocated memory after the batch.
