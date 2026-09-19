# Motivation v3: T1 and D1

Based on `main` commit `45a476c7`, on branch `motivation-v3`. These runners use
the main implementation; they do not import the v2 experiment controls.

## Measurement boundary

Every measured sample, including traced and event-instrumented samples, uses:

```text
CUDA synchronize -> WORLD barrier -> CUDA synchronize
start event -> one operation / graph replay -> end event -> wait for end
read trace / events, then gather results outside timing
```

Events are created and materialized before sampling. Warmup precedes the
sample loop. The WORLD barrier is outside CUDA Graph capture and outside the
event interval. Kernel-internal phase barriers remain part of the algorithm.
This excludes prior GPU work and the initial barrier, but does not align
different GPUs' clocks or guarantee simultaneous physical launch times.

Defaults are 40 warmups and 60 samples. Each configuration is measured
separately. Results retain every rank's samples. Aggregation takes the maximum
rank duration **for each sample**, then computes p50/p90, rather than taking
the maximum of independently calculated rank percentiles.

## T1

Global tokens = 131072; batch sizes = 1, 2, 4, 8, 16. Each sequence has
`131072 / batch_size` tokens globally and `131072 / batch_size / world_size`
tokens per rank. Inputs use BF16, causal zigzag sharding, QH=32, KVH=8, D=128.

The runners call `VarlenAllGatherForward` and `fa3_ring_forward` from
`ring_test/hybrid_forward_baselines.py`. Allgather retains the main baseline's
`index_select` reorder and KV-head pipeline with `heads_k_stride=1`.
The shared backend selector prefers external FA3; its existing min_fa3 fallback
is retained and the actual selection is recorded in `t1.json`.

| Mode | Measured work |
| --- | --- |
| comm_only | Send packing, real transfers and waits; no attention, KV reorder or output merge |
| comp_only | Real pre-gathered KV, existing reorder/slicing, FA3 and output merge; no transfers or extra snapshot copies |
| serial | Real communication and computation with overlap disabled by stream dependencies |
| overlap | Original baseline pipeline |

KV snapshots are prepared before timing. Ring snapshots are in ring-step order;
allgather snapshots are rank-major chunks. Serial is measured directly, not
estimated by adding isolated communication and computation. T1 runs the eager
Python baselines with CUDA event timing. Each non-communication-only mode is
checked against ring output before timing, using atol=rtol=0.02.

## D1

Uses `case_000003` and `case_000009` directly from
`dcp_test/mega_dcp_trace_cases.jsonl`. Their Q/history arrays match the v2
candidate manifest selected by `benchmark_logs/motivation_lab/run.log`:

| Case | Batch | Query lengths | Total Q | Total history |
| --- | ---: | --- | ---: | ---: |
| case_000003 | 42 | 42 × 16 | 672 | 404697 |
| case_000009 | 31 | 29 × 16 + 2 × 4096 | 8656 | 355534 |

Use one node with 8 Hopper GPUs: TP=8, DCP=2, QH=32, KVH=4, D=128. Each rank
has four query heads. Q=16 requests retain the source workload's chunk-path
semantics; they are not converted to Q=1 decode.

Both Mega variants use `scheduler_heuristic=False`,
`reorder_history_override=False`, native splits, FIFO and the same BlockN.
The runner compares their actual packed metadata images (excluding replay
epochs), and checks output and LSE against vLLM A2A before/after repeated graph
replay. The mixed variant retains four communication CTAs. Metadata generation
retains that configuration for both variants, including final-task granularity;
only execution changes to all CTAs in the phased variant.

`DCPMegaAttentionRunner` adds `execution_mode="mixed"|"phased"` and
`record_sm_trace=False`. Existing defaults remain mixed. Phased execution uses
fixed-shape `capture_last_forward()`, not the dynamic serving graph path or
the production phase-timestamp counters. Only the D1 DCP=2, Hq_local=4 CUDA
specializations are instantiated for phased execution, to avoid multiplying
compilation work for unrelated topologies.

The six phases are Q allgather, chunk attention, history attention, history
combine/publish, A2A pull/receive, and final combine. Chunk/history use separate
contiguous slices of the same descriptors, retaining original completion IDs.
The attention ticket counter is reset before each attention phase, and all
phase worker counts/communication strides use the full grid.
Before re-entering FA3, each attention phase consumes the outstanding
QueryEmpty signal and WG1 scheduler token left for a prospective next tile.
Without this drain, a second `mma_init()` can deadlock on named barriers,
including on CTAs that received no chunk tile. Mixed execution is unchanged.

Each phase ends with CTA convergence followed by `grid.sync()`. At Q allgather
and A2A entry, after prior local completion, CTA0 publishes the current epoch
and waits only for its own DCP peer; another `grid.sync()` releases all CTAs
before work begins. IPC slots 1 and 2 are reserved for these rendezvous; slot 0
keeps the existing launch pre/post protocol. Replay uses the existing monotonic
device epoch. All TMA completion waits and system release/acquire publication
from the original helpers remain in place.

The phased kernel uses cooperative launch. Its first eager invocation sets
shared-memory attributes and checks occupancy before capture; shared-memory
reservation enforces at most one resident CTA per SM, and occupancy must allow
exactly one. The grid contains the device's SM count. This follows the
[CUDA grid synchronization requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html).

### Five measurements

| Configuration | Purpose |
| --- | --- |
| mixed_trace_off | Original native-planned Mega total graph time |
| phased_trace_off | Phase-serialized total graph time with all required barriers |
| phased_trace_on | Same phased algorithm plus per-CTA boundary timestamps |
| vllm_a2a_events_off | Baseline total graph time without internal events |
| vllm_a2a_events_on | Separate baseline diagnostic with existing phase events |

All five use CUDA Graphs and return both output and LSE. Graph timings include
their required state resets and synchronization. Trace-on/off change only
instrumentation, not required barriers. Allocation/capture/trace copies and
the WORLD start barrier are untimed. VLLM uses its existing single-stream
`overlap_q_allgather=False` baseline; diagnostic stage rows are not assumed to
be an additive decomposition.

### Trace interpretation

The preallocated trace is int64 `[6, num_sms, 7]`, indexed by phase and CTA:
`sm_id, entry_ns, work_start_ns, work_end_ns, exit_sync_done_ns,
rank_sync_enter_ns, rank_sync_exit_ns`. Only thread 0 records boundary
`%globaltimer` timestamps, and only CTA0 records the two rank-barrier intervals.
The trace is compile-time disabled in the untraced specialization. There are
no per-task trace writes or trace atomics. Every warp starts work after the
leader's start marker; the corresponding CTA barrier also exists in trace-off.

The decoder requires distinct physical SM IDs for all CTAs, stable IDs across
phases, monotonic timestamps, and grid exit after every CTA's work end.
Physical SM IDs need not be consecutive. It reports:

- `work_span_ns`: latest work end minus earliest work start on this GPU.
- `tail_idle_sm_ns`: sum over SMs of latest work end minus that SM's work end.
- `exit_wait_sm_ns`: sum of end-to-grid-release durations, including barrier cost.
- `entry_wait_sm_ns`: sum of entry-to-start durations, including rank release wait.
- `rank_wait_ns`: CTA0's measured IPC barrier duration.

These durations overlap and must not be summed as independent time components.
Different GPUs have separate clock origins. Figures use each rank's local
origin, one panel per rank; panels are not a globally aligned timeline. The
plotted replay is the sample nearest the trace-on rank-max p50. All replay
traces are retained in per-rank JSON files.

Phased minus mixed is the cost of forced phase execution, including lost
overlap, changed worker allocation and barrier cost. It is not purely a wave
quantization metric. The paired task images and traces expose tail idle time
without claiming that all gains come from one cause. VLLM is an external
implementation reference; this experiment does not measure its per-SM
occupancy or remove the resource footprint of Mega-DCP.

## Run on a CUDA node

Use the repository's locked environment and normal build procedure:

```bash
uv sync --frozen --no-install-project --group build --group transformer-layer
make PYTHON=.venv/bin/python

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/torchrun --standalone --nproc_per_node=8 \
  -m motivation.t1 --output-dir benchmark_logs/motivation_v3

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/torchrun --standalone --nproc_per_node=8 \
  -m motivation.d1 --output-dir benchmark_logs/motivation_v3

.venv/bin/python -m motivation.analyze --input-dir benchmark_logs/motivation_v3 \
  --output-dir benchmark_logs/motivation_v3/analysis
```

Analysis writes T1/D1 CSV summaries, per-rank/per-sample D1 stage statistics,
vLLM event statistics, trace overhead in milliseconds/percent, and PNG/PDF
SM timelines for both cases. Matplotlib comes from the existing environment.
Use `--no-plots` for dependency-free table export.

### Compact plotting log

After measurement, export a single portable log using only the Python standard
library; CUDA, PyTorch and matplotlib are not needed:

```bash
python3 -m motivation.export_plot_data --input-dir benchmark_logs/motivation_v3 \
  --output benchmark_logs/motivation_v3/plot_data.log
```

Direct execution is also supported: `python3 motivation/export_plot_data.py ...`
from the repository root, or `python3 export_plot_data.py ...` from `motivation/`.
Input/output paths are relative to the current working directory. `--output-dir`
is accepted as an alias for `--output`; both take the full output **file** path.

The `.log` uses JSON Lines: parse each line with `json.loads(line)`. It supports
directories containing T1, D1, or both. Metadata lines retain the configuration,
GPU environment and timing convention. Data lines contain:

- `t1_timing`: each batch/method/mode, per-sample `rank_max_ms`, and p50/p90 in
  milliseconds. Rank reduction happens before percentiles. The measured serial
  configuration is retained independently of comm-only and comp-only.
- `d1_timeline`: one complete `phased_trace_on` timeline per case. First choose
  the sample nearest trace-on rank-max p50, then the rank with the largest CUDA
  event duration in that sample. Ties choose the lowest sample/rank index.
  `sample_index` is zero-based; `rank` is the logical distributed rank.
  `selected_graph_ms` and five-configuration p50/p90 summaries retain timing
  context, including the trace-on/off comparison.

D1 stores physical SM IDs once in `sm_ids_by_cta`. `phase_times_ns` is indexed
by `[phase][CTA][field]`, with phases and the four time fields defined in the
`d1_metadata` line. Each row retains entry, work start, work end and exit-sync
completion. `rank_sync_ns` rows contain `[phase_index, enter_ns, exit_ns]` for
CTA0 at Q allgather and A2A entry; CTA0's physical SM is `sm_ids_by_cta[0]`.
All timestamps are integer nanoseconds relative to the selected rank's earliest
phase entry; subtraction occurs before any floating-point conversion. Divide
by 1000 for a microsecond axis. Entry wait, work and exit wait can be drawn
directly from adjacent boundaries. Within each phase, `max(work_end) - work_end`
gives each SM's tail idle duration.

Only the selected rank's JSON file is read for each D1 case. Full task tables,
other ranks/replays and vLLM diagnostic events are omitted. The exported log is
sufficient for T1 time-comparison plots and one representative per-SM timeline
per D1 case; it cannot reconstruct all-rank imbalance or trace distributions.
Each timeline preserves one actual execution, without mixing per-SM maxima
across ranks or aligning independent GPU clocks. `motivation.analyze` continues
to read the original JSON directory, not this compact log.

For a four-GPU smoke test, select four non-MIG devices using
`CUDA_VISIBLE_DEVICES` and use `--nproc_per_node=4`. T1 retains 131072 global
tokens; D1 additionally requires `--smoke`, which uses TP=4, DCP=2, QH=16 and
KVH=2. This preserves four query heads per rank and the same CUDA
specializations. Use `--warmup 2 --iters 3` for both runners. Shared GPUs are
sufficient for correctness and replay checks, but their timing samples are
not formal performance measurements.

For the initial hardware check, run D1 with `--warmup 2 --iters 3
--check-rank-delay` in a separate output directory. The delay test runs after
formal samples: it sleeps rank 0 on the GPU **after** the untimed WORLD barrier,
then replays the graph. It checks that rank 1's Q-entry IPC wait increases by
more than 1 ms relative to the normal sample and exceeds the other DCP groups'
wait increments by more than 1 ms. It stores that diagnostic separately from
formal timing. The fixed sleep-cycle count is a test stimulus, not a calibrated
time measurement. The test targets Q entry; it does not inject a delay between
the history and A2A phases.

The delay diagnostic captures a separate phased graph with
`capture_last_forward(run_pre_barrier=False)`. It relies on the internal
Q-entry rendezvous and retains workspace resets, epoch increments and the
post-barrier. Otherwise the existing outer pre-barrier absorbs the delay
before it reaches Q entry. Its undelayed reference uses the same diagnostic
graph. All five measured configurations retain the outer pre-barrier.

## Local validation and remaining hardware work

```bash
python3 -m unittest motivation.test_experiments -v
python3 -m motivation.t1 --help
python3 -m motivation.d1 --help
python3 -m motivation.analyze --help
git diff --check
```

The focused CPU suite checks fixed workloads, native descriptor domains and
completion IDs for both DCP ranks, actual baseline scheduling bodies with fake
tensor/communication operations, timing-boundary order, per-sample rank-max
aggregation, trace invariants and CSV export. It does not execute GPU tensor
arithmetic. No unrelated full suite is required for this change.

The Mac development environment has no CUDA. Remote four-H20 smoke validation
has now covered compilation, cooperative graph capture, numerical correctness,
repeated IPC replay, actual SM coverage, Q-entry rank-delay behavior and timeline
export; see [SMOKE_H20.md](SMOKE_H20.md) for commands, results and the FA3
named-barrier re-entry fix. Eight-GPU formal measurements and isolated-GPU trace
overhead characterization remain pending. Shared-GPU smoke timings are not
formal performance results.
