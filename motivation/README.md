# MegaCP motivation experiments

These entry points are the small, reproducible measurements used to motivate
the paper.  Run them on an allocated full Hopper node from the repository root;
they do not SSH implicitly and they do not run the evaluation matrix.

For a formal batch (as opposed to a two-iteration smoke), use one of the
explicit repository-root launchers:

```bash
./motivation_lab_4gpu.sh
./motivation_lab_8gpu.sh
```

Both run complete T1/T2/T3/D1 batches with the paper's steady-state defaults
and distinct `motivation_4gpu_*` / `motivation_8gpu_*` result directories.
Each launcher contains its complete validation and experiment sequence; it
does not source, copy, or execute another launcher during a long batch.
The four-GPU launcher uses `motivation/t1_uniform_cases.json` and D1
CP4/KVH1,2.  The eight-GPU launcher uses
`motivation/t1_uniform_cases_cp8.json` and D1 CP8/KVH1,2,4.  T1 and T2 share
the same five uniform cases within each launcher.
Set `RUN_D1=1` and provide matching full homogeneous GPUs through
`D1_GPU_IDS_4` or `D1_GPU_IDS_8`
to run the opt-in D1 CTA/SM trace.  The current frozen D1 manifest is a DCP4
selection from the historical vLLM A2A Graph p50 median: two
`decode_only_q16` cases (`case_000017`, `case_000064`) and two mixed cases
(`case_000003`, `case_000078`).  Each case runs Mega twice (critical-wave
optimized and FA3-native/FIFO control) and vLLM A2A once in CUDA Graph mode;
the latter records its CUDA-event phase breakdown.  D1 currently traces the
Mega persistent kernel; baseline per-CTA traces and D2 Nsight Compute
collection remain outside the implemented batch and are explicitly marked
pending.  The runner supports CP4/KVH1,2 and CP8/KVH1,2,4; the default D1
list is CP4/KVH1,2 because the checked-in H20 allocation is four GPUs.  For
CP8 use `D1_TOPOLOGIES=8:1,8:2,8:4 D1_GPU_IDS_8=0,1,2,3,4,5,6,7`.
The physical topology is always `TP=CP`, `QH=32`, and
`DCP=TP/KVH`: CP4/KVH1 uses DCP4, CP4/KVH2 uses two DCP2 groups,
CP8/KVH1 uses DCP8, CP8/KVH2 uses two DCP4 groups, and CP8/KVH4 uses four
DCP2 groups.  The JSON records every actual DCP group explicitly.

Build on the CUDA node:

```bash
make PYTHON=.venv/bin/python
```

The checked-in D1 case manifest was generated reproducibly from the historical
DCP4 phase summary with:

```bash
.venv/bin/python motivation/select_d1_cases.py \
  --summary-csv benchmark_logs/bench_dcp/20260812-125820-arrival4-phases-graph/mega_phase_timestamp_summary.csv \
  --cases-jsonl benchmark_logs/bench_dcp/20260812-125820-arrival4-phases-graph/results/arrival_4/dcp_4/trace/cases.jsonl \
  --output-json motivation/d1_cases_dcp4.json --dcp-size 4
```

Core examples:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  motivation/training_overlap.py \
  --qhead 32 --kvhead 8 --headdim 128 --mode causal \
  --case-manifest motivation/t1_uniform_cases.json --case-id context65k_b4 \
  --warmup 5 --iters 20

CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  motivation/training_step.py \
  --qhead 32 --kvhead 8 --headdim 128 \
  --case-manifest motivation/t1_uniform_cases.json --case-id context65k_b4 \
  --num-comp-sm 0 --num-comm-sm 0 --warmup 5 --iters 20

`--num-comp-sm 0` selects the available SMs after the homogeneous-device
preflight (subtracting the requested communication SMs).  Do not hard-code
SM counts across GPU models.  Mega/TK VMM also requires all participating devices to be
full, non-MIG, unoccupied homogeneous GPUs.

python motivation/load_balance.py --datasets arxiv,github,pile,freelaw,prolong \
  --target-tokens 131072 --num-cases 30 --world-size 8

CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  motivation/decode_sm_trace.py \
  --case-manifest motivation/d1_cases_dcp4.json \
  --qhead 32 --kvhead 1 --tp-size 4 --warmup 5 --iters 3 \
  --output-json benchmark_logs/motivation/d1.json \
  --output-jsonl benchmark_logs/motivation/d1.jsonl
```

For the pending D2 experiment, first inspect the exact coordinated Nsight
Compute command:

```bash
./motivation/pending/profile_memory.sh --method mega --workload chunk --dcp-size 8 --dry-run
```

The output is diagnostic.  A profiler run must not be used as the normal
latency result, and unsupported DRAM/L2 counters remain explicitly missing.
On the current H20-2 setup, D2 is pending: multi-rank CUDA Graph/NCCL NCU
collection was not reliable, so these entry points do not claim measured HBM
round-trip reduction.

T1/T2 formal batches use five fixed uniform shapes borrowed from the 64K
global-context slice of the cp-uniform suite.  The CP4 manifest uses local
lengths 16K/8K/4K/2K/1K for B=1/2/4/8/16; the CP8 manifest uses
8K/4K/2K/1K/512 for the same batch sizes.  T3 uses the five dataset
distributions in one paired JSON artifact.
T2 records `step_external_reduce`, `step_fused_reduce`, and
`linear_queue_recycle`, separating the external-reduce, fused-step, and
continuous-segment execution organizations.
T1 additionally records controlled COMM-ONLY/COMP-ONLY co-run component
durations on independent CUDA streams.  Those component times are diagnostic
resource-contention measurements; the legal SERIAL/OVERLAP calls remain the
complete-time comparison and the components are never summed into a critical
path.

Motivation deliberately leaves backward four-mode overlap, ready-gate
ablations, full comm-SM sweeps, projection/MLP attribution and end-to-end
serving out of these entry points.

## Plotting a completed motivation batch

The completed H20 batch can be turned into the two compact paper figures with
the plotting entry point below.  It validates the recorded sample counts and
T2 work counters before drawing; D2 is intentionally absent because the
current run has no reliable NCU DRAM/L2 measurements.

```bash
# The plot group is optional in the locked environment.
uv sync --frozen --no-install-project --group plot
MPLCONFIGDIR=/tmp/motivation_mpl_cache \
.venv/bin/python motivation/plot_motivation.py \
  --run-dir benchmark_logs/motivation/20260906_042142 \
  --output-dir benchmark_logs/motivation/20260906_042142/figures
```

This writes `fig1_training_load_balance.{png,pdf}` (T1/T2/T3) and one
`fig2_decode_stage_trace_<topology>.{png,pdf}` per D1 topology (or the legacy
single `fig2_decode_stage_trace.{png,pdf}`).  T1/T2 use rank-max p50 CUDA-event
timings.  T3 panels are static accounting over five shared 30-case manifests,
one per dataset, not measured attention latency.  D1 shows the Mega
persistent-kernel CTA/SM trace and a vLLM A2A Graph phase timeline from one
representative critical-rank replay rather than a case-comparison bar chart;
NCCL internal CTA intervals and HBM traffic are not inferred.
