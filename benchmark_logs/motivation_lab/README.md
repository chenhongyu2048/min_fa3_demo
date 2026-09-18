# Motivation figures

Run with the project's `plot` dependency group (matplotlib 3.11.0):

```bash
python benchmark_logs/motivation_lab/plot_motivation.py
```

The script reads `run.log` next to itself and writes four figures in both PNG
and vector PDF format to the same directory. Optional `--input FILE` and
`--output-dir DIR` arguments change those paths.

- `t1_t2`: two panels; T1 has Ring and AllGather, each with three bars per
  batch shape. Hatching identifies AllGather. T2 has three execution profiles.
  T1's first bar sums independently measured communication and computation
  p50 values. Sequence length labels are global, with 128K total tokens.
- `t3`: five datasets, four placement colors, and four component hatches.
  Input values already average 20 cases per dataset/placement. Stack order is
  OtherBwd, OtherFwd, CoreFwd, CoreBwd. Labels give total layer milliseconds.
- `d1_case_000003` and `d1_case_000009`: every rank-0 CTA is drawn in increasing
  ID order, with critical-wave above FA3-native/FIFO. Both panels use the same
  time scale within a case. Actual phase starts and gaps are preserved, and
  each method's first recorded phase defines its own origin. Trace phase
  intervals include waits and synchronization; they are not useful-work counts.

The bottom A2A strip concatenates seven distinct stage durations in the local
vLLM implementation's execution order. It excludes aggregate windows, aliases,
and total time to avoid double counting. It is a cumulative-duration strip,
not a timestamped timeline: inter-stage gaps are not available in this export.
The strip has its own axis, and its sum and measured diagnostic end-to-end
duration are both labeled. A2A diagnostics are from rank 3 in this export,
whereas the CTA traces are from rank 0. Diagnostic timings and uninstrumented
graph p50 are separate measurements and must not be treated as identical.
