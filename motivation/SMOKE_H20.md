# H20 four-GPU smoke validation — 2026-09-18

Source base: `45a476c7`, branch `motivation-v3`, with the working-tree changes.
The remote repository uses the same branch and existing uv environment.

## Environment and commands

- Host: H20-2, via the supplied login node.
- Physical GPUs: `0,4,5,6`; four non-MIG NVIDIA H20 devices, 78 SMs each.
- All selected GPU pairs support CUDA peer access; topology reports NV18.
- Python 3.12, PyTorch 2.11.0+cu128, CUDA Toolkit 12.8.
- GPUs are shared with other workloads, as authorized. No existing workload
  was stopped. Timing samples validate the measurement pipeline only.
- The system nvcc symlink caused automatic CUDA_HOME detection to select
  `/usr/local`; the successful build explicitly selected the installed toolkit.

Run from the repository root on the CUDA node:

```bash
CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=8 make PYTHON=.venv/bin/python
CUDA_VISIBLE_DEVICES=0,4,5,6 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m motivation.t1 --warmup 2 --iters 3 --output-dir benchmark_logs/motivation_v3_smoke
CUDA_VISIBLE_DEVICES=0,4,5,6 .venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m motivation.d1 --smoke --warmup 2 --iters 3 --check-rank-delay \
  --output-dir benchmark_logs/motivation_v3_smoke
```

The toolkit location is this node's environment setting, not a source-code
path requirement. D1 smoke uses TP=4, DCP=2, QH=16, KVH=2 and D=128; the
per-rank head count and phased CUDA specialization match the formal setup.
T1 retains 131072 global tokens and all five batch sizes.

## Completed checks

- Ten focused CPU tests passed on both the local machine and remote environment.
- Initial CUDA compilation and extension linking passed with CUDA_HOME set.
- T1: all 40 configurations completed; 480 per-rank timing samples retained.
  All compute-containing modes passed the runner's output comparison against
  ring overlap (atol=rtol=0.02). The selected FA3 backend was `min_fa3`.

## D1 debugging evidence

The first phased eager call stalled after Q allgather and chunk attention.
Snapshots showed all 78 CTAs had completed those phases, every Q-ready count
was 2, and all 42 chunk descriptors were marked complete. CUDA-GDB identified
producers waiting on QueryEmpty while MMA warps waited on Q data in the second
attention phase. The native mainloop leaves next-tile QueryEmpty and WG1
scheduler signals on exit. Phased execution must drain these before calling
mma_init again. The fix is scoped to phased execution. An intermediate
register-allocation change did not resolve the stall and was removed.

## D1 validation results

The named-barrier drain fixed the stall. Both cases passed all five CUDA Graph
configurations with two warmups and three samples, then passed an independent
replay check with five warmups and ten samples. The latter retained 400
per-rank timing samples; the original smoke retained 120.

- Output and LSE match vLLM A2A at atol=rtol=0.02, before and after graph replay.
- Mixed/phased serialized task images match after masking replay epochs.
- Native splits, FIFO, BlockN=128 and completion ID domains are preserved.
- Cooperative launch occupancy checks pass. Every traced phase covers 78
  distinct physical SMs per rank; CTA-to-SM mapping is stable through all six
  phases. Every exit timestamp follows every CTA's work end in its phase.
- Q-entry delay test: rank 1 wait increased by 10.280736 ms (case 003) and
  10.087232 ms (case 009). The other DCP group changed by at most 5.312 µs
  in absolute magnitude. Rank 0 was delayed by 20 million GPU clock cycles.
  The separate diagnostic graph skips the outer pre-barrier to expose the
  internal Q rendezvous; all five measured configurations retain it.
- PNG/PDF timelines and CSV stage/event summaries were generated. Both
  initial-smoke timeline PNGs were visually inspected.

The ten-sample replay-check p50 values below are in milliseconds and use the
per-sample maximum across ranks. These are shared-GPU smoke observations:

| Case | vLLM events off | vLLM events on | Mixed | Phased trace off | Phased trace on |
| --- | ---: | ---: | ---: | ---: | ---: |
| case_000003 | 0.243744 | 0.298352 | 0.267824 | 0.333696 | 0.336720 |
| case_000009 | 2.012320 | 2.064720 | 1.765264 | 2.085952 | 2.085008 |

Trace-on minus trace-off was +3.024 µs (+0.906%) for case 003 and −0.944 µs
(−0.045%) for case 009 in this repeat. The original three-sample run had a
+42.65% apparent gap for case 003, including two slow traced graph samples;
all original samples are retained. Neither small-sample run establishes a
formal trace-overhead bound under shared load.

For the exercised BlockN=128 specializations, ptxas reports 168 registers
and 16 named-barrier resources for trace-on and trace-off. The split-32
specialization reports a 232-byte stack frame in both versions, with spill
store/load byte counts 770/1068 (off) and 778/1080 (on). The nonsplit version
reports 216-byte frames, with 524/748 (off) and 552/780 (on). These compiler
counts are not dynamic per-replay memory traffic. Trace-on does alter generated
spill instructions, so timing on unshared hardware is still needed.

## Artifacts and remaining scope

All results and logs are under `benchmark_logs/motivation_v3_smoke/`:

- `t1.json`, `analysis/t1.csv`: all T1 configurations and raw samples.
- `d1.json`, `case_*.rank*.json`: original D1 samples, traces, metadata and delay test.
- `analysis/`: original-smoke CSV summaries and PNG/PDF timelines.
- `replay_check/`: independent ten-sample D1 run and its analysis.
- `build_barrier_drain.log`, `d1_barrier_drain.log`: final build and successful smoke.
- Earlier failure snapshots and CUDA-GDB logs retain the evidence for the fix.

No eight-GPU run, isolated-GPU performance claim, or A2A-boundary delay injection
was performed. The A2A barrier ran in every successful phased replay, but the
controlled rank-skew injection specifically validated Q entry. Formal timings
and trace-overhead conclusions require the planned eight-GPU environment.

