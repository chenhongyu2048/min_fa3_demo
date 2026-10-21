# H20 four-GPU correctness validation

Validation date: 2026-10-21. Source base: `144e748f`, branch `motivation-v3`,
with the CP motivation working-tree changes.

## Environment

- Login node: SSH alias `H20-HKUSTGZ`; CUDA node: `h20` / hostname `H20-2`.
- Checkout: `/home/LOCAL/shixuan/hongyu/tmp/test/min_fa3_demo`.
- Physical GPUs: `CUDA_VISIBLE_DEVICES=0,2,3,6`, four non-MIG NVIDIA H20 GPUs.
- Driver: 570.133.20. Python 3.12, PyTorch 2.11.0+cu128, CUDA 12.8.
- Backend: in-repo `min_fa3`, using the existing CUDA extension and environment.
- Causal BF16 attention; QH=32, KVH=8, D=128; seed=0.
- Other workloads share the node. GPU 7 has MIG enabled and was excluded.
  These runs establish correctness and execution coverage; their timings are
  not performance results for the paper.

## Completed checks

The four focused CPU tests pass in the remote environment. They cover dense
causal attention/autograd equivalence at CP2/4/8 with MHA and GQA, repeated
execution, original sample/group order, and summary aggregation.

Both four-GPU runs cover arxiv, freelaw, github, pile, and prolong:

| Run | Target tokens | Cases per dataset | CP-sizing target per rank | Cases | Mode executions | Phase records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small workload | 32768 | 1 | 2048 | 5 | 15 | 45 |
| Default workload | 131072 | 4 | 8192 | 20 | 60 | 180 |

Every case executes `normal`, `compute-only`, and `one-step`. Before timing,
both compute controls compare O, LSE, dQ, dK, and dV against `normal` with
`atol=0.1`, `rtol=0.1`. All comparisons pass. Complete owner dK/dV for the two
controls are assembled in an untimed correctness collective.

Every mode completes forward, backward, and directly measured
forward+backward with one warmup and two measured iterations. The default
workload covers CP1/2/4. Its execution-group distribution is four cases with
one group, six with two, nine with three, and one with four. All three modes
retain the same assignments, sample order, padding, and group barriers.

## Reproduction

Run from the remote checkout on `h20`:

```bash
.venv/bin/python -m unittest ring_test.test_cp_motivation -v

CUDA_VISIBLE_DEVICES=0,2,3,6 GPUS=4 TARGET_TOKENS=32768 NUM_CASES=1 \
  MEGATRON_MAX_SEQLEN_PER_RANK=2048 WARMUP_ITERS=1 NUM_ITERS=2 CHECK=1 \
  LOG_DIR=benchmark_logs/cp_motivation_smoke_4gpu \
  bash scripts/benchmark_cp_motivation.sh

CUDA_VISIBLE_DEVICES=0,2,3,6 GPUS=4 TARGET_TOKENS=131072 NUM_CASES=4 \
  WARMUP_ITERS=1 NUM_ITERS=2 CHECK=1 \
  LOG_DIR=benchmark_logs/cp_motivation_4gpu_128k \
  bash scripts/benchmark_cp_motivation.sh
```

## Records

The remote checkout and local checkout both contain:

- [Small-workload results](../../benchmark_logs/cp_motivation_smoke_4gpu/results.jsonl)
  and [console log](../../benchmark_logs/cp_motivation_smoke_4gpu/console.log).
- [Default-workload results](../../benchmark_logs/cp_motivation_4gpu_128k/results.jsonl)
  and [console log](../../benchmark_logs/cp_motivation_4gpu_128k/console.log).

Each JSONL records the backend, selected devices, configuration, original and
execution lengths, CP assignments, correctness status, and timing samples.
Eight-GPU validation and paper performance measurements remain for the
dedicated benchmark environment.
