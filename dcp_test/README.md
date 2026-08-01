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
timing boundary, method-label, and JSON-schema details.
