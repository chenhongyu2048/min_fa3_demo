# TKParallelTensor Legacy IPC Offset Repro

This is a minimal ThunderKittens repro for a `TKParallelTensor(tensor, ...)`
legacy-CUDA-IPC offset issue.

It intentionally constructs two contiguous PyTorch views from one backing
allocation:

```python
base = torch.empty((2, rows, cols), device=device, dtype=torch.bfloat16)
k = base[0]  # storage_offset == 0
v = base[1]  # storage_offset == rows * cols, still contiguous
```

Wrapping `k` and `v` separately with `TKParallelTensor(tensor, ...)` uses the
legacy IPC constructor. The local rank stores `tensor.data_ptr()` in
`raw_ptrs_[local_rank]`, but a remote rank receives the pointer returned by
`cudaIpcOpenMemHandle`. For an offset view, that imported pointer can refer to
the base of the exported allocation rather than the original view pointer. The
view offset is not stored or re-applied by `TKParallelTensor`.

As a result:

- remote copy of `k = base[0]` succeeds because its view offset is zero;
- remote copy of `v = base[1]` reads from the wrong address on non-source ranks;
- allocating one VMM-backed `TKParallelTensor([2 * rows, cols], ...)` and using
  an explicit row offset for `v` succeeds.

## Build And Run

From this directory:

```bash
export ARCH=SM90
make
CUDA_VISIBLE_DEVICES=2,3 make run
```

Or run the test directly after building:

```bash
CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 test_repro.py
```

The test exits successfully only if it observes the expected legacy-IPC mismatch
for the non-zero-offset view and also verifies the combined-VMM workaround.
