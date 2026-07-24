# Forward Ablation Static Review

- Source content hash: `e4133178c359a6fac72494ec5d172c4ceeb600b5b703eea9b970df9f0bf20eda`
- Baseline HEAD: `48f3f20c436ff2201ff1e45193bc35810faa4794`
- Scope: causal BF16 SM90 W8 forward accumulation ablation only

| Gate | Result | Evidence |
| --- | --- | --- |
| Initial worktree and codegen environment recorded | PASS | Baseline manifest records clean status, toolchain, GPUs, environment, submodules, and setup flags. |
| Production L5/L6 mainloop, epilogue, kernel, and launch wrapper unchanged | PASS | No diff in `min_fa3_mainloop.h`, `min_fa3_epilogue.h`, `min_fa3_kernel.h`, `mega_ring_min_fa3_varlen_ring_launch.h`, or its launch TU. |
| No runtime profile branch in WGMMA/mainloop/dynamic hot path | PASS | Profile switch is host-only; new step/linear behavior is selected by explicit template instantiation. |
| L3/L4 share decoder and atomic queue | PASS | Both instantiate `KernelConfig<false, true, CollectStats>` and use virtual block first work plus `atomicAdd(...) + virtual_grid_dim_x`. |
| L3 communication CTAs exit; L4 recycles | PASS | The only kernel-wrapper delta is compile-time `RecycleComm`; the false branch ends after remote load and the true branch enters `attn_kernel(..., true)`. |
| L5/L6 use production dynamic segment path | PASS | Both profile cases call `run_mega_ring_min_fa3_varlen_ring_fwd`; only topology tensors differ. |
| L1/L2 share step work graph and communication | PASS | Both use the same causal W8 step scheduler and 16x1024 TMA step loader; only `EnableReduction` differs. |
| L1 external reduction matches fused O/LSE formula | PASS | Both use max-LSE plus `log1pf(exp(-abs(delta)))` and `prev + scale * (block - prev)` with the same all-negative-infinity sentinel. |
| Workspace and counter bounds explicit | PASS | Python hierarchy fields are checked as non-negative int32; binding validates every device workspace length; completed stats require four ints and normal execution one. |
| No out-of-scope code changed | PASS | No backward, dtype/head-dim/architecture generalization, unrelated API, generated binary, or third-party diff. |
| Formatting and CPU dispatch checks | PASS | `git diff --check`, trailing-whitespace scan, `py_compile`, and `ring_test.test_forward_ablation_profiles` all pass. |

Intentional generated artifacts are confined to `benchmark_logs/forward_ablation/`.
