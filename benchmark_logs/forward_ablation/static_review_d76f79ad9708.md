# Forward Ablation Static Review (Revised)

- Source content hash: `d76f79ad97086ef35bf99bf89d9355d02542c451a1354a81072c42c6a87e6d52`
- Baseline HEAD: `48f3f20c436ff2201ff1e45193bc35810faa4794`
- Revision from `3feea596e947`: the copied communication row constants now
  use the anonymous-enum form used by the production configuration, avoiding
  CUDA device ODR-use of class `static constexpr` members.

| Gate | Result |
| --- | --- |
| Production L5/L6 mainloop, epilogue, kernel, and dynamic launcher unchanged | PASS |
| Host-only profile switch; no runtime profile branch in WGMMA, mainloop, or dynamic scheduler | PASS |
| L3/L4 share one linear decoder, ticket ordering, and atomic queue | PASS |
| L3 communication CTAs exit; L4 alone recycles them into the compute queue | PASS |
| L5/L6 call the same production launcher and differ only through topology metadata | PASS |
| L1/L2 share the step work graph and communication; only reduction placement differs | PASS |
| External O/LSE reduction formula and all-`-inf` sentinel match the fused epilogue | PASS |
| Workspace, counter, grid, and int32 indices have explicit bounds | PASS |
| Backward, unsupported dtype/dimension/architecture paths, and third-party code untouched | PASS |
| No direct `hopper/...` include dependency added | PASS |
| `git diff --check`, trailing whitespace, Python compile, and CPU dispatch tests | PASS |

The two earlier failed post-build logs remain as diagnostic records. This
review is the required static gate for the next clean build.
