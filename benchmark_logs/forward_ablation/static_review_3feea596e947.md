# Forward Ablation Static Review (Revised)

- Source content hash: `3feea596e947ed7005d004d5b9ada4ed9dc88622fb3d03816ab08e1ba95fcd3f`
- Baseline HEAD: `48f3f20c436ff2201ff1e45193bc35810faa4794`
- Revision from the first review: include order now enters the existing
  ThunderKittens macro-hygiene layer before the new scheduler header.

| Gate | Result |
| --- | --- |
| Production L5/L6 mainloop, epilogue, kernel, dynamic launcher unchanged | PASS |
| Host-only profile switch; no runtime profile branch in WGMMA/mainloop/dynamic scheduler | PASS |
| L3/L4 share one linear decoder and atomic queue | PASS |
| L3 comm exits and L4 alone recycles | PASS |
| L5/L6 call the same production launcher and differ only in metadata | PASS |
| L1/L2 share step work/communication and differ only in reduction placement | PASS |
| External O/LSE reduction formula matches the existing fused epilogue | PASS |
| Workspace/counter/grid indices have explicit int32 and length bounds | PASS |
| Backward, unsupported dtypes/dims/architectures, third-party code untouched | PASS |
| `git diff --check`, trailing whitespace, Python compile, CPU dispatch tests | PASS |

The failed pre-revision build log is retained as diagnostic evidence. No
post-build resource result was produced by that failed attempt.
