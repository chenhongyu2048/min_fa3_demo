# Forward Ablation Static Review (Final Runner Revision)

- Source content hash: `58ff74177d4a9a1492e0da87e048c814c993cddb361aeb32e2453ac8600ef31d`
- Native-source build hash: `d76f79ad97086ef35bf99bf89d9355d02542c451a1354a81072c42c6a87e6d52`
- Baseline HEAD: `48f3f20c436ff2201ff1e45193bc35810faa4794`
- Revision from `d76f79ad9708`: the benchmark file-path entry point now
  inserts the demo root into `sys.path`, matching the existing topology
  benchmark convention. No CUDA, C++, header, setup, or extension source
  changed after the successful clean build and resource gate.

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
| Direct file-path CLI import and `--help` execution | PASS |
| `git diff --check`, trailing whitespace, Python compile, and CPU dispatch tests | PASS |

The clean native build and resource artifacts retain the `d76f79ad9708`
suffix because the final revision is Python-only and cannot affect device
code or the extension binary.
