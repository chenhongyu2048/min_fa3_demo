# Mega Ring Forward Six-Level Ablation Results

- Baseline HEAD: `48f3f20c436ff2201ff1e45193bc35810faa4794`
- Final source hash: `58ff74177d4a9a1492e0da87e048c814c993cddb361aeb32e2453ac8600ef31d`
- Native clean-build hash: `d76f79ad97086ef35bf99bf89d9355d02542c451a1354a81072c42c6a87e6d52`
- Hardware: 8 x NVIDIA H100 80GB HBM3, driver 590.48.01, CUDA 12.8
- Fixed configuration: causal BF16, QH=16, KVH=8, D=128, 116 compute SMs, 16 communication SMs

## Build Resource Gate

| Level | Registers | Stack | Spill stores | Spill loads | Shared | Local | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline L6 | 168 | 56 | 520 | 712 | 1152 | 0 | reference |
| L3 | 168 | 48 | 128 | 232 | 1152 | 0 | PASS |
| L4 | 168 | 48 | 256 | 464 | 1152 | 0 | PASS |
| L5 | 168 | 56 | 520 | 712 | 1152 | 0 | PASS |
| L6 | 168 | 56 | 520 | 712 | 1152 | 0 | PASS |

L5 and L6 resolve to the same causal W8, statistics-off production kernel
symbol. Their only distinction is topology metadata.

## Correctness And Behavior

Representative 8-GPU correctness passed for L1-L5 all-CP lengths
`(8192, 4096)` and L6 mixed lengths `(2048, 1024, 512, 256)`, each repeated
five times and checked against the hierarchical reference for O and FP32 LSE.

| Level | Attention launches | Reduction launches | Recycled work | Max span | Ring sizes |
| --- | ---: | ---: | ---: | ---: | --- |
| L1 | 8 | 8 | 0 | 1 | 8 |
| L2 | 8 | 0 | 0 | 1 | 8 |
| L3 | 1 | 0 | 0 | 1 | 8 |
| L4 | 1 | 0 | 55 | 1 | 8 |
| L5 | 1 | 0 | 0 | 4 | 8 |
| L6 | 1 | 0 | 0 | 7 | 8, 4, 2, 1 |

## Performance

The raw and canonical workload were both
`(16384, 12288, 8192, 6144, 4096, 2048, 2048, 2048)`. BR-PBS used the same
lengths without extra padding. Results use 10 warmups, 40 measured iterations,
3 interleaved rounds, CUDA events around one `plan.run()`, and the maximum
latency across all 8 ranks.

| Level | Profile | Median ms | P10 ms | P90 ms | Original-token TFLOPS | Kernels | Max span |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | step_external_reduce | 4.295840 | 2.390982 | 5.393821 | 527.944 | 16 | 1 |
| L2 | step_fused_reduce | 3.005200 | 2.091190 | 3.960624 | 754.679 | 8 | 1 |
| L3 | linear_queue_no_recycle | 1.851120 | 1.367728 | 3.266301 | 1225.183 | 1 | 1 |
| L4 | linear_queue_recycle | 1.699280 | 1.326269 | 3.199354 | 1334.660 | 1 | 1 |
| L5 | dynamic_segment_recycle | 2.114544 | 1.441050 | 3.385008 | 1072.553 | 1 | 7 |
| L6 | hybrid_br_pbs | 2.000544 | 1.195482 | 3.280355 | 1133.672 | 1 | 7 |

Adjacent effects: L2 is 30.0% lower latency than L1; L3 is 38.4% lower than
L2; L4 is 8.2% lower than L3; L5 is 24.4% higher than L4 on this workload;
L6 is 5.4% lower than L5. End to end, L6 is 2.147x faster than L1, while L4
is the fastest measured level. Per-rank Q/O visits, KV tile reads, recycled
work, span, and claim counts remain in the full performance log.
