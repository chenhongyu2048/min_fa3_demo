"""Compare UltraAttn and hybrid mega-ring on five fixed 128K workloads."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


THIS_DIR = Path(__file__).resolve().parent
RING_TEST_DIR = THIS_DIR.parent
DEMO_DIR = RING_TEST_DIR.parent
for path in (THIS_DIR, RING_TEST_DIR, DEMO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from ring_test.utils import HybridBenchmarkCase
from ring_test.ultraattn.ultraattn_forward import BLOCK_TOKENS


TOTAL_CASES = 5
FIXED_CASES = (
    HybridBenchmarkCase(
        label="fixed=1x128K, topology=1xG8",
        case_index=0,
        num_cases=TOTAL_CASES,
        global_lengths=(131072,),
        ring_sizes=(8,),
        ring_starts=(0,),
    ),
    HybridBenchmarkCase(
        label="fixed=2x64K, topology=2xG4",
        case_index=1,
        num_cases=TOTAL_CASES,
        global_lengths=(65536, 65536),
        ring_sizes=(4, 4),
        ring_starts=(0, 4),
    ),
    HybridBenchmarkCase(
        label="fixed=4x32K, topology=4xG2",
        case_index=2,
        num_cases=TOTAL_CASES,
        global_lengths=(32768,) * 4,
        ring_sizes=(2,) * 4,
        ring_starts=(0, 2, 4, 6),
    ),
    HybridBenchmarkCase(
        label="fixed=8x16K, topology=8xG1",
        case_index=3,
        num_cases=TOTAL_CASES,
        global_lengths=(16384,) * 8,
        ring_sizes=(1,) * 8,
        ring_starts=tuple(range(8)),
    ),
    HybridBenchmarkCase(
        label="fixed=16x8K, topology=2xG1 per rank",
        case_index=4,
        num_cases=TOTAL_CASES,
        global_lengths=(8192,) * 16,
        ring_sizes=(1,) * 16,
        ring_starts=tuple(range(8)) * 2,
    ),
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare UltraAttn and hybrid mega-ring on 1x128K, 2x64K, "
            "4x32K, 8x16K, and 16x8K causal workloads"
        )
    )
    parser.add_argument("--qhead", type=_positive_int, default=32)
    parser.add_argument("--kvhead", type=_positive_int, default=8)
    parser.add_argument("--headdim", type=_positive_int, default=128)
    parser.add_argument(
        "--methods", default="ultraattn,mega_ring_hybrid"
    )
    parser.add_argument("--sm-configs", default="128:4")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=_positive_int, default=40)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("--rtol", type=float, default=2e-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ultraattn-plan-dir",
        default=str(DEMO_DIR / "baseline" / "UltraAttn" / "packing_plans"),
    )
    parser.add_argument(
        "--ultraattn-workspace-mib", type=_positive_int, default=2048
    )
    parser.add_argument(
        "--ultraattn-block-tokens",
        type=int,
        choices=(BLOCK_TOKENS,),
        default=BLOCK_TOKENS,
    )
    return parser.parse_args(argv)


def _format(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    env_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if env_world_size is None:
        raise SystemExit("Run this benchmark with torchrun --nproc_per_node=8")
    if int(env_world_size) != 8:
        raise SystemExit(
            f"the fixed hierarchy requires LOCAL_WORLD_SIZE=8, got {env_world_size}"
        )
    if (args.qhead, args.kvhead, args.headdim) != (32, 8, 128):
        raise SystemExit("fixed UltraAttn graph comparison requires QH/KVH/D=32/8/128")

    first = FIXED_CASES[0]
    forwarded_argv = [
        "--global-seqlens",
        _format(first.global_lengths),
        "--ring-sizes",
        _format(first.ring_sizes),
        "--ring-starts",
        _format(first.ring_starts),
        "--qhead",
        str(args.qhead),
        "--kvhead",
        str(args.kvhead),
        "--headdim",
        str(args.headdim),
        "--mode",
        "causal",
        "--methods",
        args.methods,
        "--sm-configs",
        args.sm_configs,
        "--warmup-iters",
        str(args.warmup_iters),
        "--num-iters",
        str(args.num_iters),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--seed",
        str(args.seed),
        "--ultraattn-plan-dir",
        args.ultraattn_plan_dir,
        "--ultraattn-workspace-mib",
        str(args.ultraattn_workspace_mib),
        "--ultraattn-block-tokens",
        str(args.ultraattn_block_tokens),
        "--check" if args.check else "--no-check",
    ]

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print("Fixed 128K comparison suite:")
        for case in FIXED_CASES:
            print(
                f"  {case.label}: global_seqlens={list(case.global_lengths)}, "
                f"ring_sizes={list(case.ring_sizes)}, "
                f"ring_starts={list(case.ring_starts)}"
            )

    from ring_test.ultraattn import benchmark_hybrid_forward

    benchmark_hybrid_forward.main(
        forwarded_argv,
        workload_cases=FIXED_CASES,
        skip_incompatible_methods=False,
    )


if __name__ == "__main__":
    main()
