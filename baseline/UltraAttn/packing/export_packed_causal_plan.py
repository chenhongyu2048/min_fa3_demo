"""Export UltraAttn ILP allocations for the fixed 8K graph suite.

The block-allocation formulation is adapted from
``search_algo/workload_partition.py::Quad_LP_GUROBI_from_block_config``.
This command is an offline-only tool and intentionally requires the original
UltraAttn planner dependencies and a usable Gurobi license.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
ULTRA_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[3]
for path in (REPO_ROOT, ULTRA_ROOT, ULTRA_ROOT / "search_algo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline.UltraAttn.packing.plan_format import (  # noqa: E402
    DEFAULT_PLANNER_SOURCE_REVISION,
    PackedCausalPlan,
    build_default_cmap,
    build_packed_causal_mask,
    gqa_communication_costs,
    save_plan,
)


BLOCK_TOKENS = 8192
FIXED_GLOBAL_SEQLENS = (
    (131072,),
    (65536, 65536),
    (32768,) * 4,
    (16384,) * 8,
    (8192,) * 16,
)


def _parse_int_list(spec: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an UltraAttn ILP allocation for one fixed 8K graph case"
    )
    parser.add_argument("--global-seqlens", type=_parse_int_list, required=True)
    parser.add_argument("--world-size", type=int, choices=(8,), required=True)
    parser.add_argument("--qhead", type=int, choices=(32,), default=32)
    parser.add_argument("--kvhead", type=int, choices=(8,), default=8)
    parser.add_argument("--headdim", type=int, choices=(128,), default=128)
    parser.add_argument(
        "--block-tokens",
        type=int,
        choices=(BLOCK_TOKENS,),
        default=BLOCK_TOKENS,
    )
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument(
        "--planner-source-revision", default=DEFAULT_PLANNER_SOURCE_REVISION
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ULTRA_ROOT / "packing_plans",
    )
    return parser.parse_args(argv)


def _workloads(args: argparse.Namespace) -> list[list[int]]:
    return [list(args.global_seqlens)]


def _to_ultra_block_table(block_types: np.ndarray):
    from search_algo.utils import Block_Type

    table = np.empty(block_types.shape, dtype=Block_Type)
    for value, enum_value in (
        (0, Block_Type.EMPTY),
        (1, Block_Type.FULL),
        (2, Block_Type.CAUSAL),
    ):
        table[block_types == value] = enum_value
    return table


def _solve_one(args: argparse.Namespace, global_seqlens: list[int]) -> Path:
    if importlib.util.find_spec("gurobipy") is None:
        raise SystemExit(
            "Offline UltraAttn planning requires gurobipy and a usable Gurobi "
            "license in a separate planner environment; gurobipy is not installed."
        )
    try:
        from search_algo.bsa_config import BSA_Config, BSA_Repr
        from search_algo.workload_partition import Quad_LP_GUROBI_from_block_config
    except ImportError as exc:
        raise SystemExit(
            "Offline UltraAttn planning dependencies are unavailable. Install the "
            "vendored UltraAttn requirements plus gurobipy in a separate planner environment. "
            f"Original import error: {exc}"
        ) from exc

    block_types = build_packed_causal_mask(global_seqlens, args.block_tokens)
    par_d = block_types.shape[0]
    cmap = build_default_cmap(par_d, args.world_size)
    ultra_repr = BSA_Repr(_to_ultra_block_table(block_types), cmap.copy())
    # Preserve the exact fixed-block representation even when BSA_Repr's generic
    # simplifier finds a coarser algebraic representation.
    ultra_repr.block_table_raw = _to_ultra_block_table(block_types)
    ultra_repr.cmap_raw = cmap.copy()
    ultra_repr.cmap = cmap.copy()
    ultra_repr.minimum_Par_D = par_d
    bsa_config = BSA_Config(
        None,
        None,
        {"bsa_repr": ultra_repr, "CP": (args.world_size, 1)},
    )
    bsa_config.cmap = cmap.copy()

    communication_costs = gqa_communication_costs(
        args.qhead, args.kvhead, args.headdim, args.block_tokens
    )
    result = Quad_LP_GUROBI_from_block_config(
        bsa_config,
        fob=0,
        hierarchy=1,
        ParD=par_d,
        communication_costs=communication_costs,
        causal_pattern=True,
        time_limit=args.time_limit,
        solver_seed=args.solver_seed,
    )

    from baseline.UltraAttn.packing.plan_format import make_plan_metadata

    metadata = make_plan_metadata(
        global_seqlens=global_seqlens,
        world_size=args.world_size,
        qhead=args.qhead,
        kvhead=args.kvhead,
        headdim=args.headdim,
        planner_source_revision=args.planner_source_revision,
        solver_metadata=result["solver"],
        block_tokens=args.block_tokens,
    )
    plan = PackedCausalPlan(
        metadata=metadata,
        cmap=cmap,
        block_types=block_types,
        allocation=np.asarray(result["table"], dtype=np.int16),
    )
    return save_plan(args.output_dir / f"{plan.cache_key}.npz", plan)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.time_limit <= 0:
        raise SystemExit("--time-limit must be positive")

    workloads = _workloads(args)
    if tuple(workloads[0]) not in FIXED_GLOBAL_SEQLENS:
        raise SystemExit(
            "UltraAttn graph planning supports only 1x128K, 2x64K, 4x32K, "
            "8x16K, and 16x8K"
        )
    for case_index, global_seqlens in enumerate(workloads, start=1):
        block_types = build_packed_causal_mask(
            global_seqlens, args.block_tokens
        )
        try:
            build_default_cmap(block_types.shape[0], args.world_size)
        except ValueError as exc:
            total_tokens = sum(global_seqlens)
            raise SystemExit(
                f"preflight failed for case={case_index}: total_tokens={total_tokens}, "
                f"block_tokens={args.block_tokens}, "
                f"packed_tiles={block_types.shape[0]}, world_size={args.world_size}; "
                "UltraAttn's default contiguous cmap requires packed_tiles % "
                "world_size == 0; the fixed graph planner will not pad or "
                "substitute another placement. "
                f"Original error: {exc}"
            ) from exc
        print(
            f"preflight case={case_index}, global_seqlens={global_seqlens}, "
            f"packed_tiles={block_types.shape[0]}",
            flush=True,
        )

    for case_index, global_seqlens in enumerate(workloads, start=1):
        output = _solve_one(args, global_seqlens)
        print(
            f"case={case_index}, global_seqlens={global_seqlens}, plan={output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
