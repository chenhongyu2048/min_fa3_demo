"""The same mapped layouts feed static accounting and the layer executor."""
from dataclasses import asdict
from ring_test.load_balance_bench.topology import (
    PlannerControls, PlannerTopology, TopologySample, make_br_pbs_topology,
    make_megatron_cp_topology, make_zeppelin_topology, validate_fused_metadata,
)
from ring_test.transformer_layer_cp import build_physical_layout
from ring_test.forward_load_model import analyze_mega_ring_hybrid
from .config import STRATEGIES


def build_placements(raw_lengths, world_size):
    raw = tuple(raw_lengths)
    # Match the evaluation All-CP alignment, then use the same fused executor.
    layout = build_physical_layout("mega_ring_all_cp", raw, (1,) * len(raw),
                                   (0,) * len(raw), world_size)
    all_cp = PlannerTopology("br_pbs", True, world_size, raw, tuple(
        TopologySample(i, original, execution, world_size, 0)
        for i, (original, execution) in enumerate(zip(raw, layout.execution_lengths))
    ), 0.0, (("strategy", "all_cp"),))
    plans = (all_cp, make_br_pbs_topology(raw, world_size, True,
                                        PlannerControls(token_balance_tolerance=0.05)),
             make_megatron_cp_topology(raw, world_size, True, 8192),
             make_zeppelin_topology(raw, world_size, True, 4096))
    for plan in plans:
        validate_fused_metadata(plan)
    return dict(zip(STRATEGIES, plans))


def describe_placement(strategy, plan):
    loads = analyze_mega_ring_hybrid(plan.global_lengths, plan.ring_sizes, plan.ring_starts,
                                    plan.world_size, 32, 4, 128, True)
    rows = [asdict(record) for record in loads.records]
    for rank, row in enumerate(rows):
        owned = [sample for sample in plan.samples
                 if sample.ring_start <= rank < sample.ring_start + sample.ring_size]
        row["effective_tokens"] = sum(s.raw_length / s.ring_size for s in owned)
        row["effective_scores"] = sum(s.raw_length * (s.raw_length + 1) / (2 * s.ring_size) for s in owned)
        row["effective_flops"] = 4 * row["effective_scores"] * 32 * 128
        row["method"] = strategy
    def imbalance(key):
        values = [row[key] for row in rows]
        return max(values) / (sum(values) / len(values))
    return {"strategy": strategy, "executor": "mega_ring_hybrid", "raw_lengths": plan.raw_lengths,
            "raw_tokens": plan.raw_tokens, "execution_tokens": plan.execution_tokens,
            "execution_lengths": plan.global_lengths, "ring_sizes": plan.ring_sizes,
            "ring_starts": plan.ring_starts, "sample_ids": plan.sample_ids,
            "samples": [asdict(s) | {"mapped_group_size": s.mapped_group_size, "padding": s.padding}
                        for s in plan.samples], "padding_tokens": plan.padding_tokens,
            "rank_records": rows, "token_imbalance": imbalance("physical_tokens"),
            "attention_imbalance": imbalance("effective_scores"),
            "communication_tx_bytes": sum(r["comm_tx_bytes"] for r in rows),
            "tile_work": sum(r["kv_tile_reads"] for r in rows),
            "accounting": "mapped execution layout; forward CP payload; redistribution excluded"}
