"""T3: four placements, one MegaRing executor and the main Qwen3 MoE layer."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .config import DATASETS, provenance


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, choices=(4, 8), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--static-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist
    from ring_test import benchmark_transformer_layer as bench
    from ring_test import transformer_layer_cp as layer_cp
    from .placements import build_placements, describe_placement
    from .common import require_homogeneous_devices, environment

    manifest = json.loads(args.manifest.read_text())
    cases = manifest["datasets"][args.dataset][:args.case_limit]
    records = [(case, strategy, plan, describe_placement(strategy, plan))
               for case in cases for strategy, plan in build_placements(case["raw_lengths"], args.gpus).items()]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    origin = provenance()
    if args.static_only:
        with args.output_jsonl.open("x") as output:
            for case, strategy, plan, static in records:
                output.write(json.dumps({"schema": "motivation.v2.T3.static", **origin,
                                         "dataset": args.dataset, "case_id": case["case_id"],
                                         "gpus": args.gpus, "static": static}) + "\n")
        return
    rank, world, device = bench._init_distributed(args.gpus)
    sm_count = require_homogeneous_devices("motivation T3", world, device)
    setup = bench.parse_args(["--dataset", args.dataset, "--world-size", str(world),
                              "--output-jsonl", str(args.output_jsonl), "--methods", "mega_ring_hybrid",
                              "--seed", str(manifest["seed"])])
    inventory = bench._collect_device_inventory(rank, rank, device)
    bench._preflight(setup, ["mega_ring_hybrid"], device, inventory)
    bench._initialize_megatron(setup)
    layer = layer_cp.build_megatron_layer(setup.megatron_path, device, world)
    dispatcher = layer.self_attention.core_attention
    sm = layer_cp.SmConfig(sm_count - 8, 8)
    env = environment(device)
    output = args.output_jsonl.open("x") if rank == 0 else None
    try:
        for case, strategy, plan, static in records:
            layout = layer_cp.build_physical_layout("mega_ring_hybrid", plan.global_lengths,
                                                    plan.ring_sizes, plan.ring_starts, world)
            local_lengths = layout.local_lengths(rank)
            local_tokens = sum(local_lengths)
            q = torch.empty((local_tokens, 32, 128), device=device, dtype=torch.bfloat16)
            capacity = ((max(layout.rank_token_loads) + 127) // 128) * 128
            adapter = layer_cp._MegaRingAdapter(
                method="mega_ring_hybrid", dummy_q=q, local_lengths=local_lengths,
                rank_capacity=capacity, global_lengths=plan.global_lengths,
                ring_sizes=plan.ring_sizes, ring_starts=plan.ring_starts,
                rank=rank, world_size=world, sm_config=sm,
            )
            # Main skips per-iteration K/V population. Define the arena once so
            # the controlled performance experiment never reads allocator debris.
            begin = rank * capacity
            adapter.remote_k.data_[begin:begin + capacity].zero_()
            adapter.remote_v.data_[begin:begin + capacity].zero_()
            torch.cuda.synchronize(device)
            dist.barrier()
            dispatcher.set_adapter(adapter)
            hidden, dout = bench._make_inputs(local_tokens, device, manifest["seed"] + case["case_id"])
            packed = layer_cp.make_packed_seq_params(local_lengths, device)
            timing = bench._measure(layer, hidden, dout, packed, args.warmup, args.iters)
            record = {"schema": "motivation.v2.T3", **origin, "dataset": args.dataset,
                      "case_id": case["case_id"], "seed": manifest["seed"],
                      "target_tokens": manifest["target_tokens"], "gpus": world,
                      "model": layer_cp.QWEN3_CONFIG, "parallelism": {"tp": 1, "cp": world, "ep": world},
                      "expert_routing": "main synthetic uniform top-8", "sm_config": asdict(sm),
                      "environment": env, "warmup": args.warmup, "iters": args.iters,
                      "static": static, "timing": asdict(timing),
                      "kv_population": "one-time zero initialization; per-iteration projected K/V copy skipped, as in main",
                      "timing_semantics": "main mean of per-iteration CUDA critical-rank full/core FWD+BWD; not p50",
                      "validation": "gradient presence; not end-to-end numerical training equivalence"}
            if output:
                output.write(json.dumps(record) + "\n")
                output.flush()
            dispatcher.set_adapter(None)
            bench._zero_grads(layer, hidden)
            del hidden, dout, packed, adapter, q
            torch.cuda.empty_cache()
            dist.barrier()
    finally:
        if output:
            output.close()
        if sys.exc_info()[0] is None:
            dispatcher.set_adapter(None)
            del layer
            bench._destroy_distributed_state()


if __name__ == "__main__":
    main()
