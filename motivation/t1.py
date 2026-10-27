"""Run T1 with torchrun: main's index_select allgather and FA3 ring."""

import argparse
import sys

from .config import ROOT, add_sampling_args, t1_cases, validate_sampling


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_sampling_args(parser)
    args = parser.parse_args(argv)
    validate_sampling(args)

    import torch
    import torch.distributed as dist
    # Existing ring_test modules use these sibling imports.
    sys.path.insert(0, str(ROOT / "ring_test"))
    from allgather_attention import select_fa3_backend
    from hybrid_forward_baselines import VarlenAllGatherForward, fa3_ring_forward, make_cu_seqlens
    from dcp_test.utils import initialize_distributed_sm90
    from .common import check_all_ranks, environment, measure, write_json

    device = initialize_distributed_sm90("motivation T1")
    world, rank = dist.get_world_size(), dist.get_rank()
    backend = select_fa3_backend(dist.group.WORLD, require_backward=False)
    records = []
    try:
        for case in t1_cases(world):
            lengths = [case["local_seqlen"]] * case["batch_size"]
            tokens = sum(lengths)
            generator = torch.Generator(device=device).manual_seed(20260918 + rank)
            q = torch.randn((tokens, 32, 128), device=device, dtype=torch.bfloat16, generator=generator)
            k = torch.randn((tokens, 8, 128), device=device, dtype=torch.bfloat16, generator=generator)
            v = torch.randn(k.shape, device=device, dtype=k.dtype, generator=generator)
            cu, cu_host = make_cu_seqlens(lengths, device)
            ks, vs = ([torch.empty_like(k) for _ in range(world)] for _ in range(2))
            dist.all_gather(ks, k)
            dist.all_gather(vs, v)
            ring_kv = [(ks[(rank - step) % world], vs[(rank - step) % world])
                       for step in range(world)]
            allgather_kv = [
                (torch.cat([item[:, head:head + 1] for item in ks]),
                 torch.cat([item[:, head:head + 1] for item in vs]))
                for head in range(8)
            ]
            allgather = VarlenAllGatherForward(
                dist.group.WORLD, q, k, v, lengths, True, backend, heads_k_stride=1)

            def ring(mode):
                return fa3_ring_forward(
                    dist.group.WORLD, q, k, v, cu, cu_host, lengths, True, backend,
                    execution_mode=mode, gathered_kv=ring_kv if mode == "comp_only" else None)

            def ag(mode):
                return allgather.forward(
                    execution_mode=mode, gathered_kv=allgather_kv if mode == "comp_only" else None)

            expected = ring("overlap").clone()
            for name, forward in (("ring", ring), ("allgather", ag)):
                for mode in ("comm_only", "comp_only", "serial", "overlap"):
                    if mode != "comm_only":
                        output = forward(mode)
                        check_all_ranks(lambda: torch.testing.assert_close(
                            output, expected, atol=0.02, rtol=0.02))
                    timing, _ = measure(lambda: forward(mode), device, args.warmup, args.iters)
                    records.append({**case, "method": name, "mode": mode, "timing": timing})
                    if rank == 0:
                        print(f'T1 B={case["batch_size"]} {name}/{mode}: '
                              f'p50={timing["p50_ms"]:.6f} ms', flush=True)
        if rank == 0:
            write_json(args.output_dir / "t1.json", {
                "schema": "motivation.v3.T1", "environment": environment(device),
                "backend": backend, "q_heads": 32, "kv_heads": 8, "head_dim": 128,
                "dtype": "bfloat16", "causal": True, "heads_k_stride": 1,
                "seed_base": 20260918, "execution": "eager CUDA-event timing",
                "warmup": args.warmup, "iters": args.iters, "records": records})
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
