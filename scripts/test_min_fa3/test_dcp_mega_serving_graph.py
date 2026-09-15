"""Replay full graphs across changing packed batches on 2/4/8 Hopper GPUs.

Run with torchrun --standalone --nproc-per-node=2 -m
scripts.test_min_fa3.test_dcp_mega_serving_graph.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

import min_fa3_op
from min_fa3_dcp import DCPMegaAttentionRunner
from dcp_test.baselines import VLLMDCPAttentionRunner, VLLMA2ADCPAttentionRunner
from scripts.test_min_fa3.test_dcp_mega_varlen_multi_rank import (
    append_chunk,
    make_cu,
    randn_bf16,
    shard_history,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mega", "vllm-ag-rs", "vllm-a2a"), default="mega")
    parser.add_argument("--hq-local", type=int, choices=(4, 8), default=4)
    parser.add_argument("--scheduler-mode", choices=("auto", "native"), default="auto")
    parser.add_argument("--pipeline", action="store_true",
                        help="Submit pairs of replays without an intervening host sync")
    options = parser.parse_args()
    backend = options.backend
    hq_local = options.hq_local
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if backend == "mega":
        runner = DCPMegaAttentionRunner(
            dist.group.WORLD, dist.group.WORLD,
            max_total_q=32, max_batch=4, Hq_local=hq_local, max_num_splits=128,
            num_comm_sm=4, block_n_override=128,
        )
        q = runner.q_backing[:32]
    else:
        cls = VLLMDCPAttentionRunner if backend == "vllm-ag-rs" else VLLMA2ADCPAttentionRunner
        runner = cls(dist.group.WORLD)
        q = torch.empty((32, hq_local, 128), device=device, dtype=torch.bfloat16)
    k = torch.empty((16384, 1, 128), device=device, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    ck = torch.empty((32, 1, 128), device=device, dtype=torch.bfloat16)
    cv = torch.empty_like(ck)
    cu_q = torch.empty(5, device=device, dtype=torch.int32)
    cu_h = torch.empty_like(cu_q)

    previous_done = None
    preparation_samples = []

    def prepare(capacity, q_lengths, h_lengths, seed):
        total_q = sum(q_lengths)
        q.copy_(randn_bf16(tuple(q.shape), seed + rank, device))
        ck.copy_(randn_bf16(tuple(ck.shape), seed + 71, device))
        cv.copy_(randn_bf16(tuple(cv.shape), seed + 72, device))
        hk = randn_bf16((sum(h_lengths), 1, 128), seed + 81, device)
        hv = randn_bf16(tuple(hk.shape), seed + 82, device)
        lk, lh = shard_history(hk, h_lengths, rank, world)
        lv, _ = shard_history(hv, h_lengths, rank, world)
        k[:len(lk)].copy_(lk)
        v[:len(lv)].copy_(lv)
        dq, hq = make_cu(q_lengths, device)
        dh, hh = make_cu(lh, device)
        cu_q.fill_(total_q)
        cu_h.fill_(len(lk))
        cu_q[:len(dq)].copy_(dq)
        cu_h[:len(dh)].copy_(dh)
        reference_lengths = [h + t for h, t in zip(h_lengths, q_lengths)]
        dr, hr = make_cu(reference_lengths, device)
        expected = min_fa3_op.forward_kvcache_varlen(
            q[:total_q], append_chunk(hk, ck, h_lengths, q_lengths),
            append_chunk(hv, cv, h_lengths, q_lengths), dq, dr,
            max(q_lengths), max(reference_lengths),
            cu_seqlens_q_host=hq, cu_seqlens_k_host=hr,
            return_lse=True, is_causal=True,
        )
        tensors = q[:capacity], k, v, ck[:capacity], cv[:capacity], cu_q, cu_h
        if backend == "mega":
            pending_before = previous_done is not None and not previous_done.query()
            start = time.perf_counter()
            args = runner.prepare_graph_forward(
                *tensors, cu_seqlens_q_host=hq,
                cu_seqlens_history_local_host=hh, return_lse=True,
                scheduler_heuristic=False if options.scheduler_mode == "native" else None,
            )
            preparation_samples.append((
                (time.perf_counter() - start) * 1000,
                pending_before,
                previous_done is not None and not previous_done.query(),
            ))
        else:
            # Host mirrors are checked for layout only by the capacity API.
            hq_pad = torch.full((5,), total_q, dtype=torch.int32)
            hh_pad = torch.full((5,), len(lk), dtype=torch.int32)
            hq_pad[:len(hq)].copy_(hq)
            hh_pad[:len(hh)].copy_(hh)
            args = tensors, hq_pad, hh_pad, capacity

        return args, expected

    def forward(args):
        if backend == "mega":
            return runner.forward_prepared_graph(args)
        tensors, hq, hh, capacity = args
        return runner.forward_chunk_prefill_varlen_graph(
            *tensors, capacity, 16384,
            cu_seqlens_q_host=hq, cu_seqlens_history_local_host=hh,
            return_lse=True,
        )

    graphs = {}
    stream = torch.cuda.Stream()
    try:
        for capacity in (16, 32):
            args, _ = prepare(capacity, [1, 1], [128, 256], 10)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward(args)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                # Two layer calls share scratch and phase state in one graph.
                out1, lse1 = forward(args)
                first = out1.clone(), lse1.clone()
                out2, lse2 = forward(args)
                second = out2.clone(), lse2.clone()
            graphs[capacity] = graph, first, second, args

        cases = [
            (16, [1, 1], [128, 256]),
            (16, [1, 1], [16384, 8192]),
            (32, [2, 7, 1], [8192, 4096, 2048]),
            (16, [8], [32768]),
            (32, [1, 1, 1, 1], [512, 1024, 2048, 4096]),
            (16, [3, 5], [256, 128]),
        ]
        pending_checks = []
        for iteration, (capacity, q_lengths, h_lengths) in enumerate(cases * 2):
            _, expected = prepare(capacity, q_lengths, h_lengths, 100 + iteration)
            graph, first, second, _ = graphs[capacity]
            graph.replay()
            previous_done = torch.cuda.Event()
            previous_done.record()
            # Preserve this replay's outputs before the next batch overwrites
            # the shared graph buffers on the same stream.
            saved = tuple((out.clone(), lse.clone()) for out, lse in (first, second))
            pending_checks.append((iteration, capacity, q_lengths, h_lengths, expected, saved))
            if options.pipeline and len(pending_checks) < 2:
                continue
            torch.cuda.synchronize()
            for i, cap, ql, hl, reference, outputs in pending_checks:
                for out, lse in outputs:
                    torch.testing.assert_close(out[:sum(ql)], reference[0], atol=0.03, rtol=0.03)
                    torch.testing.assert_close(lse[:, :sum(ql)], reference[1], atol=0.03, rtol=0.03)
                if rank == 0:
                    print(f"PASS backend={backend} mode={options.scheduler_mode} pipeline={options.pipeline} replay={i} capacity={cap} q={ql} history={hl}", flush=True)
            pending_checks.clear()

        if backend == "mega":
            # A controlled pending-GPU check: metadata preparation must return
            # before prior stream work completes. This is not a speedup benchmark.
            args, _ = prepare(16, [1, 1], [128, 256], 500)
            old_host = runner._graph_metadata_host
            old_payload = old_host.clone()
            torch.cuda.synchronize()
            torch.cuda._sleep(100_000_000)
            pending_gpu = torch.cuda.Event()
            pending_gpu.record()
            start = time.perf_counter()
            runner.prepare_graph_forward(
                *args[:7], cu_seqlens_q_host=torch.tensor([0, 1, 2], dtype=torch.int32),
                cu_seqlens_history_local_host=torch.tensor([0, 128 // world, 384 // world], dtype=torch.int32),
                scheduler_heuristic=False if options.scheduler_mode == "native" else None,
                return_lse=True,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert not pending_gpu.query(), "No overlap observed during the controlled pending-GPU check"
            assert runner._graph_metadata_host is old_host
            torch.testing.assert_close(old_host, old_payload, rtol=0, atol=0)
            torch.cuda.synchronize()
            if rank == 0:
                print(f"PASS CPU preparation returned while GPU pending: {elapsed_ms:.3f} ms; immutable pinned source reused", flush=True)
                print(f"preparation_samples_ms_pending_before_after={preparation_samples}", flush=True)
    finally:
        torch.cuda.synchronize()
        for graph, *_ in graphs.values():
            graph.reset()
        if backend == "mega":
            runner.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
