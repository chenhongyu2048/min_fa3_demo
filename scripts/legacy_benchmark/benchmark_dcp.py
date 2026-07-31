import argparse
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

import min_fa3_op
from min_fa3_dcp import DCPAttentionRunner


@dataclass(frozen=True)
class DCPGroup:
    size: int
    start_rank: int
    process_group: dist.ProcessGroup


def parse_int_list(spec: str, name: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise SystemExit(f"{name} must contain at least one integer")
    return values


def make_groups(sizes: list[int], device: torch.device) -> list[DCPGroup]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_groups: list[DCPGroup] = []
    for size in sizes:
        if size <= 0 or size > world_size or world_size % size:
            raise SystemExit(
                f"every DCP size must divide torchrun world size {world_size}, got {size}"
            )
        for start_rank in range(0, world_size, size):
            ranks = list(range(start_rank, start_rank + size))
            group = dist.new_group(ranks, backend="nccl", device_id=device)
            if start_rank <= rank < start_rank + size:
                local_groups.append(DCPGroup(size, start_rank, group))
    return local_groups


def generator_for(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def randn_bf16(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)


def local_q(q_group: torch.Tensor, rank: int, h_local: int) -> torch.Tensor:
    return q_group[:, :, rank * h_local : (rank + 1) * h_local].contiguous()


def shard_kv(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    return tensor[:, rank::world_size].contiguous()


def rank_max(value: float, group: dist.ProcessGroup, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return float(tensor.item())


def quantiles(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
    }


def derived_quantiles(
    lhs: list[float], rhs: list[float], operation
) -> dict[str, float]:
    return quantiles([operation(left, right) for left, right in zip(lhs, rhs)])


def benchmark_full_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
    num_splits: int,
    warmup: int,
    iterations: int,
    group: dist.ProcessGroup,
    device: torch.device,
) -> list[float]:
    for _ in range(warmup):
        min_fa3_op.forward_kvcache(
            q, k, v, lengths, num_splits=num_splits, return_lse=False
        )
    torch.cuda.synchronize(device)
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        min_fa3_op.forward_kvcache(
            q, k, v, lengths, num_splits=num_splits, return_lse=False
        )
        end.record()
        end.synchronize()
        samples.append(rank_max(start.elapsed_time(end), group, device))
    return samples


def benchmark_runner(
    runner: DCPAttentionRunner,
    call,
    warmup: int,
    iterations: int,
    group: dist.ProcessGroup,
    device: torch.device,
) -> dict[str, list[float]]:
    for _ in range(warmup):
        call(False)
    torch.cuda.synchronize(device)
    samples: dict[str, list[float]] = {}
    for _ in range(iterations):
        call(True)
        timing = runner.last_timing_ms(synchronize=True)
        for name, value in timing.items():
            samples.setdefault(name, []).append(rank_max(value, group, device))
    return samples


def effective_gbps(payload_bytes: float, milliseconds: float) -> float:
    return payload_bytes / max(milliseconds, 1.0e-9) / 1.0e6


def build_report(
    workload: str,
    dcp_size: int,
    batch_size: int,
    sq: int,
    sk: int,
    h_local: int,
    h_kv: int,
    d: int,
    full_samples: list[float],
    sequential: dict[str, list[float]],
    overlapped: dict[str, list[float]],
) -> dict[str, object]:
    metric_names = (
        "q_allgather_and_reorder_ms",
        "local_chunk_attention_ms",
        "sequential_ag_plus_chunk_ms",
        "overlapped_ag_chunk_window_ms",
        "local_history_attention_ms",
        "lse_allgather_correct_ms",
        "output_reduce_scatter_ms",
        "state_merge_ms",
        "attention_end_to_end_ms",
    )
    metrics = {name: quantiles(overlapped[name]) for name in metric_names}
    metrics["dcp_sequential_attention_end_to_end_ms"] = quantiles(
        sequential["attention_end_to_end_ms"]
    )
    metrics["full_kv_attention_end_to_end_ms"] = quantiles(full_samples)
    metrics["sequential_to_overlap_speedup"] = derived_quantiles(
        sequential["attention_end_to_end_ms"],
        overlapped["attention_end_to_end_ms"],
        lambda seq, overlap: seq / overlap,
    )
    metrics["full_kv_to_dcp_speedup"] = derived_quantiles(
        full_samples,
        overlapped["attention_end_to_end_ms"],
        lambda full, dcp: full / dcp,
    )
    hidden_times = [
        q_ms + chunk_ms - window_ms
        for q_ms, chunk_ms, window_ms in zip(
            overlapped["q_allgather_and_reorder_ms"],
            overlapped["local_chunk_attention_ms"],
            overlapped["overlapped_ag_chunk_window_ms"],
        )
    ]
    hidden_fractions = [
        hidden / max(minimum, 1.0e-9)
        for hidden, minimum in zip(
            hidden_times,
            map(
                min,
                overlapped["q_allgather_and_reorder_ms"],
                overlapped["local_chunk_attention_ms"],
            ),
        )
    ]
    metrics["hidden_time_ms"] = quantiles(hidden_times)
    metrics["hidden_fraction"] = quantiles(hidden_fractions)

    h_group = h_local * dcp_size
    if workload == "decode":
        useful_flops = 4.0 * batch_size * h_group * d * sk
        full_kv_tokens = sk
        local_kv_tokens = math.ceil(sk / dcp_size)
        aggregate_local_kv_tokens = sk
        merge_bytes = 0.0
    else:
        useful_flops = (
            4.0
            * batch_size
            * h_group
            * d
            * (sq * sk + sq * (sq + 1) / 2.0)
        )
        full_kv_tokens = sk + sq
        local_kv_tokens = math.ceil(sk / dcp_size) + sq
        aggregate_local_kv_tokens = sk + dcp_size * sq
        merge_bytes = batch_size * sq * h_local * (3 * d * 2 + 2 * 4)

    aggregate_tflops = [
        useful_flops / (milliseconds * 1.0e-3) / 1.0e12
        for milliseconds in overlapped["attention_end_to_end_ms"]
    ]
    metrics["aggregate_useful_tflops"] = quantiles(aggregate_tflops)
    metrics["average_useful_tflops_per_gpu"] = quantiles(
        [value / dcp_size for value in aggregate_tflops]
    )

    q_local_bytes = batch_size * sq * h_local * d * 2
    q_group_bytes = batch_size * sq * h_group * d * 2
    local_kv_bytes = batch_size * local_kv_tokens * h_kv * d * 2 * 2
    full_kv_bytes = batch_size * full_kv_tokens * h_kv * d * 2 * 2
    aggregate_local_kv_bytes = (
        batch_size * aggregate_local_kv_tokens * h_kv * d * 2 * 2
    )
    partial_out_bytes = batch_size * sq * h_group * d * 2
    partial_lse_bytes = batch_size * sq * h_group * 4
    aggregate_logical_hbm_bytes = (
        dcp_size
        * (q_group_bytes + partial_out_bytes + partial_lse_bytes + merge_bytes)
        + aggregate_local_kv_bytes
    )
    average_logical_hbm_bytes_per_gpu = aggregate_logical_hbm_bytes / dcp_size
    aggregate_logical_hbm_gbps = [
        effective_gbps(aggregate_logical_hbm_bytes, milliseconds)
        for milliseconds in overlapped["attention_end_to_end_ms"]
    ]
    average_logical_hbm_gbps_per_gpu = [
        value / dcp_size for value in aggregate_logical_hbm_gbps
    ]
    metrics["aggregate_logical_effective_hbm_gbps"] = quantiles(
        aggregate_logical_hbm_gbps
    )
    metrics["average_logical_effective_hbm_gbps_per_gpu"] = quantiles(
        average_logical_hbm_gbps_per_gpu
    )
    # Backward-compatible alias for the old per-rank bandwidth field.
    metrics["logical_effective_hbm_gbps"] = metrics[
        "average_logical_effective_hbm_gbps_per_gpu"
    ]

    q_ag_receive = (dcp_size - 1) * q_local_bytes
    lse_ag_receive = (dcp_size - 1) * partial_lse_bytes
    output_rs_payload = (dcp_size - 1) / dcp_size * partial_out_bytes
    communication = {
        "q_allgather_receive_bytes_per_rank": q_ag_receive,
        "lse_allgather_receive_bytes_per_rank": lse_ag_receive,
        "output_reduce_scatter_bytes_per_rank": output_rs_payload,
        "q_allgather_effective_payload_gbps": quantiles(
            [
                effective_gbps(q_ag_receive, value)
                for value in overlapped["q_allgather_and_reorder_ms"]
            ]
        ),
        "lse_allgather_effective_payload_gbps": quantiles(
            [
                effective_gbps(lse_ag_receive, value)
                for value in overlapped["lse_allgather_correct_ms"]
            ]
        ),
        "output_reduce_scatter_effective_payload_gbps": quantiles(
            [
                effective_gbps(output_rs_payload, value)
                for value in overlapped["output_reduce_scatter_ms"]
            ]
        ),
    }
    memory = {
        "full_kv_bytes": full_kv_bytes,
        "local_kv_bytes": local_kv_bytes,
        "actual_kv_memory_reduction": full_kv_bytes / local_kv_bytes,
        "theoretical_kv_memory_reduction": float(dcp_size),
        "aggregate_logical_hbm_bytes_per_forward": aggregate_logical_hbm_bytes,
        "average_logical_hbm_bytes_per_gpu_per_forward": (
            average_logical_hbm_bytes_per_gpu
        ),
        "logical_hbm_bytes_per_forward": average_logical_hbm_bytes_per_gpu,
        "bandwidth_note": (
            "Group aggregate and group-average per-GPU logical effective bandwidth; "
            "not a hardware DRAM counter. logical_effective_hbm_gbps is a "
            "backward-compatible alias for average_logical_effective_hbm_gbps_per_gpu."
        ),
    }
    return {
        "workload": workload,
        "dcp_size": dcp_size,
        "batch_size": batch_size,
        "sq": sq,
        "sk_history_or_cache": sk,
        "h_q_local": h_local,
        "h_q_group": h_group,
        "h_kv_group": h_kv,
        "head_dim": d,
        "metrics": metrics,
        "communication": communication,
        "memory": memory,
    }


def print_report(report: dict[str, object]) -> None:
    print(
        "\nDCP attention-only benchmark: "
        f"workload={report['workload']} N={report['dcp_size']} "
        f"B={report['batch_size']} Sq={report['sq']} Sk={report['sk_history_or_cache']}",
        flush=True,
    )
    print(f"{'metric':42s} {'p50':>12s} {'p90':>12s}", flush=True)
    for name, values in report["metrics"].items():
        print(
            f"{name:42s} {values['p50']:12.4f} {values['p90']:12.4f}",
            flush=True,
        )
    print("JSON " + json.dumps(report, sort_keys=True), flush=True)


def run_case(
    workload: str,
    runner_sequential: DCPAttentionRunner,
    runner_overlap: DCPAttentionRunner,
    group_info: DCPGroup,
    batch_size: int,
    sq: int,
    sk: int,
    h_local: int,
    h_kv: int,
    d: int,
    num_splits: int,
    warmup: int,
    iterations: int,
    device: torch.device,
    profile_only: bool,
) -> dict[str, object] | None:
    generator = generator_for(
        5000
        + group_info.start_rank * 1009
        + group_info.size * 101
        + batch_size * 17
        + sq * 7
        + sk,
        device,
    )
    q_group = randn_bf16(
        (batch_size, sq, h_local * group_info.size, d), generator, device
    )
    q = local_q(q_group, runner_overlap.rank, h_local)
    k_full = randn_bf16((batch_size, sk, h_kv, d), generator, device)
    v_full = randn_bf16((batch_size, sk, h_kv, d), generator, device)
    k_local = shard_kv(k_full, runner_overlap.rank, group_info.size)
    v_local = shard_kv(v_full, runner_overlap.rank, group_info.size)
    local_length = (sk + group_info.size - 1 - runner_overlap.rank) // group_info.size
    local_lengths = torch.full(
        (batch_size,), local_length, device=device, dtype=torch.int32
    )

    if workload == "decode":
        full_lengths = torch.full((batch_size,), sk, device=device, dtype=torch.int32)

        def sequential_call(timing: bool):
            return runner_sequential.forward_decode(
                q,
                k_local,
                v_local,
                local_lengths,
                num_splits=num_splits,
                _record_timing=timing,
            )

        def overlap_call(timing: bool):
            return runner_overlap.forward_decode(
                q,
                k_local,
                v_local,
                local_lengths,
                num_splits=num_splits,
                _record_timing=timing,
            )

        k_reference, v_reference = k_full, v_full
    else:
        k_chunk = randn_bf16((batch_size, sq, h_kv, d), generator, device)
        v_chunk = randn_bf16((batch_size, sq, h_kv, d), generator, device)
        k_reference = torch.cat((k_full, k_chunk), dim=1)
        v_reference = torch.cat((v_full, v_chunk), dim=1)
        full_lengths = torch.full(
            (batch_size,), sk + sq, device=device, dtype=torch.int32
        )

        def sequential_call(timing: bool):
            return runner_sequential.forward_chunk_prefill(
                q,
                k_local,
                v_local,
                local_lengths,
                k_chunk,
                v_chunk,
                num_splits=num_splits,
                overlap_q_allgather=False,
                _record_timing=timing,
            )

        def overlap_call(timing: bool):
            return runner_overlap.forward_chunk_prefill(
                q,
                k_local,
                v_local,
                local_lengths,
                k_chunk,
                v_chunk,
                num_splits=num_splits,
                overlap_q_allgather=True,
                _record_timing=timing,
            )

    if profile_only:
        for _ in range(warmup):
            overlap_call(False)
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_push("dcp_profile_window")
        for _ in range(iterations):
            overlap_call(False)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(device)
        return None

    full_samples = benchmark_full_kv(
        q,
        k_reference,
        v_reference,
        full_lengths,
        num_splits,
        warmup,
        iterations,
        group_info.process_group,
        device,
    )
    sequential = benchmark_runner(
        runner_sequential,
        sequential_call,
        warmup,
        iterations,
        group_info.process_group,
        device,
    )
    overlapped = benchmark_runner(
        runner_overlap,
        overlap_call,
        warmup,
        iterations,
        group_info.process_group,
        device,
    )
    return build_report(
        workload,
        group_info.size,
        batch_size,
        sq,
        sk,
        h_local,
        h_kv,
        d,
        full_samples,
        sequential,
        overlapped,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attention-only benchmark for minimal FA3 head-sharded DCP."
    )
    parser.add_argument("--dcp-sizes", type=str, default="2,4,8")
    parser.add_argument("--workload", choices=("decode", "chunk", "both"), default="both")
    parser.add_argument("--decode-b", type=str, default="1,8,32")
    parser.add_argument("--chunk-b", type=str, default="1,4,16")
    parser.add_argument("--seqlen", type=str, default="4096,16384,65536")
    parser.add_argument("--sq", type=str, default="8,32,128")
    parser.add_argument("--qhead-local", type=int, default=8)
    parser.add_argument("--kvhead", type=int, default=1)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--num-splits", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Run only the overlapped path inside an NVTX dcp_profile_window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise SystemExit("This benchmark requires SM90 Hopper")
        if args.headdim != 128:
            raise SystemExit("--headdim must be 128")
        if (
            args.qhead_local <= 0
            or args.kvhead <= 0
            or args.qhead_local % args.kvhead
        ):
            raise SystemExit("--qhead-local must be positive and divisible by --kvhead")
        if not 0 <= args.num_splits <= 128:
            raise SystemExit("--num-splits must be in [0, 128]")
        if args.warmup < 0 or args.iters <= 0:
            raise SystemExit("--warmup must be nonnegative and --iters must be positive")

        group_sizes = parse_int_list(args.dcp_sizes, "--dcp-sizes")
        local_groups = make_groups(group_sizes, device)
        workloads = (
            ("decode", "chunk") if args.workload == "both" else (args.workload,)
        )
        for group_info in local_groups:
            sequential_runner = DCPAttentionRunner(group_info.process_group)
            overlap_runner = DCPAttentionRunner(group_info.process_group)
            for workload in workloads:
                batch_sizes = parse_int_list(
                    args.decode_b if workload == "decode" else args.chunk_b,
                    "--decode-b" if workload == "decode" else "--chunk-b",
                )
                sq_values = [1] if workload == "decode" else parse_int_list(args.sq, "--sq")
                for batch_size in batch_sizes:
                    for sq in sq_values:
                        for sk in parse_int_list(args.seqlen, "--seqlen"):
                            report = run_case(
                                workload,
                                sequential_runner,
                                overlap_runner,
                                group_info,
                                batch_size,
                                sq,
                                sk,
                                args.qhead_local,
                                args.kvhead,
                                args.headdim,
                                args.num_splits,
                                args.warmup,
                                args.iters,
                                device,
                                args.profile_only,
                            )
                            if report is not None and overlap_runner.rank == 0:
                                print_report(report)
            dist.barrier(group=group_info.process_group)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
