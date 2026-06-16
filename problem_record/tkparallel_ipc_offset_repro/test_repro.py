import argparse
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

import _C


@dataclass
class Result:
    name: str
    ok: bool
    actual_head: list[float]
    expected_head: list[float]
    actual_storage_offset: int
    expected_storage_offset: int


def init_distributed() -> tuple[int, int]:
    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit("Run with torchrun so LOCAL_RANK and LOCAL_WORLD_SIZE are set")

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    if dist.get_world_size() != local_world_size:
        raise SystemExit(
            f"single-node repro only: world_size={dist.get_world_size()}, "
            f"local_world_size={local_world_size}"
        )
    if local_world_size < 2:
        raise SystemExit("This repro requires at least two local ranks")

    return local_rank, local_world_size


def make_expected(local_tensor: torch.Tensor, src_rank: int) -> torch.Tensor:
    expected = local_tensor.clone()
    dist.broadcast(expected, src=src_rank)
    return expected


def head_values(tensor: torch.Tensor) -> list[float]:
    return [float(x) for x in tensor.detach().flatten()[:8].float().cpu().tolist()]


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> Result:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    return Result(
        name=name,
        ok=torch.equal(actual_cpu, expected_cpu),
        actual_head=head_values(actual),
        expected_head=head_values(expected),
        actual_storage_offset=int(actual.storage_offset()),
        expected_storage_offset=int(expected.storage_offset()),
    )


def print_result(local_rank: int, result: Result) -> None:
    print(
        f"[rank {local_rank}] {result.name}: ok={result.ok}, "
        f"actual_head={result.actual_head}, expected_head={result.expected_head}, "
        f"actual_storage_offset={result.actual_storage_offset}, "
        f"expected_storage_offset={result.expected_storage_offset}",
        flush=True,
    )


def all_reduce_count(value: bool) -> int:
    tensor = torch.tensor([1 if value else 0], device="cuda", dtype=torch.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def legacy_offset_repro(rows: int, cols: int, src_rank: int, local_rank: int, local_world_size: int) -> None:
    base = torch.empty((2, rows, cols), device="cuda", dtype=torch.bfloat16)
    base[0].fill_(1.0 + 10.0 * local_rank)
    base[1].fill_(7.0 + 10.0 * local_rank)
    torch.cuda.synchronize()
    first = base[0]
    second = base[1]

    assert first.is_contiguous()
    assert second.is_contiguous()
    assert first.storage_offset() == 0
    assert second.storage_offset() == rows * cols

    first_remote = _C.TKParallelTensor(first, local_rank, local_world_size, False)
    second_remote = _C.TKParallelTensor(second, local_rank, local_world_size, False)

    expected_first = make_expected(first, src_rank)
    expected_second = make_expected(second, src_rank)

    first_out = torch.empty_like(first)
    second_out = torch.empty_like(second)
    _C.remote_copy(first_out, first_remote, src_rank)
    _C.remote_copy(second_out, second_remote, src_rank)
    torch.cuda.synchronize()

    first_result = compare("legacy first view, storage_offset=0", first_out, expected_first)
    second_result = compare("legacy second view, storage_offset>0", second_out, expected_second)
    print_result(local_rank, first_result)
    print_result(local_rank, second_result)

    first_failures = all_reduce_count(not first_result.ok)
    src_second_failures = all_reduce_count(local_rank == src_rank and not second_result.ok)
    non_src_second_mismatches = all_reduce_count(local_rank != src_rank and not second_result.ok)
    non_src_second_unexpected_ok = all_reduce_count(local_rank != src_rank and second_result.ok)

    if local_rank == 0:
        print(
            "legacy-offset summary: "
            f"first_failures={first_failures}, "
            f"src_second_failures={src_second_failures}, "
            f"non_src_second_mismatches={non_src_second_mismatches}, "
            f"non_src_second_unexpected_ok={non_src_second_unexpected_ok}",
            flush=True,
        )

    if first_failures:
        raise AssertionError("zero-offset view unexpectedly failed")
    if src_second_failures:
        raise AssertionError("source rank should read its own non-zero-offset view correctly")
    if non_src_second_mismatches != local_world_size - 1:
        raise AssertionError("legacy IPC offset issue was not reproduced on every non-source rank")
    if non_src_second_unexpected_ok:
        raise AssertionError("a non-source rank unexpectedly read the non-zero-offset view correctly")


def combined_vmm_workaround(rows: int, cols: int, src_rank: int, local_rank: int, local_world_size: int) -> None:
    combined_remote = _C.TKParallelTensor([2 * rows, cols], torch.bfloat16, local_rank, local_world_size, False)
    combined_remote.data_[:rows].fill_(1.0 + 10.0 * local_rank)
    combined_remote.data_[rows:].fill_(7.0 + 10.0 * local_rank)
    torch.cuda.synchronize()

    expected_first = make_expected(combined_remote.data_[:rows], src_rank)
    expected_second = make_expected(combined_remote.data_[rows:], src_rank)

    first_out = torch.empty((rows, cols), device="cuda", dtype=torch.bfloat16)
    second_out = torch.empty((rows, cols), device="cuda", dtype=torch.bfloat16)
    _C.remote_copy(first_out, combined_remote, src_rank, 0)
    _C.remote_copy(second_out, combined_remote, src_rank, rows)
    torch.cuda.synchronize()

    first_result = compare("combined VMM first half", first_out, expected_first)
    second_result = compare("combined VMM second half with explicit row_offset", second_out, expected_second)
    print_result(local_rank, first_result)
    print_result(local_rank, second_result)

    failures = all_reduce_count((not first_result.ok) or (not second_result.ok))
    if failures:
        raise AssertionError("combined VMM row-offset workaround failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal TKParallelTensor legacy IPC offset repro")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--src-rank", type=int, default=0)
    parser.add_argument(
        "--case",
        choices=("legacy", "combined", "both"),
        default="both",
        help="legacy reproduces the bad offset view; combined verifies the explicit-offset workaround",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank, local_world_size = init_distributed()
    if args.src_rank < 0 or args.src_rank >= local_world_size:
        raise SystemExit(f"--src-rank must be in [0, {local_world_size}), got {args.src_rank}")

    try:
        if args.case in ("legacy", "both"):
            legacy_offset_repro(args.rows, args.cols, args.src_rank, local_rank, local_world_size)
            dist.barrier()
        if args.case in ("combined", "both"):
            combined_vmm_workaround(args.rows, args.cols, args.src_rank, local_rank, local_world_size)
            dist.barrier()
        if local_rank == 0:
            print("TKParallelTensor IPC offset repro completed as expected", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
