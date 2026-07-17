import argparse
import os

import torch
import torch.distributed as dist

import min_fa3_op


def parse_shape_spec(spec: str) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise SystemExit("--shape must provide comma-separated ROWSxCOLS cases, for example 256x384,512x512")
        rows_str, cols_str = token.split("x", 1)
        rows = int(rows_str)
        cols = int(cols_str)
        cases.append((rows, cols))
    if not cases:
        raise SystemExit("--shape must provide at least one ROWSxCOLS case")
    return cases


def init_distributed() -> tuple[int, int]:
    if "LOCAL_RANK" not in os.environ or "LOCAL_WORLD_SIZE" not in os.environ:
        raise SystemExit("Run this test with torchrun so LOCAL_RANK and LOCAL_WORLD_SIZE are set")

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    if dist.get_world_size() != local_world_size:
        raise SystemExit(
            "ThunderKittens remote-load demo is single-node only: "
            f"world_size={dist.get_world_size()}, local_world_size={local_world_size}"
        )

    return local_rank, local_world_size


def make_rank_local_tensor(rows: int, cols: int, local_rank: int) -> torch.Tensor:
    base = torch.arange(rows * cols, device="cuda", dtype=torch.float32).reshape(rows, cols)
    local = base + local_rank * 1000.0
    return local.to(dtype=torch.bfloat16).contiguous()


def prepare_case_tensors(
    rows: int,
    cols: int,
    num_blocks: int | None,
    local_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    local_tensor = make_rank_local_tensor(rows, cols, local_rank)
    output = torch.empty_like(local_tensor)
    expected = torch.empty_like(local_tensor)
    resolved_num_blocks = num_blocks
    if resolved_num_blocks is None:
        resolved_num_blocks = torch.cuda.get_device_properties(local_rank).multi_processor_count
    return local_tensor, output, expected, resolved_num_blocks


def run_case(
    local_tensor: torch.Tensor,
    output: torch.Tensor,
    expected: torch.Tensor,
    src_rank: int,
    num_blocks: int,
    local_rank: int,
    local_world_size: int,
) -> None:
    input_tk = min_fa3_op.create_parallel_tensor(
        local_tensor,
        local_rank=local_rank,
        local_world_size=local_world_size,
    )

    dist.barrier()
    min_fa3_op.parallel_remote_load(output=output, input_tensor=input_tk, src_rank=src_rank, num_blocks=num_blocks)
    expected.copy_(local_tensor)
    dist.broadcast(expected, src=src_rank)
    torch.cuda.synchronize()
    dist.barrier()

    assert output.shape == local_tensor.shape, (output.shape, local_tensor.shape)
    torch.testing.assert_close(output.float(), expected.float(), atol=0.0, rtol=0.0)

    if local_rank == 0:
        print(
            f"remote-load case src_rank={src_rank}: ok "
            f"(world_size={local_world_size}, rows={local_tensor.size(0)}, cols={local_tensor.size(1)}, dtype=bf16, num_blocks={num_blocks})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal ThunderKittens remote-load demo test.")
    parser.add_argument(
        "--shape",
        type=str,
        default="256x384",
        help="Comma-separated ROWSxCOLS cases. Both dimensions must be multiples of 128.",
    )
    parser.add_argument("--src-rank", type=int, default=0, help="Source rank to read from.")
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Fixed thread-block count for the remote-load kernel. Defaults to the current device SM count.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    shape_cases = parse_shape_spec(args.shape)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (9, 0):
        raise SystemExit(f"This demo requires SM90 Hopper, got {(major, minor)}")

    local_rank, local_world_size = init_distributed()
    if args.src_rank < 0 or args.src_rank >= local_world_size:
        raise SystemExit(
            f"--src-rank must be in [0, {local_world_size}), got src_rank={args.src_rank}"
        )
    if args.num_blocks is not None and args.num_blocks <= 0:
        raise SystemExit(f"--num-blocks must be positive when provided, got num_blocks={args.num_blocks}")

    try:
        for rows, cols in shape_cases:
            if rows <= 0 or cols <= 0:
                raise SystemExit(f"Each shape dimension must be positive, got {rows}x{cols}")
            if rows % 128 != 0 or cols % 128 != 0:
                raise SystemExit(
                    f"This demo requires rows and cols to be multiples of 128, got {rows}x{cols}"
                )

            local_tensor, output, expected, resolved_num_blocks = prepare_case_tensors(
                rows,
                cols,
                args.num_blocks,
                local_rank,
            )

            run_case(
                local_tensor,
                output,
                expected,
                args.src_rank,
                resolved_num_blocks,
                local_rank,
                local_world_size,
            )
            dist.barrier()
            local_tensor = None
            output = None
            expected = None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
