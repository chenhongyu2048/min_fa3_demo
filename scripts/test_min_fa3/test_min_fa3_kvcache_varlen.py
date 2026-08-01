import argparse
import math
from collections.abc import Callable

import torch

import min_fa3_op


def parse_lengths(spec: str, batch_size: int, name: str) -> list[int]:
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if len(values) == 1:
        values *= batch_size
    if len(values) != batch_size:
        raise SystemExit(f"{name} must contain one value or exactly B={batch_size} values")
    if any(value <= 0 for value in values):
        raise SystemExit(f"{name} values must be positive")
    return values


def make_cu_seqlens(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    host = torch.tensor(offsets, dtype=torch.int32)
    return host.cuda(), host


def make_inputs(
    q_lengths: list[int],
    k_lengths: list[int],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, ...]:
    cu_q, cu_q_host = make_cu_seqlens(q_lengths)
    cu_k, cu_k_host = make_cu_seqlens(k_lengths)
    q = torch.randn(sum(q_lengths), q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(k_lengths), kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    return q, k, v, cu_q, cu_k, cu_q_host, cu_k_host


def reference_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lengths: list[int],
    k_lengths: list[int],
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    q_start = 0
    k_start = 0
    qhead_per_khead = q.size(1) // k.size(1)
    scale = 1.0 / math.sqrt(q.size(2))
    for q_len, k_len in zip(q_lengths, k_lengths):
        q_i = q[q_start : q_start + q_len].float()
        k_i = k[k_start : k_start + k_len].float().repeat_interleave(qhead_per_khead, dim=1)
        v_i = v[k_start : k_start + k_len].float().repeat_interleave(qhead_per_khead, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * scale
        if is_causal:
            query_idx = torch.arange(q_len, device=q.device)[:, None]
            key_idx = torch.arange(k_len, device=q.device)[None, :]
            scores.masked_fill_(key_idx > k_len - q_len + query_idx, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probabilities, v_i))
        lses.append(torch.logsumexp(scores, dim=-1))
        q_start += q_len
        k_start += k_len
    return torch.cat(outputs, dim=0), torch.cat(lses, dim=1)


def run_packed(
    tensors: tuple[torch.Tensor, ...],
    *,
    num_splits: int,
    is_causal: bool | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v, cu_q, cu_k, cu_q_host, cu_k_host = tensors
    return min_fa3_op.forward_kvcache_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        int((cu_q_host[1:] - cu_q_host[:-1]).max()),
        int((cu_k_host[1:] - cu_k_host[:-1]).max()),
        cu_seqlens_q_host=cu_q_host,
        cu_seqlens_k_host=cu_k_host,
        num_splits=num_splits,
        return_lse=True,
        is_causal=is_causal,
    )


def run_dense_loop(
    tensors: tuple[torch.Tensor, ...],
    q_lengths: list[int],
    k_lengths: list[int],
    *,
    num_splits: int,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v, *_ = tensors
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    q_start = 0
    k_start = 0
    for q_len, k_len in zip(q_lengths, k_lengths):
        q_i = q[q_start : q_start + q_len].unsqueeze(0).contiguous()
        k_i = k[k_start : k_start + k_len].unsqueeze(0).contiguous()
        v_i = v[k_start : k_start + k_len].unsqueeze(0).contiguous()
        cache_seqlens = torch.tensor([k_len], device="cuda", dtype=torch.int32)
        out_i, lse_i = min_fa3_op.forward_kvcache(
            q_i,
            k_i,
            v_i,
            cache_seqlens,
            num_splits=num_splits,
            return_lse=True,
            is_causal=is_causal,
        )
        outputs.append(out_i.squeeze(0))
        lses.append(lse_i.squeeze(0))
        q_start += q_len
        k_start += k_len
    return torch.cat(outputs, dim=0), torch.cat(lses, dim=1)


def assert_results_close(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    torch.testing.assert_close(actual[0].float(), expected[0].float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=5e-3, rtol=3e-3)


def check_case(
    name: str,
    q_lengths: list[int],
    k_lengths: list[int],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    split_values: list[int],
    is_causal: bool,
    *,
    compare_dense: bool = True,
) -> None:
    tensors = make_inputs(q_lengths, k_lengths, q_heads, kv_heads, head_dim)
    reference = reference_packed(tensors[0], tensors[1], tensors[2], q_lengths, k_lengths, is_causal)
    for num_splits in split_values:
        actual = run_packed(tensors, num_splits=num_splits, is_causal=is_causal)
        assert_results_close(actual, reference)
        if compare_dense:
            dense = run_dense_loop(
                tensors,
                q_lengths,
                k_lengths,
                num_splits=num_splits,
                is_causal=is_causal,
            )
            assert_results_close(actual, dense)
    print(
        f"{name}: ok (q_lengths={q_lengths}, k_lengths={k_lengths}, "
        f"QH={q_heads}, KVH={kv_heads}, causal={is_causal}, splits={split_values})"
    )


def check_default_mask_semantics(q_heads: int, kv_heads: int, head_dim: int) -> None:
    decode_q = [1, 1, 1]
    decode_k = [129, 257, 513]
    decode = make_inputs(decode_q, decode_k, q_heads, kv_heads, head_dim)
    for num_splits in (0, 1, 2):
        assert_results_close(
            run_packed(decode, num_splits=num_splits, is_causal=None),
            run_packed(decode, num_splits=num_splits, is_causal=False),
        )

    mixed_q = [1, 8, 32]
    mixed_k = [129, 257, 513]
    mixed = make_inputs(mixed_q, mixed_k, q_heads, kv_heads, head_dim)
    for num_splits in (0, 1, 2):
        assert_results_close(
            run_packed(mixed, num_splits=num_splits, is_causal=None),
            run_packed(mixed, num_splits=num_splits, is_causal=True),
        )

    context_q = [5, 7]
    context_k = [3, 4]
    context = make_inputs(context_q, context_k, q_heads, kv_heads, head_dim)
    reference = reference_packed(
        context[0], context[1], context[2], context_q, context_k, is_causal=False
    )
    for num_splits in (0, 1, 2):
        actual = run_packed(context, num_splits=num_splits, is_causal=False)
        assert_results_close(actual, reference)
        assert_results_close(
            actual,
            run_dense_loop(
                context,
                context_q,
                context_k,
                num_splits=num_splits,
                is_causal=False,
            ),
        )
    print("default and explicit mask semantics: ok")


def expect_error(name: str, expected: str, fn: Callable[[], object]) -> None:
    try:
        fn()
    except (RuntimeError, TypeError, ValueError) as error:
        if expected not in str(error):
            raise AssertionError(f"{name}: expected error containing {expected!r}, got {error!r}") from error
        return
    raise AssertionError(f"{name}: expected an exception")


def check_failures(head_dim: int) -> None:
    tensors = make_inputs([2, 3], [4, 5], 4, 2, head_dim)
    q, k, v, cu_q, cu_k, cu_q_host, cu_k_host = tensors

    def call(
        q_arg: torch.Tensor = q,
        k_arg: torch.Tensor = k,
        v_arg: torch.Tensor = v,
        cu_q_arg: torch.Tensor = cu_q,
        cu_k_arg: torch.Tensor = cu_k,
        cu_q_host_arg: torch.Tensor = cu_q_host,
        cu_k_host_arg: torch.Tensor = cu_k_host,
        max_q: int = 3,
        max_k: int = 5,
        num_splits: int = 1,
        is_causal: bool | None = False,
    ) -> object:
        return min_fa3_op.forward_kvcache_varlen(
            q_arg,
            k_arg,
            v_arg,
            cu_q_arg,
            cu_k_arg,
            max_q,
            max_k,
            cu_seqlens_q_host=cu_q_host_arg,
            cu_seqlens_k_host=cu_k_host_arg,
            num_splits=num_splits,
            is_causal=is_causal,
        )

    expect_error("q dtype", "torch.bfloat16", lambda: call(q_arg=q.half()))
    expect_error("q device", "CUDA tensor", lambda: call(q_arg=q.cpu()))
    expect_error("q rank", "[total_tokens, H, 128]", lambda: call(q_arg=q.unsqueeze(0)))
    expect_error("cu dtype", "torch.int32", lambda: call(cu_q_arg=cu_q.long()))
    expect_error("cu device", "CUDA tensor", lambda: call(cu_q_arg=cu_q_host))
    expect_error("host device", "CPU tensor", lambda: call(cu_q_host_arg=cu_q))
    expect_error("kv tokens", "same total token count", lambda: call(v_arg=v[:-1].contiguous()))
    expect_error("kv heads", "same KV head count", lambda: call(v_arg=v[:, :1].contiguous()))

    q_bad_heads = torch.randn(q.size(0), 3, head_dim, device="cuda", dtype=torch.bfloat16)
    expect_error("head divisibility", "QH must be divisible by KVH", lambda: call(q_arg=q_bad_heads))
    expect_error("cu batch", "same length", lambda: call(cu_k_arg=cu_k[:-1].contiguous()))

    bad_start = cu_q_host.clone()
    bad_start[0] = 1
    expect_error("nonzero start", "start with 0", lambda: call(cu_q_host_arg=bad_start))
    bad_end = cu_q_host.clone()
    bad_end[-1] += 1
    expect_error("wrong endpoint", "total token count", lambda: call(cu_q_host_arg=bad_end))
    nonincreasing = cu_q_host.clone()
    nonincreasing[1] = 0
    expect_error("nonincreasing", "strictly increasing", lambda: call(cu_q_host_arg=nonincreasing))
    expect_error("wrong max", "maximum length", lambda: call(max_q=4))

    causal = make_inputs([5, 2], [4, 3], 4, 2, head_dim)
    expect_error(
        "causal q > k",
        "q_len <= k_len",
        lambda: min_fa3_op.forward_kvcache_varlen(
            causal[0],
            causal[1],
            causal[2],
            causal[3],
            causal[4],
            5,
            4,
            cu_seqlens_q_host=causal[5],
            cu_seqlens_k_host=causal[6],
            is_causal=True,
        ),
    )
    expect_error("negative split", "num_splits", lambda: call(num_splits=-1))
    expect_error("large split", "num_splits", lambda: call(num_splits=129))
    print("failure validation: ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test packed-varlen Hopper FA3 KV-cache forward.")
    parser.add_argument("--b", type=int, default=3, help="Batch size B")
    parser.add_argument(
        "--sq",
        type=str,
        default="1,8,32",
        help="One broadcast query length or exactly B comma-separated lengths",
    )
    parser.add_argument(
        "--seqlen",
        type=str,
        default="129,1024,3131",
        help="One broadcast KV length or exactly B comma-separated lengths",
    )
    parser.add_argument("--qhead", type=int, default=8)
    parser.add_argument("--kvhead", type=int, default=2)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--mode", choices=("auto", "nosplit", "split", "all"), default="all")
    parser.add_argument("--num-splits", type=str, default="2,3,8,128")
    parser.add_argument("--skip-failure-tests", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("This test requires an SM90 Hopper GPU")
    if args.b <= 0:
        raise SystemExit("--b must be positive")
    if args.headdim != 128:
        raise SystemExit("This test requires --headdim 128")
    if args.qhead <= 0 or args.kvhead <= 0 or args.qhead % args.kvhead:
        raise SystemExit("qhead and kvhead must be positive and qhead must be divisible by kvhead")

    q_lengths = parse_lengths(args.sq, args.b, "--sq")
    k_lengths = parse_lengths(args.seqlen, args.b, "--seqlen")
    if any(q_len > k_len for q_len, k_len in zip(q_lengths, k_lengths)):
        raise SystemExit("default causal workload requires each --sq length <= its --seqlen length")

    requested_splits = [int(token.strip()) for token in args.num_splits.split(",") if token.strip()]
    if not requested_splits or any(value < 2 or value > 128 for value in requested_splits):
        raise SystemExit("--num-splits must contain values in [2, 128]")
    split_values = {
        "auto": [0],
        "nosplit": [1],
        "split": requested_splits,
        "all": [0, 1, *requested_splits],
    }[args.mode]

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    check_case(
        "primary mixed workload",
        q_lengths,
        k_lengths,
        args.qhead,
        args.kvhead,
        args.headdim,
        split_values,
        is_causal=True,
    )

    supplemental_splits = [1, 2]
    head_cases = [(args.qhead, args.qhead), (args.qhead, args.kvhead), (args.qhead, 1)]
    seen: set[tuple[int, int]] = set()
    for q_heads, kv_heads in head_cases:
        if (q_heads, kv_heads) in seen:
            continue
        seen.add((q_heads, kv_heads))
        check_case(
            "head-mode coverage",
            [1, 8, 32],
            [129, 257, 513],
            q_heads,
            kv_heads,
            args.headdim,
            supplemental_splits,
            is_causal=True,
        )

    uniform_q_len = min(8, min(k_lengths))
    check_case(
        "uniform chunk",
        [uniform_q_len] * args.b,
        k_lengths,
        args.qhead,
        args.kvhead,
        args.headdim,
        supplemental_splits,
        is_causal=True,
    )
    check_default_mask_semantics(args.qhead, args.kvhead, args.headdim)
    if not args.skip_failure_tests:
        check_failures(args.headdim)
