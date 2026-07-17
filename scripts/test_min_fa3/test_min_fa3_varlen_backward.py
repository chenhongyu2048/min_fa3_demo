import argparse

import torch
import torch.nn.functional as F

import min_fa3_op


def parse_seqlen_spec(spec: str) -> list[int]:
    lengths = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not lengths:
        raise SystemExit("--seqlen must provide at least one case")
    return lengths


def make_lengths(batch_size: int, max_seqlen: int) -> list[int]:
    return [max(1, max_seqlen - (index % 3) * 17) for index in range(batch_size)]


def make_cu_seqlens(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return host.cuda(), host


def reference_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    cu_q_host: torch.Tensor,
    cu_k_host: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_ref = q.detach().float().requires_grad_()
    k_ref = k.detach().float().requires_grad_()
    v_ref = v.detach().float().requires_grad_()
    outputs = []
    for index in range(cu_q_host.numel() - 1):
        qs, qe = int(cu_q_host[index]), int(cu_q_host[index + 1])
        ks, ke = int(cu_k_host[index]), int(cu_k_host[index + 1])
        qt = q_ref[qs:qe].transpose(0, 1).unsqueeze(0)
        kt = k_ref[ks:ke].transpose(0, 1).unsqueeze(0)
        vt = v_ref[ks:ke].transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            qt,
            kt,
            vt,
            is_causal=is_causal,
            enable_gqa=qt.size(1) != kt.size(1),
        )
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.autograd.grad(torch.cat(outputs), (q_ref, k_ref, v_ref), dout.float())


def assert_grads_close(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for name, grad, ref in zip(("dq", "dk", "dv"), actual, expected):
        torch.testing.assert_close(grad.float(), ref, atol=0.3, rtol=0.3, msg=lambda msg: f"{name}: {msg}")


def run_case(args: argparse.Namespace, max_seqlen: int, is_causal: bool) -> None:
    lengths = make_lengths(args.b, max_seqlen)
    cu_q, cu_q_host = make_cu_seqlens(lengths)
    cu_k, cu_k_host = make_cu_seqlens(lengths)
    total = sum(lengths)
    q = torch.randn(total, args.qhead, args.headdim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, args.kvhead, args.headdim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    dout = torch.randn_like(q)

    out, lse = min_fa3_op.forward_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen,
        max_seqlen,
        is_causal,
        cu_seqlens_q_host=cu_q_host,
        cu_seqlens_k_host=cu_k_host,
        return_lse=True,
    )
    grads = min_fa3_op.backward_varlen(
        dout,
        q,
        k,
        v,
        out,
        lse,
        cu_q,
        cu_k,
        max_seqlen,
        max_seqlen,
        is_causal,
        deterministic=args.deterministic,
    )
    expected = reference_backward(q, k, v, dout, cu_q_host, cu_k_host, is_causal)
    assert_grads_close(grads, expected)

    buffers = (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))
    buffered = min_fa3_op.backward_varlen(
        dout,
        q,
        k,
        v,
        out,
        lse,
        cu_q,
        cu_k,
        max_seqlen,
        max_seqlen,
        is_causal,
        deterministic=args.deterministic,
        dq=buffers[0],
        dk=buffers[1],
        dv=buffers[2],
    )
    assert all(result.data_ptr() == buffer.data_ptr() for result, buffer in zip(buffered, buffers))
    assert_grads_close(buffered, expected)

    if args.deterministic:
        baseline = tuple(grad.clone() for grad in buffered)
        for _ in range(4):
            repeated = min_fa3_op.backward_varlen(
                dout,
                q,
                k,
                v,
                out,
                lse,
                cu_q,
                cu_k,
                max_seqlen,
                max_seqlen,
                is_causal,
                deterministic=True,
                dq=buffers[0],
                dk=buffers[1],
                dv=buffers[2],
            )
            assert all(torch.equal(first, current) for first, current in zip(baseline, repeated))

    print(
        f"varlen backward causal={is_causal} deterministic={args.deterministic}: ok "
        f"(lengths={lengths}, QH={args.qhead}, KVH={args.kvhead}, D={args.headdim})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the minimal Hopper FA3 varlen backward demo")
    parser.add_argument("--b", type=int, default=3)
    parser.add_argument("--seqlen", type=str, default="128,129")
    parser.add_argument("--qhead", type=int, default=8)
    parser.add_argument("--kvhead", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--mode", choices=("noncausal", "causal", "both"), default="both")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit(f"This demo requires SM90, got {torch.cuda.get_device_capability()}")
    if args.headdim != 128:
        raise SystemExit(f"This demo requires D=128, got {args.headdim}")
    if args.qhead % args.kvhead != 0:
        raise SystemExit("qhead must be divisible by kvhead")
    for seqlen in parse_seqlen_spec(args.seqlen):
        if args.mode in ("noncausal", "both"):
            run_case(args, seqlen, False)
        if args.mode in ("causal", "both"):
            run_case(args, seqlen, True)
