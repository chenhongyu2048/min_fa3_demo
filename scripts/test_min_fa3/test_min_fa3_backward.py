import argparse

import torch
import torch.nn.functional as F

import min_fa3_op


def parse_seqlen_spec(spec: str) -> list[int]:
    lengths = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not lengths:
        raise SystemExit("--seqlen must provide at least one case")
    return lengths


def reference_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_ref = q.detach().float().requires_grad_()
    k_ref = k.detach().float().requires_grad_()
    v_ref = v.detach().float().requires_grad_()
    qt = q_ref.transpose(1, 2)
    kt = k_ref.transpose(1, 2)
    vt = v_ref.transpose(1, 2)
    out = F.scaled_dot_product_attention(
        qt,
        kt,
        vt,
        is_causal=is_causal,
        enable_gqa=qt.size(1) != kt.size(1),
    ).transpose(1, 2)
    return torch.autograd.grad(out, (q_ref, k_ref, v_ref), dout.float())


def assert_grads_close(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for name, grad, ref in zip(("dq", "dk", "dv"), actual, expected):
        torch.testing.assert_close(grad.float(), ref, atol=0.3, rtol=0.3, msg=lambda msg: f"{name}: {msg}")


def run_case(args: argparse.Namespace, seqlen: int, is_causal: bool) -> None:
    q = torch.randn(args.b, seqlen, args.qhead, args.headdim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(args.b, seqlen, args.kvhead, args.headdim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    dout = torch.randn_like(q)

    out, lse = min_fa3_op.forward(q, k, v, is_causal, return_lse=True)
    grads = min_fa3_op.backward(
        dout, q, k, v, out, lse, is_causal, deterministic=args.deterministic
    )
    expected = reference_backward(q, k, v, dout, is_causal)
    assert_grads_close(grads, expected)

    buffers = (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))
    buffered = min_fa3_op.backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
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
            repeated = min_fa3_op.backward(
                dout,
                q,
                k,
                v,
                out,
                lse,
                is_causal,
                deterministic=True,
                dq=buffers[0],
                dk=buffers[1],
                dv=buffers[2],
            )
            assert all(torch.equal(first, current) for first, current in zip(baseline, repeated))

    bad_lse = torch.empty(
        (args.b, args.qhead, seqlen + 1), device="cuda", dtype=torch.float32
    )
    try:
        min_fa3_op.backward(dout, q, k, v, out, bad_lse, is_causal)
    except RuntimeError:
        pass
    else:
        raise AssertionError("backward accepted an invalid LSE shape")

    print(
        f"backward causal={is_causal} deterministic={args.deterministic}: ok "
        f"(B={args.b}, S={seqlen}, QH={args.qhead}, KVH={args.kvhead}, D={args.headdim})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the minimal Hopper FA3 backward demo")
    parser.add_argument("--b", type=int, default=2)
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
