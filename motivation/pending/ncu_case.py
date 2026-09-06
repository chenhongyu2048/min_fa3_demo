"""Build or execute one NCU case command for D2."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("vllm_a2a", "mega"), required=True)
    parser.add_argument("--workload", choices=("decode", "chunk"), required=True)
    parser.add_argument("--dcp-size", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--qhead", type=int, default=32)
    parser.add_argument("--kvhead", type=int, default=1)
    parser.add_argument("--history", type=int, default=4096)
    parser.add_argument("--sq", type=int, default=None)
    parser.add_argument("--metrics", default="dram__bytes_read.sum,dram__bytes_write.sum")
    parser.add_argument("--ncu", default="/usr/local/cuda-12.8/bin/ncu")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_logs/motivation/ncu"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method == "mega" and args.workload == "decode":
        raise SystemExit("DCP Mega NCU path supports chunk-prefill only")
    q_len = 1 if args.sq is None and args.workload == "decode" else (16 if args.sq is None else args.sq)
    if args.workload == "decode" and q_len != 1:
        raise SystemExit("decode NCU cases require --sq 1")
    output_dir = args.output_dir / f"{args.workload}_{args.method}_dcp{args.dcp_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    implementations = "vllm_a2a" if args.method == "vllm_a2a" else "mega"
    command = [
        args.ncu, "--set", "full", "--target-processes", "all",
        "--graph-profiling", "graph", "--metrics", args.metrics,
        "--csv", "--log-file", str(output_dir / "ncu.csv"),
        "torchrun", "--standalone", f"--nproc_per_node={args.dcp_size}",
        "--module", "dcp_test.benchmark_dcp_varlen",
        "--b", "1", "--sq", str(q_len), "--seqlen", str(args.history),
        "--qhead", str(args.qhead), "--kvhead", str(args.kvhead),
        "--tp-size", str(args.dcp_size), "--dcp-size", str(args.dcp_size),
        "--workload", args.workload, "--implementations", implementations,
        "--no-check", "--cuda-graph", "--warmup", "2", "--iters", "1",
    ]
    payload = {"schema_version": 1, "method": args.method, "workload": args.workload, "metrics": args.metrics.split(","), "command": command, "command_shell": shlex.join(command), "scope": "single coordinated multi-rank case; profiler duration is diagnostic only", "metric_query": f"{args.ncu} --query-metrics", "execution_mode": "cuda_graph"}
    (output_dir / "command.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["command_shell"])
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
