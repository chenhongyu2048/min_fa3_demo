"""Run motivation v2 from the repository environment, with one GPU-count choice."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .config import ROOT, DATASETS, d1_config, provenance, t3_manifest, uniform_manifest, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, choices=(4, 8), required=True)
    parser.add_argument("--experiments", default="T1,T2,T3,D1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case-limit", type=int, help="Explicit smoke subset, preserving full case lengths")
    parser.add_argument("--warmup", type=int, help="Explicit override for every experiment")
    parser.add_argument("--iters", type=int, help="Explicit override for every experiment")
    parser.add_argument("--d1-manifest", type=Path, default=ROOT / "motivation" / "d1_candidates.json")
    parser.add_argument("--d1-run-id", default="0", help="Identifier for this independent process run")
    parser.add_argument("--d1-trace", action="store_true", help="Collect diagnostic replay after uninstrumented timings")
    args = parser.parse_args(argv)
    args.experiments = args.experiments.upper().split(",")
    if any(x not in ("T1", "T2", "T3", "D1") for x in args.experiments):
        parser.error("experiments must be a comma-separated subset of T1,T2,T3,D1")
    if args.case_limit is not None and args.case_limit <= 0:
        parser.error("case-limit must be positive")
    if args.warmup is not None and args.warmup < 0 or args.iters is not None and args.iters <= 0:
        parser.error("warmup must be nonnegative and iters positive")
    return args


def build_commands(args, output):
    prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone",
              f"--nproc_per_node={args.gpus}", "--module"]
    commands = []
    timings = {"T1": (40, 60), "T2": (40, 60), "T3": (10, 40), "D1": (20, 30)}
    for experiment in args.experiments:
        warmup, iters = timings[experiment]
        warmup = warmup if args.warmup is None else args.warmup
        iters = iters if args.iters is None else args.iters
        common = ["--warmup", str(warmup), "--iters", str(iters)]
        if experiment in ("T1", "T2"):
            module = "training_overlap" if experiment == "T1" else "training_step"
            for case in uniform_manifest(args.gpus)["cases"][:args.case_limit]:
                commands.append(prefix + [f"motivation.{module}", *common,
                    "--case-manifest", str(output / "uniform.json"), "--case-id", case["case_id"],
                    "--seed", str(args.seed), "--output-json", str(output / experiment / f'{case["case_id"]}.json')])
        elif experiment == "T3":
            for dataset in DATASETS:
                command = prefix + ["motivation.transformer_layer", *common, "--gpus", str(args.gpus),
                    "--manifest", str(output / "t3_cases.json"), "--dataset", dataset,
                    "--output-jsonl", str(output / "T3" / f"{dataset}.jsonl")]
                if args.case_limit is not None:
                    command += ["--case-limit", str(args.case_limit)]
                commands.append(command)
        else:
            command = prefix + ["motivation.decode_sm_trace", *common, "--gpus", str(args.gpus),
                "--case-manifest", str(args.d1_manifest.resolve()), "--run-id", args.d1_run_id,
                "--output-jsonl", str(output / "D1" / "cases.jsonl")]
            if args.case_limit is not None:
                command += ["--case-limit", str(args.case_limit)]
            if args.d1_trace:
                command += ["--trace"]
            commands.append(command)
    return commands


def main(argv=None):
    args = parse_args(argv)
    output = (args.output_dir or ROOT / "benchmark_logs" / "motivation_v2" /
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")).resolve()
    commands = build_commands(args, output)
    uniform = uniform_manifest(args.gpus)
    layer_cases = t3_manifest(args.seed) if "T3" in args.experiments else None
    print(json.dumps({"schema": "motivation.v2.run", **provenance(), "gpus": args.gpus,
                      "target_tokens": 131072, "d1": d1_config(args.gpus),
                      "uniform": uniform, "t3_manifest": layer_cases,
                      "attention": {"qhead": 32, "kvhead": 8, "headdim": 128},
                      "communication_sms": {"T1": 8, "T2": 0, "T3": 8, "D1": 4},
                      "commands": commands}, indent=2))
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", {"schema": "motivation.v2.run", **provenance(),
                                    "config": vars(args), "d1": d1_config(args.gpus), "commands": commands})
    write_json(output / "uniform.json", uniform)
    if layer_cases is not None:
        write_json(output / "t3_cases.json", layer_cases)
    env = dict(os.environ)
    for name, default in {"CUDA_DEVICE_MAX_CONNECTIONS": "8", "NCCL_CGA_CLUSTER_SIZE": "1",
                          "TORCH_NCCL_HIGH_PRIORITY": "1", "OMP_NUM_THREADS": "1"}.items():
        env.setdefault(name, default)
    for command in commands:
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
