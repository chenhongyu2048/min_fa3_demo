"""Build the pinned vLLM serve command for one DCP backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MEGA_BACKENDS = ("mega", "mega-fa3-native")
BACKENDS = ("vllm-ag-rs", "vllm-a2a", "mega-fa3-native", "mega")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_dir(kv_heads: int) -> Path:
    if kv_heads not in (1, 2, 4):
        raise ValueError("kv_heads must be one of 1, 2, or 4")
    return repo_root() / "infer" / "vllm_bench" / "models" / (
        f"qwen3-30b-a3b-kvh{kv_heads}"
    )


def dcp_size(kv_heads: int, tp_size: int = 8) -> int:
    if tp_size not in (4, 8) or kv_heads not in (1, 2, 4) or tp_size // kv_heads < 2:
        raise ValueError("requires TP=4/8 and DCP=TP/KVH in {2, 4, 8}")
    return tp_size // kv_heads


def build_serve_command(args: argparse.Namespace) -> tuple[dict[str, str], list[str]]:
    root = repo_root()
    plugin_src = root / "infer" / "vllm_plugin" / "src"
    python = root / ".venv" / "bin" / "python"
    if args.backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if not 1 <= args.mega_max_num_splits <= 128:
        raise ValueError("mega_max_num_splits must be in [1, 128]")
    # ``hf_overrides`` is intentionally used for smoke runs instead of
    # maintaining a second model config for each layer count. vLLM applies
    # this to the HuggingFace config before constructing the model, so changing the layer
    # count does not affect the Q/KV head topology exercised by the benchmark.
    num_hidden_layers = getattr(args, "num_hidden_layers", 48)
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be a positive integer")
    kv_cache_memory_bytes = getattr(args, "kv_cache_memory_bytes", None)
    if kv_cache_memory_bytes is not None:
        if (
            not isinstance(kv_cache_memory_bytes, int)
            or isinstance(kv_cache_memory_bytes, bool)
            or kv_cache_memory_bytes <= 0
        ):
            raise ValueError("kv_cache_memory_bytes must be a positive integer")
    hf_overrides = {"num_hidden_layers": num_hidden_layers}
    command = [
        str(python),
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model_dir(args.kv_heads)),
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--load-format",
        "dummy",
        "--skip-tokenizer-init",
        "--generation-config",
        "vllm",
        "--hf-overrides",
        json.dumps(hf_overrides, separators=(",", ":")),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(args.tp_size),
        "--enable-expert-parallel",
        "--decode-context-parallel-size",
        str(dcp_size(args.kv_heads, args.tp_size)),
        "--cp-kv-cache-interleave-size",
        "1",
        "--max-model-len",
        "131072",
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        "4096",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--compilation-config",
        json.dumps(
            {"mode": 0, "cudagraph_mode": "FULL",
             "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]},
            separators=(",", ":"),
        ),
        # The CUSTOM backend does not use FlashInfer.  Leaving this enabled on
        # Hopper makes vLLM execute an unrelated full-prefill dummy attention
        # batch during FlashInfer autotuning, which violates this benchmark's
        # decode-side-only history contract.
        "--no-enable-flashinfer-autotune",
        "--cudagraph-metrics",
        "--stream-interval",
        "1",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--kv-transfer-config",
        json.dumps(
            {
                "kv_connector": "HistoryDecodeBenchConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "fill_mean": args.fill_mean,
                    "fill_std": 0.0,
                },
            },
            separators=(",", ":"),
        ),
    ]
    command.extend(["--attention-backend", "CUSTOM"])
    if kv_cache_memory_bytes is not None:
        command.extend(["--kv-cache-memory-bytes", str(kv_cache_memory_bytes)])
    env = os.environ.copy()
    inherited_pythonpath = env.get("PYTHONPATH")
    pythonpath = [str(root), str(root / "infer"), str(plugin_src)]
    if inherited_pythonpath:
        pythonpath.append(inherited_pythonpath)
    env.update(
        {
            "VLLM_PLUGINS": "min_fa3_dcp",
            "MIN_FA3_DCP_BACKEND": args.backend,
            # The pinned MoE wheel predates padding-mask support in top-k.
            "VLLM_MOE_SKIP_PADDING": "0",
            "VLLM_MOE_ROUTING_SIMULATION_STRATEGY": env.get(
                "VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "min_fa3_balanced"
            ),
            # The benchmark uses a CUSTOM attention backend and does not
            # measure FlashInfer's unrelated sampling kernel.  Disable its
            # optional JIT by default; callers with a complete CUDA toolkit
            # can opt back in with VLLM_USE_FLASHINFER_SAMPLER=1.
            "VLLM_USE_FLASHINFER_SAMPLER": env.get(
                "VLLM_USE_FLASHINFER_SAMPLER", "0"
            ),
            # Keep all three source trees importable in the vLLM API process:
            # repository modules, the benchmark package, and the plugin's
            # src-layout package.  The explicit plugin path also repairs
            # environments whose editable-install .pth still points to the
            # pre-migration top-level vllm_plugin/src directory.
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "MEGA_DCP_MAX_TOTAL_Q": str(args.mega_max_total_q),
            "MEGA_DCP_MAX_BATCH": "64",
            "MEGA_DCP_MAX_NUM_SPLITS": str(args.mega_max_num_splits),
            "MEGA_DCP_NUM_COMM_SM": str(args.mega_num_comm_sm),
            "MEGA_DCP_BLOCK_N": args.mega_block_n,
            "MEGA_DCP_SCHEDULER_HEURISTIC": (
                "0" if args.backend == "mega-fa3-native" else "auto"
            ),
        }
    )
    return env, command


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--backend", choices=BACKENDS, required=True)
    result.add_argument("--tp-size", type=int, choices=(4, 8), default=8)
    result.add_argument("--kv-heads", type=int, choices=(1, 2, 4), default=4)
    result.add_argument("--host", default="0.0.0.0")
    result.add_argument("--port", type=int, default=8000)
    result.add_argument("--served-model-name", default="qwen3-30b-a3b-dummy")
    result.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    result.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=None,
        help=(
            "Explicit per-GPU KV-cache size in bytes. This skips vLLM's "
            "free-memory-derived KV-cache sizing/profile assertion and is "
            "useful when other processes change GPU allocations during startup."
        ),
    )
    result.add_argument(
        "--num-hidden-layers",
        type=int,
        default=48,
        help="Override the synthetic model layer count (use 1 for a low-memory smoke run).",
    )
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--fill-mean", type=float, default=0.015)
    result.add_argument("--mega-max-total-q", type=int, default=4096)
    result.add_argument("--mega-max-num-splits", type=int, default=128)
    result.add_argument("--mega-num-comm-sm", type=int, default=8)
    result.add_argument(
        "--mega-block-n", choices=("auto", "128", "176"), default="auto"
    )
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    env, command = build_serve_command(args)
    if args.dry_run:
        benchmark_env = {
            key: value
            for key, value in env.items()
            if key
            in (
                "VLLM_PLUGINS",
                "MIN_FA3_DCP_BACKEND",
                "VLLM_USE_FLASHINFER_SAMPLER",
                "VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
            )
            or key.startswith("MEGA_DCP_")
        }
        print(json.dumps({"env": benchmark_env, "command": command}, indent=2))
        return
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
