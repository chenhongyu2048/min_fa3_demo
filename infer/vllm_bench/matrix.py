"""Run the formal backend/load matrix with isolated vLLM server restarts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from .client import _write_csv, _write_results, execute, prepare_requests, summarize
from .serve import (
    BACKENDS,
    MEGA_BACKENDS,
    build_serve_command,
    parser as serve_parser,
)
from .workload import load_manifest


def _parse_positive_int_list(value: str) -> tuple[int, ...]:
    """Parse a comma- or whitespace-separated list of communication SMs."""
    tokens = value.replace(",", " ").split()
    if not tokens:
        raise argparse.ArgumentTypeError("list must not be empty")
    result: list[int] = []
    for token in tokens:
        try:
            parsed = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid integer value {token!r}"
            ) from exc
        if parsed <= 0 or parsed >= 132:
            raise argparse.ArgumentTypeError(
                "values must be positive integers below 132"
            )
        if parsed in result:
            raise argparse.ArgumentTypeError(f"duplicate value {parsed}")
        result.append(parsed)
    return tuple(result)


def _healthy(url: str) -> bool:
    try:
        with urlopen(f"{url.rstrip('/')}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def _serves_model(url: str, model: str) -> bool:
    try:
        with urlopen(f"{url.rstrip('/')}/v1/models", timeout=2) as response:
            if response.status != 200:
                return False
            payload = json.load(response)
    except (OSError, URLError, json.JSONDecodeError):
        return False
    models = payload.get("data") if isinstance(payload, dict) else None
    return isinstance(models, list) and any(
        isinstance(item, dict) and item.get("id") == model for item in models
    )


def _port_in_use(url: str) -> bool:
    parsed = urlsplit(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _server_log_tail(path: Path, lines: int = 80) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def _wait_for_server(
    process: subprocess.Popen,
    url: str,
    timeout_s: float,
    log_path: Path | None = None,
    expected_model: str | None = None,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            message = f"vLLM server exited with code {process.returncode}"
            if log_path is not None:
                tail = _server_log_tail(log_path)
                if tail:
                    message += f"\nLast server log lines:\n{tail}"
            raise RuntimeError(message)
        if _healthy(url) and (
            expected_model is None or _serves_model(url, expected_model)
        ):
            return
        time.sleep(1)
    raise TimeoutError(f"vLLM server did not become healthy within {timeout_s}s")


def _stop(process: subprocess.Popen, timeout_s: float = 60) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _runtime_versions(root: Path) -> dict:
    code = """
import json
import platform
from importlib.metadata import version

import torch

devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": props.name,
            "capability": [props.major, props.minor],
            "total_memory": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
        })
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "vllm": version("vllm"),
    "devices": devices,
}))
"""
    raw = subprocess.check_output(
        [str(root / ".venv/bin/python"), "-c", code], text=True
    )
    return json.loads(raw)


def _run_client(
    manifest, backend: str, scale: float, url: str, model: str, output: Path
):
    warmup = prepare_requests(
        manifest.warmup,
        scale=scale,
        model=model,
        token_seed=manifest.token_seed,
        request_prefix=f"warmup-{backend}-{scale}",
    )
    warmup_results = asyncio.run(execute(warmup, url))
    _write_results(output / "warmup.jsonl", warmup_results)
    failures = [result for result in warmup_results if not result.success]
    if failures:
        details = "; ".join(
            f"request {result.request_id}: HTTP {result.http_status}: "
            f"{result.error or 'unknown error'}"
            for result in failures[:3]
        )
        raise RuntimeError(
            "warmup failed; measured workload was not started; " + details
        )
    measured = prepare_requests(
        manifest.measured,
        scale=scale,
        model=model,
        token_seed=manifest.token_seed,
        request_prefix=f"measured-{backend}-{scale}",
    )
    start = time.monotonic()
    results = asyncio.run(execute(measured, url))
    elapsed = time.monotonic() - start
    _write_results(output / "requests.jsonl", results)
    _write_csv(output / "requests.csv", results)
    return summarize(results, backend=backend, scale=scale, elapsed_s=elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    parser.add_argument(
        "--arrival-time-scales", nargs="+", type=float, default=[1, 2, 4]
    )
    parser.add_argument("--kv-heads", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument(
        "--mega-num-comm-sms",
        type=_parse_positive_int_list,
        default=None,
        help=(
            "Comma- or whitespace-separated Mega-backend communication-SM "
            "sweep; vLLM-style baselines run once per load point."
        ),
    )
    parser.add_argument(
        "--mega-max-num-splits",
        type=int,
        default=128,
        help="Preallocated Mega split-workspace capacity in [1, 128].",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="vLLM per-GPU memory target; lower this when GPUs are shared.",
    )
    parser.add_argument(
        "--kv-cache-memory-bytes", type=int, default=None,
        help="Explicit per-GPU KV-cache size; useful with changing co-tenant usage.",
    )
    parser.add_argument(
        "--num-hidden-layers", type=int, default=32,
        help="Synthetic Llama layer count passed through --hf-overrides.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-start-timeout", type=float, default=900)
    args = parser.parse_args()
    if args.mega_num_comm_sms is not None and not any(
        backend in MEGA_BACKENDS for backend in args.backends
    ):
        parser.error(
            "--mega-num-comm-sms requires a Mega backend "
            "(mega or mega-fa3-native)"
        )
    if not 1 <= args.mega_max_num_splits <= 128:
        parser.error("--mega-max-num-splits must be in [1, 128]")
    # ``infer/`` is an archive directory; the repository root is two levels
    # above this module (``infer/vllm_bench/matrix.py``).
    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(args.workload)
    workload_hash = hashlib.sha256(args.workload.read_bytes()).hexdigest()
    runtime_versions = _runtime_versions(root)
    url = f"http://{args.host}:{args.port}"
    summaries = []
    backend_variants: list[tuple[str, int | None]] = []
    for backend in args.backends:
        if backend in MEGA_BACKENDS and args.mega_num_comm_sms is not None:
            backend_variants.extend(
                (backend, comm_sm) for comm_sm in args.mega_num_comm_sms
            )
        else:
            backend_variants.append(
                (backend, 8 if backend in MEGA_BACKENDS else None)
            )
    for backend, comm_sm in backend_variants:
        for scale in args.arrival_time_scales:
            run_dir_name = f"{backend}-scale{scale:g}"
            if backend in MEGA_BACKENDS and args.mega_num_comm_sms is not None:
                run_dir_name = f"{backend}-comm_sm{comm_sm}-scale{scale:g}"
            run_dir = args.result_dir / run_dir_name
            run_dir.mkdir(parents=True, exist_ok=True)
            serve_args = serve_parser().parse_args(
                [
                    "--backend",
                    backend,
                    "--kv-heads",
                    str(args.kv_heads),
                    "--mega-num-comm-sm",
                    str(comm_sm or 8),
                    "--mega-max-num-splits",
                    str(args.mega_max_num_splits),
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                    *(
                        ["--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes)]
                        if args.kv_cache_memory_bytes is not None
                        else []
                    ),
                    "--num-hidden-layers",
                    str(args.num_hidden_layers),
                ]
            )
            env, command = build_serve_command(serve_args)
            if _port_in_use(url):
                raise RuntimeError(
                    f"port {serve_args.port} is already in use at {url}; "
                    "choose another --port or set PORT"
                )
            run_manifest = {
                "backend": backend,
                "arrival_time_scale": scale,
                "kv_heads": args.kv_heads,
                "tp_size": 8,
                "dcp_size": 8 // args.kv_heads,
                "mega_num_comm_sm": comm_sm,
                "mega_max_num_splits": args.mega_max_num_splits,
                "num_hidden_layers": args.num_hidden_layers,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
                "workload_sha256": workload_hash,
                "trace_sha256": manifest.trace_sha256,
                "repository_commit": _git_revision(root),
                "vllm_commit": _git_revision(root / "third_party" / "vllm"),
                "model_config_sha256": hashlib.sha256(
                    (
                        root
                        / "infer"
                        / "vllm_bench"
                        / "models"
                        / f"llama-3.1-8b-kvh{args.kv_heads}"
                        / "config.json"
                    ).read_bytes()
                ).hexdigest(),
                "runtime": runtime_versions,
                "command": command,
                "backend_env": {
                    key: value
                    for key, value in env.items()
                    if key == "MIN_FA3_DCP_BACKEND"
                    or key.startswith("MEGA_DCP_")
                },
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
            )
            with (run_dir / "server.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    env=env,
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    _wait_for_server(
                        process,
                        url,
                        args.server_start_timeout,
                        run_dir / "server.log",
                        serve_args.served_model_name,
                    )
                    summary = _run_client(
                        manifest,
                        backend,
                        scale,
                        url,
                        serve_args.served_model_name,
                        run_dir,
                    )
                    summary["workload_sha256"] = workload_hash
                    summary["mega_num_comm_sm"] = comm_sm
                    (run_dir / "summary.json").write_text(
                        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
                    )
                    summaries.append(summary)
                except Exception as exc:
                    run_manifest["failure"] = f"{type(exc).__name__}: {exc}"
                    (run_dir / "manifest.json").write_text(
                        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
                    )
                    raise
                finally:
                    _stop(process)
    (args.result_dir / "matrix_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
