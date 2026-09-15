"""Replay a fixed token-ID workload and measure token-level vLLM latency."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .workload import WorkloadRequest, load_manifest, prompt_token_ids


@dataclass(frozen=True)
class PreparedRequest:
    request: WorkloadRequest
    scheduled_offset_s: float
    body: bytes


@dataclass
class RequestResult:
    request_id: int
    source_timestamp_us: int
    input_length: int
    history_tokens: int
    chunk_tokens: int
    requested_output_length: int
    scheduled_offset_s: float
    dispatch_lag_s: float
    request_start_s: float
    token_ids: list[int]
    token_timestamps_s: list[float]
    ttft_s: float | None
    itl_s: list[float]
    e2e_s: float
    http_status: int | None
    success: bool
    tbt_valid: bool
    error: str | None


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "p99": _quantile(values, 0.99),
    }


def _relative_offsets(
    requests: tuple[WorkloadRequest, ...], scale: float
) -> list[float]:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("arrival_time_scale must be positive and finite")
    if not requests:
        return []
    origin = requests[0].source_timestamp_us
    return [
        (request.source_timestamp_us - origin) / 1_000_000 / scale
        for request in requests
    ]


def prepare_requests(
    requests: tuple[WorkloadRequest, ...],
    *,
    scale: float,
    model: str,
    token_seed: int,
    request_prefix: str,
) -> list[PreparedRequest]:
    offsets = _relative_offsets(requests, scale)
    prepared: list[PreparedRequest] = []
    for request, offset in zip(requests, offsets):
        payload = {
            "model": model,
            "prompt": prompt_token_ids(request, token_seed),
            "max_tokens": request.output_length,
            "min_tokens": request.output_length,
            "temperature": 0.0,
            "ignore_eos": True,
            "add_special_tokens": False,
            "stream": True,
            "return_token_ids": True,
            "request_id": f"{request_prefix}-{request.request_id}",
            "kv_transfer_params": {"history_tokens": request.history_tokens},
        }
        prepared.append(
            PreparedRequest(
                request=request,
                scheduled_offset_s=offset,
                body=json.dumps(payload, separators=(",", ":")).encode(),
            )
        )
    return prepared


async def _send_one(
    session, url: str, item: PreparedRequest, epoch: float
) -> RequestResult:
    scheduled = epoch + item.scheduled_offset_s
    await asyncio.sleep(max(0.0, scheduled - time.monotonic()))
    start = time.monotonic()
    token_ids: list[int] = []
    timestamps: list[float] = []
    status: int | None = None
    error: str | None = None
    tbt_valid = True
    success = False
    try:
        async with session.post(
            url,
            data=item.body,
            headers={"Content-Type": "application/json"},
        ) as response:
            status = response.status
            if status != 200:
                error = (await response.text())[:2000]
            else:
                buffer = b""
                async for chunk in response.content.iter_any():
                    buffer += chunk
                    while b"\n\n" in buffer:
                        event, buffer = buffer.split(b"\n\n", 1)
                        for line in event.splitlines():
                            if not line.startswith(b"data:"):
                                continue
                            data = line[5:].strip()
                            if not data or data == b"[DONE]":
                                continue
                            message = json.loads(data)
                            if "error" in message:
                                error = json.dumps(message["error"], ensure_ascii=False)
                                continue
                            choices = message.get("choices") or []
                            for choice in choices:
                                delta = choice.get("token_ids") or []
                                if len(delta) > 1:
                                    tbt_valid = False
                                now = time.monotonic()
                                token_ids.extend(int(token) for token in delta)
                                timestamps.extend(now for _ in delta)
                success = len(token_ids) == item.request.output_length
                if not success:
                    error = (
                        f"expected {item.request.output_length} output tokens, "
                        f"received {len(token_ids)}"
                    )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    end = time.monotonic()
    if not success:
        tbt_valid = False
    ttft = timestamps[0] - start if timestamps else None
    itl = [end - begin for begin, end in zip(timestamps, timestamps[1:])]
    request = item.request
    return RequestResult(
        request_id=request.request_id,
        source_timestamp_us=request.source_timestamp_us,
        input_length=request.input_length,
        history_tokens=request.history_tokens,
        chunk_tokens=request.chunk_tokens,
        requested_output_length=request.output_length,
        scheduled_offset_s=item.scheduled_offset_s,
        dispatch_lag_s=start - scheduled,
        request_start_s=start - epoch,
        token_ids=token_ids,
        token_timestamps_s=[stamp - epoch for stamp in timestamps],
        ttft_s=ttft,
        itl_s=itl,
        e2e_s=end - start,
        http_status=status,
        success=success,
        tbt_valid=tbt_valid,
        error=error,
    )


async def execute(
    prepared: list[PreparedRequest], server_url: str
) -> list[RequestResult]:
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError(
            "aiohttp is required; run this in the vLLM environment"
        ) from exc
    timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=None)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        epoch = time.monotonic() + 0.1
        tasks = [
            asyncio.create_task(
                _send_one(
                    session,
                    f"{server_url.rstrip('/')}/v1/completions",
                    item,
                    epoch,
                )
            )
            for item in prepared
        ]
        return await asyncio.gather(*tasks)


def summarize(
    results: list[RequestResult], *, backend: str, scale: float, elapsed_s: float
) -> dict[str, Any]:
    successful = [result for result in results if result.success]
    tbt_valid = [result for result in successful if result.tbt_valid]
    ttft = [result.ttft_s for result in successful if result.ttft_s is not None]
    pooled_itl = [value for result in tbt_valid for value in result.itl_s]
    per_request_tbt = [
        statistics.fmean(result.itl_s) for result in tbt_valid if result.itl_s
    ]
    e2e = [result.e2e_s for result in successful]
    dispatch_lag = [result.dispatch_lag_s for result in results]
    scheduled_span = max((result.scheduled_offset_s for result in results), default=0.0)
    output_tokens = sum(len(result.token_ids) for result in successful)
    offered_rps = (
        (len(results) - 1) / scheduled_span
        if scheduled_span > 0 and len(results) > 1
        else None
    )
    return {
        "backend": backend,
        "arrival_time_scale": scale,
        "scheduled_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "tbt_valid_requests": len(tbt_valid),
        "scheduled_span_s": scheduled_span,
        "elapsed_s": elapsed_s,
        "offered_rps": offered_rps,
        "achieved_rps": len(successful) / elapsed_s if elapsed_s > 0 else None,
        "output_tokens": output_tokens,
        "output_token_throughput": output_tokens / elapsed_s if elapsed_s > 0 else None,
        "ttft_s": _stats(ttft),
        "pooled_itl_s": _stats(pooled_itl),
        "per_request_mean_tbt_s": _stats(per_request_tbt),
        "e2e_s": _stats(e2e),
        "dispatch_lag_s": _stats(dispatch_lag),
    }


def _write_results(path: Path, results: list[RequestResult]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), separators=(",", ":")) + "\n")


def _write_csv(path: Path, results: list[RequestResult]) -> None:
    columns = [
        "request_id",
        "input_length",
        "history_tokens",
        "chunk_tokens",
        "requested_output_length",
        "success",
        "tbt_valid",
        "ttft_s",
        "e2e_s",
        "http_status",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            raw = asdict(result)
            writer.writerow({key: raw[key] for key in columns})


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workload", type=Path, required=True)
    result.add_argument("--server-url", default="http://127.0.0.1:8000")
    result.add_argument("--model", default="qwen3-30b-a3b-dummy")
    result.add_argument("--backend", required=True)
    result.add_argument("--arrival-time-scale", type=float, required=True)
    result.add_argument("--result-dir", type=Path, required=True)
    result.add_argument("--skip-warmup", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    manifest = load_manifest(args.workload)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_warmup and manifest.warmup:
        warmup = prepare_requests(
            manifest.warmup,
            scale=args.arrival_time_scale,
            model=args.model,
            token_seed=manifest.token_seed,
            request_prefix=f"warmup-{args.backend}-{args.arrival_time_scale}",
        )
        warmup_results = asyncio.run(execute(warmup, args.server_url))
        _write_results(args.result_dir / "warmup.jsonl", warmup_results)
        if not all(result.success for result in warmup_results):
            failures = sum(not result.success for result in warmup_results)
            raise RuntimeError(f"{failures} warmup requests failed")

    prepared = prepare_requests(
        manifest.measured,
        scale=args.arrival_time_scale,
        model=args.model,
        token_seed=manifest.token_seed,
        request_prefix=f"measured-{args.backend}-{args.arrival_time_scale}",
    )
    start = time.monotonic()
    results = asyncio.run(execute(prepared, args.server_url))
    elapsed = time.monotonic() - start
    summary = summarize(
        results, backend=args.backend, scale=args.arrival_time_scale, elapsed_s=elapsed
    )
    summary["workload_sha256"] = hashlib.sha256(args.workload.read_bytes()).hexdigest()
    _write_results(args.result_dir / "requests.jsonl", results)
    _write_csv(args.result_dir / "requests.csv", results)
    (args.result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
