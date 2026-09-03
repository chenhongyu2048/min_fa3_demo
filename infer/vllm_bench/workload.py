"""Convert the Mooncake conversation trace into decode-side HTTP requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable

BLOCK_SIZE = 512
TOKEN_UPPER_BOUND = 128000


class WorkloadError(ValueError):
    """Raised when a source trace or workload manifest is invalid."""


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: int
    source_timestamp_us: int
    input_length: int
    output_length: int
    history_tokens: int
    hash_ids: tuple[int, ...]
    lineage_request_id: int | None

    @property
    def chunk_tokens(self) -> int:
        return self.input_length - self.history_tokens


@dataclass(frozen=True)
class WorkloadManifest:
    trace_sha256: str
    token_seed: int
    max_model_len: int
    ambiguous_dropped: int
    model_length_dropped: int
    warmup: tuple[WorkloadRequest, ...]
    measured: tuple[WorkloadRequest, ...]


@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"]
    latest_request_id: int | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_us(value: object, line_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkloadError(f"line {line_number}: timestamp must be numeric")
    try:
        timestamp = Decimal(str(value))
    except InvalidOperation as exc:
        raise WorkloadError(f"line {line_number}: invalid timestamp") from exc
    if not timestamp.is_finite() or timestamp < 0:
        raise WorkloadError(f"line {line_number}: timestamp must be finite and >= 0")
    return int((timestamp * 1000).to_integral_value(rounding=ROUND_HALF_UP))


def _positive_int(row: dict[str, object], name: str, line_number: int) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkloadError(f"line {line_number}: {name} must be a positive integer")
    return value


def load_trace(path: str | Path) -> tuple[str, list[WorkloadRequest], int, int]:
    """Load rows, infer prior conversation lineage, and drop ambiguous rows."""
    trace_path = Path(path)
    root = _TrieNode(children={})
    requests: list[WorkloadRequest] = []
    ambiguous_dropped = 0
    model_length_dropped = 0
    prior_timestamp = -1
    with trace_path.open("r", encoding="utf-8") as handle:
        for request_id, text in enumerate(handle):
            line_number = request_id + 1
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise WorkloadError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise WorkloadError(f"line {line_number}: row must be an object")
            timestamp_us = _timestamp_us(row.get("timestamp"), line_number)
            if timestamp_us < prior_timestamp:
                raise WorkloadError("trace timestamps must be non-decreasing")
            prior_timestamp = timestamp_us
            input_length = _positive_int(row, "input_length", line_number)
            output_length = _positive_int(row, "output_length", line_number)
            hashes = row.get("hash_ids")
            if not isinstance(hashes, list) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in hashes
            ):
                raise WorkloadError(
                    f"line {line_number}: hash_ids must contain non-negative integers"
                )
            expected = (input_length + BLOCK_SIZE - 1) // BLOCK_SIZE
            if len(hashes) != expected:
                raise WorkloadError(
                    f"line {line_number}: expected {expected} hashes, "
                    f"found {len(hashes)}"
                )
            if input_length + output_length > 131072:
                model_length_dropped += 1
                continue

            node = root
            lineage_id: int | None = None
            matched_blocks = 0
            for depth, block_hash in enumerate(hashes, start=1):
                child = node.children.get(block_hash)
                if child is None:
                    break
                node = child
                if depth >= 2 and node.latest_request_id is not None:
                    lineage_id = node.latest_request_id
                    matched_blocks = depth
            history_tokens = matched_blocks * BLOCK_SIZE
            if lineage_id is None:
                history_tokens = input_length - 1
            elif matched_blocks == len(hashes):
                ambiguous_dropped += 1
                history_tokens = -1
            if history_tokens >= 0:
                requests.append(
                    WorkloadRequest(
                        request_id=request_id,
                        source_timestamp_us=timestamp_us,
                        input_length=input_length,
                        output_length=output_length,
                        history_tokens=history_tokens,
                        hash_ids=tuple(hashes),
                        lineage_request_id=lineage_id,
                    )
                )

            node = root
            for block_hash in hashes:
                node = node.children.setdefault(block_hash, _TrieNode(children={}))
                node.latest_request_id = request_id
    return _sha256(trace_path), requests, ambiguous_dropped, model_length_dropped


def build_manifest(
    trace_path: str | Path,
    *,
    warmup_requests: int = 100,
    measured_requests: int = 1000,
    token_seed: int = 42,
    seed: int = 42,
) -> WorkloadManifest:
    if warmup_requests < 0 or measured_requests <= 0:
        raise WorkloadError("warmup must be non-negative and measured must be positive")
    digest, requests, ambiguous, model_length = load_trace(trace_path)
    required = warmup_requests + measured_requests
    if len(requests) < required:
        raise WorkloadError(f"need {required} eligible requests, found {len(requests)}")
    start = random.Random(seed).randrange(len(requests) - required + 1)
    selected = requests[start : start + required]
    return WorkloadManifest(
        trace_sha256=digest,
        token_seed=token_seed,
        max_model_len=131072,
        ambiguous_dropped=ambiguous,
        model_length_dropped=model_length,
        warmup=tuple(selected[:warmup_requests]),
        measured=tuple(selected[warmup_requests:]),
    )


def _block_tokens(block_hash: int, token_seed: int) -> list[int]:
    digest = hashlib.sha256(f"{token_seed}:{block_hash}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:16], "big"))
    return [rng.randrange(TOKEN_UPPER_BOUND) for _ in range(BLOCK_SIZE)]


def prompt_token_ids(request: WorkloadRequest, token_seed: int) -> list[int]:
    tokens: list[int] = []
    for block_hash in request.hash_ids:
        tokens.extend(_block_tokens(block_hash, token_seed))
    return tokens[: request.input_length]


def request_to_dict(request: WorkloadRequest) -> dict[str, object]:
    raw = asdict(request)
    raw["hash_ids"] = list(request.hash_ids)
    raw["chunk_tokens"] = request.chunk_tokens
    return raw


def manifest_to_dict(manifest: WorkloadManifest) -> dict[str, object]:
    return {
        "schema_version": 1,
        "trace_sha256": manifest.trace_sha256,
        "token_seed": manifest.token_seed,
        "max_model_len": manifest.max_model_len,
        "ambiguous_dropped": manifest.ambiguous_dropped,
        "model_length_dropped": manifest.model_length_dropped,
        "warmup": [request_to_dict(request) for request in manifest.warmup],
        "measured": [request_to_dict(request) for request in manifest.measured],
    }


def _requests(raw: Iterable[dict[str, object]]) -> tuple[WorkloadRequest, ...]:
    return tuple(
        WorkloadRequest(
            request_id=int(item["request_id"]),
            source_timestamp_us=int(item["source_timestamp_us"]),
            input_length=int(item["input_length"]),
            output_length=int(item["output_length"]),
            history_tokens=int(item["history_tokens"]),
            hash_ids=tuple(int(value) for value in item["hash_ids"]),
            lineage_request_id=(
                None
                if item.get("lineage_request_id") is None
                else int(item["lineage_request_id"])
            ),
        )
        for item in raw
    )


def load_manifest(path: str | Path) -> WorkloadManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("schema_version") != 1:
        raise WorkloadError("unsupported workload manifest schema")
    return WorkloadManifest(
        trace_sha256=str(raw["trace_sha256"]),
        token_seed=int(raw["token_seed"]),
        max_model_len=int(raw["max_model_len"]),
        ambiguous_dropped=int(raw["ambiguous_dropped"]),
        model_length_dropped=int(raw["model_length_dropped"]),
        warmup=_requests(raw["warmup"]),
        measured=_requests(raw["measured"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-requests", type=int, default=100)
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--token-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_manifest(
        args.trace,
        warmup_requests=args.warmup_requests,
        measured_requests=args.num_requests,
        token_seed=args.token_seed,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest_to_dict(manifest), indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(manifest.warmup)} warmup and {len(manifest.measured)} "
        f"measured requests to {args.output}"
    )
    print(
        f"dropped ambiguous={manifest.ambiguous_dropped}, "
        f"model_length={manifest.model_length_dropped}"
    )


if __name__ == "__main__":
    main()
