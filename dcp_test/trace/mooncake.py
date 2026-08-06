"""Mooncake FAST'25 conversation trace parsing and prefix-cache modeling."""

from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

from .models import ReplayConfig, TraceLoadResult, TraceRequest


BLOCK_SIZE = 512


class TraceError(ValueError):
    """Raised when trace provenance or row structure is invalid."""


@dataclass(frozen=True)
class _RawRequest:
    request_id: int
    timestamp_us: int
    input_length: int
    output_length: int
    hash_ids: tuple[int, ...]


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TraceError(f"cannot read trace {path}: {exc}") from exc
    return digest.hexdigest()


def _timestamp_us(value: object, line_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceError(f"line {line_number}: timestamp must be numeric")
    try:
        timestamp = Decimal(str(value))
    except InvalidOperation as exc:
        raise TraceError(f"line {line_number}: invalid timestamp") from exc
    if not timestamp.is_finite() or timestamp < 0:
        raise TraceError(f"line {line_number}: timestamp must be finite and >= 0")
    # Mooncake's public trace stores elapsed time in milliseconds.
    return int((timestamp * 1_000).to_integral_value(rounding=ROUND_HALF_UP))


def _nonnegative_int(row: dict[str, object], name: str, line_number: int) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TraceError(f"line {line_number}: {name} must be a non-negative integer")
    return value


def _parse_rows(path: Path) -> list[_RawRequest]:
    rows: list[_RawRequest] = []
    prior_timestamp = -1
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise TraceError(f"cannot read trace {path}: {exc}") from exc
    with handle:
        for request_id, text in enumerate(handle):
            line_number = request_id + 1
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise TraceError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise TraceError(f"line {line_number}: row must be a JSON object")
            missing = {
                "timestamp",
                "input_length",
                "output_length",
                "hash_ids",
            } - row.keys()
            if missing:
                raise TraceError(
                    f"line {line_number}: missing fields: {', '.join(sorted(missing))}"
                )
            timestamp_us = _timestamp_us(row["timestamp"], line_number)
            if timestamp_us < prior_timestamp:
                raise TraceError(
                    f"line {line_number}: timestamps must be non-decreasing"
                )
            prior_timestamp = timestamp_us
            input_length = _nonnegative_int(row, "input_length", line_number)
            if input_length == 0:
                raise TraceError(f"line {line_number}: input_length must be positive")
            output_length = _nonnegative_int(row, "output_length", line_number)
            hashes = row["hash_ids"]
            if not isinstance(hashes, list):
                raise TraceError(f"line {line_number}: hash_ids must be a list")
            hash_ids: list[int] = []
            for index, block_hash in enumerate(hashes):
                if (
                    isinstance(block_hash, bool)
                    or not isinstance(block_hash, int)
                    or block_hash < 0
                ):
                    raise TraceError(
                        f"line {line_number}: hash_ids[{index}] must be a "
                        "non-negative integer"
                    )
                hash_ids.append(block_hash)
            expected_hashes = (input_length + BLOCK_SIZE - 1) // BLOCK_SIZE
            if len(hash_ids) != expected_hashes:
                raise TraceError(
                    f"line {line_number}: expected {expected_hashes} hash_ids, "
                    f"found {len(hash_ids)}"
                )
            rows.append(
                _RawRequest(
                    request_id=request_id,
                    timestamp_us=timestamp_us,
                    input_length=input_length,
                    output_length=output_length,
                    hash_ids=tuple(hash_ids),
                )
            )
    if not rows:
        raise TraceError("trace contains no requests")
    return rows


def _median_width(widths: list[int]) -> int:
    ordered = sorted(widths)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return max((ordered[middle - 1] + ordered[middle]) // 2, 1)


def _arrival_times(
    rows: list[_RawRequest], config: ReplayConfig, rng: random.Random
) -> dict[int, int]:
    if config.timestamp_policy == "preserve":
        absolute = {row.request_id: row.timestamp_us for row in rows}
    else:
        timestamps = sorted({row.timestamp_us for row in rows})
        widths = [end - begin for begin, end in zip(timestamps, timestamps[1:])]
        positive_widths = [width for width in widths if width > 0]
        fallback_width = _median_width(positive_widths) if positive_widths else 1
        width_for_timestamp = {
            timestamp: (
                timestamps[index + 1] - timestamp
                if index + 1 < len(timestamps)
                else fallback_width
            )
            for index, timestamp in enumerate(timestamps)
        }
        absolute = {
            row.request_id: row.timestamp_us
            + rng.randrange(max(width_for_timestamp[row.timestamp_us], 1))
            for row in rows
        }
    origin = min(absolute.values())
    scaled: dict[int, int] = {}
    for request_id, timestamp in absolute.items():
        elapsed = Decimal(timestamp - origin) / config.arrival_time_scale
        scaled[request_id] = int(elapsed.to_integral_value(rounding=ROUND_HALF_UP))
    return scaled


def load_mooncake_trace(
    config: ReplayConfig, arrival_rng: random.Random
) -> TraceLoadResult:
    """Verify and normalize the public Mooncake conversation trace."""
    actual_sha256 = compute_sha256(config.trace_path)
    if actual_sha256 != config.trace_sha256:
        raise TraceError(
            "trace SHA-256 mismatch: expected "
            f"{config.trace_sha256}, found {actual_sha256}"
        )
    rows = _parse_rows(config.trace_path)
    arrival_times = _arrival_times(rows, config, arrival_rng)
    dropped_model_len = 0
    requests: list[TraceRequest] = []
    for row in rows:
        if row.input_length + row.output_length > config.max_model_len:
            dropped_model_len += 1
            continue
        requests.append(
            TraceRequest(
                request_id=row.request_id,
                source_timestamp_us=row.timestamp_us,
                arrival_us=arrival_times[row.request_id],
                input_length=row.input_length,
                output_length=row.output_length,
                hash_ids=row.hash_ids,
            )
        )
    requests.sort(key=lambda request: (request.arrival_us, request.request_id))
    return TraceLoadResult(
        requests=tuple(requests),
        trace_sha256=actual_sha256,
        total_rows=len(rows),
        dropped_model_len=dropped_model_len,
    )


class PrefixBlockCache:
    """Finite LRU for trace-proven complete prompt blocks."""

    def __init__(self, capacity_blocks: int) -> None:
        self.capacity_blocks = capacity_blocks
        self._blocks: OrderedDict[int, None] = OrderedDict()

    def lookup_prefix(self, request: TraceRequest) -> int:
        max_blocks = (request.input_length - 1) // BLOCK_SIZE
        matched = 0
        for block_hash in request.hash_ids[:max_blocks]:
            if block_hash not in self._blocks:
                break
            self._blocks.move_to_end(block_hash)
            matched += 1
        return matched * BLOCK_SIZE

    def insert_completed(
        self, request: TraceRequest, old_computed: int, new_computed: int
    ) -> None:
        full_blocks = request.input_length // BLOCK_SIZE
        for index in range(full_blocks):
            block_end = (index + 1) * BLOCK_SIZE
            if old_computed < block_end <= new_computed:
                self._insert(request.hash_ids[index])

    def _insert(self, block_hash: int) -> None:
        if self.capacity_blocks == 0:
            return
        if block_hash in self._blocks:
            self._blocks.move_to_end(block_hash)
        else:
            self._blocks[block_hash] = None
        while len(self._blocks) > self.capacity_blocks:
            self._blocks.popitem(last=False)

    def __len__(self) -> int:
        return len(self._blocks)
