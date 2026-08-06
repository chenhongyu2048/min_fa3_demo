"""Validated configuration and trace records for workload replay."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mega_dcp_workload/v1"
SOURCE_NAME = "mooncake_kimi_conversation_fast25"


class ConfigError(ValueError):
    """Raised when a replay configuration is missing or inconsistent."""


@dataclass(frozen=True)
class ReplayConfig:
    trace_path: Path
    trace_sha256: str
    timestamp_policy: str
    arrival_time_scale: Decimal
    sampling_start_ms: int
    sampling_end_ms: int
    fixed_step_us: int
    num_cases: int
    seed: int
    max_num_seqs: int
    max_num_batched_tokens: int
    prefill_chunk_size: int
    max_model_len: int
    dcp_size: int
    prefix_cache_capacity_blocks: int
    num_speculative_tokens: int
    scheduled_q_rule: str
    accepted_draft_pmf: tuple[float, ...]
    config_sha256: str

    @property
    def mtp_query_len(self) -> int:
        if self.scheduled_q_rule == "target_plus_drafts":
            return 1 + self.num_speculative_tokens
        return self.num_speculative_tokens

    @property
    def max_accepted_drafts(self) -> int:
        return len(self.accepted_draft_pmf) - 1


@dataclass(frozen=True)
class TraceRequest:
    request_id: int
    source_timestamp_us: int
    arrival_us: int
    input_length: int
    output_length: int
    hash_ids: tuple[int, ...]


@dataclass(frozen=True)
class TraceLoadResult:
    requests: tuple[TraceRequest, ...]
    trace_sha256: str
    total_rows: int
    dropped_model_len: int


_CONFIG_FIELDS = {
    "trace_path",
    "trace_sha256",
    "timestamp_policy",
    "arrival_time_scale",
    "sampling_start_ms",
    "sampling_end_ms",
    "fixed_step_us",
    "num_cases",
    "seed",
    "max_num_seqs",
    "max_num_batched_tokens",
    "prefill_chunk_size",
    "max_model_len",
    "dcp_size",
    "prefix_cache_capacity_blocks",
    "num_speculative_tokens",
    "scheduled_q_rule",
    "accepted_draft_pmf",
}


def _integer(raw: dict[str, Any], name: str, *, minimum: int) -> int:
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def _decimal(raw: dict[str, Any], name: str) -> Decimal:
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"{name} must be a positive number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ConfigError(f"{name} must be a positive number") from exc
    if not result.is_finite() or result <= 0:
        raise ConfigError(f"{name} must be a positive finite number")
    return result


def _probabilities(raw: dict[str, Any], expected: int) -> tuple[float, ...]:
    values = raw["accepted_draft_pmf"]
    if not isinstance(values, list) or len(values) != expected:
        raise ConfigError(
            "accepted_draft_pmf must contain exactly "
            f"{expected} probabilities for this scheduled_q_rule"
        )
    probabilities: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"accepted_draft_pmf[{index}] must be numeric")
        probability = float(value)
        if not math.isfinite(probability) or probability < 0:
            raise ConfigError(
                f"accepted_draft_pmf[{index}] must be finite and non-negative"
            )
        probabilities.append(probability)
    if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-9):
        raise ConfigError("accepted_draft_pmf probabilities must sum to 1")
    return tuple(probabilities)


def load_config(path: str | Path) -> ReplayConfig:
    """Load a strict JSON config, resolving trace_path beside the config file."""
    config_path = Path(path).resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a JSON object")
    missing = sorted(_CONFIG_FIELDS - raw.keys())
    extra = sorted(raw.keys() - _CONFIG_FIELDS)
    if missing:
        raise ConfigError(f"missing configuration fields: {', '.join(missing)}")
    if extra:
        raise ConfigError(f"unknown configuration fields: {', '.join(extra)}")

    trace_path_value = raw["trace_path"]
    if not isinstance(trace_path_value, str) or not trace_path_value:
        raise ConfigError("trace_path must be a non-empty string")
    trace_path = Path(trace_path_value)
    if not trace_path.is_absolute():
        trace_path = config_path.parent / trace_path
    trace_path = trace_path.resolve()

    trace_sha256 = raw["trace_sha256"]
    if (
        not isinstance(trace_sha256, str)
        or len(trace_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in trace_sha256)
    ):
        raise ConfigError("trace_sha256 must be a 64-character hexadecimal digest")
    trace_sha256 = trace_sha256.lower()

    timestamp_policy = raw["timestamp_policy"]
    if timestamp_policy not in {"preserve", "uniform_bucket_jitter"}:
        raise ConfigError(
            "timestamp_policy must be preserve or uniform_bucket_jitter"
        )
    scheduled_q_rule = raw["scheduled_q_rule"]
    if scheduled_q_rule not in {"target_plus_drafts", "drafts_only"}:
        raise ConfigError(
            "scheduled_q_rule must be target_plus_drafts or drafts_only"
        )

    num_speculative_tokens = _integer(raw, "num_speculative_tokens", minimum=0)
    if scheduled_q_rule == "target_plus_drafts":
        pmf_entries = num_speculative_tokens + 1
    else:
        if num_speculative_tokens < 1:
            raise ConfigError(
                "drafts_only requires num_speculative_tokens to be at least 1"
            )
        # In this rule the configured count is the total verification query
        # length, leaving q_len - 1 positions that can enter history as drafts.
        pmf_entries = num_speculative_tokens
    accepted_draft_pmf = _probabilities(raw, pmf_entries)

    sampling_start_ms = _integer(raw, "sampling_start_ms", minimum=0)
    sampling_end_ms = _integer(raw, "sampling_end_ms", minimum=1)
    if sampling_end_ms <= sampling_start_ms:
        raise ConfigError("sampling_end_ms must be greater than sampling_start_ms")

    canonical = json.dumps(
        raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    config_sha256 = hashlib.sha256(canonical).hexdigest()
    config = ReplayConfig(
        trace_path=trace_path,
        trace_sha256=trace_sha256,
        timestamp_policy=timestamp_policy,
        arrival_time_scale=_decimal(raw, "arrival_time_scale"),
        sampling_start_ms=sampling_start_ms,
        sampling_end_ms=sampling_end_ms,
        fixed_step_us=_integer(raw, "fixed_step_us", minimum=1),
        num_cases=_integer(raw, "num_cases", minimum=1),
        seed=_integer(raw, "seed", minimum=0),
        max_num_seqs=_integer(raw, "max_num_seqs", minimum=1),
        max_num_batched_tokens=_integer(
            raw, "max_num_batched_tokens", minimum=1
        ),
        prefill_chunk_size=_integer(raw, "prefill_chunk_size", minimum=1),
        max_model_len=_integer(raw, "max_model_len", minimum=1),
        dcp_size=_integer(raw, "dcp_size", minimum=1),
        prefix_cache_capacity_blocks=_integer(
            raw, "prefix_cache_capacity_blocks", minimum=0
        ),
        num_speculative_tokens=num_speculative_tokens,
        scheduled_q_rule=scheduled_q_rule,
        accepted_draft_pmf=accepted_draft_pmf,
        config_sha256=config_sha256,
    )
    if config.dcp_size not in {2, 4, 8}:
        raise ConfigError("dcp_size must be one of 2, 4, or 8")
    if config.mtp_query_len > config.max_num_batched_tokens:
        raise ConfigError(
            "the configured MTP query cannot fit max_num_batched_tokens"
        )
    return config


def derived_rng(seed: int, stream: str) -> random.Random:
    """Build independent deterministic RNG streams without shared call order."""
    digest = hashlib.sha256(f"{seed}:{stream}".encode("ascii")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))
