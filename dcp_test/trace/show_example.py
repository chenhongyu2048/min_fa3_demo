"""Select and display reproducible examples from a workload JSONL file."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import TextIO


DEFAULT_DISPLAY_SEED = 20260806

DISPLAY_FIELDS = (
    "schema_version",
    "case_id",
    "source",
    "trace_sha256",
    "config_sha256",
    "sampled_time_us",
    "scheduler_step",
    "batch_size",
    "request_ids",
    "phases",
    "q_lens",
    "history_lens",
    "total_kv_lens",
    "prompt_lengths",
    "output_lengths",
    "cached_prefix_lengths",
    "generated_tokens_before",
    "num_speculative_tokens",
    "accepted_drafts",
    "fixed_step_us",
    "timestamp_policy",
    "seed",
)


class ExampleError(ValueError):
    """Raised when an input workload cannot be displayed safely."""


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display random examples from a Mega DCP workload JSONL"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--num-examples",
        type=_positive_integer,
        default=5,
        help="number of cases to display (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_DISPLAY_SEED,
        help=f"display selection seed (default: {DEFAULT_DISPLAY_SEED})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="print complete cases instead of the focused field subset",
    )
    return parser


def load_cases(path: str | Path) -> tuple[dict[str, object], ...]:
    input_path = Path(path).resolve()
    cases: list[dict[str, object]] = []
    try:
        handle = input_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ExampleError(f"cannot read workload {input_path}: {exc}") from exc
    with handle:
        for line_number, text in enumerate(handle, start=1):
            if not text.strip():
                continue
            try:
                case = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ExampleError(
                    f"line {line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(case, dict):
                raise ExampleError(
                    f"line {line_number}: workload case must be a JSON object"
                )
            missing = [field for field in DISPLAY_FIELDS if field not in case]
            if missing:
                raise ExampleError(
                    f"line {line_number}: missing display fields: "
                    f"{', '.join(missing)}"
                )
            cases.append(case)
    if not cases:
        raise ExampleError(f"workload contains no cases: {input_path}")
    return tuple(cases)


def show_examples(
    input_path: str | Path,
    *,
    num_examples: int,
    seed: int,
    full: bool,
    output: TextIO,
) -> dict[str, object]:
    """Write selected cases as JSONL and return a machine-readable summary."""
    if num_examples <= 0:
        raise ExampleError("num_examples must be greater than zero")
    resolved_path = Path(input_path).resolve()
    cases = load_cases(resolved_path)
    display_count = min(num_examples, len(cases))
    selected = random.Random(seed).sample(cases, display_count)
    for case in selected:
        displayed = case if full else {field: case[field] for field in DISPLAY_FIELDS}
        output.write(
            json.dumps(
                displayed,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )
        output.write("\n")
    return {
        "input": str(resolved_path),
        "total_cases": len(cases),
        "examples_displayed": display_count,
        "display_seed": seed,
        "full": full,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = show_examples(
            args.input,
            num_examples=args.num_examples,
            seed=args.seed,
            full=args.full,
            output=sys.stdout,
        )
    except (ExampleError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
