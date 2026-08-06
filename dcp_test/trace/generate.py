"""CLI for producing Mega DCP JSONL cases from Mooncake traces."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .models import ConfigError, derived_rng, load_config
from .mooncake import TraceError, load_mooncake_trace
from .replay import ReplayError, replay_trace, stats_dict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract deterministic Mega DCP workloads from Mooncake trace"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--num-cases",
        type=_positive_integer,
        default=None,
        help="override num_cases from the trace replay config",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output file"
    )
    return parser


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def write_cases(cases: tuple[dict[str, object], ...], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for case in cases:
                handle.write(
                    json.dumps(
                        case,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def run(
    config_path: Path,
    output: Path,
    *,
    force: bool,
    num_cases: int | None = None,
) -> dict[str, object]:
    config = load_config(config_path, num_cases=num_cases)
    output = output.resolve()
    if output.exists() and not force:
        raise ConfigError(f"output already exists: {output}; pass --force to replace it")
    trace = load_mooncake_trace(config, derived_rng(config.seed, "arrival"))
    result = replay_trace(
        trace,
        config,
        derived_rng(config.seed, "acceptance"),
        derived_rng(config.seed, "sampling"),
    )
    write_cases(result.cases, output)
    return {
        "output": str(output),
        "cases_written": len(result.cases),
        "num_cases": config.num_cases,
        "config_sha256": config.config_sha256,
        "stats": stats_dict(result.stats),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run(
            args.config,
            args.output,
            force=args.force,
            num_cases=args.num_cases,
        )
    except (ConfigError, TraceError, ReplayError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
