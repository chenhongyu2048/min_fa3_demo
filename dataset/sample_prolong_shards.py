#!/usr/bin/env python3
"""Sample ProLong-64K MDS shards and save sample_length-style outputs.

ProLong is already tokenized. Each MDS sample stores one packed sequence in
``input_ids`` and its constituent sequence ranges in ``indices``. This script
uses those ranges to recover sequence lengths without running another
tokenizer.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import posixpath
import random
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from streaming import LocalDataset
from tqdm import tqdm
from urllib3.util.retry import Retry


REPO_ID = "princeton-nlp/prolong-data-64K"
REVISION = "main"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DATASET_SERVER_FIRST_ROWS = "https://datasets-server.huggingface.co/first-rows"
DATASET_DIR = Path(__file__).resolve().parent

RECIPE_WEIGHT_BPS = {
    "thestackv1_concat_by_repo-65536": 3_000,
    "book-65536": 3_000,
    "textbooks": 300,
    "fineweb-edu": 999,
    "fineweb-2023-50": 999,
    "tuluv2": 407,
    "stackexchange": 407,
    "dolmawiki": 296,
    "openwebmath": 296,
    "arxiv": 296,
}
DEFAULT_SUBSETS = list(RECIPE_WEIGHT_BPS)
CANDIDATE_POOL_MULTIPLIER = 2

# Keep these exactly aligned with dataset/sample_length.py.
BINS = [
    0,
    1_000,
    2_000,
    4_000,
    8_000,
    16_000,
    32_000,
    64_000,
    128_000,
    256_000,
]
BIN_LABELS = [
    "<1",
    "1-2",
    "2-4",
    "4-8",
    "8-16",
    "16-32",
    "32-64",
    "64-128",
    "128-256",
]


@dataclass(frozen=True)
class ShardCandidate:
    subset: str
    index_path: str
    index_version: int
    shard_position: int
    shard: dict[str, Any]
    basename: str
    path_in_repo: str
    bytes: int
    samples: int
    weight_fallback: bool


@dataclass(frozen=True)
class SubsetResult:
    subset: str
    lengths: np.ndarray
    target_sequences: int
    pool_sequences: int
    processed_rows: int


def create_http_session(
    retries: int = 5,
    backoff_factor: float = 1.0,
) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def build_headers(token: str | None) -> dict[str, str]:
    headers = {"User-Agent": "prolong-length-distribution/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_hf_endpoint() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid Hugging Face endpoint: {endpoint!r}")
    return endpoint


def replace_url_endpoint(url: str, endpoint: str) -> str:
    """Keep a pagination URL on endpoint even if a mirror links upstream."""

    parsed_url = urlsplit(url)
    parsed_endpoint = urlsplit(endpoint)
    return urlunsplit(
        (
            parsed_endpoint.scheme,
            parsed_endpoint.netloc,
            parsed_url.path,
            parsed_url.query,
            parsed_url.fragment,
        )
    )


def make_resolve_url(repo_id: str, revision: str, path_in_repo: str) -> str:
    encoded_path = quote(path_in_repo, safe="/")
    return (
        f"{get_hf_endpoint()}/datasets/{repo_id}/resolve/"
        f"{revision}/{encoded_path}?download=true"
    )


def probe_dataset_viewer(
    session: requests.Session,
    headers: dict[str, str],
) -> None:
    params = {"dataset": REPO_ID, "config": "default", "split": "train"}
    print("Checking Hugging Face Dataset Viewer...")
    try:
        response = session.get(
            DATASET_SERVER_FIRST_ROWS,
            params=params,
            headers=headers,
            timeout=60,
        )
        if response.ok:
            rows = response.json().get("rows", [])
            print(f"Dataset Viewer is available (HTTP {response.status_code}, {len(rows)} rows).")
        else:
            print(
                f"Dataset Viewer is unavailable (HTTP {response.status_code}); "
                "continuing with Hub files."
            )
    except (requests.RequestException, ValueError) as exc:
        print(f"Dataset Viewer probe failed ({exc}); continuing with Hub files.")


def get_json_file(
    session: requests.Session,
    headers: dict[str, str],
    path_in_repo: str,
    timeout: int = 120,
) -> dict[str, Any]:
    response = session.get(
        make_resolve_url(REPO_ID, REVISION, path_in_repo),
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError(f"{path_in_repo} did not return a JSON object")
    return body


def download_file(
    session: requests.Session,
    headers: dict[str, str],
    path_in_repo: str,
    output_path: Path,
    timeout: int = 600,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    response = session.get(
        make_resolve_url(REPO_ID, REVISION, path_in_repo),
        headers=headers,
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    try:
        with open(temporary_path, "wb") as output_file:
            with tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=output_path.name,
                leave=False,
            ) as progress:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    output_file.write(chunk)
                    downloaded += len(chunk)
                    progress.update(len(chunk))
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return downloaded


def list_repository_files(
    session: requests.Session,
    headers: dict[str, str],
    requested_subsets: list[str],
) -> list[str]:
    """List requested subtrees while keeping mirror pagination on the mirror."""

    endpoint = get_hf_endpoint()
    repo_path = quote(REPO_ID, safe="/")
    revision_path = quote(REVISION, safe="")
    paths: list[str] = []
    print("Fetching repository file lists...")

    for subset in requested_subsets:
        subset_path = quote(subset, safe="/")
        next_url: str | None = (
            f"{endpoint}/api/datasets/{repo_path}/tree/"
            f"{revision_path}/{subset_path}"
        )
        params: dict[str, str | int] | None = {
            "recursive": "true",
            "expand": "false",
            "limit": 1000,
        }
        subset_entries = 0

        while next_url is not None:
            response = session.get(
                next_url,
                params=params,
                headers=headers,
                timeout=120,
            )
            if response.status_code == 404:
                print(f"Warning: subset {subset} was not found on the Hub", file=sys.stderr)
                break
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"Hub tree response for {subset} was not a list")

            for item in payload:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    raise ValueError(f"Hub tree response for {subset} has a malformed item")
                paths.append(item["path"])
                subset_entries += 1

            upstream_next_url = response.links.get("next", {}).get("url")
            next_url = (
                replace_url_endpoint(upstream_next_url, endpoint)
                if upstream_next_url
                else None
            )
            params = None

        print(f"  {subset}: {subset_entries:,} repository entries")

    print(f"Repository entries: {len(paths):,}")
    return paths


def find_index_files_by_subset(
    repo_files: list[str],
    requested_subsets: list[str],
) -> dict[str, list[str]]:
    result = {subset: [] for subset in requested_subsets}
    for path in repo_files:
        if not path.endswith("/index.json"):
            continue
        top_level = path.split("/", maxsplit=1)[0]
        if top_level in result:
            result[top_level].append(path)

    for paths in result.values():
        paths.sort()
    return result


def raw_data_file_info(
    index_path: str,
    shard: dict[str, Any],
) -> tuple[str, str, int]:
    raw_data = shard.get("raw_data")
    if not isinstance(raw_data, dict) or not raw_data.get("basename"):
        if isinstance(shard.get("zip_data"), dict):
            raise ValueError(f"compressed zip_data shard is unsupported: {index_path}")
        raise ValueError(f"shard has no raw_data basename: {index_path}")

    basename = str(raw_data["basename"])
    path_in_repo = posixpath.join(posixpath.dirname(index_path), basename)
    try:
        byte_count = int(raw_data.get("bytes", 0) or 0)
    except (TypeError, ValueError):
        byte_count = 0
    return basename, path_in_repo, byte_count


def parse_shard_samples(shard: dict[str, Any]) -> tuple[int, bool]:
    try:
        samples = int(shard.get("samples", 0) or 0)
    except (TypeError, ValueError):
        return 1, True
    if samples <= 0:
        return 1, True
    return samples, False


def enumerate_shards(
    session: requests.Session,
    headers: dict[str, str],
    subset: str,
    index_paths: list[str],
) -> list[ShardCandidate]:
    candidates: list[ShardCandidate] = []
    seen_paths: set[str] = set()
    for index_path in tqdm(index_paths, desc=f"{subset}: reading index.json", unit="index"):
        try:
            index = get_json_file(session, headers, index_path)
        except Exception as exc:
            print(f"Warning: failed to read {index_path}: {exc}", file=sys.stderr)
            continue

        try:
            index_version = int(index.get("version", 2))
        except (TypeError, ValueError):
            index_version = 2
        shards = index.get("shards", [])
        if not isinstance(shards, list):
            print(f"Warning: {index_path} has a non-list shards field", file=sys.stderr)
            continue

        for shard_position, shard in enumerate(shards):
            if not isinstance(shard, dict):
                print(f"Warning: malformed shard in {index_path}", file=sys.stderr)
                continue
            try:
                basename, path_in_repo, byte_count = raw_data_file_info(index_path, shard)
            except ValueError as exc:
                print(f"Warning: {exc}", file=sys.stderr)
                continue
            if path_in_repo in seen_paths:
                continue
            seen_paths.add(path_in_repo)
            samples, weight_fallback = parse_shard_samples(shard)
            candidates.append(
                ShardCandidate(
                    subset=subset,
                    index_path=index_path,
                    index_version=index_version,
                    shard_position=shard_position,
                    shard=shard,
                    basename=basename,
                    path_in_repo=path_in_repo,
                    bytes=byte_count,
                    samples=samples,
                    weight_fallback=weight_fallback,
                )
            )
    return candidates


def weighted_shard_order(
    candidates: list[ShardCandidate],
    rng: random.Random,
    max_shards: int,
) -> list[ShardCandidate]:
    if max_shards <= 0:
        raise ValueError("max_shards must be positive")

    remaining = list(candidates)
    selected: list[ShardCandidate] = []
    while remaining and len(selected) < max_shards:
        weights = [max(1, candidate.samples) for candidate in remaining]
        chosen = rng.choices(remaining, weights=weights, k=1)[0]
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def allocate_recipe_targets(
    total_sequences: int,
) -> dict[str, int]:
    """Allocate the fixed ProLong recipe with the largest-remainder method."""

    if total_sequences <= 0:
        raise ValueError("total_sequences must be positive")

    targets: dict[str, int] = {}
    remainders: list[tuple[int, int, str]] = []
    allocated = 0
    total_weight = sum(RECIPE_WEIGHT_BPS.values())
    for position, (subset, weight) in enumerate(RECIPE_WEIGHT_BPS.items()):
        numerator = total_sequences * weight
        target, remainder = divmod(numerator, total_weight)
        targets[subset] = target
        allocated += target
        remainders.append((remainder, -position, subset))

    for _remainder, _position, subset in sorted(remainders, reverse=True)[
        : total_sequences - allocated
    ]:
        targets[subset] += 1

    zero_targets = [subset for subset, target in targets.items() if target == 0]
    if zero_targets:
        names = ", ".join(zero_targets)
        raise ValueError(
            "--num-sequences is too small to sample every recipe subset; "
            f"zero-sequence subsets: {names}"
        )
    return targets


def extract_document_lengths(
    sample: dict[str, Any],
    quality: Counter[str],
) -> list[int]:
    indices = sample.get("indices")
    if indices is None:
        quality["samples_without_indices"] += 1
        return []

    try:
        array = np.asarray(indices)
        if array.size == 0:
            quality["samples_without_indices"] += 1
            return []
        if array.size % 2 != 0:
            quality["malformed_indices"] += 1
            return []
        pairs = array.reshape(-1, 2)
    except Exception:
        quality["malformed_indices"] += 1
        return []

    document_lengths: list[int] = []
    for start, end in pairs:
        try:
            document_length = int(end) - int(start)
        except (TypeError, ValueError, OverflowError):
            quality["malformed_indices"] += 1
            continue
        if document_length <= 0:
            quality["invalid_document_ranges"] += 1
            continue
        document_lengths.append(document_length)
    return document_lengths


def process_one_shard(
    session: requests.Session,
    headers: dict[str, str],
    candidate: ShardCandidate,
    document_lengths: list[int],
    quality: Counter[str],
    rng: random.Random,
    max_rows_per_shard: int,
    temp_root: Path,
    max_sequence_lengths: int | None = None,
) -> tuple[int, int]:
    working_directory = Path(
        tempfile.mkdtemp(prefix=f"{candidate.subset}_", dir=temp_root)
    )
    try:
        local_shard_path = working_directory / candidate.basename
        downloaded_bytes = download_file(
            session=session,
            headers=headers,
            path_in_repo=candidate.path_in_repo,
            output_path=local_shard_path,
        )
        sampled_index = {
            "version": candidate.index_version,
            "shards": [candidate.shard],
        }
        (working_directory / "index.json").write_text(
            json.dumps(sampled_index, ensure_ascii=False),
            encoding="utf-8",
        )

        dataset = LocalDataset(local=str(working_directory))
        shard_document_lengths: list[int] = []
        shard_quality: Counter[str] = Counter()
        dataset_size = len(dataset)
        if max_rows_per_shard > 0 and dataset_size > max_rows_per_shard:
            row_indices = rng.sample(
                range(dataset_size),
                max_rows_per_shard,
            )
        else:
            row_indices = list(range(dataset_size))
            rng.shuffle(row_indices)

        processed_rows = 0
        for row_index in tqdm(
            row_indices,
            total=len(row_indices),
            desc=candidate.basename,
            unit="row",
            leave=False,
        ):
            sample = dataset[row_index]
            if not isinstance(sample, dict):
                raise TypeError(f"expected dict sample, got {type(sample).__name__}")
            shard_document_lengths.extend(
                extract_document_lengths(sample, shard_quality)
            )
            del sample
            processed_rows += 1
            if (
                max_sequence_lengths is not None
                and len(shard_document_lengths) >= max_sequence_lengths
            ):
                break
        del dataset
        document_lengths.extend(shard_document_lengths)
        quality.update(shard_quality)
        return downloaded_bytes, processed_rows
    finally:
        shutil.rmtree(working_directory, ignore_errors=True)


def calculate_distribution(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if lengths.ndim != 1 or lengths.size == 0:
        raise ValueError("lengths must be a non-empty one-dimensional array")
    if not np.issubdtype(lengths.dtype, np.integer) or np.any(lengths <= 0):
        raise ValueError("lengths must contain positive integers")

    # sample_length.py places every sequence >=256K in its final 128K-256K bin.
    clipped = np.clip(lengths, 0, BINS[-1] - 1)
    counts, _ = np.histogram(clipped, bins=BINS)
    proportions = counts / counts.sum()
    return counts, proportions


def print_distribution(
    subset: str,
    lengths: np.ndarray,
    counts: np.ndarray,
    proportions: np.ndarray,
) -> None:
    display_name = "ProLong" if subset == "prolong" else f"ProLong {subset}"
    print()
    print(
        f"===== {display_name} sequence length distribution "
        "(aligned with Zeppelin Table 2) ====="
    )
    print(f"{'Range (k tokens)':<18}{'Prop':<10}{'Count':<10}")
    for label, proportion, count in zip(BIN_LABELS, proportions, counts):
        print(f"{label:<18}{proportion:<10.3f}{count:<10}")

    print(f"\nSequences  : {len(lengths)}")
    print(f"Mean length: {lengths.mean():.0f} tokens")
    print(f"Median(p50): {np.percentile(lengths, 50):.0f}")
    print(f"p90        : {np.percentile(lengths, 90):.0f}")
    print(f"p99        : {np.percentile(lengths, 99):.0f}")
    print(f"Max        : {lengths.max()}")


def save_distribution_outputs(
    subset: str,
    lengths: np.ndarray,
    counts: np.ndarray,
    proportions: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lengths_path = output_dir / f"{subset}_doc_lengths.npy"
    proportions_path = output_dir / f"{subset}_bin_proportions.npy"
    figure_path = output_dir / f"{subset}_length_hist_log.png"

    np.save(lengths_path, lengths)
    np.save(proportions_path, proportions)

    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(BIN_LABELS, counts.astype(float), color="#4C72B0", edgecolor="black")
    axis.set_yscale("log")
    axis.set_xlabel("Sequence length range (k tokens)")
    axis.set_ylabel("Count (log scale)")
    display_name = "ProLong" if subset == "prolong" else f"ProLong {subset}"
    axis.set_title(
        f"{display_name} sequence length distribution "
        f"(n={len(lengths)}, pretokenized)"
    )
    axis.grid(axis="y", which="both", linestyle="--", alpha=0.4)
    for bar, count, proportion in zip(bars, counts, proportions):
        if count > 0:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count}\n({proportion * 100:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    return lengths_path, proportions_path, figure_path


def sample_sequence_pool(
    sequence_lengths: list[int],
    target_sequences: int,
    rng: random.Random,
) -> np.ndarray:
    if target_sequences <= 0:
        raise ValueError("target_sequences must be positive")
    if len(sequence_lengths) < target_sequences:
        raise ValueError(
            f"sequence pool has {len(sequence_lengths):,} entries, "
            f"but {target_sequences:,} are required"
        )

    selected_indices = rng.sample(range(len(sequence_lengths)), target_sequences)
    selected_indices.sort()
    return np.asarray(
        [sequence_lengths[index] for index in selected_indices],
        dtype=np.int64,
    )


def process_subset(
    session: requests.Session,
    headers: dict[str, str],
    subset: str,
    index_paths: list[str],
    target_sequences: int,
    max_shards_per_subset: int,
    max_rows_per_shard: int,
    rng: random.Random,
    temp_root: Path,
    candidates: list[ShardCandidate] | None = None,
) -> SubsetResult | None:
    print()
    print("=" * 80)
    print(f"Subset: {subset}")
    print("=" * 80)
    print(f"index.json files: {len(index_paths):,}")
    if candidates is None:
        candidates = enumerate_shards(session, headers, subset, index_paths)
    print(f"Available raw shards: {len(candidates):,}")
    if not candidates:
        print(f"Warning: no usable raw shards found for {subset}", file=sys.stderr)
        return None

    pool_target = CANDIDATE_POOL_MULTIPLIER * target_sequences
    selected = weighted_shard_order(
        candidates,
        rng,
        max_shards_per_subset,
    )
    print(f"Recipe target sequences: {target_sequences:,}")
    print(f"Candidate pool target  : {pool_target:,}")
    print(f"Candidate shards       : up to {len(selected):,}")
    print(
        "Candidate shard 64K rows: "
        f"{sum(candidate.samples for candidate in selected):,}"
    )
    print(
        "Maximum download       : "
        f"{sum(candidate.bytes for candidate in selected) / 1024**3:.3f} GiB"
    )
    fallback_weight_shards = sum(
        candidate.weight_fallback for candidate in candidates
    )
    if fallback_weight_shards:
        print(f"Shards using fallback 64K-row weight: {fallback_weight_shards:,}")

    document_lengths: list[int] = []
    quality: Counter[str] = Counter()
    downloaded_bytes = 0
    processed_rows = 0
    processed_shards = 0
    failed_shards = 0
    for position, candidate in enumerate(selected, start=1):
        if len(document_lengths) >= pool_target:
            break
        print(
            f"[{position}/{len(selected)}] {candidate.path_in_repo} "
            f"({candidate.bytes / 1024**2:.1f} MiB, "
            f"{candidate.samples:,} 64K rows)"
        )
        try:
            remaining_sequences = pool_target - len(document_lengths)
            shard_bytes, shard_rows = process_one_shard(
                session,
                headers,
                candidate,
                document_lengths,
                quality,
                rng,
                max_rows_per_shard,
                temp_root,
                max_sequence_lengths=remaining_sequences,
            )
            downloaded_bytes += shard_bytes
            processed_rows += shard_rows
            processed_shards += 1
        except Exception as exc:
            if isinstance(exc, MemoryError) or (
                isinstance(exc, OSError) and exc.errno == errno.ENOMEM
            ):
                raise RuntimeError(
                    "memory exhausted while processing "
                    f"{candidate.path_in_repo}; aborting without trying more shards"
                ) from exc
            failed_shards += 1
            error_text = str(exc).strip() or repr(exc)
            print(
                f"Warning: failed to process {candidate.path_in_repo}: "
                f"{type(exc).__name__}: {error_text}",
                file=sys.stderr,
            )

    if len(document_lengths) < pool_target:
        print(
            f"Warning: subset {subset} collected {len(document_lengths):,} "
            f"sequences, but its 2x candidate pool requires {pool_target:,}",
            file=sys.stderr,
        )
        return None

    pool_sequences = len(document_lengths)
    lengths = sample_sequence_pool(document_lengths, target_sequences, rng)

    print(f"\nProcessed shards: {processed_shards}; failed shards: {failed_shards}")
    print(f"Sampled 64K rows: {processed_rows:,}")
    print(f"Candidate sequences: {pool_sequences:,}")
    print(f"Selected sequences : {len(lengths):,}")
    print(f"Downloaded      : {downloaded_bytes / 1024**3:.3f} GiB")
    if quality:
        quality_text = ", ".join(
            f"{key}={value}" for key, value in sorted(quality.items())
        )
        print(f"Data quality    : {quality_text}")
    return SubsetResult(
        subset=subset,
        lengths=lengths,
        target_sequences=target_sequences,
        pool_sequences=pool_sequences,
        processed_rows=processed_rows,
    )


def save_merged_distribution(
    subset_results: list[SubsetResult],
    output_dir: Path = DATASET_DIR,
) -> tuple[Path, Path, Path]:
    if not subset_results:
        raise ValueError("subset_results must not be empty")

    merged_lengths = np.concatenate(
        [result.lengths for result in subset_results]
    ).astype(np.int64, copy=False)
    expected_sequences = sum(result.target_sequences for result in subset_results)
    if len(merged_lengths) != expected_sequences:
        raise RuntimeError(
            f"merged sequence count {len(merged_lengths):,} does not match "
            f"the allocated target {expected_sequences:,}"
        )
    counts, proportions = calculate_distribution(merged_lengths)
    print_distribution("prolong", merged_lengths, counts, proportions)
    paths = save_distribution_outputs(
        "prolong",
        merged_lengths,
        counts,
        proportions,
        output_dir,
    )

    print("\nMerged subset contributions:")
    print(
        f"{'Subset':<38}{'Recipe share':>14}"
        f"{'Actual share':>14}{'64K rows':>12}"
        f"{'Pool':>12}{'Sequences':>12}"
    )
    for result in subset_results:
        recipe_share = RECIPE_WEIGHT_BPS[result.subset] / sum(
            RECIPE_WEIGHT_BPS.values()
        )
        actual_share = len(result.lengths) / len(merged_lengths)
        print(
            f"{result.subset:<38}{recipe_share:>13.3%}"
            f"{actual_share:>13.3%}{result.processed_rows:>12,}"
            f"{result.pool_sequences:>12,}{len(result.lengths):>12,}"
        )
    print(f"Merged bar chart: {paths[2]}")
    print(f"Merged arrays   : {paths[0]} / {paths[1]}")
    return paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample ProLong-64K MDS shards and save sequence-length outputs "
            "matching dataset/sample_length.py."
        )
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        required=True,
        help="Total independent sequence lengths to sample from the ProLong recipe.",
    )
    parser.add_argument(
        "--max-shards-per-subset",
        type=int,
        default=100,
        help="Maximum shards to download for each subset (default: 100).",
    )
    parser.add_argument(
        "--max-rows-per-shard",
        type=int,
        default=0,
        help=(
            "Maximum random 64K rows read from each selected shard; 0 has no "
            "per-shard cap (default: 0)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help=(
            "Hugging Face base URL. Overrides HF_ENDPOINT; for example, "
            "https://hf-mirror.com."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Merged output directory (default: {DATASET_DIR}).",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Temporary shard directory (default: system temporary directory).",
    )
    parser.add_argument(
        "--skip-viewer-probe",
        action="store_true",
        help="Skip the optional datasets-server health check.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep an automatically created top-level temporary directory.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.num_sequences <= 0:
        raise ValueError("--num-sequences must be greater than 0")
    if args.max_shards_per_subset <= 0:
        raise ValueError("--max-shards-per-subset must be greater than 0")
    if args.max_rows_per_shard < 0:
        raise ValueError("--max-rows-per-shard cannot be negative")


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    targets = allocate_recipe_targets(args.num_sequences)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    hf_endpoint = get_hf_endpoint()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.temp_dir is None:
        temp_root = Path(tempfile.mkdtemp(prefix="prolong_shards_"))
        created_temp_root = True
    else:
        temp_root = args.temp_dir
        temp_root.mkdir(parents=True, exist_ok=True)
        created_temp_root = False

    session = create_http_session()
    headers = build_headers(token)
    print(f"Temporary directory: {temp_root}")
    print(f"Output directory   : {args.output_dir}")
    print(f"Hugging Face URL   : {hf_endpoint}")
    rng = random.Random(args.seed)
    subset_results: list[SubsetResult] = []
    candidates_by_subset: dict[str, list[ShardCandidate]] = {}

    print("\nRecipe sequence allocation:")
    print(f"{'Subset':<38}{'Recipe share':>14}{'Target sequences':>20}")
    total_weight = sum(RECIPE_WEIGHT_BPS.values())
    for subset in DEFAULT_SUBSETS:
        recipe_share = RECIPE_WEIGHT_BPS[subset] / total_weight
        print(f"{subset:<38}{recipe_share:>13.3%}{targets[subset]:>20,}")
    print(f"{'TOTAL':<38}{1:>13.3%}{sum(targets.values()):>20,}")

    try:
        if not args.skip_viewer_probe:
            probe_dataset_viewer(session, headers)
        repo_files = list_repository_files(session, headers, DEFAULT_SUBSETS)
        index_files = find_index_files_by_subset(repo_files, DEFAULT_SUBSETS)
        print("Discovered index.json files:")
        for subset in DEFAULT_SUBSETS:
            print(f"  {subset}: {len(index_files[subset]):,}")

        missing_indexes = [
            subset for subset in DEFAULT_SUBSETS if not index_files[subset]
        ]
        if missing_indexes:
            raise RuntimeError(
                "cannot sample the complete recipe because these subsets have "
                "no index.json: " + ", ".join(missing_indexes)
            )

        print("\nReading all subset indexes...")
        for subset in DEFAULT_SUBSETS:
            candidates = enumerate_shards(
                session,
                headers,
                subset,
                index_files[subset],
            )
            if not candidates:
                raise RuntimeError(
                    f"subset {subset} has no usable raw shards"
                )
            candidates_by_subset[subset] = candidates

        for subset in DEFAULT_SUBSETS:
            paths = index_files[subset]
            result = process_subset(
                session,
                headers,
                subset,
                paths,
                targets[subset],
                args.max_shards_per_subset,
                args.max_rows_per_shard,
                rng,
                temp_root,
                candidates=candidates_by_subset[subset],
            )
            if result is not None:
                subset_results.append(result)
    finally:
        session.close()
        if created_temp_root and not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)

    completed_subsets = {result.subset for result in subset_results}
    missing_subsets = [
        subset for subset in DEFAULT_SUBSETS if subset not in completed_subsets
    ]
    if missing_subsets:
        raise RuntimeError(
            "merged output was not written because these subsets failed: "
            + ", ".join(missing_subsets)
        )

    if sum(len(result.lengths) for result in subset_results) != args.num_sequences:
        raise RuntimeError("sampled sequence count does not match --num-sequences")
    save_merged_distribution(subset_results, args.output_dir)

    print(
        f"\nDone. Saved {args.num_sequences:,} merged ProLong sequence "
        f"lengths to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
