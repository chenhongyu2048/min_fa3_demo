"""Plot training sequence lengths and Mooncake chunk-prefill cache CDFs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


DATASET_DIR = Path(__file__).resolve().parent
TRACE_PATH = DATASET_DIR.parent / "dcp_test/trace/conversation_trace.jsonl"
BLOCK_SIZE = 512
DATASETS = (
    ("prolong", "ProLong", "#4C78A8", "-"),
    ("arxiv", "ArXiv", "#59A14F", "--"),
    ("freelaw", "FreeLaw", "#9C755F", "-."),
    ("github", "GitHub", "#F28E2B", ":"),
    ("pile", "Pile-CC", "#17A2B8", (0, (4, 1, 1, 1))),
)


def prefix_statistics(path: Path) -> dict[str, np.ndarray]:
    """Match full prefix blocks against strictly earlier request timestamps.

    Earlier full prompt blocks are assumed retained, with no eviction or
    completion-time model. Requests sharing a timestamp do not supply one
    another's prefixes. The remainder is input work, not a user-turn length.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    trie: dict = {}
    records = []
    for _, group in itertools.groupby(rows, key=lambda row: row["timestamp"]):
        group = list(group)
        for row in group:
            length = row["input_length"]
            node = trie
            matched = 0
            for block in row["hash_ids"][:length // BLOCK_SIZE]:
                if block not in node:
                    break
                node = node[block]
                matched += 1
            prefix = matched * BLOCK_SIZE
            records.append((length, prefix, length - prefix))
        for row in group:
            node = trie
            for block in row["hash_ids"][:row["input_length"] // BLOCK_SIZE]:
                node = node.setdefault(block, {})
    values = np.asarray(records, dtype=np.int64)
    return {
        "context": values[:, 0],
        "prefix": values[:, 1],
        "suffix": values[:, 2],
        "reuse_fraction": values[:, 1] / values[:, 0],
    }


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, counts = np.unique(values, return_counts=True)
    return unique, np.cumsum(counts) / len(values)


def chunk_prefill_statistics(
    requests: dict[str, np.ndarray], chunk_size: int
) -> dict[str, np.ndarray]:
    """Expand each uncached suffix into successive chunk-prefill observations."""
    records = []
    for request_id, (length, prefix) in enumerate(
        zip(requests["context"], requests["prefix"])
    ):
        for chunk_index, cache_length in enumerate(range(prefix, length, chunk_size)):
            chunk_length = min(chunk_size, length - cache_length)
            records.append((request_id, chunk_index, cache_length, chunk_length))
    values = np.asarray(records, dtype=np.int64)
    context = values[:, 2] + values[:, 3]
    return {
        "source_request_index": values[:, 0],
        "chunk_index": values[:, 1],
        "cache_length": values[:, 2],
        "chunk_length": values[:, 3],
        "context": context,
        "cache_ratio": values[:, 2] / context,
    }


def describe(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p50": float(np.quantile(values, .5)),
        "p90": float(np.quantile(values, .9)),
        "p99": float(np.quantile(values, .99)),
        "max": float(np.max(values)),
    }


def style_axis(axis, ticks=(0, .5, 1)) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_linewidth(.6)
    axis.set_ylim(0, 1.02)
    axis.set_yticks(ticks)
    axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=""))
    axis.tick_params(axis="both", labelsize=7.2, length=2.2, width=.6, pad=1.7)
    axis.minorticks_off()
    axis.grid(axis="y", color="#DADADA", linewidth=.45, alpha=.75)
    axis.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--trace", type=Path, default=TRACE_PATH)
    parser.add_argument("--prefill-chunk-size", type=int, default=4096,
                        help="Maximum prefill chunk length, matching the trace evaluation.")
    parser.add_argument("--output-prefix", type=Path,
                        default=DATASET_DIR / "workload_distributions")
    args = parser.parse_args()
    training = {
        name: np.load(args.dataset_dir / f"{name}_doc_lengths.npy", allow_pickle=False)
        for name, _, _, _ in DATASETS
    }
    requests = prefix_statistics(args.trace)
    trace = chunk_prefill_statistics(requests, args.prefill_chunk_size)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.8,
        "axes.labelsize": 7.8, "axes.titlesize": 7.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "path.simplify": True,
    })
    figure_size = (3.25, 1.42)
    fig = plt.figure(figsize=figure_size)
    train_ax = fig.add_axes((.12, .34, .40, .62))
    request_ax = fig.add_axes((.57, .34, .40, .62), sharey=train_ax)
    fig.text(.32, .015, "(a) Training datasets", ha="center", va="bottom", fontsize=7.3)
    fig.text(.77, .015, "(b) Inference trace", ha="center", va="bottom", fontsize=7.3)

    for name, label, color, linestyle in DATASETS:
        x, y = ecdf(training[name])
        train_ax.step(x, y, where="post", label=label, color=color,
                      linewidth=1.25, linestyle=linestyle)
    train_ax.set_xscale("log", base=2)
    train_ax.set_xlim(1, 1048576)
    train_ax.set_xticks((1, 128, 16384, 1048576), ("1", "128", "16K", "1M"))
    train_ax.set_xlabel("Sequence tokens", labelpad=3)
    train_ax.set_ylabel("CDF (%)", labelpad=2)
    style_axis(train_ax, (0, .25, .5, .75, 1))
    train_ax.legend(loc="upper left", ncol=1, fontsize=5.7,
                    frameon=True, facecolor="white", edgecolor="none", framealpha=.94,
                    handlelength=.95, handletextpad=.25,
                    columnspacing=.4, labelspacing=.15, borderpad=.15, borderaxespad=.3)

    length_color, ratio_color = "#4C78A8", "#D55E00"
    length_x, length_y = ecdf(trace["cache_length"])
    length_curve, = request_ax.step(
        np.r_[0, length_x, 131072], np.r_[0, length_y, 1], where="post",
        label="Cache length (tok.)", color=length_color, linewidth=1.3,
    )
    # Keep zero-prefix requests visible before the logarithmic positive range.
    request_ax.set_xscale("symlog", base=2, linthresh=512, linscale=.5)
    request_ax.set_xlim(0, 131072)
    request_ax.set_xticks((0, 512, 8192, 131072), ("0", "512", "8K", "128K"))
    style_axis(request_ax, (0, .25, .5, .75, 1))
    request_ax.tick_params(axis="y", left=False, labelleft=False)
    request_ax.tick_params(axis="x", colors=length_color, labelsize=6.8)
    request_ax.spines["left"].set_visible(False)
    request_ax.spines["bottom"].set_color(length_color)
    request_ax.get_xticklabels()[0].set_ha("right")
    request_ax.get_xticklabels()[-1].set_ha("right")

    ratio_ax = request_ax.twiny()
    ratio_x, ratio_y = ecdf(trace["cache_ratio"])
    ratio_curve, = ratio_ax.step(
        np.r_[0, ratio_x, 1], np.r_[0, ratio_y, 1], where="post",
        label="Cache ratio (%)", color=ratio_color, linewidth=1.3, linestyle="--",
    )
    ratio_ax.set_xlim(0, 1)
    ratio_ax.set_xticks((0, .5, 1), ("0", "50", "100"))
    ratio_ax.xaxis.set_ticks_position("bottom")
    ratio_ax.tick_params(axis="x", labelsize=6.8, length=2.2, width=.6,
                        pad=1.7, colors=ratio_color)
    ratio_ax.spines[["left", "right", "top"]].set_visible(False)
    ratio_ax.spines["bottom"].set_position(("outward", 12))
    ratio_ax.spines["bottom"].set_linewidth(.6)
    ratio_ax.spines["bottom"].set_color(ratio_color)
    ratio_ax.get_xticklabels()[0].set_ha("left")
    ratio_ax.get_xticklabels()[-1].set_ha("right")
    ratio_ax.legend(handles=(length_curve, ratio_curve), loc="upper left",
                    fontsize=5.7, frameon=False, handlelength=1.3, handletextpad=.35,
                    labelspacing=.2, borderpad=.2, borderaxespad=.35)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)

    report = {
        "figure_size_inches": list(figure_size),
        "training": {
            label: {**describe(training[name]),
                    "fraction_below_8192": float(np.mean(training[name] < 8192))}
            for name, label, _, _ in DATASETS
        },
        "mooncake": {
            name: describe(trace[name])
            for name in ("cache_length", "chunk_length", "context", "cache_ratio")
        },
        "cache_cdfs": {
            "source_requests": int(len(requests["prefix"])),
            "prefill_chunk_size": args.prefill_chunk_size,
            "chunk_prefill_observations": int(len(trace["cache_length"])),
            "multi_chunk_requests": int(np.sum(requests["suffix"] > args.prefill_chunk_size)),
            "zero_cache_observations": int(np.sum(trace["cache_length"] == 0)),
            "fraction_ratio_ge_90_percent": float(np.mean(trace["cache_ratio"] >= .9)),
        },
        "definitions": {
            "initial_prefix": "Longest full 512-token prefix seen at a strictly earlier timestamp, assuming retention without eviction; not measured cache residency.",
            "cache_length": "Initial reusable prefix plus all earlier completed chunks of the same request, measured before the current chunk executes.",
            "chunk_length": "Tokens processed in this chunk: min(prefill_chunk_size, remaining uncached input).",
            "context": "cache_length + chunk_length; context visible through the end of the current chunk, excluding future chunks.",
            "cache_ratio": "cache_length / (cache_length + chunk_length), one equal-weight observation per chunk-prefill execution.",
            "source_request_index": "Zero-based index of the original request in the trace.",
            "chunk_index": "Zero-based prefill chunk index within the source request.",
            "turn_count": "Not available: the public trace has no conversation or turn identifiers.",
            "sampling": f"Expand all {len(requests['prefix'])} source requests into chunks of at most {args.prefill_chunk_size} tokens after prefix reuse; no synthetic first-turn history and no decode-step observations. This is fixed-size chunk expansion, without batch-budget scheduling.",
            "training_cdf": "Empirical cumulative fraction, one equal-weight observation per sequence.",
            "mooncake_plot": "Two empirical CDFs share the training panel's CDF axis. Both horizontal axes are below the panel: the upper blue row shows cache length (linear from 0 to 512, then logarithmic base 2), and the lower orange row shows cache ratio on a linear percentage scale. Every chunk-prefill observation has equal weight, including those with zero cache length.",
            "training_range": "Raw lengths are retained, including values above 128K; the displayed log axis spans 1 to 1M tokens.",
        },
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output_prefix.with_suffix(".npz"), **trace)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
