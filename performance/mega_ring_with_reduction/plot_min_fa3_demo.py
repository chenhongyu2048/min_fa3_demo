from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


THIS_DIR = Path(__file__).resolve().parent
INPUT_PATH = THIS_DIR / "min_fa3_demo_h100-91351.out"
RUN_ID = INPUT_PATH.stem.rsplit("-", 1)[-1]
SEQLENS = [512, 1024, 2048, 4096, 8192]

METHODS = [
    "PyTorch_SDPA",
    "FA2_varlen",
    "FA3_varlen",
    "min_fa3_varlen",
    "min_fa3_varlen_ring",
    "min_fa3_varlen_mega_ring",
]

METHOD_LABELS = {
    "PyTorch_SDPA": "PyTorch SDPA",
    "FA2_varlen": "FA2 varlen",
    "FA3_varlen": "FA3 varlen",
    "min_fa3_varlen": "min_fa3 varlen",
    "min_fa3_varlen_ring": "min_fa3 ring",
    "min_fa3_varlen_mega_ring": "min_fa3 mega-ring",
}

METHOD_COLORS = {
    "PyTorch_SDPA": "#4C78A8",
    "FA2_varlen": "#F58518",
    "FA3_varlen": "#54A24B",
    "min_fa3_varlen": "#E45756",
    "min_fa3_varlen_ring": "#72B7B2",
    "min_fa3_varlen_mega_ring": "#B279A2",
}

SETTING_LABELS = {
    "comp128_comm0": "comp=128, comm=0",
    "comp128_comm4": "comp=128, comm=4",
    "comp124_comm8": "comp=124, comm=8",
    "comp116_comm16": "comp=116, comm=16",
    "comp108_comm24": "comp=108, comm=24",
}

DEFAULT_SETTINGS = [
    "comp128_comm0",
    "comp128_comm4",
    "comp124_comm8",
    "comp116_comm16",
    "comp108_comm24",
]

_SETTING_RE = re.compile(r"num_comp_sm=(\d+), num_comm_sm=(\d+)")
_PROFILE_RE = re.compile(r"profile=(True|False)")
_CASE_RE = re.compile(r"^B=\d+,S=(\d+),QH=\d+,KVH=\d+,D=\d+,causal=(True|False)$")
_RESULT_RE = re.compile(
    r"^(\S+)\s+"
    r"(N/A|\d+(?:\.\d+)?)\s+"
    r"(N/A|\d+(?:\.\d+)?)\s+"
    r"(\d+(?:\.\d+)?)\s+"
    r"(\d+(?:\.\d+)?)\s*$"
)


def _optional_float(value: str) -> float | None:
    return None if value == "N/A" else float(value)


def _finalize_data(raw_data: dict) -> dict:
    data = {}
    missing = []

    for setting in DEFAULT_SETTINGS:
        data[setting] = {}
        for method in METHODS:
            method_data = raw_data.get(setting, {}).get(method)
            if method_data is None:
                missing.append(f"{setting}/{method}")
                continue

            data[setting][method] = {}
            for metric in ("time", "attn", "reduce", "tflops"):
                values_by_seqlen = method_data[metric]
                absent_seqlens = [seqlen for seqlen in SEQLENS if seqlen not in values_by_seqlen]
                if absent_seqlens:
                    missing.append(f"{setting}/{method}/{metric}:{absent_seqlens}")
                    continue

                values = [values_by_seqlen[seqlen] for seqlen in SEQLENS]
                if metric in ("attn", "reduce") and all(value is None for value in values):
                    data[setting][method][metric] = None
                elif any(value is None for value in values):
                    missing.append(f"{setting}/{method}/{metric}:N/A")
                else:
                    data[setting][method][metric] = values

    if missing:
        details = "\n".join(missing)
        raise ValueError(f"missing benchmark data in {INPUT_PATH}:\n{details}")

    return data


def load_causal_data(input_path: Path = INPUT_PATH) -> dict:
    raw_data = {}
    current_setting = None
    current_case = None

    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Config:"):
            match = _SETTING_RE.search(line)
            profile_match = _PROFILE_RE.search(line)
            if match is None or profile_match is None or profile_match.group(1) != "False":
                current_setting = None
                current_case = None
                continue
            current_setting = f"comp{match.group(1)}_comm{match.group(2)}"
            raw_data.setdefault(current_setting, {})
            current_case = None
            continue

        match = _CASE_RE.match(line)
        if match is not None:
            current_case = (int(match.group(1)), match.group(2) == "True")
            continue

        match = _RESULT_RE.match(line)
        if match is None or current_setting is None or current_case is None:
            continue
        seqlen, is_causal = current_case
        method = match.group(1)
        if not is_causal or method not in METHODS:
            continue

        method_data = raw_data[current_setting].setdefault(
            method,
            {
                "time": {},
                "attn": {},
                "reduce": {},
                "tflops": {},
            },
        )
        method_data["attn"][seqlen] = _optional_float(match.group(2))
        method_data["reduce"][seqlen] = _optional_float(match.group(3))
        method_data["time"][seqlen] = float(match.group(4))
        method_data["tflops"][seqlen] = float(match.group(5))

    return _finalize_data(raw_data)


DATA = load_causal_data()


def _make_subplots(settings: list[str], *, sharey: bool, height: float):
    fig, axes = plt.subplots(1, len(settings), figsize=(6.2 * len(settings), height), sharey=sharey)
    return fig, np.atleast_1d(axes)


def plot_causal_time_breakdown(
    settings: list[str] | None = None,
    output_path: Path | None = None,
) -> Path:
    settings = list(DEFAULT_SETTINGS if settings is None else settings)
    output_path = THIS_DIR / f"causal_time_breakdown_{RUN_ID}.png" if output_path is None else output_path

    fig, axes = _make_subplots(settings, sharey=True, height=5.5)
    x = np.arange(len(SEQLENS))
    group_width = 0.84
    bar_width = group_width / len(METHODS)

    for ax, setting in zip(axes, settings):
        for method_idx, method in enumerate(METHODS):
            color = METHOD_COLORS[method]
            offset = (method_idx - (len(METHODS) - 1) / 2) * bar_width
            positions = x + offset
            total_values = DATA[setting][method]["time"]
            attn_values = DATA[setting][method]["attn"]
            reduce_values = DATA[setting][method]["reduce"]

            if attn_values is not None and reduce_values is not None:
                ax.bar(
                    positions,
                    total_values,
                    width=bar_width * 0.92,
                    color=color,
                    alpha=0.22,
                    edgecolor=color,
                    linewidth=1.0,
                    zorder=1,
                )
                ax.bar(
                    positions,
                    attn_values,
                    width=bar_width * 0.60,
                    color=color,
                    alpha=0.95,
                    zorder=3,
                )
                ax.bar(
                    positions,
                    reduce_values,
                    width=bar_width * 0.80,
                    bottom=attn_values,
                    facecolor="none",
                    edgecolor=color,
                    hatch="///",
                    linewidth=0.0,
                    zorder=4,
                )
            else:
                ax.bar(
                    positions,
                    total_values,
                    width=bar_width * 0.92,
                    color=color,
                    alpha=0.92,
                    zorder=2,
                )

        ax.set_title(SETTING_LABELS[setting], fontsize=12)
        ax.set_xticks(x, [str(seqlen) for seqlen in SEQLENS])
        ax.set_xlabel("Seqlen")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)

    axes[0].set_ylabel("Time (ms, log scale)")
    fig.suptitle("Causal=True Time Breakdown", fontsize=16, y=1.05)

    method_handles = [
        Patch(facecolor=METHOD_COLORS[method], edgecolor="none", label=METHOD_LABELS[method])
        for method in METHODS
    ]
    style_handles = [
        Patch(facecolor="black", alpha=0.22, edgecolor="black", label="Total time"),
        Patch(facecolor="black", alpha=0.95, edgecolor="none", label="Attn time"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Reduction time"),
    ]
    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(METHODS),
        frameon=False,
    )
    fig.legend(handles=style_handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.01,
        "Each seqlen group contains 6 bars in this order: "
        "PyTorch SDPA, FA2 varlen, FA3 varlen, min_fa3 varlen, min_fa3 ring, min_fa3 mega-ring.",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_causal_tflops(
    settings: list[str] | None = None,
    output_path: Path | None = None,
) -> Path:
    settings = list(DEFAULT_SETTINGS if settings is None else settings)
    output_path = THIS_DIR / f"causal_tflops_{RUN_ID}.png" if output_path is None else output_path

    fig, axes = _make_subplots(settings, sharey=True, height=5.0)

    for ax, setting in zip(axes, settings):
        for method in METHODS:
            ax.plot(
                SEQLENS,
                DATA[setting][method]["tflops"],
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_title(SETTING_LABELS[setting], fontsize=12)
        ax.set_xticks(SEQLENS, [str(seqlen) for seqlen in SEQLENS], rotation=0)
        ax.set_xlabel("Seqlen")
        ax.grid(alpha=0.25, linewidth=0.8)

    axes[0].set_ylabel("TFLOPS")
    fig.suptitle("Causal=True TFLOPS", fontsize=16, y=1.03)
    line_handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], marker="o", linewidth=2.2, label=METHOD_LABELS[method])
        for method in METHODS
    ]
    fig.legend(handles=line_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(METHODS), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    breakdown_path = plot_causal_time_breakdown()
    tflops_path = plot_causal_tflops()
    print(f"saved {breakdown_path}")
    print(f"saved {tflops_path}")


if __name__ == "__main__":
    main()
