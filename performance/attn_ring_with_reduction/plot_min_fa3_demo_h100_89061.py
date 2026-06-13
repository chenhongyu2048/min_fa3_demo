from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


THIS_DIR = Path(__file__).resolve().parent
SEQLENS = [512, 1024, 2048, 4096, 8192]

METHODS = [
    "PyTorch_SDPA",
    "FA2_varlen",
    "FA3_varlen",
    "min_fa3_varlen",
    "min_fa3_varlen_ring",
]

METHOD_LABELS = {
    "PyTorch_SDPA": "PyTorch SDPA",
    "FA2_varlen": "FA2 varlen",
    "FA3_varlen": "FA3 varlen",
    "min_fa3_varlen": "min_fa3 varlen",
    "min_fa3_varlen_ring": "min_fa3 ring",
}

METHOD_COLORS = {
    "PyTorch_SDPA": "#4C78A8",
    "FA2_varlen": "#F58518",
    "FA3_varlen": "#54A24B",
    "min_fa3_varlen": "#E45756",
    "min_fa3_varlen_ring": "#72B7B2",
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

DATA = {
    "comp128_comm0": {
        "PyTorch_SDPA": {
            "time": [0.522, 1.422, 3.519, 10.143, 32.865],
            "attn": [0.366, 0.809, 2.311, 7.689, 27.903],
            "reduce": [0.156, 0.613, 1.207, 2.454, 4.961],
            "tflops": [65.8, 96.6, 156.2, 216.8, 267.6],
        },
        "FA2_varlen": {
            "time": [0.347, 0.933, 2.490, 7.993, 27.596],
            "attn": [0.199, 0.508, 1.648, 6.262, 24.113],
            "reduce": [0.148, 0.425, 0.843, 1.727, 3.490],
            "tflops": [99.1, 147.4, 220.8, 275.1, 318.7],
        },
        "FA3_varlen": {
            "time": [0.271, 0.739, 1.894, 5.385, 17.466],
            "attn": [0.124, 0.316, 1.012, 3.548, 13.632],
            "reduce": [0.146, 0.423, 0.883, 1.837, 3.842],
            "tflops": [126.9, 185.9, 290.2, 408.4, 503.6],
        },
        "min_fa3_varlen": {
            "time": [0.161, 0.348, 1.162, 3.968, 14.689],
            "attn": None,
            "reduce": None,
            "tflops": [213.5, 394.6, 473.2, 554.2, 598.8],
        },
        "min_fa3_varlen_ring": {
            "time": [0.216, 0.457, 1.296, 4.238, 15.187],
            "attn": None,
            "reduce": None,
            "tflops": [158.8, 300.8, 424.1, 518.9, 579.2],
        },
    },
    "comp128_comm4": {
        "PyTorch_SDPA": {
            "time": [0.502, 1.422, 3.503, 10.408, 32.799],
            "attn": [0.350, 0.810, 2.302, 7.902, 27.840],
            "reduce": [0.153, 0.612, 1.201, 2.506, 4.962],
            "tflops": [68.4, 96.7, 156.9, 211.3, 268.2],
        },
        "FA2_varlen": {
            "time": [0.348, 0.929, 2.507, 7.948, 27.609],
            "attn": [0.200, 0.506, 1.661, 6.231, 24.119],
            "reduce": [0.148, 0.423, 0.846, 1.720, 3.484],
            "tflops": [98.6, 147.9, 219.3, 276.7, 318.6],
        },
        "FA3_varlen": {
            "time": [0.271, 0.739, 1.845, 5.371, 17.502],
            "attn": [0.124, 0.316, 0.982, 3.541, 13.651],
            "reduce": [0.147, 0.422, 0.862, 1.831, 3.843],
            "tflops": [126.7, 186.1, 298.0, 409.5, 502.6],
        },
        "min_fa3_varlen": {
            "time": [0.164, 0.353, 1.320, 3.938, 14.690],
            "attn": None,
            "reduce": None,
            "tflops": [209.1, 389.5, 416.5, 558.5, 598.8],
        },
        "min_fa3_varlen_ring": {
            "time": [0.562, 1.061, 2.201, 4.977, 15.390],
            "attn": None,
            "reduce": None,
            "tflops": [61.2, 129.5, 249.7, 441.9, 571.5],
        },
    },
    "comp124_comm8": {
        "PyTorch_SDPA": {
            "time": [0.504, 1.422, 3.510, 10.266, 32.907],
            "attn": [0.351, 0.809, 2.307, 7.789, 27.965],
            "reduce": [0.153, 0.613, 1.204, 2.472, 4.964],
            "tflops": [68.2, 96.6, 156.6, 214.2, 267.3],
        },
        "FA2_varlen": {
            "time": [0.346, 0.929, 2.492, 7.986, 27.601],
            "attn": [0.198, 0.506, 1.649, 6.258, 24.118],
            "reduce": [0.148, 0.424, 0.842, 1.724, 3.496],
            "tflops": [99.2, 147.9, 220.6, 275.4, 318.7],
        },
        "FA3_varlen": {
            "time": [0.270, 0.738, 1.871, 5.395, 17.457],
            "attn": [0.124, 0.316, 0.996, 3.557, 13.631],
            "reduce": [0.146, 0.422, 0.874, 1.838, 3.839],
            "tflops": [127.3, 186.1, 293.9, 407.6, 503.9],
        },
        "min_fa3_varlen": {
            "time": [0.166, 0.355, 1.220, 3.987, 14.988],
            "attn": None,
            "reduce": None,
            "tflops": [206.7, 387.5, 450.6, 551.6, 586.9],
        },
        "min_fa3_varlen_ring": {
            "time": [0.335, 0.683, 1.425, 4.372, 15.617],
            "attn": None,
            "reduce": None,
            "tflops": [102.6, 201.2, 385.8, 503.0, 563.2],
        },
    },
    "comp116_comm16": {
        "PyTorch_SDPA": {
            "time": [0.502, 1.421, 3.507, 10.184, 32.870],
            "attn": [0.350, 0.809, 2.305, 7.726, 27.927],
            "reduce": [0.152, 0.612, 1.201, 2.457, 4.963],
            "tflops": [68.4, 96.7, 156.8, 215.9, 267.6],
        },
        "FA2_varlen": {
            "time": [0.346, 0.933, 2.520, 7.977, 27.573],
            "attn": [0.199, 0.508, 1.673, 6.252, 24.090],
            "reduce": [0.147, 0.425, 0.848, 1.722, 3.484],
            "tflops": [99.2, 147.3, 218.2, 275.7, 319.0],
        },
        "FA3_varlen": {
            "time": [0.270, 0.738, 1.918, 5.395, 17.578],
            "attn": [0.124, 0.316, 1.024, 3.558, 13.734],
            "reduce": [0.146, 0.422, 0.894, 1.838, 3.865],
            "tflops": [127.2, 186.2, 286.7, 407.6, 500.4],
        },
        "min_fa3_varlen": {
            "time": [0.175, 0.377, 1.218, 4.147, 15.402],
            "attn": None,
            "reduce": None,
            "tflops": [196.2, 364.7, 451.5, 530.3, 571.1],
        },
        "min_fa3_varlen_ring": {
            "time": [0.239, 0.497, 1.392, 4.534, 16.047],
            "attn": None,
            "reduce": None,
            "tflops": [143.7, 276.6, 394.8, 485.0, 548.2],
        },
    },
    "comp108_comm24": {
        "PyTorch_SDPA": {
            "time": [0.514, 1.420, 3.498, 10.133, 32.787],
            "attn": [0.360, 0.808, 2.298, 7.681, 27.834],
            "reduce": [0.154, 0.612, 1.200, 2.452, 4.964],
            "tflops": [66.9, 96.8, 157.2, 217.0, 268.3],
        },
        "FA2_varlen": {
            "time": [0.343, 0.929, 2.503, 8.018, 27.581],
            "attn": [0.196, 0.506, 1.657, 6.282, 24.091],
            "reduce": [0.147, 0.424, 0.845, 1.737, 3.487],
            "tflops": [100.3, 147.9, 219.6, 274.3, 318.9],
        },
        "FA3_varlen": {
            "time": [0.269, 0.738, 1.903, 5.384, 17.471],
            "attn": [0.123, 0.316, 1.016, 3.547, 13.647],
            "reduce": [0.146, 0.422, 0.886, 1.836, 3.840],
            "tflops": [127.6, 186.2, 288.9, 408.4, 503.5],
        },
        "min_fa3_varlen": {
            "time": [0.182, 0.396, 1.281, 4.329, 16.087],
            "attn": None,
            "reduce": None,
            "tflops": [189.0, 346.8, 429.2, 508.0, 546.8],
        },
        "min_fa3_varlen_ring": {
            "time": [0.243, 0.521, 1.449, 4.680, 16.742],
            "attn": None,
            "reduce": None,
            "tflops": [141.3, 263.9, 379.4, 469.9, 525.4],
        },
    },
}


def _make_subplots(settings: list[str], *, sharey: bool, height: float):
    fig, axes = plt.subplots(1, len(settings), figsize=(6.0 * len(settings), height), sharey=sharey)
    return fig, np.atleast_1d(axes)


def plot_causal_time_breakdown(
    settings: list[str] | None = None,
    output_path: Path | None = None,
) -> Path:
    settings = list(DEFAULT_SETTINGS if settings is None else settings)
    output_path = THIS_DIR / "causal_time_breakdown_89061.png" if output_path is None else output_path

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
    fig.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False)
    fig.legend(handles=style_handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.01,
        "Each seqlen group contains 5 bars in this order: PyTorch SDPA, FA2 varlen, FA3 varlen, min_fa3 varlen, min_fa3 ring.",
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
    output_path = THIS_DIR / "causal_tflops_89061.png" if output_path is None else output_path

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
    fig.legend(handles=line_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
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
