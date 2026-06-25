from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
INPUT_PATH = THIS_DIR / "min_fa3_demo_h100-93782.out"
RUN_ID = INPUT_PATH.stem.rsplit("-", 1)[-1]

WORLD_SIZE = 2
CHECK = False
SEQLENS = [512, 1024, 2048, 4096, 8192]
PANEL_CONFIGS = [
    (128, 4),
    (124, 8),
    (116, 16),
]
MODES = [True, False]

METHODS = [
    "pytorch",
    "fa2",
    "fa3",
    "min_varlen",
    "min_varlen_ring",
    "min_varlen_mega_ring",
]

METHOD_LABELS = {
    "pytorch": "PyTorch",
    "fa2": "FA2",
    "fa3": "FA3",
    "min_varlen": "min_varlen",
    "min_varlen_ring": "min_varlen_ring",
    "min_varlen_mega_ring": "min_varlen_mega_ring",
}

METHOD_COLORS = {
    "pytorch": "#4C78A8",
    "fa2": "#F58518",
    "fa3": "#54A24B",
    "min_varlen": "#E45756",
    "min_varlen_ring": "#72B7B2",
    "min_varlen_mega_ring": "#B279A2",
}

MODE_LABELS = {
    True: "Causal",
    False: "Noncausal",
}

CONFIG_RE = re.compile(
    r"^Config: world_size=(\d+), .* num_comp_sm=(\d+), num_comm_sm=(\d+), .* check=(True|False)$"
)
CASE_RE = re.compile(r"^B=\d+, local_S=(\d+), QH=\d+, KVH=\d+, D=\d+, mode=(noncausal|causal)$")
RESULT_RE = re.compile(
    r"^(\S+)\s+"
    r"t0=\d+(?:\.\d+)?, t1=\d+(?:\.\d+)? \| max_across_ranks=(\d+(?:\.\d+)?)\s+"
    r"(\d+(?:\.\d+)?)\s+"
    r"(\d+(?:\.\d+)?)\s+"
    r"\S+"
)


def load_data(
    input_path: Path = INPUT_PATH,
) -> dict[tuple[int, int], dict[bool, dict[str, dict[str, list[float]]]]]:
    data = {
        config: {
            is_causal: {
                method: {"time_ms": [], "tflops_per_gpu": []}
                for method in METHODS
            }
            for is_causal in MODE_LABELS
        }
        for config in PANEL_CONFIGS
    }
    active_config = False
    current_case: tuple[int, bool] | None = None
    current_panel: tuple[int, int] | None = None

    for line in input_path.read_text(encoding="utf-8").splitlines():
        config_match = CONFIG_RE.match(line)
        if config_match is not None:
            config = (int(config_match.group(2)), int(config_match.group(3)))
            active_config = (
                int(config_match.group(1)) == WORLD_SIZE
                and config in PANEL_CONFIGS
                and (config_match.group(4) == "True") == CHECK
            )
            current_panel = config if active_config else None
            current_case = None
            continue

        if not active_config or current_panel is None:
            continue

        case_match = CASE_RE.match(line)
        if case_match is not None:
            seqlen = int(case_match.group(1))
            is_causal = case_match.group(2) == "causal"
            current_case = (seqlen, is_causal)
            continue

        result_match = RESULT_RE.match(line)
        if result_match is None or current_case is None:
            continue

        method = result_match.group(1)
        if method not in METHODS:
            continue

        seqlen, is_causal = current_case
        if seqlen not in SEQLENS:
            continue

        data[current_panel][is_causal][method]["time_ms"].append(float(result_match.group(2)))
        data[current_panel][is_causal][method]["tflops_per_gpu"].append(float(result_match.group(4)))

    for config in PANEL_CONFIGS:
        for is_causal in MODE_LABELS:
            for method in METHODS:
                for metric in ("time_ms", "tflops_per_gpu"):
                    values = data[config][is_causal][method][metric]
                    if len(values) != len(SEQLENS):
                        raise ValueError(
                            "missing data for "
                            f"comp={config[0]}, comm={config[1]}, mode={MODE_LABELS[is_causal]}, "
                            f"method={method}, metric={metric}: expected {len(SEQLENS)} points, got {len(values)}"
                        )

    return data


DATA = load_data()


def _make_figure(metric_key: str, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 8.6), sharex=True, sharey="row")
    for col_idx, config in enumerate(PANEL_CONFIGS):
        for row_idx, is_causal in enumerate(MODES):
            ax = axes[row_idx, col_idx]
            for method in METHODS:
                ax.plot(
                    SEQLENS,
                    DATA[config][is_causal][method][metric_key],
                    marker="o",
                    linewidth=2.2,
                    markersize=5.5,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
            if row_idx == 0:
                ax.set_title(f"comp={config[0]}, comm={config[1]}", fontsize=15)
            if col_idx == 0:
                ax.set_ylabel(f"{MODE_LABELS[is_causal]}\n{title}", fontsize=14)
            ax.set_xticks(SEQLENS, [str(seqlen) for seqlen in SEQLENS])
            ax.set_xlabel("Seqlen", fontsize=14)
            ax.tick_params(axis="both", labelsize=13)
            ax.grid(alpha=0.25, linewidth=0.8)
    fig.suptitle(
        f"world_size={WORLD_SIZE}",
        fontsize=18,
        y=0.99,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(METHODS),
        frameon=False,
        prop={"size": 13},
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_time(output_path: Path | None = None) -> Path:
    output_path = THIS_DIR / f"time_{RUN_ID}.png" if output_path is None else output_path
    fig = _make_figure("time_ms", "Time (ms)")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_tflops(output_path: Path | None = None) -> Path:
    output_path = THIS_DIR / f"tflops_per_gpu_{RUN_ID}.png" if output_path is None else output_path
    fig = _make_figure("tflops_per_gpu", "Per GPU TFLOPS")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    time_path = plot_time()
    tflops_path = plot_tflops()
    print(f"saved {time_path}")
    print(f"saved {tflops_path}")


if __name__ == "__main__":
    main()
