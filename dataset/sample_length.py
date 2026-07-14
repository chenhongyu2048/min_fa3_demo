import numpy as np
import matplotlib
matplotlib.use("Agg")          # Works on headless servers (saves to file)
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter

# ======================================================================
# Config
# ======================================================================
DATASET_CHOICE = "github"        # options: "pile" / "arxiv" / "github"
NUM_SAMPLES = 20_000            # number of samples to collect
TOKENIZER_NAME = "gpt2"

# Bin edges from Table 2 (unit: tokens)
BINS = [0, 1_000, 2_000, 4_000, 8_000, 16_000,
        32_000, 64_000, 128_000, 256_000]
BIN_LABELS = ["<1", "1-2", "2-4", "4-8", "8-16",
              "16-32", "32-64", "64-128", "128-256"]

# ----------------------------------------------------------------------
# Per-dataset load & filter config
# ----------------------------------------------------------------------
DATASET_CONFIG = {
    # pile: Pile-CC subset only
    "pile": {
        "path": "monology/pile-uncopyrighted",
        "name": None,
        "text_field": "text",
        "filter_fn": lambda ex:
            ex.get("meta", {}).get("pile_set_name", "") == "Pile-CC",
    },
    # arxiv: ArXiv subset of Pile
    "arxiv": {
        "path": "monology/pile-uncopyrighted",
        "name": None,
        "text_field": "text",
        "filter_fn": lambda ex:
            ex.get("meta", {}).get("pile_set_name", "") == "ArXiv",
    },
    # github: Github subset of Pile
    "github": {
        "path": "monology/pile-uncopyrighted",
        "name": None,
        "text_field": "text",
        "filter_fn": lambda ex:
            ex.get("meta", {}).get("pile_set_name", "") == "Github",
    },
}

# ======================================================================
# Main
# ======================================================================
cfg = DATASET_CONFIG[DATASET_CHOICE]
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
# GPT-2 defaults to model_max_length=1024; bump it to silence long-text warnings
tokenizer.model_max_length = int(1e9)

print(f"Collecting dataset: {DATASET_CHOICE}  (source: {cfg['path']})")

ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

# ---- Stream, filter, and measure token length ----
lengths = []
scanned = 0
for example in ds:
    scanned += 1
    if not cfg["filter_fn"](example):
        continue

    text = example[cfg["text_field"]]
    token_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    lengths.append(token_len)

    if len(lengths) % 1_000 == 0:
        print(f"Collected {len(lengths)} (scanned {scanned}), "
              f"running mean length ~ {np.mean(lengths):.0f} tokens")

    if len(lengths) >= NUM_SAMPLES:
        break

lengths = np.array(lengths)

# ---- Bin per Table 2 and compute proportions ----
# Anything above 256k goes into the last bin (long seqs truncated in the paper)
clipped = np.clip(lengths, 0, BINS[-1] - 1)
counts, _ = np.histogram(clipped, bins=BINS)
proportions = counts / counts.sum()

# ---- Print results (Table 2 style) ----
print(f"\n===== {DATASET_CHOICE} sequence length distribution "
      f"(aligned with Zeppelin Table 2) =====")
print(f"{'Range (k tokens)':<18}{'Prop':<10}{'Count':<10}")
for label, p, c in zip(BIN_LABELS, proportions, counts):
    print(f"{label:<18}{p:<10.3f}{c:<10}")

print(f"\nSamples    : {len(lengths)}")
print(f"Mean length: {lengths.mean():.0f} tokens")
print(f"Median(p50): {np.percentile(lengths, 50):.0f}")
print(f"p90        : {np.percentile(lengths, 90):.0f}")
print(f"p99        : {np.percentile(lengths, 99):.0f}")
print(f"Max        : {lengths.max()}")

# ======================================================================
# Plot log-scale bar chart
# ======================================================================
fig, ax = plt.subplots(figsize=(10, 6))

plot_counts = counts.astype(float)
bars = ax.bar(BIN_LABELS, plot_counts, color="#4C72B0", edgecolor="black")

ax.set_yscale("log")                       # log-scale y axis
ax.set_xlabel("Sequence length range (k tokens)")
ax.set_ylabel("Count (log scale)")
ax.set_title(f"{DATASET_CHOICE} sequence length distribution "
             f"(n={len(lengths)}, tokenizer={TOKENIZER_NAME})")
ax.grid(axis="y", which="both", linestyle="--", alpha=0.4)

# Annotate each bar with count and percentage
for bar, c, p in zip(bars, counts, proportions):
    if c > 0:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{c}\n({p*100:.1f}%)",
                ha="center", va="bottom", fontsize=8)

plt.tight_layout()
fig_path = f"{DATASET_CHOICE}_length_hist_log.png"
plt.savefig(fig_path, dpi=150)
print(f"\nSaved bar chart: {fig_path}")

# ---- Save arrays ----
np.save(f"{DATASET_CHOICE}_doc_lengths.npy", lengths)
np.save(f"{DATASET_CHOICE}_bin_proportions.npy", proportions)
print(f"Saved: {DATASET_CHOICE}_doc_lengths.npy / "
      f"{DATASET_CHOICE}_bin_proportions.npy")