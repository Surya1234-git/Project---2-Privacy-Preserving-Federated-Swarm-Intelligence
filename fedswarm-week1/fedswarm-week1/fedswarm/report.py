"""CSV / plot helpers (matplotlib is imported lazily with a headless backend)."""
from __future__ import annotations

import csv
import json
import os

HISTORY_FIELDS = [
    "round", "test_loss", "test_acc", "train_loss",
    "nodes_selected", "nodes_reported", "mb_up", "mb_down", "seconds",
]


def save_history_csv(history, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(history)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_history(runs: dict, path: str, title: str = "Federated training"):
    """runs: {label: history_rows}. Draws test accuracy and test loss per round."""
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for label, hist in runs.items():
        r = [h["round"] for h in hist]
        ax1.plot(r, [h["test_acc"] * 100 for h in hist], marker="o", ms=3, label=label)
        ax2.plot(r, [h["test_loss"] for h in hist], marker="o", ms=3, label=label)
    ax1.set(xlabel="Round", ylabel="Global test accuracy (%)", title="Accuracy")
    ax2.set(xlabel="Round", ylabel="Global test loss", title="Loss")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_partition(matrix, path, max_nodes=60, title="Non-IID data split"):
    """Heatmap of class counts per node + histogram of node sizes."""
    import numpy as np
    plt = _plt()
    matrix = np.asarray(matrix)
    shown = matrix[:max_nodes]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})
    im = ax1.imshow(shown, aspect="auto", cmap="viridis")
    ax1.set(xlabel="Class", ylabel="Edge node", title=f"Samples per class (first {len(shown)} nodes)")
    fig.colorbar(im, ax=ax1, label="samples")
    ax2.hist(matrix.sum(axis=1), bins=20, color="#4c72b0")
    ax2.set(xlabel="Samples on node", ylabel="Number of nodes", title="Node sizes")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
