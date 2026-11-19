"""Create isolated per-node shards on disk.

    python -m swarm.data.prepare --nodes 120 --alpha 0.2 --out data/shards

Each node later receives *only its own* ``node-XXXX.npz`` (bind-mounted read-only into exactly
one container), so "raw data never leaves the device" is true by construction in the Docker
topology. A held-out ``global_test.npz`` is written for the in-process simulator only; it is
never mounted into any node or the aggregator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from swarm.data.datasets import load_dataset
from swarm.data.partition import capacity_dirichlet_partition, heterogeneity_report
from swarm.profiles import DEFAULT_MIX, DEVICE_PROFILES, assign_device_classes, parse_mix


def node_id(i: int) -> str:
    return f"node-{i:04d}"


def prepare(
    out: Path,
    *,
    dataset: str = "synthetic",
    n_samples: int = 30_000,
    nodes: int = 100,
    alpha: float = 0.3,
    mix: str = DEFAULT_MIX,
    seed: int = 42,
    val_frac: float = 0.2,
    test_frac: float = 0.15,
    min_size: int = 30,
    feature_shift: float = 0.0,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("node-*.npz"):
        old.unlink()
    rng = np.random.default_rng(seed)

    x, y = load_dataset(dataset, n_samples=n_samples, seed=seed)
    num_classes = int(y.max()) + 1
    order = rng.permutation(len(y))
    n_test = int(len(y) * test_frac)
    test_idx, pool_idx = order[:n_test], order[n_test:]

    # Standardise using pool statistics (in a real deployment: pre-agreed public normalisation).
    mu, sd = x[pool_idx].mean(axis=0), x[pool_idx].std(axis=0) + 1e-8
    x = (x - mu) / sd
    np.savez_compressed(out / "global_test.npz", x=x[test_idx].astype(np.float32), y=y[test_idx])

    classes = assign_device_classes(nodes, parse_mix(mix), seed)
    caps = np.array([DEVICE_PROFILES[c].capacity for c in classes])
    xp, yp = x[pool_idx], y[pool_idx]
    parts = capacity_dirichlet_partition(yp, caps, alpha, rng, min_size=min_size)
    report = heterogeneity_report(yp, parts, num_classes)

    h = hashlib.sha256()
    manifest_nodes = []
    for i, part in enumerate(parts):
        nid = node_id(i + 1)
        h.update(nid.encode() + part.tobytes())
        xi, yi = xp[part].copy(), yp[part].copy()
        if feature_shift > 0:  # per-device sensor drift: gain and bias
            gain = rng.normal(1.0, feature_shift)
            bias = rng.normal(0.0, feature_shift, size=xi.shape[1])
            xi = xi * gain + bias
        perm = rng.permutation(len(yi))
        n_val = max(4, int(len(yi) * val_frac))
        v, t = perm[:n_val], perm[n_val:]
        np.savez_compressed(out / f"{nid}.npz", x_train=xi[t].astype(np.float32), y_train=yi[t],
                            x_val=xi[v].astype(np.float32), y_val=yi[v])
        manifest_nodes.append({
            "node_id": nid, "device_class": classes[i], "n_train": int(len(t)),
            "n_val": int(len(v)),
            "label_counts": np.bincount(yi, minlength=num_classes).tolist(),
        })

    manifest = {
        "dataset": dataset, "input_dim": int(x.shape[1]), "num_classes": num_classes,
        "n_nodes": nodes, "alpha": alpha, "mix": mix, "seed": seed,
        "feature_shift": feature_shift, "partition_sha256": h.hexdigest(),
        "heterogeneity": report, "nodes": manifest_nodes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        _plot_heatmap(manifest, out / "heterogeneity.png")
    except Exception as exc:  # matplotlib is optional
        print(f"[prepare] heatmap skipped: {exc}")
    return manifest


def _plot_heatmap(manifest: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.array([n["label_counts"] for n in manifest["nodes"]], dtype=float)
    shares = counts / counts.sum(axis=1, keepdims=True)
    cls = [n["device_class"] for n in manifest["nodes"]]
    order = np.lexsort((-counts.sum(axis=1), cls))
    fig, ax = plt.subplots(figsize=(8, max(4, len(order) * 0.06 + 2)))
    im = ax.imshow(shares[order], aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("class label")
    ax.set_ylabel("edge nodes (sorted by device class, then data volume)")
    hz = manifest["heterogeneity"]
    ax.set_title(f"Label share per node  |  non-IID score {hz['noniid_score']:.2f}, "
                 f"size Gini {hz['size_gini']:.2f}")
    fig.colorbar(im, ax=ax, label="share of node's data")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "digits", "mnist"])
    ap.add_argument("--samples", type=int, default=30_000)
    ap.add_argument("--nodes", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.3, help="Dirichlet label skew (lower = harsher)")
    ap.add_argument("--mix", default=DEFAULT_MIX)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-size", type=int, default=30)
    ap.add_argument("--feature-shift", type=float, default=0.0)
    a = ap.parse_args()
    m = prepare(Path(a.out), dataset=a.dataset, n_samples=a.samples, nodes=a.nodes, alpha=a.alpha,
                mix=a.mix, seed=a.seed, min_size=a.min_size, feature_shift=a.feature_shift)
    hz = m["heterogeneity"]
    print(f"[prepare] {m['n_nodes']} shards -> {a.out}")
    print(f"[prepare] non-IID score={hz['noniid_score']:.3f}  mean effective classes="
          f"{hz['mean_effective_classes']:.2f}/{m['num_classes']}  size gini={hz['size_gini']:.2f}  "
          f"sizes min/median/max={hz['size_min']}/{hz['size_median']:.0f}/{hz['size_max']}")
    print(f"[prepare] partition sha256={m['partition_sha256'][:16]}...")


if __name__ == "__main__":
    main()
