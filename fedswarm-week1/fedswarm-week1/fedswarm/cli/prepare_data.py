"""Create the private per-node datasets.

    python -m fedswarm.cli.prepare_data --dataset mnist --num-nodes 100 --partition dirichlet --alpha 0.3

Writes:
    data/nodes/node_0000.npz ...   one private shard per node  (mounted ONLY into client containers)
    data/server/global_test.npz    held-out test set           (mounted ONLY into the server)
    data/server/meta.json          dataset / model spec
    results/partition_report.json  statistics proving the split is non-IID
    results/partition_heatmap.png  picture for your report
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from fedswarm.data import load_dataset
from fedswarm.partition import heterogeneity_stats, label_matrix, make_partition
from fedswarm.report import plot_partition, save_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", default="mnist",
                   help="synthetic | mnist | fmnist | cifar10 | bloodmnist | pathmnist | ...")
    p.add_argument("--num-nodes", type=int, default=100)
    p.add_argument("--partition", choices=["dirichlet", "shards", "iid"], default="dirichlet")
    p.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration (lower = more non-IID)")
    p.add_argument("--shards-per-node", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--raw-dir", default=None, help="download cache (default: <data-dir>/raw)")
    p.add_argument("--results-dir", default="results")
    a = p.parse_args(argv)

    raw_dir = a.raw_dir or os.path.join(a.data_dir, "raw")
    print(f"Loading dataset '{a.dataset}' ...")
    data = load_dataset(a.dataset, raw_dir, seed=a.seed)
    meta = data["meta"]
    y = data["y_train"]
    print(f"  train={len(y)} test={len(data['y_test'])} classes={meta['num_classes']} "
          f"image={meta['in_channels']}x{meta['image_size']}x{meta['image_size']}")

    parts = make_partition(y, a.partition, a.num_nodes, a.alpha, a.shards_per_node, a.seed)

    nodes_dir = os.path.join(a.data_dir, "nodes")
    server_dir = os.path.join(a.data_dir, "server")
    os.makedirs(nodes_dir, exist_ok=True)
    os.makedirs(server_dir, exist_ok=True)
    for old in glob.glob(os.path.join(nodes_dir, "node_*.npz")):
        os.remove(old)                      # never leave shards from a previous run behind
    for i, idx in enumerate(parts):
        np.savez_compressed(os.path.join(nodes_dir, f"node_{i:04d}.npz"),
                            x=data["x_train"][idx], y=y[idx])
    np.savez_compressed(os.path.join(server_dir, "global_test.npz"),
                        x=data["x_test"], y=data["y_test"])
    with open(os.path.join(server_dir, "meta.json"), "w") as f:
        json.dump({**meta, "num_nodes": a.num_nodes}, f, indent=2)

    matrix = label_matrix(y, parts, meta["num_classes"])
    stats = heterogeneity_stats(matrix)
    save_json({"dataset": meta, "partition": a.partition, "alpha": a.alpha,
               "shards_per_node": a.shards_per_node, "seed": a.seed,
               "stats": stats, "label_matrix": matrix.tolist()},
              os.path.join(a.results_dir, "partition_report.json"))
    title = f"{a.dataset}: {a.partition}" + (f" (alpha={a.alpha})" if a.partition == "dirichlet" else "")
    plot_partition(matrix, os.path.join(a.results_dir, "partition_heatmap.png"), title=title)

    print(f"\nWrote {a.num_nodes} private node shards to {nodes_dir}")
    print(f"  samples/node: min {stats['samples_min']}, mean {stats['samples_mean']:.0f}, max {stats['samples_max']}")
    print(f"  classes/node (mean): {stats['classes_per_node_mean']:.1f} of {meta['num_classes']}")
    print(f"  mean TV distance from global label distribution: {stats['mean_tv_distance_from_global']:.3f} "
          f"(0 = IID, 1 = fully disjoint)")


if __name__ == "__main__":
    main()
