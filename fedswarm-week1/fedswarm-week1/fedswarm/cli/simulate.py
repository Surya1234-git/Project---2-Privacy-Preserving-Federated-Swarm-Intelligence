"""Run FedAvg over many logical nodes in a single process (no Docker).

    python -m fedswarm.cli.simulate --dataset mnist --num-nodes 200 --rounds 30 --compare-iid
"""
from __future__ import annotations

import argparse
import os

from fedswarm.config import FLConfig
from fedswarm.data import load_dataset
from fedswarm.partition import heterogeneity_stats, label_matrix, make_partition
from fedswarm.report import plot_history, save_history_csv
from fedswarm.simulation import run_fedavg_simulation


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--num-nodes", type=int, default=100)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--fraction", type=float, default=0.1, help="fraction of nodes per round")
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--mu", type=float, default=0.0, help="FedProx strength (0 = FedAvg)")
    p.add_argument("--partition", choices=["dirichlet", "shards", "iid"], default="dirichlet")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--shards-per-node", type=int, default=2)
    p.add_argument("--compare-iid", action="store_true", help="also run an IID control")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="results")
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)

    cfg = FLConfig(num_nodes=a.num_nodes, rounds=a.rounds, client_fraction=a.fraction,
                   local_epochs=a.local_epochs, batch_size=a.batch_size, lr=a.lr,
                   mu=a.mu, seed=a.seed)
    data = load_dataset(a.dataset, a.raw_dir, seed=a.seed)
    classes = data["meta"]["num_classes"]

    def run(strategy):
        parts = make_partition(data["y_train"], strategy, a.num_nodes, a.alpha,
                               a.shards_per_node, a.seed)
        stats = heterogeneity_stats(label_matrix(data["y_train"], parts, classes))
        label = strategy + (f" (alpha={a.alpha})" if strategy == "dirichlet" else "")
        print(f"\n=== {label}: {a.num_nodes} nodes, "
              f"{stats['classes_per_node_mean']:.1f} classes/node, "
              f"TV={stats['mean_tv_distance_from_global']:.3f} ===")
        hist = run_fedavg_simulation(data, parts, cfg, a.device)
        save_history_csv(hist, os.path.join(a.out, f"simulation_{strategy}.csv"))
        return label, hist

    runs = dict([run(a.partition)])
    if a.compare_iid and a.partition != "iid":
        runs.update([run("iid")])

    path = os.path.join(a.out, "simulation_curves.png")
    plot_history(runs, path, title=f"FedAvg on {a.dataset} - {a.num_nodes} nodes, "
                                   f"{int(a.fraction * 100)}% per round")
    for label, hist in runs.items():
        print(f"final [{label}]: accuracy {hist[-1]['test_acc'] * 100:.2f}%")
    print(f"\nSaved curves to {path}")


if __name__ == "__main__":
    main()
