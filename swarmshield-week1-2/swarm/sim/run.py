"""CLI: python -m swarm.sim.run --shards data/shards --mode both --rounds 20"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm.config import ExperimentConfig
from swarm.sim.engine import run_simulation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/shards")
    ap.add_argument("--mode", choices=["plain", "fhe", "both"], default="both")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--participation", type=float, default=0.3)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--prox-mu", type=float, default=0.0)
    ap.add_argument("--no-dropouts", action="store_true")
    ap.add_argument("--out", default="runs/sim")
    a = ap.parse_args()

    modes = ["plain", "fhe"] if a.mode == "both" else [a.mode]
    results = {}
    for m in modes:
        cfg = ExperimentConfig(mode=m, rounds=a.rounds, participation=a.participation,
                               local_epochs=a.local_epochs, lr=a.lr, prox_mu=a.prox_mu)
        results[m] = run_simulation(a.shards, cfg, dropouts=not a.no_dropouts, verbose=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sim_results.json").write_text(json.dumps({k: v.to_dict() for k, v in results.items()},
                                                     indent=2))
    if len(results) == 2:
        d = abs(results["plain"].final_acc - results["fhe"].final_acc)
        print(f"\nfinal test acc  plain={results['plain'].final_acc:.4f}  "
              f"fhe={results['fhe'].final_acc:.4f}  |diff|={d:.5f}")
    try:
        _plot(results, out / "convergence.png")
    except Exception as exc:
        print(f"[sim] plot skipped: {exc}")


def _plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for m, res in results.items():
        ax.plot([h["round"] for h in res.history], [h["test_acc"] for h in res.history],
                marker="o", ms=3, label=m)
    ax.set_xlabel("communication round")
    ax.set_ylabel("global test accuracy")
    ax.set_title("FedAvg convergence on non-IID swarm")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
