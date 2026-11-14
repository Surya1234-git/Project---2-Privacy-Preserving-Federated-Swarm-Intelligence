"""In-process FedAvg simulation: hundreds/thousands of logical nodes on one machine,
no Docker needed. Same maths and same node-sampling as the Docker deployment, so the
two are directly comparable."""
from __future__ import annotations

import time

import numpy as np

from .aggregation import fedavg, select_nodes
from .config import FLConfig
from .data import model_spec
from .models import SmallCNN, get_weights, set_weights
from .training import evaluate, local_train


def run_fedavg_simulation(data: dict, parts, cfg: FLConfig, device="cpu", log=print):
    """data: output of load_dataset; parts: list of index arrays (one per node)."""
    x, y = data["x_train"], data["y_train"]
    model = SmallCNN(**model_spec(data["meta"]))
    global_w = get_weights(model)
    num_nodes = len(parts)
    history = []

    for r in range(1, cfg.rounds + 1):
        t0 = time.time()
        selected = select_nodes(num_nodes, cfg.client_fraction, r, cfg.seed)
        updates, losses = [], []
        for nid in selected:
            idx = parts[nid]
            set_weights(model, global_w)
            loss = local_train(
                model, x[idx], y[idx], cfg.local_epochs, cfg.batch_size, cfg.lr,
                cfg.momentum, cfg.mu, device, seed=cfg.seed + r * 1000 + nid,
            )
            updates.append((get_weights(model), len(idx)))
            losses.append((loss, len(idx)))
        global_w = fedavg(updates)
        set_weights(model, global_w)
        test_loss, test_acc = evaluate(model, data["x_test"], data["y_test"], device)
        total = sum(n for _, n in losses)
        row = {
            "round": r, "test_loss": test_loss, "test_acc": test_acc,
            "train_loss": sum(l * n for l, n in losses) / total,
            "nodes_selected": len(selected), "nodes_reported": len(selected),
            "mb_up": 0.0, "mb_down": 0.0, "seconds": time.time() - t0,
        }
        history.append(row)
        log(f"round {r:>3}/{cfg.rounds} | acc {test_acc * 100:6.2f}% | "
            f"test loss {test_loss:.4f} | train loss {row['train_loss']:.4f} | "
            f"{len(selected)} nodes | {row['seconds']:.1f}s")
    return history
