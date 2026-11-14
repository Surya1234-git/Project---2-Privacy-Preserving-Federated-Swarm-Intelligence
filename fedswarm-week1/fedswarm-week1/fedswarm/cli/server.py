"""Federated server entry point (runs inside the `server` container)."""
from __future__ import annotations

import json
import logging
import os
import threading
import time

import numpy as np

from fedswarm.config import FLConfig
from fedswarm.coordinator import Coordinator, create_app
from fedswarm.data import model_spec
from fedswarm.models import SmallCNN, get_weights, set_weights
from fedswarm.report import plot_history, save_history_csv, save_json
from fedswarm.training import evaluate


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [server] %(message)s")
    cfg = FLConfig.from_env()
    cfg.validate_for_cluster()
    data_dir = os.environ.get("DATA_DIR", "data/server")
    out_dir = os.environ.get("RESULTS_DIR", "results")

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    if meta["num_nodes"] != cfg.num_nodes:
        raise SystemExit(f"data was prepared for {meta['num_nodes']} nodes but NUM_NODES={cfg.num_nodes}")
    test = np.load(os.path.join(data_dir, "global_test.npz"))
    x_test, y_test = test["x"], test["y"]

    model = SmallCNN(**model_spec(meta))
    torch_threads = int(os.environ.get("TORCH_THREADS", "2"))
    import torch
    torch.set_num_threads(torch_threads)

    def evaluate_fn(weights):
        set_weights(model, weights)
        return evaluate(model, x_test, y_test)

    def on_round(row, history):
        save_history_csv(history, os.path.join(out_dir, "history.csv"))

    coord = Coordinator(cfg, get_weights(model), evaluate_fn, model_spec(meta), on_round)
    app = create_app(coord)

    def lifecycle():
        coord.run()
        if coord.history:
            save_history_csv(coord.history, os.path.join(out_dir, "history.csv"))
            plot_history({"FedAvg (docker swarm)": coord.history},
                         os.path.join(out_dir, "fedavg_curve.png"),
                         title=f"FedAvg - {cfg.num_nodes} nodes in {cfg.num_containers} containers")
            last = coord.history[-1]
            save_json({"dataset": meta["name"], "config": cfg.__dict__,
                       "final_test_acc": last["test_acc"], "final_test_loss": last["test_loss"],
                       "total_mb_up": sum(h["mb_up"] for h in coord.history),
                       "total_mb_down": sum(h["mb_down"] for h in coord.history),
                       "rounds_completed": len(coord.history)},
                      os.path.join(out_dir, "summary.json"))
        time.sleep(cfg.shutdown_grace)      # let clients read the "done" message
        os._exit(0 if coord.state == "done" else 1)

    threading.Thread(target=lifecycle, daemon=True).start()
    logging.info("waiting for %d client containers on port %d ...", cfg.num_containers, cfg.port)
    app.run(host="0.0.0.0", port=cfg.port, threaded=True)


if __name__ == "__main__":
    main()
