"""In-process swarm simulator (no network, no Docker).

Runs the *same* learning, selection, dropout and aggregation code as the networked system, so
results here (plain vs. encrypted, FedAvg vs. FedProx, IID vs. non-IID) carry over to the
Docker swarm. Because everything lives in one process this class may decrypt the aggregated
model to score it on a global test set - a privilege the real aggregator never has.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from swarm.config import ExperimentConfig
from swarm.crypto import ckks
from swarm.fl.aggregation import EncryptedAggregator, PlainAggregator
from swarm.fl.selection import dropout_draw, select_participants
from swarm.models.mlp import MLP
from swarm.node.trainer import LocalTrainer, Shard
from swarm.profiles import DEVICE_PROFILES


@dataclass
class SimResult:
    mode: str
    history: list[dict]
    final_acc: float
    final_loss: float
    global_weights: list[np.ndarray] = field(default_factory=list)  # per version, if kept

    def to_dict(self) -> dict:
        return {"mode": self.mode, "final_acc": self.final_acc, "final_loss": self.final_loss,
                "history": self.history}


def load_swarm(shards_dir: str | Path) -> tuple[dict, dict[str, Shard], tuple[np.ndarray, np.ndarray]]:
    d = Path(shards_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    shards = {n["node_id"]: Shard.load(d / f"{n['node_id']}.npz") for n in manifest["nodes"]}
    t = np.load(d / "global_test.npz")
    return manifest, shards, (t["x"].astype(np.float64), t["y"].astype(np.int64))


def run_simulation(shards_dir: str | Path, cfg: ExperimentConfig, *, mode: str | None = None,
                   dropouts: bool = True, keep_weights: bool = False,
                   verbose: bool = False) -> SimResult:
    mode = mode or cfg.mode
    manifest, shards, (xt, yt) = load_swarm(shards_dir)
    cfg.input_dim, cfg.num_classes = manifest["input_dim"], manifest["num_classes"]
    classes = {n["node_id"]: n["device_class"] for n in manifest["nodes"]}
    trainers = {nid: LocalTrainer(nid, sh, cfg) for nid, sh in shards.items()}
    node_ids = sorted(trainers)

    scorer = MLP(cfg.dims, seed=cfg.model_seed)
    global_w = scorer.get_flat()
    versions = [global_w.copy()] if keep_weights else []

    secret_ctx = codec = enc_agg = None
    if mode == "fhe":
        secret_ctx = ckks.new_secret_context(cfg.poly_modulus_degree, cfg.coeff_mod_bit_sizes,
                                             cfg.scale_bits)
        pub = ckks.load_context(ckks.export_public(secret_ctx), require_secret=False)
        enc_agg = EncryptedAggregator(pub)
        codec = ckks.CKKSCodec(secret_ctx, scorer.n_params, cfg.poly_modulus_degree)
    plain_agg = PlainAggregator()

    def score(w):
        scorer.set_flat(w)
        return scorer.evaluate(xt, yt)

    loss0, acc0 = score(global_w)
    history = [{"round": 0, "test_acc": acc0, "test_loss": loss0}]
    for r in range(1, cfg.rounds + 1):
        t_round = time.perf_counter()
        attempt, got = 0, []
        while True:
            selected = select_participants(node_ids, cfg.participation, cfg.min_selected,
                                           cfg.selection_seed, r, attempt)
            got = [n for n in selected if not (dropouts and dropout_draw(
                trainers[n].crc, r, attempt, DEVICE_PROFILES[classes[n]].dropout_p))]
            if len(got) >= max(1, math.ceil(cfg.quorum * len(selected))):
                break
            attempt += 1
            if attempt >= cfg.max_attempts:
                raise RuntimeError(f"round {r}: quorum never reached")
        dropped = [n for n in selected if n not in got]

        local_w, n_samples, train_losses = [], [], []
        for nid in got:
            tr = trainers[nid]
            tr.set_global(global_w)
            w, loss = tr.train(r)
            local_w.append(w)
            n_samples.append(tr.n_train)
            train_losses.append(loss)

        info: dict = {}
        if mode == "fhe":
            t0 = time.perf_counter()
            uploads = [codec.encrypt(w) for w in local_w]
            info["encrypt_s_per_node"] = (time.perf_counter() - t0) / len(uploads)
            info["upload_bytes_per_node"] = sum(len(c) for c in uploads[0])
            t0 = time.perf_counter()
            agg_ct = enc_agg.aggregate(uploads, n_samples)
            info["aggregate_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            global_w = codec.decrypt(agg_ct)
            info["decrypt_s"] = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            global_w = plain_agg.aggregate(local_w, n_samples)
            info["aggregate_s"] = time.perf_counter() - t0
            info["upload_bytes_per_node"] = global_w.nbytes
        if keep_weights:
            versions.append(global_w.copy())
        loss, acc = score(global_w)
        by_class: dict[str, int] = {}
        for n in dropped:
            by_class[classes[n]] = by_class.get(classes[n], 0) + 1
        tot = float(sum(n_samples))
        history.append({
            "round": r, "attempt": attempt, "selected": len(selected), "received": len(got),
            "dropped": len(dropped), "dropped_by_class": by_class,
            "train_loss": float(np.dot(train_losses, n_samples) / tot),
            "test_acc": acc, "test_loss": loss, "wall_s": time.perf_counter() - t_round, **info,
        })
        if verbose:
            print(f"[{mode:5}] round {r:3d}  recv {len(got):3d}/{len(selected):3d}  "
                  f"train_loss {history[-1]['train_loss']:.3f}  test_acc {acc:.4f}")
    return SimResult(mode, history, history[-1]["test_acc"], history[-1]["test_loss"], versions)
