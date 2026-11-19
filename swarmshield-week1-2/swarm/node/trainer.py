"""Pure client-side learning logic, shared by the networked EdgeNode and the simulator."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from swarm.config import ExperimentConfig
from swarm.models.mlp import MLP, local_sgd


@dataclass
class Shard:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> Shard:
        d = np.load(path)
        return cls(d["x_train"].astype(np.float64), d["y_train"].astype(np.int64),
                   d["x_val"].astype(np.float64), d["y_val"].astype(np.int64))


def node_crc(node_id: str) -> int:
    return zlib.crc32(node_id.encode())


class LocalTrainer:
    def __init__(self, node_id: str, shard: Shard, cfg: ExperimentConfig):
        self.node_id, self.shard, self.cfg = node_id, shard, cfg
        self.crc = node_crc(node_id)
        self.model = MLP(cfg.dims, seed=cfg.model_seed)  # identical init on every node

    @property
    def n_train(self) -> int:
        return int(len(self.shard.y_train))

    def set_global(self, flat: np.ndarray) -> None:
        self.model.set_flat(flat)

    def evaluate_global(self) -> dict:
        loss, acc = self.model.evaluate(self.shard.x_val, self.shard.y_val)
        return {"loss": loss, "acc": acc, "n": int(len(self.shard.y_val))}

    def train(self, round_id: int) -> tuple[np.ndarray, float]:
        anchor = self.model.get_flat()
        rng = np.random.default_rng([self.crc, round_id])
        loss = local_sgd(self.model, self.shard.x_train, self.shard.y_train,
                         epochs=self.cfg.local_epochs, batch_size=self.cfg.batch_size,
                         lr=self.cfg.lr, rng=rng, prox_mu=self.cfg.prox_mu, anchor=anchor)
        return self.model.get_flat(), loss
