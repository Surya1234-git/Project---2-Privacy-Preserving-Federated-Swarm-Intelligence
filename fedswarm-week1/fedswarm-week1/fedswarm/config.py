"""Run configuration. Every field can be overridden by an UPPER_CASE env var
(this is how docker-compose passes settings into the containers)."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass


def _env(name, default, cast):
    value = os.environ.get(name)
    return default if value is None or value == "" else cast(value)


@dataclass
class FLConfig:
    num_nodes: int = 100          # logical edge nodes (each owns one private data shard)
    num_containers: int = 8       # docker containers; each hosts num_nodes/num_containers nodes
    rounds: int = 30              # federated rounds
    client_fraction: float = 0.1  # fraction of nodes sampled per round (the "C" in FedAvg)
    local_epochs: int = 1         # local epochs per selected node per round (the "E" in FedAvg)
    batch_size: int = 32
    lr: float = 0.01
    momentum: float = 0.9
    mu: float = 0.0               # FedProx strength; 0.0 = plain FedAvg (Week 1 baseline)
    seed: int = 42
    round_timeout: float = 600.0  # seconds to wait for stragglers before aggregating what arrived
    port: int = 5000
    shutdown_grace: float = 10.0  # seconds the server stays up after the last round

    def __post_init__(self):
        if not 0 < self.client_fraction <= 1:
            raise ValueError("client_fraction must be in (0, 1]")
        if self.num_nodes < 1 or self.rounds < 1:
            raise ValueError("num_nodes and rounds must be >= 1")

    def validate_for_cluster(self):
        if self.num_containers > self.num_nodes:
            raise ValueError("NUM_CONTAINERS cannot exceed NUM_NODES")

    @classmethod
    def from_env(cls) -> "FLConfig":
        defaults = cls()
        values = {
            name: _env(name.upper(), default, type(default))
            for name, default in asdict(defaults).items()
        }
        return cls(**values)

    def hyperparams(self) -> dict:
        return {
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "momentum": self.momentum,
            "mu": self.mu,
            "seed": self.seed,
        }
