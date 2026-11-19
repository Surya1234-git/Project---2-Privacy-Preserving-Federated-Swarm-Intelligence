"""Single source of truth for an experiment. The aggregator owns it and ships it to nodes."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass
class ExperimentConfig:
    mode: str = "fhe"                 # "plain" (Week-1 FedAvg baseline) | "fhe" (Week-2 CKKS)
    rounds: int = 15
    participation: float = 0.3        # fraction of registered nodes sampled per round
    min_selected: int = 3
    quorum: float = 0.6               # fraction of selected nodes needed to close a round
    round_timeout_s: float = 90.0
    straggler_grace_s: float = 3.0    # extra wait after quorum is reached
    max_attempts: int = 5
    # model / training
    input_dim: int = 32
    num_classes: int = 10
    hidden: tuple[int, ...] = (64, 32)
    local_epochs: int = 2
    batch_size: int = 32
    lr: float = 0.05
    prox_mu: float = 0.0              # >0 => FedProx term
    model_seed: int = 1234
    selection_seed: int = 7
    # CKKS
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: tuple[int, ...] = (60, 40, 40, 60)
    scale_bits: int = 40
    # orchestration
    expected_nodes: int = 0           # start as soon as this many registered (0 = use grace)
    min_nodes: int = 3
    start_grace_s: float = 5.0

    @property
    def dims(self) -> list[int]:
        return [self.input_dim, *self.hidden, self.num_classes]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        d["coeff_mod_bit_sizes"] = list(self.coeff_mod_bit_sizes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        d = dict(d)
        d["hidden"] = tuple(d.get("hidden", (64, 32)))
        d["coeff_mod_bit_sizes"] = tuple(d.get("coeff_mod_bit_sizes", (60, 40, 40, 60)))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_env(cls, prefix: str = "SWARM_") -> ExperimentConfig:
        base = cls()
        kw: dict = {}
        for name in cls.__dataclass_fields__:
            raw = os.environ.get(prefix + name.upper())
            if raw is None or raw == "":
                continue
            cur = getattr(base, name)
            if isinstance(cur, tuple):
                kw[name] = tuple(int(p) for p in raw.split(","))
            elif isinstance(cur, bool):
                kw[name] = raw.lower() in ("1", "true", "yes")
            elif isinstance(cur, int):
                kw[name] = int(raw)
            elif isinstance(cur, float):
                kw[name] = float(raw)
            else:
                kw[name] = raw
        return cls(**kw)

