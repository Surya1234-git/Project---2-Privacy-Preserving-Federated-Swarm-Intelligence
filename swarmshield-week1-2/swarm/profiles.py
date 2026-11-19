"""Device-class profiles: what makes a *swarm* different from a uniform pool of clients.

Every simulated edge device belongs to a class. The class drives four things at once, so the
data layer, the Docker layer and the runtime behaviour stay consistent with each other:

  * capacity   - relative share of the training data the device holds (data-volume skew)
  * dropout_p  - probability that a *selected* device silently disappears for a round
  * compute_x  - slowdown factor of local training versus the fastest device (stragglers)
  * bw_kbps    - uplink bandwidth used to emulate upload time of the (large) ciphertexts
  * cpus/mem   - hard container limits generated into docker-compose.yml
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    capacity: float
    dropout_p: float
    compute_x: float
    bw_kbps: int
    cpus: float
    mem_mb: int


DEVICE_PROFILES: dict[str, DeviceProfile] = {
    "phone": DeviceProfile("phone", capacity=1.0, dropout_p=0.10, compute_x=1.0,
                           bw_kbps=8_000, cpus=0.50, mem_mb=160),
    "iot": DeviceProfile("iot", capacity=0.35, dropout_p=0.20, compute_x=2.5,
                         bw_kbps=500, cpus=0.25, mem_mb=128),
    "gateway": DeviceProfile("gateway", capacity=3.0, dropout_p=0.03, compute_x=0.6,
                             bw_kbps=50_000, cpus=1.00, mem_mb=192),
}

DEFAULT_MIX = "phone=0.55,iot=0.40,gateway=0.05"


def parse_mix(text: str) -> dict[str, float]:
    """Parse 'phone=0.5,iot=0.4,gateway=0.1' into a normalised dict."""
    mix: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, val = part.partition("=")
        name = name.strip()
        if name not in DEVICE_PROFILES:
            raise ValueError(f"unknown device class {name!r}; known: {sorted(DEVICE_PROFILES)}")
        mix[name] = float(val)
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("device mix must have positive total weight")
    return {k: v / total for k, v in mix.items()}


def assign_device_classes(n_nodes: int, mix: dict[str, float], seed: int) -> list[str]:
    """Deterministically assign a class to each node (largest-remainder rounding, then shuffle)."""
    names = list(mix)
    raw = np.array([mix[k] * n_nodes for k in names])
    counts = np.floor(raw).astype(int)
    remainder = n_nodes - counts.sum()
    order = np.argsort(-(raw - counts))
    for i in range(remainder):
        counts[order[i % len(names)]] += 1
    classes = [n for n, c in zip(names, counts, strict=True) for _ in range(c)]
    rng = np.random.default_rng(seed)
    rng.shuffle(classes)
    return classes
