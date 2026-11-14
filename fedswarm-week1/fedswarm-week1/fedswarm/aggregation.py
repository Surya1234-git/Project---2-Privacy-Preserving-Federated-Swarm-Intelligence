"""Server-side aggregation. NumPy only.

Week 2 will swap `fedavg` for an encrypted version (TenSEAL) - keep this the single
place where client updates are combined.
"""
from __future__ import annotations

import numpy as np


def select_nodes(num_nodes: int, fraction: float, round_idx: int, seed: int):
    """Deterministic client sampling: same (seed, round) => same nodes."""
    m = max(1, int(round(fraction * num_nodes)))
    rng = np.random.default_rng(seed * 100003 + round_idx)
    return sorted(int(i) for i in rng.choice(num_nodes, size=m, replace=False))


def fedavg(updates):
    """Federated Averaging (McMahan et al., 2017).

    updates: list of (weights_dict, num_samples). Returns
        w_global = sum_k (n_k / n) * w_k
    """
    if not updates:
        raise ValueError("no updates to aggregate")
    total = float(sum(n for _, n in updates))
    if total <= 0:
        raise ValueError("total sample count must be positive")
    keys = list(updates[0][0].keys())
    for w, _ in updates:
        if list(w.keys()) != keys:
            raise ValueError("updates have mismatching parameter names")
    out = {}
    for k in keys:
        ref = updates[0][0][k]
        acc = np.zeros(ref.shape, dtype=np.float64)
        for w, n in updates:
            if w[k].shape != ref.shape:
                raise ValueError(f"shape mismatch for {k}")
            acc += w[k].astype(np.float64) * (n / total)
        out[k] = acc.astype(ref.dtype)
    return out
