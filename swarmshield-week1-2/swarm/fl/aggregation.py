"""FedAvg in two flavours that must agree: plaintext baseline and encrypted aggregation."""

from __future__ import annotations

import numpy as np
import tenseal as ts

from swarm.crypto import ckks


def fedavg_weights(n_samples: list[int]) -> list[float]:
    total = float(sum(n_samples))
    if total <= 0:
        raise ValueError("total sample count must be positive")
    w = [n / total for n in n_samples]
    w[-1] = 1.0 - sum(w[:-1])  # make the sum exactly 1.0 despite float rounding
    return w


class PlainAggregator:
    """Week-1 baseline: the server sees every client's raw weights."""

    def aggregate(self, weights: list[np.ndarray], n_samples: list[int]) -> np.ndarray:
        coef = fedavg_weights(n_samples)
        out = np.zeros_like(weights[0], dtype=np.float64)
        for c, w in zip(coef, weights, strict=True):
            out += c * np.asarray(w, dtype=np.float64)
        return out


class EncryptedAggregator:
    """Week-2: weighted averaging directly on CKKS ciphertexts (public context only)."""

    def __init__(self, public_ctx: ts.Context):
        if public_ctx.has_secret_key():
            raise ckks.CryptoError("EncryptedAggregator must not hold the secret key")
        self.ctx = public_ctx

    def aggregate(self, uploads: list[list[bytes]], n_samples: list[int]) -> list[bytes]:
        return ckks.aggregate_weighted(self.ctx, uploads, fedavg_weights(n_samples))
