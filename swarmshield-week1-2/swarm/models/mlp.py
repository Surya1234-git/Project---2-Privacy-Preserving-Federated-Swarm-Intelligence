"""A small numpy MLP with a *flat parameter vector* interface.

Why numpy and not PyTorch on the edge? The swarm runs hundreds of containers; an edge device
should not need a 1 GB ML framework just to train a tiny network. The aggregator and the
cryptography only ever see a flat float64 vector, so a PyTorch model can be swapped in later by
implementing the same four members: ``n_params``, ``get_flat``, ``set_flat``, ``loss_and_grad``.
"""

from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, dims: list[int] | tuple[int, ...], seed: int = 0):
        if len(dims) < 2:
            raise ValueError("dims needs at least input and output size")
        self.dims = tuple(int(d) for d in dims)
        self._shapes: list[tuple[int, int]] = list(zip(self.dims[:-1], self.dims[1:], strict=True))
        self.n_params = sum(i * o + o for i, o in self._shapes)
        self.theta = np.zeros(self.n_params, dtype=np.float64)
        rng = np.random.default_rng(seed)
        for (w, b), (i, o) in zip(self._views(), self._shapes, strict=True):
            w[...] = rng.normal(0.0, np.sqrt(2.0 / i), size=(i, o))  # He init
            b[...] = 0.0

    # ---- flat parameter interface -------------------------------------------------------
    def get_flat(self) -> np.ndarray:
        return self.theta.copy()

    def set_flat(self, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float64).ravel()
        if vec.shape[0] != self.n_params:
            raise ValueError(f"expected {self.n_params} params, got {vec.shape[0]}")
        self.theta[:] = vec

    def _views(self, flat: np.ndarray | None = None):
        flat = self.theta if flat is None else flat
        out, pos = [], 0
        for i, o in self._shapes:
            w = flat[pos:pos + i * o].reshape(i, o)
            pos += i * o
            b = flat[pos:pos + o]
            pos += o
            out.append((w, b))
        return out

    # ---- forward / backward -------------------------------------------------------------
    def _forward(self, x: np.ndarray):
        acts, pre = [x], []
        h = x
        layers = self._views()
        for li, (w, b) in enumerate(layers):
            z = h @ w + b
            pre.append(z)
            h = np.maximum(z, 0.0) if li < len(layers) - 1 else z
            acts.append(h)
        return acts, pre

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[0][-1]

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def loss_and_grad(self, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        n = x.shape[0]
        acts, pre = self._forward(x)
        p = self._softmax(acts[-1])
        loss = float(-np.mean(np.log(p[np.arange(n), y] + 1e-12)))
        dz = p.copy()
        dz[np.arange(n), y] -= 1.0
        dz /= n
        grad = np.zeros_like(self.theta)
        gviews = self._views(grad)
        layers = self._views()
        for li in range(len(layers) - 1, -1, -1):
            gw, gb = gviews[li]
            gw[...] = acts[li].T @ dz
            gb[...] = dz.sum(axis=0)
            if li > 0:
                dz = (dz @ layers[li][0].T) * (pre[li - 1] > 0)
        return loss, grad

    # ---- evaluation ---------------------------------------------------------------------
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        if len(y) == 0:
            return float("nan"), float("nan")
        logits = self.logits(x)
        p = self._softmax(logits)
        loss = float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))
        acc = float(np.mean(logits.argmax(axis=1) == y))
        return loss, acc


def local_sgd(model: MLP, x: np.ndarray, y: np.ndarray, *, epochs: int, batch_size: int,
              lr: float, rng: np.random.Generator, prox_mu: float = 0.0,
              anchor: np.ndarray | None = None) -> float:
    """Mini-batch SGD on the local shard. ``prox_mu`` > 0 turns FedAvg into FedProx.

    The proximal term mu/2 * ||w - w_global||^2 limits client drift on non-IID data.
    Returns the mean training loss of the last epoch.
    """
    n = x.shape[0]
    if n == 0:
        return float("nan")
    if prox_mu > 0 and anchor is None:
        anchor = model.get_flat()
    last = float("nan")
    for _ in range(epochs):
        order = rng.permutation(n)
        losses = []
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            loss, grad = model.loss_and_grad(x[idx], y[idx])
            if prox_mu > 0:
                grad = grad + prox_mu * (model.theta - anchor)
            model.theta -= lr * grad
            losses.append(loss * len(idx))
        last = float(np.sum(losses) / n)
    return last
