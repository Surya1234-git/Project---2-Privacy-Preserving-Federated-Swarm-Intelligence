"""Local training on one node, and evaluation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def to_tensor(x_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(x_uint8)).float()
    return (x / 255.0 - 0.5) / 0.5


def local_train(model, x, y, epochs, batch_size, lr, momentum=0.9, mu=0.0,
                device="cpu", seed=0) -> float:
    """Train `model` in place on one node's private data. Returns mean training loss.

    mu > 0 adds the FedProx proximal term (mu/2)*||w - w_global||^2, which stabilises
    training on non-IID data. mu = 0 is plain FedAvg.
    """
    device = torch.device(device)
    model.to(device).train()
    gen = torch.Generator().manual_seed(int(seed))
    X = to_tensor(x).to(device)
    Y = torch.as_tensor(y, dtype=torch.long).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    anchor = [p.detach().clone() for p in model.parameters()] if mu > 0 else None

    n = X.shape[0]
    total_loss, seen = 0.0, 0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size].to(device)
            xb, yb = X[idx], Y[idx]
            opt.zero_grad()
            ce = F.cross_entropy(model(xb), yb)
            loss = ce
            if anchor is not None:
                prox = sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
                loss = ce + 0.5 * mu * prox
            loss.backward()
            opt.step()
            total_loss += ce.item() * len(idx)
            seen += len(idx)
    return total_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model, x, y, device="cpu", batch_size=1000):
    """Returns (loss, accuracy) on the given arrays."""
    device = torch.device(device)
    model.to(device).eval()
    loss_sum, correct, n = 0.0, 0, len(y)
    for i in range(0, n, batch_size):
        xb = to_tensor(x[i:i + batch_size]).to(device)
        yb = torch.as_tensor(y[i:i + batch_size], dtype=torch.long).to(device)
        out = model(xb)
        loss_sum += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return loss_sum / n, correct / n
