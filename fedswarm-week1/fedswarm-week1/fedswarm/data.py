"""Dataset loading. Always returns uint8 NCHW images + int64 labels.

Supported names:
  synthetic            tiny offline dataset (tests / CI, no download)
  mnist, fmnist        torchvision
  cifar10              torchvision
  bloodmnist, pathmnist, dermamnist, pneumoniamnist, ...   MedMNIST medical images
                       (needs `pip install medmnist`) - the "healthcare" demo
"""
from __future__ import annotations

import os

import numpy as np

TORCHVISION_SETS = {
    "mnist": "MNIST",
    "fmnist": "FashionMNIST",
    "cifar10": "CIFAR10",
}


def _to_nchw(x) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3:                      # (N, H, W) grayscale
        x = x[:, None, :, :]
    elif x.ndim == 4 and x.shape[-1] in (1, 3):  # (N, H, W, C)
        x = x.transpose(0, 3, 1, 2)
    return np.ascontiguousarray(x.astype(np.uint8))


def make_synthetic(n_train=6000, n_test=1000, num_classes=10, size=28, seed=0):
    rng = np.random.default_rng(seed)
    protos = rng.normal(0, 1, (num_classes, 1, size, size))

    def make(n):
        y = rng.integers(0, num_classes, n)
        x = protos[y] + rng.normal(0, 1.0, (n, 1, size, size))
        return np.clip(x * 50 + 128, 0, 255).astype(np.uint8), y.astype(np.int64)

    x_tr, y_tr = make(n_train)
    x_te, y_te = make(n_test)
    return x_tr, y_tr, x_te, y_te


def load_dataset(name: str, root: str = "data/raw", seed: int = 0) -> dict:
    name = name.lower()
    if name == "synthetic":
        x_tr, y_tr, x_te, y_te = make_synthetic(seed=seed)
        num_classes = 10
    elif name in TORCHVISION_SETS:
        from torchvision import datasets

        cls = getattr(datasets, TORCHVISION_SETS[name])
        tr = cls(root, train=True, download=True)
        te = cls(root, train=False, download=True)
        x_tr, y_tr = _to_nchw(tr.data), np.asarray(tr.targets, dtype=np.int64)
        x_te, y_te = _to_nchw(te.data), np.asarray(te.targets, dtype=np.int64)
        num_classes = 10
    else:
        try:
            import medmnist
            from medmnist import INFO
        except ImportError as e:
            raise ImportError(
                f"'{name}' needs MedMNIST: pip install -r requirements-medical.txt"
            ) from e
        if name not in INFO:
            raise ValueError(
                f"unknown dataset '{name}'. Built-in: synthetic, {', '.join(TORCHVISION_SETS)}; "
                f"MedMNIST: {', '.join(sorted(INFO))}"
            )
        info = INFO[name]
        if "multi-label" in info["task"]:
            raise ValueError("multi-label MedMNIST sets are not supported")
        os.makedirs(root, exist_ok=True)
        cls = getattr(medmnist, info["python_class"])
        tr = cls(split="train", root=root, download=True)
        te = cls(split="test", root=root, download=True)
        x_tr, y_tr = _to_nchw(tr.imgs), tr.labels.reshape(-1).astype(np.int64)
        x_te, y_te = _to_nchw(te.imgs), te.labels.reshape(-1).astype(np.int64)
        num_classes = len(info["label"])

    meta = {
        "name": name,
        "num_classes": int(num_classes),
        "in_channels": int(x_tr.shape[1]),
        "image_size": int(x_tr.shape[-1]),
    }
    return {"x_train": x_tr, "y_train": y_tr, "x_test": x_te, "y_test": y_te, "meta": meta}


def model_spec(meta: dict) -> dict:
    return {
        "in_channels": meta["in_channels"],
        "num_classes": meta["num_classes"],
        "image_size": meta["image_size"],
    }
