"""Dataset loaders. Everything except ``mnist`` works fully offline."""

from __future__ import annotations

import numpy as np


def load_dataset(name: str, n_samples: int = 30_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if name == "synthetic":
        # 'IoT sensor' style tabular data: 32 features, 10 classes, 2 sub-clusters per class.
        from sklearn.datasets import make_classification

        x, y = make_classification(
            n_samples=n_samples, n_features=32, n_informative=24, n_redundant=4,
            n_classes=10, n_clusters_per_class=2, class_sep=2.0, flip_y=0.01,
            random_state=seed,
        )
        return x.astype(np.float64), y.astype(np.int64)
    if name == "digits":
        from sklearn.datasets import load_digits

        d = load_digits()
        return d.data.astype(np.float64), d.target.astype(np.int64)
    if name == "mnist":  # needs internet on first use (cached by scikit-learn afterwards)
        from sklearn.datasets import fetch_openml

        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        x = d.data.astype(np.float64) / 255.0
        y = d.target.astype(np.int64)
        if n_samples and n_samples < len(y):
            idx = np.random.default_rng(seed).choice(len(y), n_samples, replace=False)
            x, y = x[idx], y[idx]
        return x, y
    raise ValueError(f"unknown dataset {name!r} (synthetic | digits | mnist)")
