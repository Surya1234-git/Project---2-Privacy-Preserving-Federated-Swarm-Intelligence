"""Non-IID partitioning: how the training set is split across edge nodes.

Pure NumPy on purpose - no torch needed, easy to unit test.

Three strategies:
  * dirichlet - per class, node shares ~ Dirichlet(alpha). Small alpha => highly skewed
                (alpha 0.1 = extreme, 0.3 = strong, 1.0 = mild, 100 = nearly IID).
  * shards    - McMahan et al. "pathological" split: sort by label, cut into shards,
                give each node a few shards, so each node sees only a couple of classes.
  * iid       - uniform random split (control experiment).
"""
from __future__ import annotations

import numpy as np


def iid_partition(labels, num_nodes, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    return [np.sort(p).astype(np.int64) for p in np.array_split(idx, num_nodes)]


def dirichlet_partition(labels, num_nodes, alpha=0.3, seed=0, min_size=10, max_tries=200):
    labels = np.asarray(labels)
    n = len(labels)
    if n < num_nodes * min_size:
        raise ValueError(f"{n} samples cannot give {num_nodes} nodes >= {min_size} samples each")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)

    for _ in range(max_tries):
        parts = [[] for _ in range(num_nodes)]
        for c in classes:
            idx = np.flatnonzero(labels == c)
            rng.shuffle(idx)
            p = rng.dirichlet(np.full(num_nodes, alpha))
            # Keep node sizes roughly balanced: nodes already above average stop receiving.
            p = p * np.array([len(part) < n / num_nodes for part in parts])
            p = np.ones(num_nodes) / num_nodes if p.sum() == 0 else p / p.sum()
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for part, chunk in zip(parts, np.split(idx, cuts)):
                part.extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.array(p, dtype=np.int64)) for p in parts]
    raise RuntimeError("could not satisfy min_size; use a larger alpha or fewer nodes")


def shard_partition(labels, num_nodes, shards_per_node=2, seed=0):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    num_shards = num_nodes * shards_per_node
    if len(labels) < num_shards:
        raise ValueError("not enough samples for that many shards")
    order = np.argsort(labels, kind="stable")
    shards = np.array_split(order, num_shards)
    perm = rng.permutation(num_shards)
    parts = []
    for i in range(num_nodes):
        mine = perm[i * shards_per_node:(i + 1) * shards_per_node]
        parts.append(np.sort(np.concatenate([shards[s] for s in mine])).astype(np.int64))
    return parts


def make_partition(labels, strategy, num_nodes, alpha=0.3, shards_per_node=2, seed=0):
    if strategy == "iid":
        return iid_partition(labels, num_nodes, seed)
    if strategy == "dirichlet":
        return dirichlet_partition(labels, num_nodes, alpha, seed)
    if strategy == "shards":
        return shard_partition(labels, num_nodes, shards_per_node, seed)
    raise ValueError(f"unknown partition strategy: {strategy}")


def label_matrix(labels, parts, num_classes):
    """(num_nodes, num_classes) matrix of per-node class counts."""
    labels = np.asarray(labels)
    m = np.zeros((len(parts), num_classes), dtype=np.int64)
    for i, idx in enumerate(parts):
        m[i] = np.bincount(labels[idx], minlength=num_classes)
    return m


def heterogeneity_stats(matrix):
    """Numbers you can quote in your report to prove the data really is non-IID."""
    matrix = np.asarray(matrix, dtype=np.float64)
    sizes = matrix.sum(axis=1)
    global_dist = matrix.sum(axis=0) / matrix.sum()
    node_dist = matrix / np.maximum(sizes[:, None], 1)
    tv = 0.5 * np.abs(node_dist - global_dist).sum(axis=1)  # total-variation distance, 0..1
    return {
        "num_nodes": int(len(sizes)),
        "samples_min": int(sizes.min()),
        "samples_max": int(sizes.max()),
        "samples_mean": float(sizes.mean()),
        "samples_std": float(sizes.std()),
        "classes_per_node_mean": float((matrix > 0).sum(axis=1).mean()),
        "mean_tv_distance_from_global": float(tv.mean()),
    }
