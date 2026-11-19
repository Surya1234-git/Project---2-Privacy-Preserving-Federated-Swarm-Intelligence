"""Non-IID partitioning for a heterogeneous swarm.

The usual benchmark recipe (Dirichlet label skew) gives every client the same expected data
volume. Real swarms are not like that: a gateway holds far more data than a battery sensor.
``capacity_dirichlet_partition`` fuses both effects in one distribution:

    for every class c:  p_c ~ Dirichlet(alpha * K * w)       w_k = capacity_k / sum(capacity)

* E[share of node k] = w_k  -> data *volume* follows device capacity (quantity skew)
* alpha controls label skew; with equal capacities it reduces to the standard Dir(alpha).

``heterogeneity_report`` then quantifies how non-IID the result really is, so "highly non-IID"
is a measured number in the report, not a claim.
"""

from __future__ import annotations

import numpy as np


def capacity_dirichlet_partition(
    y: np.ndarray,
    capacities: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    min_size: int = 30,
    max_tries: int = 200,
) -> list[np.ndarray]:
    y = np.asarray(y)
    k = len(capacities)
    if len(y) < k * min_size:
        raise ValueError(f"{len(y)} samples cannot give {k} nodes at least {min_size} each")
    w = np.asarray(capacities, dtype=np.float64)
    w = w / w.sum()
    conc = np.maximum(alpha * k * w, 1e-2)
    classes = np.unique(y)

    best: list[list[int]] | None = None
    for _ in range(max_tries):
        parts: list[list[int]] = [[] for _ in range(k)]
        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(conc)
            if not np.all(np.isfinite(p)) or p.sum() <= 0:
                p = w
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for node, chunk in enumerate(np.split(idx, cuts)):
                parts[node].extend(chunk.tolist())
        best = parts
        if min(len(p) for p in parts) >= min_size:
            break

    assert best is not None
    # Guarantee: nobody is below min_size. Move samples from the largest node(s) to the needy.
    sizes = [len(p) for p in best]
    for node in np.argsort(sizes):
        while len(best[node]) < min_size:
            donor = int(np.argmax([len(p) for p in best]))
            if len(best[donor]) <= min_size:
                break
            best[node].append(best[donor].pop(int(rng.integers(len(best[donor])))))
    return [np.array(sorted(p), dtype=np.int64) for p in best]


def heterogeneity_report(y: np.ndarray, parts: list[np.ndarray], num_classes: int) -> dict:
    """Measured non-IID-ness of a partition.

    * mean_jsd / max_jsd   Jensen-Shannon divergence (base 2, in [0,1]) between each node's label
                           distribution and the global one. 0 = perfectly IID.
    * noniid_score         mean_jsd, i.e. 0 (IID) .. 1 (every node holds a disjoint label set)
    * effective_classes    exp(entropy) per node: how many classes a node 'really' sees
    * size_gini            Gini coefficient of node data sizes (quantity skew)
    """
    y = np.asarray(y)
    glob = np.bincount(y, minlength=num_classes).astype(np.float64)
    glob /= glob.sum()

    def kl(p, q):
        m = p > 0
        return float(np.sum(p[m] * np.log2(p[m] / q[m])))

    jsds, eff, sizes = [], [], []
    for part in parts:
        cnt = np.bincount(y[part], minlength=num_classes).astype(np.float64)
        sizes.append(len(part))
        p = cnt / max(cnt.sum(), 1.0)
        m = 0.5 * (p + glob)
        jsds.append(0.5 * kl(p, m) + 0.5 * kl(glob, m))
        nz = p[p > 0]
        eff.append(float(np.exp(-np.sum(nz * np.log(nz)))))
    s = np.sort(np.array(sizes, dtype=np.float64))
    n = len(s)
    gini = float((2 * np.sum((np.arange(1, n + 1)) * s) / (n * s.sum())) - (n + 1) / n)
    return {
        "n_nodes": n,
        "noniid_score": float(np.mean(jsds)),
        "mean_jsd": float(np.mean(jsds)),
        "max_jsd": float(np.max(jsds)),
        "mean_effective_classes": float(np.mean(eff)),
        "min_effective_classes": float(np.min(eff)),
        "size_gini": gini,
        "size_min": int(s.min()),
        "size_median": float(np.median(s)),
        "size_max": int(s.max()),
    }
