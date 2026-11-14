import numpy as np

from fedswarm.partition import (dirichlet_partition, heterogeneity_stats, iid_partition,
                                label_matrix, shard_partition)

LABELS = np.random.default_rng(0).integers(0, 10, 20000)


def _covers_exactly_once(parts, n):
    all_idx = np.concatenate(parts)
    return len(all_idx) == n and len(np.unique(all_idx)) == n


def test_dirichlet_covers_every_sample_exactly_once():
    parts = dirichlet_partition(LABELS, 50, alpha=0.3, seed=1)
    assert len(parts) == 50
    assert _covers_exactly_once(parts, len(LABELS))


def test_dirichlet_respects_min_size():
    parts = dirichlet_partition(LABELS, 100, alpha=0.1, seed=2, min_size=10)
    assert min(len(p) for p in parts) >= 10


def test_lower_alpha_is_more_non_iid():
    def tv(alpha):
        parts = dirichlet_partition(LABELS, 40, alpha=alpha, seed=3)
        return heterogeneity_stats(label_matrix(LABELS, parts, 10))["mean_tv_distance_from_global"]
    assert tv(0.1) > tv(1.0) > tv(100.0)


def test_iid_is_close_to_global_distribution():
    parts = iid_partition(LABELS, 40, seed=4)
    assert _covers_exactly_once(parts, len(LABELS))
    stats = heterogeneity_stats(label_matrix(LABELS, parts, 10))
    assert stats["mean_tv_distance_from_global"] < 0.1


def test_shards_limit_classes_per_node():
    parts = shard_partition(LABELS, 50, shards_per_node=2, seed=5)
    assert _covers_exactly_once(parts, len(LABELS))
    m = label_matrix(LABELS, parts, 10)
    assert ((m > 0).sum(axis=1) <= 4).all()   # a shard can straddle at most 2 classes


def test_partition_is_deterministic():
    a = dirichlet_partition(LABELS, 20, 0.3, seed=7)
    b = dirichlet_partition(LABELS, 20, 0.3, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
