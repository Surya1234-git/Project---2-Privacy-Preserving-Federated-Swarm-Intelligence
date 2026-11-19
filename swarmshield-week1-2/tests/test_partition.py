import numpy as np
import pytest

from swarm.data.partition import capacity_dirichlet_partition, heterogeneity_report
from swarm.profiles import assign_device_classes, parse_mix


def _labels(n=6000, k=10, seed=0):
    return np.random.default_rng(seed).integers(0, k, n)


def test_partition_is_a_true_partition_with_min_size():
    y = _labels()
    caps = np.array([1.0] * 30 + [0.35] * 20)
    parts = capacity_dirichlet_partition(y, caps, 0.2, np.random.default_rng(1), min_size=30)
    allidx = np.concatenate(parts)
    assert len(allidx) == len(y) and len(set(allidx.tolist())) == len(y)
    assert min(len(p) for p in parts) >= 30


def test_lower_alpha_is_more_non_iid():
    y = _labels()
    caps = np.ones(40)
    s = {a: heterogeneity_report(y, capacity_dirichlet_partition(
        y, caps, a, np.random.default_rng(2), 20), 10)["noniid_score"] for a in (0.05, 0.5, 50)}
    assert s[0.05] > s[0.5] > s[50]
    assert s[50] < 0.05


def test_capacity_drives_data_volume():
    y = _labels(20000)
    caps = np.array([3.0] * 10 + [0.35] * 10)
    parts = capacity_dirichlet_partition(y, caps, 5.0, np.random.default_rng(3), 30)
    big = np.mean([len(p) for p in parts[:10]])
    small = np.mean([len(p) for p in parts[10:]])
    assert big > 4 * small


def test_infeasible_request_fails_loudly():
    with pytest.raises(ValueError):
        capacity_dirichlet_partition(_labels(100), np.ones(50), 0.3, np.random.default_rng(0), 30)


def test_device_class_assignment_is_deterministic_and_proportional():
    mix = parse_mix("phone=0.5,iot=0.3,gateway=0.2")
    a = assign_device_classes(100, mix, 7)
    assert a == assign_device_classes(100, mix, 7)
    assert a.count("phone") == 50 and a.count("iot") == 30 and a.count("gateway") == 20
    with pytest.raises(ValueError):
        parse_mix("toaster=1")


def test_prepare_writes_isolated_shards(shards):
    import json

    m = json.loads((shards / "manifest.json").read_text())
    assert m["n_nodes"] == 12 and len(m["partition_sha256"]) == 64
    total = 0
    for n in m["nodes"]:
        d = np.load(shards / f"{n['node_id']}.npz")
        assert d["x_train"].shape[0] == n["n_train"]
        total += n["n_train"] + n["n_val"]
    assert total == 4000 - int(4000 * 0.15)
