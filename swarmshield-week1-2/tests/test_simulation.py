import numpy as np

from swarm.config import ExperimentConfig
from swarm.fl.selection import dropout_draw, select_participants
from swarm.sim.engine import run_simulation


def test_selection_is_deterministic_and_sized():
    ids = [f"node-{i:04d}" for i in range(40)]
    a = select_participants(ids, 0.25, 3, 7, 2, 0)
    assert a == select_participants(list(reversed(ids)), 0.25, 3, 7, 2, 0)
    assert len(a) == 10 and len(set(a)) == 10
    assert a != select_participants(ids, 0.25, 3, 7, 3, 0)
    assert a != select_participants(ids, 0.25, 3, 7, 2, 1)


def test_dropout_rate_matches_probability():
    hits = sum(dropout_draw(c, r, 0, 0.2) for c in range(200) for r in range(50))
    assert abs(hits / 10000 - 0.2) < 0.02


def test_federated_training_learns_and_plain_equals_fhe(shards):
    res = {}
    for mode in ("plain", "fhe"):
        cfg = ExperimentConfig(mode=mode, rounds=5, participation=0.6, local_epochs=2)
        res[mode] = run_simulation(shards, cfg, keep_weights=True)
    assert res["plain"].history[-1]["test_acc"] > res["plain"].history[0]["test_acc"] + 0.15
    drift = max(np.max(np.abs(a - b)) for a, b in
                zip(res["plain"].global_weights, res["fhe"].global_weights, strict=True))
    assert drift < 1e-4
    assert abs(res["plain"].final_acc - res["fhe"].final_acc) < 0.01
    assert res["fhe"].history[1]["upload_bytes_per_node"] > 10 * res["plain"].history[1][
        "upload_bytes_per_node"]
