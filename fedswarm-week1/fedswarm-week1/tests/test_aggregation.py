import numpy as np

from fedswarm.aggregation import fedavg, select_nodes
from fedswarm.serialization import bytes_to_weights, weights_to_bytes


def test_fedavg_is_sample_weighted_mean():
    a = {"w": np.array([[1.0, 1.0]], dtype=np.float32), "b": np.array([0.0], dtype=np.float32)}
    b = {"w": np.array([[4.0, 7.0]], dtype=np.float32), "b": np.array([3.0], dtype=np.float32)}
    out = fedavg([(a, 100), (b, 300)])          # weights 0.25 / 0.75
    assert np.allclose(out["w"], [[3.25, 5.5]])
    assert np.allclose(out["b"], [2.25])
    assert out["w"].dtype == np.float32


def test_fedavg_single_update_is_identity():
    a = {"w": np.arange(6, dtype=np.float32).reshape(2, 3)}
    assert np.allclose(fedavg([(a, 5)])["w"], a["w"])


def test_fedavg_rejects_bad_input():
    a = {"w": np.zeros(2, dtype=np.float32)}
    for bad in ([], [(a, 0)], [(a, 1), ({"x": np.zeros(2, dtype=np.float32)}, 1)],
                [(a, 1), ({"w": np.zeros(3, dtype=np.float32)}, 1)]):
        try:
            fedavg(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_serialization_roundtrip():
    w = {"features.0.weight": np.random.rand(4, 1, 3, 3).astype(np.float32),
         "classifier.1.bias": np.random.rand(7).astype(np.float32)}
    back = bytes_to_weights(weights_to_bytes(w))
    assert set(back) == set(w)
    assert all(np.array_equal(back[k], w[k]) for k in w)


def test_select_nodes_is_deterministic_and_sized():
    a = select_nodes(100, 0.1, round_idx=3, seed=42)
    assert a == select_nodes(100, 0.1, round_idx=3, seed=42)
    assert len(a) == 10 and len(set(a)) == 10
    assert a != select_nodes(100, 0.1, round_idx=4, seed=42)
    assert len(select_nodes(5, 0.01, 1, 0)) == 1   # always at least one node
