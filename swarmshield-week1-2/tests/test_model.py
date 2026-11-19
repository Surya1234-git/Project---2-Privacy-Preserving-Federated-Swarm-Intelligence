import numpy as np

from swarm.models.mlp import MLP, local_sgd


def test_gradient_matches_finite_differences():
    m = MLP([5, 7, 4, 3], seed=1)
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(11, 5)), rng.integers(0, 3, size=11)
    _, g = m.loss_and_grad(x, y)
    num = np.zeros_like(g)
    for i in range(m.n_params):
        t = m.theta[i]
        m.theta[i] = t + 1e-6
        lp, _ = m.loss_and_grad(x, y)
        m.theta[i] = t - 1e-6
        lm, _ = m.loss_and_grad(x, y)
        m.theta[i] = t
        num[i] = (lp - lm) / 2e-6
    assert np.max(np.abs(num - g)) < 1e-6


def test_flat_roundtrip_and_seeded_init():
    a, b = MLP([4, 6, 3], seed=9), MLP([4, 6, 3], seed=9)
    assert np.array_equal(a.get_flat(), b.get_flat())
    v = np.arange(a.n_params, dtype=float)
    a.set_flat(v)
    assert np.array_equal(a.get_flat(), v)


def test_local_sgd_reduces_loss_and_prox_pulls_toward_anchor():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6))
    y = (x[:, 0] > 0).astype(int)
    m = MLP([6, 8, 2], seed=0)
    before = m.evaluate(x, y)[0]
    anchor = m.get_flat()
    local_sgd(m, x, y, epochs=5, batch_size=32, lr=0.1, rng=np.random.default_rng(1))
    assert m.evaluate(x, y)[0] < before
    free = np.linalg.norm(m.get_flat() - anchor)
    m2 = MLP([6, 8, 2], seed=0)
    local_sgd(m2, x, y, epochs=5, batch_size=32, lr=0.1, rng=np.random.default_rng(1),
              prox_mu=5.0, anchor=anchor)
    assert np.linalg.norm(m2.get_flat() - anchor) < free
