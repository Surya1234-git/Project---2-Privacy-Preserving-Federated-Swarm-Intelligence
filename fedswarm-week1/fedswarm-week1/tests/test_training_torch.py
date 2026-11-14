"""Needs PyTorch (skipped automatically if it is not installed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fedswarm.config import FLConfig                      # noqa: E402
from fedswarm.data import load_dataset, model_spec         # noqa: E402
from fedswarm.models import SmallCNN, get_weights, set_weights  # noqa: E402
from fedswarm.partition import make_partition              # noqa: E402
from fedswarm.simulation import run_fedavg_simulation      # noqa: E402
from fedswarm.training import evaluate, local_train        # noqa: E402


def test_model_output_shape():
    for ch, size in ((1, 28), (3, 32), (3, 28)):
        out = SmallCNN(ch, 7, size)(torch.zeros(4, ch, size, size))
        assert out.shape == (4, 7)


def test_get_set_weights_roundtrip():
    a, b = SmallCNN(), SmallCNN()
    set_weights(b, get_weights(a))
    for (_, p), (_, q) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(p, q)


def test_local_training_reduces_loss():
    data = load_dataset("synthetic", seed=0)
    x, y = data["x_train"][:600], data["y_train"][:600]
    torch.manual_seed(0)
    model = SmallCNN(**model_spec(data["meta"]))
    before, _ = evaluate(model, x, y)
    local_train(model, x, y, epochs=3, batch_size=32, lr=0.01, seed=0)
    after, _ = evaluate(model, x, y)
    assert after < before


def test_fedprox_term_runs():
    data = load_dataset("synthetic", seed=0)
    model = SmallCNN(**model_spec(data["meta"]))
    loss = local_train(model, data["x_train"][:128], data["y_train"][:128],
                       epochs=1, batch_size=32, lr=0.01, mu=0.1, seed=0)
    assert np.isfinite(loss)


def test_federated_simulation_learns_on_non_iid_data():
    torch.manual_seed(0)
    data = load_dataset("synthetic", seed=0)
    parts = make_partition(data["y_train"], "dirichlet", 10, alpha=0.5, seed=0)
    cfg = FLConfig(num_nodes=10, rounds=6, client_fraction=0.5, local_epochs=1, seed=0)
    hist = run_fedavg_simulation(data, parts, cfg, log=lambda *_: None)
    assert len(hist) == 6
    assert hist[-1]["test_acc"] > 0.3          # chance is 0.10
