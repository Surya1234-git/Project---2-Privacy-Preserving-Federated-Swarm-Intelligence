import pytest

from swarm.sim.local_network import run_local_network


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["plain", "fhe"])
def test_real_processes_train_over_http(shards, tmp_path, mode):
    s = run_local_network(shards, nodes=8, rounds=3, mode=mode, out=tmp_path / mode,
                          participation=0.75, timeout_s=240)
    assert s["final_global_acc"] is not None and s["final_global_acc"] > 0.2
    evals = [h["eval_acc"] for h in s["history"] if h.get("eval_acc") is not None]
    assert evals[-1] > evals[0]
    if mode == "fhe":
        # every upload was ciphertext-sized (>> a plain float64 vector of the model)
        assert s["bytes_in_total"] > 10 * s["model_params"] * 8
