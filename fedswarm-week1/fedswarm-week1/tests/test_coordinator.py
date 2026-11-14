"""Protocol tests for the federated server - no torch needed."""
import threading
import time

import numpy as np

from fedswarm.config import FLConfig
from fedswarm.coordinator import Coordinator, create_app
from fedswarm.serialization import bytes_to_weights, weights_to_bytes

SPEC = {"in_channels": 1, "num_classes": 10, "image_size": 28}


def _make(rounds=2, nodes=4, containers=2, timeout=10.0):
    cfg = FLConfig(num_nodes=nodes, num_containers=containers, rounds=rounds,
                   client_fraction=1.0, round_timeout=timeout, seed=1)
    init = {"w": np.zeros((2, 2), dtype=np.float32), "b": np.zeros(2, dtype=np.float32)}
    coord = Coordinator(cfg, init, lambda w: (0.1, 0.9), SPEC)
    app = create_app(coord)
    return coord, app.test_client()


def _post_update(c, rnd, nid, weights, n=10, loss="0.5"):
    return c.post("/update", data=weights_to_bytes(weights),
                  headers={"X-Round": str(rnd), "X-Node-Id": str(nid),
                           "X-Num-Samples": str(n), "X-Train-Loss": loss})


def _wait_for(pred, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_registration_is_idempotent_and_capped():
    coord, c = _make()
    r1 = c.post("/register", json={"hostname": "a"}).get_json()
    r2 = c.post("/register", json={"hostname": "a"}).get_json()
    assert r1["container_idx"] == r2["container_idx"] == 0
    assert r1["node_ids"] == [0, 2]
    assert c.post("/register", json={"hostname": "b"}).get_json()["node_ids"] == [1, 3]
    assert c.post("/register", json={"hostname": "c"}).status_code == 409
    assert c.post("/register", json={}).status_code == 400


def test_full_two_round_run_averages_correctly():
    coord, c = _make(rounds=2)
    threading.Thread(target=coord.run, daemon=True).start()
    c.post("/register", json={"hostname": "a"})
    c.post("/register", json={"hostname": "b"})

    deadline = time.time() + 20
    while coord.state != "done" and time.time() < deadline:
        for idx in (0, 1):
            task = c.get(f"/task?container_idx={idx}").get_json()
            if task["status"] != "train":
                continue
            resp = c.get("/global_model")
            rnd = int(resp.headers["X-Round"])
            gw = bytes_to_weights(resp.data)
            for nid in task["node_ids"]:
                r = _post_update(c, rnd, nid, {k: v + 1.0 for k, v in gw.items()})
                assert r.status_code == 200, r.get_json()
        time.sleep(0.01)

    assert coord.state == "done"
    assert len(coord.history) == 2
    assert all(h["nodes_reported"] == 4 for h in coord.history)
    # every node adds +1 per round => global model is +1 per round
    assert np.allclose(coord.global_weights["w"], 2.0)
    assert coord.history[0]["mb_up"] > 0 and coord.history[0]["mb_down"] > 0
    assert c.get("/task?container_idx=0").get_json()["status"] == "done"


def _start_training(rounds=3):
    coord, c = _make(rounds=rounds, timeout=2.0)
    threading.Thread(target=coord.run, daemon=True).start()
    c.post("/register", json={"hostname": "a"})
    c.post("/register", json={"hostname": "b"})
    assert _wait_for(lambda: coord.state == "training")
    return coord, c


def test_bad_updates_are_rejected():
    coord, c = _start_training()
    good = {"w": np.ones((2, 2), dtype=np.float32), "b": np.ones(2, dtype=np.float32)}
    rnd = coord.round

    nan = {"w": np.full((2, 2), np.nan, dtype=np.float32), "b": np.ones(2, dtype=np.float32)}
    assert _post_update(c, rnd, 0, nan).status_code == 400            # non-finite
    wrong_shape = {"w": np.ones((3, 3), dtype=np.float32), "b": np.ones(2, dtype=np.float32)}
    assert _post_update(c, rnd, 0, wrong_shape).status_code == 400    # shape
    wrong_keys = {"x": np.ones(2, dtype=np.float32)}
    assert _post_update(c, rnd, 0, wrong_keys).status_code == 400     # keys
    assert c.post("/update", data=b"garbage",
                  headers={"X-Round": str(rnd), "X-Node-Id": "0", "X-Num-Samples": "5"}
                  ).status_code == 400                                  # malformed
    assert _post_update(c, rnd + 5, 0, good).status_code == 409       # wrong round
    assert _post_update(c, rnd, 99, good).status_code == 400          # not selected
    assert c.post("/update", data=b"x").status_code == 400            # missing headers

    assert _post_update(c, rnd, 0, good).status_code == 200
    assert _post_update(c, rnd, 0, good).status_code == 200           # duplicate is harmless
    assert len(coord.updates) == 1


def test_round_timeout_aggregates_partial_updates():
    coord, c = _start_training(rounds=1)          # 2 s timeout, only 1 of 4 nodes reports
    good = {"w": np.full((2, 2), 4.0, dtype=np.float32), "b": np.full(2, 4.0, dtype=np.float32)}
    assert _post_update(c, coord.round, 0, good).status_code == 200
    assert _wait_for(lambda: coord.state == "done", timeout=10)
    assert coord.history[0]["nodes_reported"] == 1
    assert np.allclose(coord.global_weights["w"], 4.0)
