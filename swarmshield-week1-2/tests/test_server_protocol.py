"""Protocol + adversarial tests against the real FastAPI aggregator (in-process)."""

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from swarm import security, wire
from swarm.config import ExperimentConfig
from swarm.crypto import ckks
from swarm.server.app import create_app

SECRET = "test-secret"


def _cfg(mode):
    return ExperimentConfig(mode=mode, rounds=2, participation=1.0, min_selected=3, quorum=1.0,
                            input_dim=4, num_classes=3, hidden=(5,), expected_nodes=3,
                            round_timeout_s=20, straggler_grace_s=0.5)


@pytest.fixture()
def fhe_env(tmp_path):
    sk = ckks.new_secret_context()
    (tmp_path / "public.ctx").write_bytes(ckks.export_public(sk))
    cfg = _cfg("fhe")
    app = create_app(cfg, public_ctx_path=str(tmp_path / "public.ctx"), out_dir=str(tmp_path / "o"),
                     enroll_secret=SECRET)
    with TestClient(app) as c:
        yield c, cfg, sk, app.state.coordinator


def _enroll(c, nid):
    r = c.post("/register", json={"node_id": nid, "device_class": "phone", "n_train": 10,
                                  "n_val": 5, "enroll_secret": SECRET})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['token']}", "X-Node-Id": nid}, r.json()["token"]


def _wait_phase(c, h, phase, t=10):
    end = time.time() + t
    while time.time() < end:
        s = c.get("/status", headers=h).json()
        if s["phase"] == phase:
            return s
        time.sleep(0.1)
    raise AssertionError(f"never reached {phase}: {s}")


def _upload(c, h, tok, nid, rnd, att, chunks, n=10, sign_body=None, ev=None):
    meta = {"n_samples": n, "train_loss": 1.0}
    if ev:
        meta["eval_prev_global"] = ev
    body = wire.pack([json.dumps(meta).encode(), *chunks])
    sig = security.sign_update(tok, nid, rnd, att, sign_body or body)
    return c.post("/update", content=body, headers={**h, "X-Round": str(rnd), "X-Attempt": str(att),
                                                    "X-Signature": sig})


def test_bad_enrollment_and_unauthenticated_access(fhe_env):
    c, *_ = fhe_env
    assert c.post("/register", json={"node_id": "x", "n_train": 1, "n_val": 1,
                                     "enroll_secret": "wrong"}).status_code == 401
    assert c.get("/status").status_code == 401
    assert c.get("/status", headers={"X-Node-Id": "x", "Authorization": "Bearer nope"}
                 ).status_code == 401


def test_aggregator_refuses_to_start_with_secret_key(tmp_path):
    sk = ckks.new_secret_context()
    (tmp_path / "secret.ctx").write_bytes(ckks.export_secret(sk))
    with pytest.raises(ckks.CryptoError):
        create_app(_cfg("fhe"), public_ctx_path=str(tmp_path / "secret.ctx"),
                   out_dir=str(tmp_path / "o"), enroll_secret=SECRET)


def test_full_encrypted_round_and_adversarial_uploads(fhe_env):
    c, cfg, sk, coord = fhe_env
    n_params = coord.n_params
    codec = ckks.CKKSCodec(sk, n_params)
    nodes = {n: _enroll(c, n) for n in ("node-1", "node-2", "node-3")}
    h1, t1 = nodes["node-1"]
    s = _wait_phase(c, h1, "collecting")
    assert s["round"] == 1 and s["selected"]
    assert c.get("/global", headers=h1).status_code == 204  # version 0: seeded init, nothing sent

    rng = np.random.default_rng(0)
    ws = {n: rng.normal(size=n_params) for n in nodes}
    cts = {n: codec.encrypt(w) for n, w in ws.items()}

    # --- attacks that must all be rejected -------------------------------------------------
    assert _upload(c, h1, t1, "node-1", 1, 0, [b"junk"]).status_code == 422          # not a ciphertext
    assert _upload(c, h1, t1, "node-1", 1, 0, cts["node-1"], sign_body=b"other").status_code == 401
    assert _upload(c, h1, t1, "node-1", 2, 0, cts["node-1"]).status_code == 409      # wrong round
    plain = [np.ones(n_params).tobytes()]
    assert _upload(c, h1, t1, "node-1", 1, 0, plain).status_code == 422              # plaintext refused
    # --- honest uploads ---------------------------------------------------------------------
    for n, (h, t) in nodes.items():
        assert _upload(c, h, t, n, 1, 0, cts[n], n=10).status_code == 200
    assert _upload(c, h1, t1, "node-1", 1, 0, cts["node-1"]).status_code == 409      # duplicate

    s = _wait_phase(c, h1, "collecting")
    while s["round"] != 2:
        time.sleep(0.1)
        s = c.get("/status", headers=h1).json()
    r = c.get("/global", headers=h1)
    assert r.status_code == 200 and r.headers["x-model-version"] == "1"
    got = codec.decrypt(wire.unpack(r.content))
    expected = np.mean([ws[n] for n in nodes], axis=0)  # equal n_samples -> plain mean
    assert np.max(np.abs(got - expected)) < 1e-5
    # the aggregator object really never held plaintext weights or the secret key
    assert not coord.pub_ctx.has_secret_key()


def test_plain_mode_baseline_fedavg(tmp_path):
    cfg = _cfg("plain")
    app = create_app(cfg, public_ctx_path=None, out_dir=str(tmp_path), enroll_secret=SECRET)
    with TestClient(app) as c:
        nodes = {n: _enroll(c, n) for n in ("a", "b", "c")}
        ha, _ = nodes["a"]
        _wait_phase(c, ha, "collecting")
        n_params = app.state.coordinator.n_params
        vecs = {"a": np.full(n_params, 1.0), "b": np.full(n_params, 2.0), "c": np.full(n_params, 6.0)}
        ns = {"a": 10, "b": 10, "c": 20}
        for n, (h, t) in nodes.items():
            assert _upload(c, h, t, n, 1, 0, [vecs[n].tobytes()], n=ns[n]).status_code == 200
        s = c.get("/status", headers=ha).json()
        while s["round"] != 2:
            time.sleep(0.1)
            s = c.get("/status", headers=ha).json()
        got = np.frombuffer(wire.unpack(c.get("/global", headers=ha).content)[0])
        assert np.allclose(got, (1 * 10 + 2 * 10 + 6 * 20) / 40)
        assert _upload(c, ha, nodes["a"][1], "a", 2, 0, [np.full(n_params, np.nan).tobytes()]
                       ).status_code == 422
