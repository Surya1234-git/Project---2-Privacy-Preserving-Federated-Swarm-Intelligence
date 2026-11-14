"""The federated server: round state machine + HTTP API.

Torch-free on purpose (evaluation is injected as a callback), so the whole protocol is
unit-testable without a GPU / torch install.

Protocol (clients poll, server never connects out - works through NAT / Docker networks):
    POST /register       {hostname}              -> container_idx, node_ids, hyperparams, model_spec
    GET  /task?container_idx=i                   -> wait | train(round, node_ids) | done
    GET  /global_model                           -> npz bytes (+ X-Round header)
    POST /update  headers X-Round,X-Node-Id,X-Num-Samples,X-Train-Loss; body = npz weights
    GET  /status, GET /health
Raw training data never reaches this process: only model weights do.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
from flask import Flask, Response, jsonify, request

from .aggregation import fedavg, select_nodes
from .config import FLConfig
from .serialization import bytes_to_weights, weights_to_bytes

log = logging.getLogger("fedswarm.coordinator")


class Coordinator:
    def __init__(self, cfg: FLConfig, init_weights: dict, evaluate_fn, model_spec: dict,
                 on_round=None):
        self.cfg = cfg
        self.model_spec = dict(model_spec)
        self.evaluate_fn = evaluate_fn        # weights -> (loss, accuracy)
        self.on_round = on_round              # callback(row, history) after every round
        self.global_weights = init_weights
        self._global_bytes = weights_to_bytes(init_weights)

        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.hosts: dict[str, int] = {}
        self.state = "waiting_for_clients"    # -> training <-> aggregating -> done | error
        self.round = 0
        self.selected: list[int] = []
        self.updates: dict[int, tuple] = {}   # node_id -> (weights, n_samples, train_loss)
        self.history: list[dict] = []
        self.bytes_up = 0
        self.bytes_down = 0

    # ------------------------------------------------------------------ registration
    def register(self, hostname: str) -> int:
        with self.cv:
            if hostname in self.hosts:
                return self.hosts[hostname]
            if len(self.hosts) >= self.cfg.num_containers:
                raise RuntimeError("all container slots are already taken")
            idx = len(self.hosts)
            self.hosts[hostname] = idx
            log.info("registered %s as container %d (%d/%d)", hostname, idx,
                     len(self.hosts), self.cfg.num_containers)
            self.cv.notify_all()
            return idx

    def node_ids_for(self, idx: int) -> list[int]:
        return [n for n in range(self.cfg.num_nodes) if n % self.cfg.num_containers == idx]

    # ------------------------------------------------------------------ client-facing calls
    def get_task(self, idx: int) -> dict:
        with self.lock:
            if self.state in ("done", "error"):
                return {"status": "done"}
            if self.state != "training":
                return {"status": "wait"}
            mine = [n for n in self.selected
                    if n % self.cfg.num_containers == idx and n not in self.updates]
            if not mine:
                return {"status": "wait"}
            return {"status": "train", "round": self.round, "node_ids": mine}

    def get_global_model(self):
        with self.lock:
            if self.state == "training":
                self.bytes_down += len(self._global_bytes)
            return self.round, self._global_bytes

    def _validate(self, w: dict) -> str | None:
        ref = self.global_weights
        if set(w) != set(ref):
            return "parameter names do not match the global model"
        for k, v in w.items():
            if v.shape != ref[k].shape:
                return f"shape mismatch for {k}"
            if not np.all(np.isfinite(v)):
                return f"non-finite values in {k}"
        return None

    def submit_update(self, round_idx, node_id, n_samples, train_loss, body: bytes):
        """Returns (http_status, message)."""
        with self.lock:
            if self.state != "training" or round_idx != self.round:
                return 409, "stale or inactive round"
            if node_id not in self.selected:
                return 400, "node was not selected this round"
            if node_id in self.updates:
                return 200, "duplicate ignored"
        if n_samples <= 0:
            return 400, "num_samples must be positive"
        try:
            w = bytes_to_weights(body)
        except Exception:
            return 400, "malformed weights payload"
        err = self._validate(w)
        if err:
            return 400, err
        with self.cv:
            if self.state != "training" or round_idx != self.round:
                return 409, "stale or inactive round"
            self.updates[node_id] = (w, int(n_samples), float(train_loss))
            self.bytes_up += len(body)
            self.cv.notify_all()
        return 200, "ok"

    def status(self) -> dict:
        with self.lock:
            return {
                "state": self.state,
                "round": self.round,
                "total_rounds": self.cfg.rounds,
                "registered_containers": len(self.hosts),
                "expected_containers": self.cfg.num_containers,
                "updates_received": len(self.updates),
                "nodes_selected": len(self.selected),
                "last": self.history[-1] if self.history else None,
            }

    # ------------------------------------------------------------------ main loop
    def run(self):
        try:
            with self.cv:
                self.cv.wait_for(lambda: len(self.hosts) >= self.cfg.num_containers)
            log.info("all %d containers registered - starting training", self.cfg.num_containers)
            for r in range(1, self.cfg.rounds + 1):
                self._run_round(r)
            with self.lock:
                self.state = "done"
            log.info("training finished")
        except Exception:
            log.exception("coordinator crashed")
            with self.lock:
                self.state = "error"

    def _run_round(self, r: int):
        t0 = time.time()
        selected = select_nodes(self.cfg.num_nodes, self.cfg.client_fraction, r, self.cfg.seed)
        with self.cv:
            self.round, self.selected, self.updates = r, selected, {}
            self.bytes_up = self.bytes_down = 0
            self.state = "training"
            self.cv.notify_all()
            self.cv.wait_for(lambda: len(self.updates) >= len(self.selected),
                             timeout=self.cfg.round_timeout)
            updates = dict(self.updates)
            up_mb, down_mb = self.bytes_up / 1e6, self.bytes_down / 1e6
            self.state = "aggregating"

        if not updates:
            log.warning("round %d: no updates arrived before the timeout - skipping", r)
            return

        merged = fedavg([(w, n) for (w, n, _) in updates.values()])
        with self.lock:
            self.global_weights = merged
            self._global_bytes = weights_to_bytes(merged)
        test_loss, test_acc = self.evaluate_fn(merged)

        finite = [(l, n) for (_, n, l) in updates.values() if np.isfinite(l)]
        train_loss = (sum(l * n for l, n in finite) / sum(n for _, n in finite)) if finite else float("nan")
        row = {
            "round": r, "test_loss": test_loss, "test_acc": test_acc, "train_loss": train_loss,
            "nodes_selected": len(selected), "nodes_reported": len(updates),
            "mb_up": up_mb, "mb_down": down_mb, "seconds": time.time() - t0,
        }
        self.history.append(row)
        log.info("round %d/%d | acc %.2f%% | test loss %.4f | train loss %.4f | %d/%d nodes | %.1fs",
                 r, self.cfg.rounds, test_acc * 100, test_loss, train_loss,
                 len(updates), len(selected), row["seconds"])
        if self.on_round:
            self.on_round(row, self.history)


def create_app(coord: Coordinator) -> Flask:
    app = Flask("fedswarm")
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024

    @app.get("/health")
    def health():
        return jsonify(ok=True)

    @app.post("/register")
    def register():
        data = request.get_json(silent=True) or {}
        hostname = data.get("hostname")
        if not hostname:
            return jsonify(error="hostname required"), 400
        try:
            idx = coord.register(str(hostname))
        except RuntimeError as e:
            return jsonify(error=str(e)), 409
        return jsonify(container_idx=idx, node_ids=coord.node_ids_for(idx),
                       hyperparams=coord.cfg.hyperparams(), model_spec=coord.model_spec)

    @app.get("/task")
    def task():
        try:
            idx = int(request.args["container_idx"])
        except (KeyError, ValueError):
            return jsonify(error="container_idx required"), 400
        return jsonify(coord.get_task(idx))

    @app.get("/global_model")
    def global_model():
        r, data = coord.get_global_model()
        return Response(data, mimetype="application/octet-stream", headers={"X-Round": str(r)})

    @app.post("/update")
    def update():
        try:
            round_idx = int(request.headers["X-Round"])
            node_id = int(request.headers["X-Node-Id"])
            n_samples = int(request.headers["X-Num-Samples"])
            loss = float(request.headers.get("X-Train-Loss", "nan"))
        except (KeyError, ValueError):
            return jsonify(error="missing or invalid X- headers"), 400
        code, msg = coord.submit_update(round_idx, node_id, n_samples, loss, request.get_data())
        return jsonify(message=msg), code

    @app.get("/status")
    def status():
        return jsonify(coord.status())

    return app
