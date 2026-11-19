"""Edge node: one process per simulated device (one container in the Docker swarm).

Life cycle: wait for aggregator -> enrol -> poll status -> when selected: download global model,
decrypt (fhe), score it on local validation data, train locally, encrypt, sign, upload.
Raw data never leaves this process; in fhe mode neither do plaintext weights.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from swarm import security, wire
from swarm.config import ExperimentConfig
from swarm.crypto import ckks
from swarm.fl.selection import dropout_draw
from swarm.node.trainer import LocalTrainer, Shard
from swarm.profiles import DEVICE_PROFILES

log = logging.getLogger("swarm.node")


class EdgeNode:
    def __init__(self, node_id: str, shard_path: str, server_url: str, enroll_secret: str,
                 device_class: str = "phone", secret_ctx_path: str | None = None,
                 time_scale: float = 1.0, poll_s: float = 1.0, emulate: bool = True):
        self.node_id = node_id
        self.shard = Shard.load(shard_path)
        self.profile = DEVICE_PROFILES[device_class]
        self.url = server_url.rstrip("/")
        self.enroll_secret = enroll_secret
        self.secret_ctx_path = secret_ctx_path
        self.time_scale, self.poll_s, self.emulate = time_scale, poll_s, emulate
        self.http = httpx.Client(timeout=60.0)
        self.token = ""
        self.cfg: ExperimentConfig | None = None
        self.trainer: LocalTrainer | None = None
        self.codec: ckks.CKKSCodec | None = None
        self.done: set[tuple] = set()

    # ------------------------------------------------------------------ plumbing
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "X-Node-Id": self.node_id}

    def _wait_for_server(self, max_wait: float = 180.0) -> None:
        t0, delay = time.time(), 0.5
        while True:
            try:
                if self.http.get(f"{self.url}/healthz").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if time.time() - t0 > max_wait:
                raise SystemExit(f"{self.node_id}: aggregator unreachable at {self.url}")
            time.sleep(delay)
            delay = min(delay * 1.5, 3.0)

    def register(self) -> None:
        r = self.http.post(f"{self.url}/register", json={
            "node_id": self.node_id, "device_class": self.profile.name,
            "n_train": len(self.shard.y_train), "n_val": len(self.shard.y_val),
            "enroll_secret": self.enroll_secret})
        r.raise_for_status()
        data = r.json()
        self.token = data["token"]
        self.cfg = ExperimentConfig.from_dict(data["config"])
        cfg = self.cfg
        if self.shard.x_train.shape[1] != cfg.input_dim:
            raise SystemExit(f"shard has {self.shard.x_train.shape[1]} features but the experiment "
                             f"expects {cfg.input_dim} (set SWARM_INPUT_DIM / SWARM_NUM_CLASSES)")
        self.trainer = LocalTrainer(self.node_id, self.shard, cfg)
        if cfg.mode == "fhe":
            if not self.secret_ctx_path:
                raise SystemExit("fhe mode needs the secret context (SWARM_SECRET_CTX)")
            ctx = ckks.load_context(Path(self.secret_ctx_path).read_bytes(), require_secret=True)
            self.codec = ckks.CKKSCodec(ctx, self.trainer.model.n_params, cfg.poly_modulus_degree)

    def _sleep(self, seconds: float) -> None:
        if self.emulate and seconds > 0:
            time.sleep(seconds * self.time_scale)

    # ------------------------------------------------------------------ one round
    def _download_global(self, expect_version: int) -> bool:
        r = self.http.get(f"{self.url}/global", headers=self._headers())
        r.raise_for_status()
        version = int(r.headers.get("x-model-version", "0"))
        if version != expect_version:
            log.warning("%s: global version %d != expected %d, skipping", self.node_id, version,
                        expect_version)
            return False
        if r.status_code == 204:
            return True  # version 0: every node already holds the same seeded initial model
        self._sleep(len(r.content) * 8 / (self.profile.bw_kbps * 1000))
        chunks = wire.unpack(r.content)
        if self.cfg.mode == "fhe":
            flat = self.codec.decrypt(chunks)
        else:
            flat = np.frombuffer(chunks[0], dtype=np.float64)
        self.trainer.set_global(flat)
        return True

    def do_train_round(self, round_id: int, attempt: int) -> None:
        if not self._download_global(round_id - 1):
            return
        ev = self.trainer.evaluate_global()  # score the model we just received (version r-1)
        if dropout_draw(self.trainer.crc, round_id, attempt, self.profile.dropout_p) \
                and self.emulate:
            log.info("%s (%s) went offline in round %d", self.node_id, self.profile.name, round_id)
            return
        t0 = time.perf_counter()
        weights, loss = self.trainer.train(round_id)
        took = time.perf_counter() - t0
        self._sleep(max(0.0, (self.profile.compute_x - 1.0) * took))  # straggler emulation
        if self.cfg.mode == "fhe":
            chunks = self.codec.encrypt(weights)  # encrypted BEFORE leaving the device
        else:
            chunks = [weights.astype(np.float64).tobytes()]
        meta = {"n_samples": self.trainer.n_train, "train_loss": loss, "eval_prev_global": ev}
        body = wire.pack([json.dumps(meta).encode(), *chunks])
        self._sleep(len(body) * 8 / (self.profile.bw_kbps * 1000))  # uplink emulation
        sig = security.sign_update(self.token, self.node_id, round_id, attempt, body)
        r = self.http.post(f"{self.url}/update", content=body, headers={
            **self._headers(), "X-Round": str(round_id), "X-Attempt": str(attempt),
            "X-Signature": sig, "Content-Type": "application/octet-stream"})
        if r.status_code != 200:
            log.warning("%s: update rejected (%d): %s", self.node_id, r.status_code, r.text)

    def do_eval(self, round_id: int, attempt: int, version: int) -> None:
        if not self._download_global(version):
            return
        ev = self.trainer.evaluate_global()
        r = self.http.post(f"{self.url}/eval", headers=self._headers(),
                           json={"round": round_id, "attempt": attempt, "eval": ev})
        if r.status_code != 200:
            log.warning("%s: eval rejected (%d): %s", self.node_id, r.status_code, r.text)

    # ------------------------------------------------------------------ main loop
    def run(self) -> int:
        self._wait_for_server()
        self.register()
        log.info("%s (%s) enrolled: %d train / %d val samples", self.node_id, self.profile.name,
                 len(self.shard.y_train), len(self.shard.y_val))
        failures = 0
        while True:
            try:
                st = self.http.get(f"{self.url}/status", headers=self._headers())
                st.raise_for_status()
                s = st.json()
                failures = 0
            except httpx.HTTPError as exc:
                failures += 1
                if failures > 40:
                    log.error("%s: lost the aggregator (%s)", self.node_id, exc)
                    return 1
                time.sleep(1.0)
                continue
            if s["phase"] == "finished":
                log.info("%s: swarm finished", self.node_id)
                return 0
            key = (s["phase"], s["round"], s["attempt"])
            if s["selected"] and not s["already_submitted"] and key not in self.done:
                self.done.add(key)
                try:
                    if s["phase"] == "collecting":
                        self.do_train_round(s["round"], s["attempt"])
                    else:
                        self.do_eval(s["round"], s["attempt"], s["model_version"])
                except httpx.HTTPError as exc:
                    log.warning("%s: network error in %s: %s", self.node_id, key, exc)
            time.sleep(self.poll_s * (0.6 + 0.8 * random.random()))  # jitter: no thundering herd


def main() -> None:
    logging.basicConfig(level=os.environ.get("SWARM_LOG", "INFO"),
                        format="%(asctime)s %(name)s %(message)s")
    e = os.environ.get
    node = EdgeNode(
        node_id=e("SWARM_NODE_ID", "node-0001"),
        shard_path=e("SWARM_SHARD", "/data/shard.npz"),
        server_url=e("SWARM_SERVER_URL", "http://aggregator:8080"),
        enroll_secret=e("SWARM_ENROLL_SECRET", "dev-enroll-secret-change-me"),
        device_class=e("SWARM_DEVICE_CLASS", "phone"),
        secret_ctx_path=e("SWARM_SECRET_CTX"),
        time_scale=float(e("SWARM_TIME_SCALE", "1.0")),
        poll_s=float(e("SWARM_POLL_S", "1.0")),
        emulate=e("SWARM_EMULATE", "1") == "1",
    )
    sys.exit(node.run())


if __name__ == "__main__":
    main()
