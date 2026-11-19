"""Round orchestration for the aggregator (semi-synchronous federated rounds).

Phases:  waiting -> collecting(round 1..R) -> aggregating -> ... -> eval -> finished

A round closes as soon as every selected node has reported, OR the quorum has been reached and
``straggler_grace_s`` has passed, OR the deadline passed with a quorum. If the deadline passes
without a quorum the attempt is discarded and a fresh sample is drawn (same round number).

Privacy boundary: in ``fhe`` mode this class only ever holds ciphertext bytes and a public
CKKS context. The only plaintext it handles is non-sensitive metadata (n_samples, metrics).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from swarm import security
from swarm.config import ExperimentConfig
from swarm.crypto import ckks
from swarm.fl.aggregation import EncryptedAggregator, PlainAggregator
from swarm.fl.selection import select_participants

log = logging.getLogger("swarm.coordinator")


class Reject(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


@dataclass
class NodeInfo:
    node_id: str
    token: str
    device_class: str
    n_train: int
    n_val: int


@dataclass
class Update:
    n_samples: int
    payload: list[bytes] | np.ndarray
    train_loss: float
    nbytes: int
    t_received: float


@dataclass
class Coordinator:
    cfg: ExperimentConfig
    pub_ctx: object | None
    out_dir: Path
    enroll_secret: str
    nodes: dict[str, NodeInfo] = field(default_factory=dict)
    phase: str = "waiting"
    round: int = 0
    attempt: int = 0
    selected: list[str] = field(default_factory=list)
    updates: dict[str, Update] = field(default_factory=dict)
    global_payload: list[bytes] | np.ndarray | None = None
    global_version: int = 0
    deadline: float = 0.0
    quorum_at: float | None = None
    round_started: float = 0.0
    last_registration: float = 0.0
    evals: dict[int, dict[str, dict]] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    bytes_in_total: int = 0
    bytes_out_total: int = 0
    finished_at: float | None = None

    def __post_init__(self):
        self.slots = self.cfg.poly_modulus_degree // 2
        dims = self.cfg.dims
        self.n_params = sum(i * o + o for i, o in zip(dims[:-1], dims[1:], strict=True))
        if self.cfg.mode == "fhe":
            if self.pub_ctx is None:
                raise ValueError("fhe mode needs the public CKKS context")
            self.aggregator = EncryptedAggregator(self.pub_ctx)  # raises if secret key present
        elif self.cfg.mode == "plain":
            self.aggregator = PlainAggregator()
        else:
            raise ValueError(f"unknown mode {self.cfg.mode!r}")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))
        (self.out_dir / "history.jsonl").write_text("")

    # ------------------------------------------------------------------ registration / auth
    def register(self, node_id: str, device_class: str, n_train: int, n_val: int,
                 secret: str) -> NodeInfo:
        if not security.constant_time_equals(secret, self.enroll_secret):
            raise Reject(401, "bad enrollment secret")
        if not node_id or len(node_id) > 64:
            raise Reject(422, "invalid node_id")
        info = NodeInfo(node_id, security.new_token(), device_class, int(n_train), int(n_val))
        self.nodes[node_id] = info
        self.last_registration = time.monotonic()
        return info

    def authenticate(self, node_id: str | None, token: str | None) -> NodeInfo:
        info = self.nodes.get(node_id or "")
        if info is None or not security.constant_time_equals(token or "", info.token):
            raise Reject(401, "unknown node or bad token")
        return info

    # ------------------------------------------------------------------------ state machine
    def _start_round(self, r: int, attempt: int) -> None:
        self.round, self.attempt = r, attempt
        self.selected = select_participants(list(self.nodes), self.cfg.participation,
                                            self.cfg.min_selected, self.cfg.selection_seed,
                                            r, attempt)
        self.updates = {}
        self.quorum_at = None
        self.round_started = time.monotonic()
        self.deadline = self.round_started + self.cfg.round_timeout_s
        self.phase = "collecting"
        log.info("round %d attempt %d: selected %d/%d nodes", r, attempt, len(self.selected),
                 len(self.nodes))

    def _start_eval(self) -> None:
        self.round, self.attempt = self.cfg.rounds + 1, 0
        self.selected = sorted(self.nodes)
        self.updates = {}
        self.round_started = time.monotonic()
        self.deadline = self.round_started + self.cfg.round_timeout_s
        self.phase = "eval"
        log.info("final evaluation of global model v%d by %d nodes", self.global_version,
                 len(self.selected))

    def quorum_needed(self) -> int:
        return max(1, math.ceil(self.cfg.quorum * len(self.selected)))

    async def tick(self) -> None:
        now = time.monotonic()
        if self.phase == "waiting":
            n = len(self.nodes)
            ready = (self.cfg.expected_nodes and n >= self.cfg.expected_nodes) or (
                n >= self.cfg.min_nodes and now - self.last_registration >= self.cfg.start_grace_s)
            if ready:
                self._start_round(1, 0)
        elif self.phase == "collecting":
            got = len(self.updates)
            if got >= self.quorum_needed() and self.quorum_at is None:
                self.quorum_at = now
            close = (got == len(self.selected)
                     or (self.quorum_at is not None
                         and now >= self.quorum_at + self.cfg.straggler_grace_s)
                     or (now >= self.deadline and got >= self.quorum_needed()))
            if close:
                await self._aggregate()
            elif now >= self.deadline:
                log.warning("round %d attempt %d: quorum missed (%d/%d) - resampling", self.round,
                            self.attempt, got, self.quorum_needed())
                if self.attempt + 1 >= self.cfg.max_attempts:
                    raise RuntimeError(f"round {self.round}: quorum never reached")
                self._start_round(self.round, self.attempt + 1)
        elif self.phase == "eval":
            got = len(self.evals.get(self.global_version, {}))
            if got >= len(self.selected) or (now >= self.deadline and got > 0):
                self._finish()

    async def _aggregate(self) -> None:
        self.phase = "aggregating"
        r = self.round
        ids = sorted(self.updates)
        ups = [self.updates[i] for i in ids]
        n_samples = [u.n_samples for u in ups]
        t0 = time.perf_counter()
        if self.cfg.mode == "fhe":
            new_global = await asyncio.to_thread(self.aggregator.aggregate,
                                                 [u.payload for u in ups], n_samples)
        else:
            new_global = await asyncio.to_thread(self.aggregator.aggregate,
                                                 [u.payload for u in ups], n_samples)
        agg_s = time.perf_counter() - t0
        self.global_payload, self.global_version = new_global, r

        tot = float(sum(n_samples))
        prev = self.evals.get(r - 1, {})
        ev_n = sum(e["n"] for e in prev.values())
        dropped = [n for n in self.selected if n not in self.updates]
        by_class: dict[str, int] = {}
        for n in dropped:
            c = self.nodes[n].device_class
            by_class[c] = by_class.get(c, 0) + 1
        entry = {
            "round": r, "attempt": self.attempt, "selected": len(self.selected),
            "received": len(ids), "dropped": len(dropped), "dropped_by_class": by_class,
            "n_samples": int(tot),
            "train_loss": float(sum(u.train_loss * u.n_samples for u in ups) / tot),
            "bytes_in": sum(u.nbytes for u in ups), "aggregate_s": agg_s,
            "wall_s": time.monotonic() - self.round_started,
            "eval_of_version": r - 1,
            "eval_acc": (sum(e["acc"] * e["n"] for e in prev.values()) / ev_n) if ev_n else None,
            "eval_loss": (sum(e["loss"] * e["n"] for e in prev.values()) / ev_n) if ev_n else None,
            "eval_nodes": len(prev),
        }
        self._log_history(entry)
        self.updates = {}
        if r < self.cfg.rounds:
            self._start_round(r + 1, 0)
        else:
            self._start_eval()

    def _finish(self) -> None:
        v = self.global_version
        ev = self.evals.get(v, {})
        n = sum(e["n"] for e in ev.values())
        entry = {
            "round": self.cfg.rounds + 1, "eval_of_version": v, "eval_nodes": len(ev),
            "eval_acc": (sum(e["acc"] * e["n"] for e in ev.values()) / n) if n else None,
            "eval_loss": (sum(e["loss"] * e["n"] for e in ev.values()) / n) if n else None,
        }
        self._log_history(entry)
        self.phase = "finished"
        self.finished_at = time.monotonic()
        summary = {
            "mode": self.cfg.mode, "rounds": self.cfg.rounds, "nodes": len(self.nodes),
            "final_global_acc": entry["eval_acc"], "final_global_loss": entry["eval_loss"],
            "bytes_in_total": self.bytes_in_total, "bytes_out_total": self.bytes_out_total,
            "model_params": self.n_params,
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log.info("finished: final federated accuracy %s", entry["eval_acc"])

    def _log_history(self, entry: dict) -> None:
        self.history.append(entry)
        with open(self.out_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        acc = entry.get("eval_acc")
        log.info("round %s done: %s", entry["round"],
                 f"global v{entry['eval_of_version']} acc={acc:.4f}" if acc is not None
                 else "no eval yet")

    # ------------------------------------------------------------------------- node-facing
    def status(self, node_id: str) -> dict:
        return {
            "phase": self.phase, "round": self.round, "attempt": self.attempt,
            "selected": node_id in self.selected and self.phase in ("collecting", "eval"),
            "already_submitted": node_id in self.updates,
            "total_rounds": self.cfg.rounds, "model_version": self.global_version,
            "seconds_left": max(0.0, self.deadline - time.monotonic()),
        }

    def current_global(self) -> tuple[int, list[bytes] | None]:
        if self.global_payload is None:
            return 0, None
        if self.cfg.mode == "plain":
            return self.global_version, [np.asarray(self.global_payload, np.float64).tobytes()]
        return self.global_version, list(self.global_payload)

    def _decode_payload(self, chunks: list[bytes]) -> list[bytes] | np.ndarray:
        if self.cfg.mode == "plain":
            if len(chunks) != 1 or len(chunks[0]) != self.n_params * 8:
                raise Reject(422, "plain update must be one float64 vector of the model size")
            arr = np.frombuffer(chunks[0], dtype=np.float64).copy()
            if not np.all(np.isfinite(arr)):
                raise Reject(422, "non-finite weights")
            return arr
        try:
            ckks.validate_ciphertext_chunks(self.pub_ctx, chunks, self.n_params, self.slots)
        except ckks.CryptoError as exc:
            raise Reject(422, str(exc)) from exc
        return chunks

    def _check_open(self, node_id: str, round_id: int, attempt: int, phase: str) -> None:
        if self.phase != phase or round_id != self.round or attempt != self.attempt:
            raise Reject(409, f"not accepting {phase} for round {round_id}/{attempt} "
                              f"(server: {self.phase} {self.round}/{self.attempt})")
        if node_id not in self.selected:
            raise Reject(403, "node was not selected for this round")

    async def submit_update(self, node: NodeInfo, round_id: int, attempt: int, signature: str,
                            body: bytes, meta: dict, chunks: list[bytes]) -> None:
        self._check_open(node.node_id, round_id, attempt, "collecting")
        if node.node_id in self.updates:
            raise Reject(409, "duplicate update")
        if not security.verify_update(node.token, node.node_id, round_id, attempt, body,
                                      signature):
            raise Reject(401, "bad update signature")
        n_samples = int(meta.get("n_samples", 0))
        if not (0 < n_samples <= 10_000_000):
            raise Reject(422, "invalid n_samples")
        payload = await asyncio.to_thread(self._decode_payload, chunks)
        # state may have changed while validating
        self._check_open(node.node_id, round_id, attempt, "collecting")
        if node.node_id in self.updates:
            raise Reject(409, "duplicate update")
        self.updates[node.node_id] = Update(n_samples, payload, float(meta.get("train_loss", 0.0)),
                                            len(body), time.monotonic())
        self.bytes_in_total += len(body)
        ev = meta.get("eval_prev_global")
        if ev and round_id - 1 >= 0:
            self._store_eval(round_id - 1, node.node_id, ev)

    def submit_eval(self, node: NodeInfo, round_id: int, attempt: int, ev: dict) -> None:
        self._check_open(node.node_id, round_id, attempt, "eval")
        self._store_eval(self.global_version, node.node_id, ev)

    def _store_eval(self, version: int, node_id: str, ev: dict) -> None:
        try:
            rec = {"loss": float(ev["loss"]), "acc": float(ev["acc"]), "n": int(ev["n"])}
        except (KeyError, TypeError, ValueError) as exc:
            raise Reject(422, "bad eval payload") from exc
        if not (0.0 <= rec["acc"] <= 1.0) or rec["n"] <= 0 or not math.isfinite(rec["loss"]):
            raise Reject(422, "eval values out of range")
        self.evals.setdefault(version, {})[node_id] = rec

    def snapshot(self) -> dict:
        by_class: dict[str, int] = {}
        for n in self.nodes.values():
            by_class[n.device_class] = by_class.get(n.device_class, 0) + 1
        return {"phase": self.phase, "round": self.round, "mode": self.cfg.mode,
                "registered": len(self.nodes), "by_class": by_class,
                "model_version": self.global_version, "history": self.history,
                "bytes_in_total": self.bytes_in_total, "bytes_out_total": self.bytes_out_total}
