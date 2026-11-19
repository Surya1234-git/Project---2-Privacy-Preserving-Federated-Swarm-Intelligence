"""Run the REAL aggregator + N edge-node processes on localhost (no Docker needed).

    python -m swarm.sim.local_network --shards data/shards --nodes 12 --rounds 5 --mode fhe
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_local_network(shards: Path, nodes: int, rounds: int, mode: str, out: Path, *,
                      participation: float = 0.5, time_scale: float = 0.02, timeout_s: float = 300,
                      extra_env: dict | None = None, emulate: bool = True) -> dict:
    manifest = json.loads((shards / "manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    env = {**os.environ, "SWARM_MODE": mode, "SWARM_ROUNDS": str(rounds),
           "SWARM_PARTICIPATION": str(participation), "SWARM_EXPECTED_NODES": str(nodes),
           "SWARM_INPUT_DIM": str(manifest["input_dim"]),
           "SWARM_NUM_CLASSES": str(manifest["num_classes"]),
           "SWARM_ROUND_TIMEOUT_S": "60", "SWARM_STRAGGLER_GRACE_S": "1.5",
           "SWARM_MIN_NODES": "2", "SWARM_MIN_SELECTED": "2", "SWARM_PORT": str(port),
           "SWARM_HOST": "127.0.0.1", "SWARM_OUT_DIR": str(out / "aggregator"),
           "SWARM_EXIT_WHEN_DONE": "1", "SWARM_TIME_SCALE": str(time_scale),
           "SWARM_POLL_S": "0.3", "SWARM_EMULATE": "1" if emulate else "0",
           **(extra_env or {})}
    if mode == "fhe":
        keys = out / "keys"
        subprocess.run([sys.executable, "-m", "swarm.crypto.keygen", "--public",
                        str(keys / "public.ctx"), "--secret", str(keys / "secret.ctx")],
                       check=True, env=env)
        env["SWARM_PUBLIC_CTX"] = str(keys / "public.ctx")
        env["SWARM_SECRET_CTX"] = str(keys / "secret.ctx")  # only passed to nodes below
    server_env = {k: v for k, v in env.items() if k != "SWARM_SECRET_CTX"}  # aggregator: no sk
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    procs: list[subprocess.Popen] = []
    server = subprocess.Popen([sys.executable, "-m", "swarm.server.main"], env=server_env,
                              stdout=open(logs / "aggregator.log", "w"), stderr=subprocess.STDOUT)
    try:
        time.sleep(1.0)
        for n in manifest["nodes"][:nodes]:
            nenv = {**env, "SWARM_NODE_ID": n["node_id"], "SWARM_DEVICE_CLASS": n["device_class"],
                    "SWARM_SHARD": str(shards / f"{n['node_id']}.npz"),
                    "SWARM_SERVER_URL": f"http://127.0.0.1:{port}"}
            procs.append(subprocess.Popen([sys.executable, "-m", "swarm.node.client"], env=nenv,
                                          stdout=open(logs / f"{n['node_id']}.log", "w"),
                                          stderr=subprocess.STDOUT))
        t0 = time.time()
        while server.poll() is None:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"swarm did not finish within {timeout_s}s; see {logs}")
            time.sleep(0.5)
        if server.returncode != 0:
            raise RuntimeError(f"aggregator exited with {server.returncode}; see {logs}")
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
    finally:
        for p in [server, *procs]:
            if p.poll() is None:
                p.terminate()
    summary = json.loads((out / "aggregator" / "summary.json").read_text())
    summary["history"] = [json.loads(line) for line in
                          (out / "aggregator" / "history.jsonl").read_text().splitlines()]
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/shards")
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--mode", choices=["plain", "fhe"], default="fhe")
    ap.add_argument("--participation", type=float, default=0.5)
    ap.add_argument("--out", default="runs/local_network")
    a = ap.parse_args()
    s = run_local_network(Path(a.shards), a.nodes, a.rounds, a.mode, Path(a.out),
                          participation=a.participation)
    print(json.dumps({k: v for k, v in s.items() if k != "history"}, indent=2))
    for h in s["history"]:
        if h.get("eval_acc") is not None:
            print(f"global v{h['eval_of_version']:<3} federated acc = {h['eval_acc']:.4f}")


if __name__ == "__main__":
    main()
