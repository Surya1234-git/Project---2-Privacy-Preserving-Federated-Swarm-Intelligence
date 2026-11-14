"""Edge-node container entry point.

One container hosts several *logical* edge nodes (so a laptop can simulate hundreds of
nodes with ~8 containers). Each logical node only ever reads its own private shard.
"""
from __future__ import annotations

import os
import socket
import time

import numpy as np
import requests
import torch

from fedswarm.models import SmallCNN, get_weights, set_weights
from fedswarm.serialization import bytes_to_weights, weights_to_bytes
from fedswarm.training import local_train


def log(msg):
    print(f"[{socket.gethostname()}] {msg}", flush=True)


def retry(fn, what, delay=1.0, max_delay=10.0):
    while True:
        try:
            return fn()
        except requests.RequestException as e:
            log(f"{what} failed ({e.__class__.__name__}); retrying in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, max_delay)


def main():
    server = os.environ.get("SERVER_URL", "http://localhost:5000").rstrip("/")
    host = os.environ.get("NODE_HOST") or socket.gethostname()
    data_dir = os.environ.get("DATA_DIR", "data/nodes")
    poll = float(os.environ.get("POLL_SECONDS", "1"))
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))

    def register():
        r = requests.post(f"{server}/register", json={"hostname": host}, timeout=30)
        if r.status_code == 409:
            raise SystemExit(f"server refused registration: {r.json().get('error')}")
        r.raise_for_status()
        return r.json()

    reg = retry(register, "registration")
    idx, hp, spec = reg["container_idx"], reg["hyperparams"], reg["model_spec"]
    log(f"container #{idx} hosting {len(reg['node_ids'])} logical nodes "
        f"({reg['node_ids'][0]}..{reg['node_ids'][-1]})")

    model = SmallCNN(**spec)
    shards = {}

    def load_node(nid):
        if nid not in shards:
            z = np.load(os.path.join(data_dir, f"node_{nid:04d}.npz"))
            shards[nid] = (z["x"], z["y"])
        return shards[nid]

    while True:
        task = retry(lambda: requests.get(f"{server}/task", params={"container_idx": idx},
                                          timeout=30).json(), "task poll")
        if task["status"] == "done":
            log("server says training is done - exiting")
            return
        if task["status"] == "wait":
            time.sleep(poll)
            continue

        rnd = task["round"]
        resp = retry(lambda: requests.get(f"{server}/global_model", timeout=120), "model download")
        if int(resp.headers.get("X-Round", -1)) != rnd:
            continue                         # round moved on while we were asking; re-poll
        global_w = bytes_to_weights(resp.content)

        for nid in task["node_ids"]:
            x, y = load_node(nid)
            set_weights(model, global_w)     # every node starts from the same global model
            loss = local_train(model, x, y, hp["local_epochs"], hp["batch_size"], hp["lr"],
                               hp["momentum"], hp["mu"], "cpu", seed=hp["seed"] + rnd * 1000 + nid)
            body = weights_to_bytes(get_weights(model))
            headers = {"X-Round": str(rnd), "X-Node-Id": str(nid),
                       "X-Num-Samples": str(len(y)), "X-Train-Loss": f"{loss:.6f}",
                       "Content-Type": "application/octet-stream"}
            r = retry(lambda: requests.post(f"{server}/update", data=body, headers=headers,
                                            timeout=120), "update upload")
            log(f"round {rnd} node {nid}: loss {loss:.4f}, {len(y)} samples -> {r.status_code}")
            if r.status_code == 409:         # round already closed (we were too slow)
                break


if __name__ == "__main__":
    main()
