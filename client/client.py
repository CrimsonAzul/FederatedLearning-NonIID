"""Federated client: owns one private data shard and trains when the server asks.

It only ever talks to the server. It never sees another client's data or
weights, and the server never sees its data, only the weights it sends back.

Lifecycle:
  1. POST /register      -> client id (which shard to load) and the run spec
  2. load data/shards/<run>/client_<id>.npz, POST /ready with its size
  3. loop: GET /task (long-poll) -> GET /weights -> train -> POST /update
  4. exit when the server says "stop" (or disappears)
"""
from __future__ import annotations

import argparse
import logging
import os
import threading
import time

import numpy as np
import requests
import torch

from fl.data import load_npz, shard_path
from fl.model import SmallCNN
from fl.serialization import bytes_to_state, state_to_bytes
from fl.train import get_device, seed_everything, to_tensors, train_local

log = logging.getLogger("client")


class Heartbeat:
    """Background thread that tells the server "still alive" while we train.

    Without it the server could not tell a slow client from a crashed one.
    It uses its own requests.Session because sessions are not thread-safe.
    """

    def __init__(self, url: str, cid: int, interval: float):
        self.url, self.cid, self.interval = url, cid, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._session = requests.Session()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._session.post(f"{self.url}/heartbeat", params={"client_id": self.cid}, timeout=5)
            except requests.RequestException:
                pass   # a missed beat is fine; the server tolerates heartbeat_timeout of silence

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def register(session: requests.Session, url: str, timeout: float = 300) -> dict:
    """POST /register, retrying while the server is still starting (normal under Compose).

    A 409 means every client id is taken. A restarted container keeps asking:
    once a crashed client has been silent long enough, the server hands its id
    (and shard) to the newcomer.
    """
    t0 = time.time()
    while True:
        try:
            r = session.post(f"{url}/register", timeout=10)
            if r.status_code != 409:
                r.raise_for_status()
                return r.json()
            reason = f"server refused registration: {r.json()['error']}"
        except requests.ConnectionError:
            reason = f"server at {url} not reachable"
        if time.time() - t0 > timeout:
            raise SystemExit(f"{reason} (gave up after {timeout:.0f}s)")
        time.sleep(1 if "reachable" in reason else 5)


def simulate_dropout(task: dict, cid: int) -> bool:
    """Does this client "crash" this round? Deterministic per (seed, client, round).

    A dropped client simply goes quiet: it does not fetch weights, does not
    send heartbeats and does not upload. From the server's side that is
    indistinguishable from a real crash or network failure.
    """
    if task["dropout"] <= 0:
        return False
    rng = np.random.default_rng([task["seed"], cid, task["round"]])
    return rng.random() < task["dropout"]


def main() -> None:
    p = argparse.ArgumentParser(description="Federated learning client")
    p.add_argument("--server", default=os.environ.get("FL_SERVER", "http://localhost:8000"))
    p.add_argument("--device", default=os.environ.get("FL_DEVICE", "auto"))
    p.add_argument("--threads", type=int, default=int(os.environ.get("FL_THREADS", "1")),
                   help="CPU threads for torch (only matters when training on CPU)")
    args = p.parse_args()
    url = args.server.rstrip("/")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    torch.set_num_threads(args.threads)

    session = requests.Session()
    reg = register(session, url)
    cid, spec = reg["client_id"], reg["spec"]
    tag = f"client {cid:02d}"

    device = get_device(args.device)
    x_np, y_np = load_npz(shard_path(spec, cid))     # this client's shard and nothing else
    num_samples = len(y_np)
    x, y = to_tensors(x_np, y_np, device)
    counts = np.bincount(y_np, minlength=10)
    log.info("%s: %d samples on %s, label counts %s", tag, num_samples, device, counts.tolist())

    model = SmallCNN().to(device)
    session.post(f"{url}/ready", params={"client_id": cid, "num_samples": num_samples},
                 timeout=10).raise_for_status()
    beat_every = spec["heartbeat_timeout"] / 4

    while True:
        try:
            r = session.get(f"{url}/task", params={"client_id": cid}, timeout=60)
        except requests.ConnectionError:
            log.info("%s: server gone, exiting", tag)
            return
        if r.status_code == 404:
            log.warning("%s: server does not know this client (restarted?), exiting", tag)
            return
        task = r.json()
        if task["action"] == "stop":
            log.info("%s: training finished, exiting", tag)
            return
        if task["action"] == "wait":
            continue

        rnd = task["round"]
        if simulate_dropout(task, cid):
            log.info("%s: round %d: simulating a disconnect", tag, rnd)
            continue

        w = session.get(f"{url}/weights", params={"client_id": cid}, timeout=60)
        if int(w.headers["X-Round"]) != rnd:
            continue                                  # that round already closed without us
        model.load_state_dict(bytes_to_state(w.content))

        seed_everything(task["seed"] * 1_000_003 + cid * 10_007 + rnd)
        t0 = time.time()
        with Heartbeat(url, cid, beat_every):
            stats = train_local(model, x, y, epochs=task["local_epochs"], batch_size=task["batch_size"],
                                lr=task["lr"], momentum=task["momentum"],
                                weight_decay=task["weight_decay"], mu=task["mu"],
                                use_augment=task["augment"])
        resp = session.post(
            f"{url}/update",
            params={"client_id": cid, "round": rnd, "num_samples": num_samples, "loss": stats["train_loss"]},
            data=state_to_bytes(model.state_dict()),
            headers={"Content-Type": "application/octet-stream"}, timeout=60)
        result = resp.json().get("result", resp.status_code)
        log.info("%s: round %d: %d steps in %.1fs, loss %.3f, update %s",
                 tag, rnd, stats["steps"], time.time() - t0, stats["train_loss"], result)


if __name__ == "__main__":
    main()
