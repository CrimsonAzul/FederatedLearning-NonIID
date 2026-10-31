"""Federated learning server: a Flask app plus one coordinator thread.

The coordinator thread runs the training loop (select clients -> wait for their
updates -> aggregate -> evaluate). Flask request handlers only read and change
the shared state. All shared state is guarded by a single threading.Condition,
which is also how the handlers and the coordinator wake each other up.

Clients pull and the server never connects to a client, which is why
`docker compose up --scale client=N` works without a list of client addresses.

One round, from a client's point of view:

    GET  /task?client_id=k        long-poll; returns {"action": "train", ...} once k is selected
    GET  /weights?client_id=k     the current global weights (raw bytes)
    POST /heartbeat?client_id=k   every 0.5 s while training, to prove it is alive
    POST /update?client_id=k&...  the trained weights (raw bytes) + its sample count

The server aggregates once every selected client has either reported or gone
silent for `heartbeat_timeout` seconds; silent clients count as dropped and a
late update from them is rejected.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.serving import make_server

from fl.aggregate import fedavg
from fl.config import load_spec, run_dir, run_name
from fl.data import load_npz, test_path
from fl.metrics import MetricsLogger, summarize, write_json
from fl.model import SmallCNN
from fl.serialization import bytes_to_state, state_to_bytes
from fl.train import evaluate, get_device, seed_everything, to_tensors

log = logging.getLogger("server")

POLL_SECONDS = 15        # longest a /task request blocks before answering "wait"
RECLAIM_SECONDS = 60     # a client silent this long can have its id taken by a newcomer


class Coordinator:
    def __init__(self, spec: dict, out_dir: Path, device):
        self.spec = spec
        self.n = spec["num_clients"]
        self.out_dir = out_dir
        self.cond = threading.Condition()
        self.rng = np.random.default_rng(spec["seed"])   # drives client selection only
        self.on_finish = lambda: None                      # set by main() to stop the HTTP server

        # ---- shared state (only touched while holding self.cond) ----
        self.clients: dict[int, dict] = {}   # id -> {"last_contact", "num_samples", "ready"}
        self.phase = "registering"           # registering -> training -> done
        self.round = 0
        self.selected: set[int] = set()      # asked to train this round
        self.dispatched: set[int] = set()    # selected and has fetched its task
        self.pending: set[int] = set()       # selected, not reported yet, not dropped
        self.dropped: set[int] = set()       # went silent this round
        self.last_beat: dict[int, float] = {}
        self.updates: dict[int, tuple[int, dict, float]] = {}   # id -> (n_samples, weights, loss)
        self.stopped: set[int] = set()       # clients that have been told to stop
        self.history: list[dict] = []

        # ---- model and held-out test set (the server never sees training data) ----
        self.device = device
        seed_everything(spec["seed"])
        self.model = SmallCNN().to(device)
        self.shapes = {k: v.shape for k, v in self.model.state_dict().items()}
        self.global_bytes = state_to_bytes(self.model.state_dict())
        x, y = load_npz(test_path(spec))
        self.x_test, self.y_test = to_tensors(x, y, device)
        self.metrics = MetricsLogger(out_dir / "metrics.jsonl")

    # ================================================================ request handlers

    def _touch(self, cid: int) -> None:
        self.clients[cid]["last_contact"] = time.time()

    def _beat(self, cid: int) -> None:
        """Liveness for the current round: only counts for clients we are waiting on."""
        if cid in self.pending:
            self.last_beat[cid] = time.time()

    def register(self) -> int | None:
        """Hand out the lowest free client id (= which shard the client loads).

        If every id is taken, a newcomer may take over the id of a client that
        has been silent for RECLAIM_SECONDS, so a crashed container that
        restarts can rejoin with its own shard.
        """
        with self.cond:
            if self.phase == "done":
                return None
            now = time.time()
            free = [i for i in range(self.n) if i not in self.clients]
            if not free:
                free = [i for i, c in sorted(self.clients.items())
                        if now - c["last_contact"] > RECLAIM_SECONDS]
            if not free:
                return None
            cid = free[0]
            self.clients[cid] = {"last_contact": now, "num_samples": None, "ready": False}
            self.stopped.discard(cid)
            log.info("client %d registered", cid)
            return cid

    def ready(self, cid: int, num_samples: int) -> bool:
        with self.cond:
            if cid not in self.clients:
                return False
            self.clients[cid].update(num_samples=num_samples, ready=True)
            self._touch(cid)
            self.cond.notify_all()
            return True

    def next_task(self, cid: int) -> dict | None:
        """Long-poll: block until this client has work, training ends, or POLL_SECONDS pass."""
        deadline = time.time() + POLL_SECONDS
        with self.cond:
            if cid not in self.clients:
                return None
            self._touch(cid)
            while True:
                if self.phase == "done":
                    self.stopped.add(cid)
                    return {"action": "stop"}
                if cid in self.pending and cid not in self.dispatched:
                    self.dispatched.add(cid)
                    self._beat(cid)
                    s = self.spec
                    return {"action": "train", "round": self.round,
                            "local_epochs": s["local_epochs"], "batch_size": s["batch_size"],
                            "lr": s["lr"], "momentum": s["momentum"], "weight_decay": s["weight_decay"],
                            "mu": s["mu"], "augment": s["augment"], "dropout": s["dropout"],
                            "seed": s["seed"]}
                remaining = deadline - time.time()
                if remaining <= 0:
                    return {"action": "wait"}
                self.cond.wait(remaining)

    def get_weights(self, cid: int) -> tuple[int, bytes] | None:
        with self.cond:
            if cid not in self.clients:
                return None
            self._touch(cid)
            self._beat(cid)
            return self.round, self.global_bytes

    def heartbeat(self, cid: int) -> bool:
        with self.cond:
            if cid not in self.clients:
                return False
            self._touch(cid)
            self._beat(cid)
            return True

    def submit_update(self, cid: int, rnd: int, num_samples: int, loss: float, body: bytes) -> str:
        """Returns "ok", "stale" (round closed or client already declared dropped) or "invalid"."""
        try:
            state = bytes_to_state(body)          # deserialise outside the lock
        except Exception:
            return "invalid"
        if set(state) != set(self.shapes) or any(state[k].shape != s for k, s in self.shapes.items()):
            return "invalid"
        if num_samples <= 0:
            return "invalid"
        with self.cond:
            if cid not in self.clients:
                return "invalid"
            self._touch(cid)
            if rnd != self.round or cid not in self.pending:
                return "stale"
            self.updates[cid] = (num_samples, state, loss)
            self.pending.discard(cid)
            self.cond.notify_all()
            return "ok"

    def status(self) -> dict:
        """Snapshot for the dashboard and for humans with curl."""
        with self.cond:
            now = time.time()
            clients = []
            for cid in range(self.n):
                c = self.clients.get(cid)
                if c is None:
                    state = "absent"
                elif cid in self.dropped:
                    state = "dropped"
                elif cid in self.updates:
                    state = "reported"
                elif cid in self.pending:
                    state = "training" if cid in self.dispatched else "selected"
                elif now - c["last_contact"] > POLL_SECONDS + 5:
                    state = "offline"
                else:
                    state = "idle" if c["ready"] else "loading"
                clients.append({"id": cid, "state": state,
                                "num_samples": c["num_samples"] if c else None})
            s = self.spec
            return {"run": run_name(s), "phase": self.phase, "round": self.round,
                    "rounds": s["rounds"], "algorithm": s["algorithm"], "alpha": s["alpha"],
                    "mu": s["mu"], "participation": s["participation"], "dropout": s["dropout"],
                    "clients": clients,
                    "history": [{k: h.get(k) for k in ("round", "test_acc", "received", "dropped")}
                                for h in self.history]}

    # ================================================================ training loop

    def run(self) -> None:
        try:
            self._wait_for_clients()
            for r in range(1, self.spec["rounds"] + 1):
                self._run_round(r)
            self._finish(ok=True)
        except Exception:
            log.exception("training loop crashed")
            self._finish(ok=False)

    def _wait_for_clients(self) -> None:
        with self.cond:
            last_report = 0.0
            while sum(c["ready"] for c in self.clients.values()) < self.n:
                if time.time() - last_report > 10:
                    ready = sum(c["ready"] for c in self.clients.values())
                    log.info("waiting for clients: %d/%d ready (start them with --scale client=%d)",
                             ready, self.n, self.n)
                    last_report = time.time()
                self.cond.wait(timeout=1)
            self.phase = "training"
        log.info("all %d clients ready; %s for %d rounds", self.n, run_name(self.spec), self.spec["rounds"])

    def _run_round(self, r: int) -> None:
        s = self.spec
        k = max(1, round(s["participation"] * self.n))
        selected = sorted(self.rng.choice(self.n, size=k, replace=False).tolist())
        t0 = time.time()

        with self.cond:
            # Open the round. Each selected client gets heartbeat_timeout seconds
            # to pick up its task; after that it must keep sending heartbeats.
            self.round = r
            self.selected, self.pending = set(selected), set(selected)
            self.dispatched, self.dropped, self.updates = set(), set(), {}
            for cid in selected:
                self.last_beat[cid] = t0
            self.cond.notify_all()                      # wake clients blocked in /task

            # Wait until every selected client has reported or gone silent.
            while self.pending:
                self.cond.wait(timeout=0.25)
                now = time.time()
                for cid in list(self.pending):
                    if now - self.last_beat[cid] > s["heartbeat_timeout"]:
                        self.pending.discard(cid)
                        self.dropped.add(cid)
                        log.info("round %d: client %d went silent, dropped", r, cid)
            received = [self.updates[cid] for cid in sorted(self.updates)]
            n_dropped = len(self.dropped)
        wait_time = time.time() - t0

        # Aggregate outside the lock: handlers keep answering meanwhile.
        row = {"round": r, "selected": len(selected), "received": len(received), "dropped": n_dropped}
        if received:
            new_state = fedavg([(n, state) for n, state, _ in received])
            self.model.load_state_dict(new_state)
            new_bytes = state_to_bytes(new_state)
            total = sum(n for n, _, _ in received)
            row["train_loss"] = sum(n * loss for n, _, loss in received) / total
        else:
            new_bytes = self.global_bytes               # everyone dropped: keep the old model
            log.warning("round %d: no updates received, global model unchanged", r)
        if r % s["eval_every"] == 0 or r == s["rounds"]:
            row["test_acc"], row["test_loss"] = evaluate(self.model, self.x_test, self.y_test)
        row["wait_time"] = round(wait_time, 3)
        row["round_time"] = round(time.time() - t0, 3)

        with self.cond:
            self.global_bytes = new_bytes
            self.history.append(row)
        self.metrics.log(**row)
        acc = f"acc {row['test_acc']:.4f}" if "test_acc" in row else "acc   -   "
        log.info("round %3d/%d %s | %d/%d updates, %d dropped | %.1fs",
                 r, s["rounds"], acc, len(received), len(selected), n_dropped, row["round_time"])

    def _finish(self, ok: bool) -> None:
        with self.cond:
            self.phase = "done"
            self.cond.notify_all()
            history = list(self.history)
        summary = summarize(history, self.spec, ok=ok and len(history) == self.spec["rounds"],
                            total_dropped=sum(h["dropped"] for h in history))
        write_json(self.out_dir / "summary.json", summary)
        self.metrics.close()
        log.info("done: final acc %s, summary in %s", summary["final_acc"], self.out_dir)

        # Give connected clients a moment to receive "stop" and exit cleanly.
        deadline = time.time() + 10
        with self.cond:
            while time.time() < deadline:
                alive = {cid for cid, c in self.clients.items()
                         if time.time() - c["last_contact"] < POLL_SECONDS + 5}
                if alive <= self.stopped:
                    break
                self.cond.wait(0.5)
        self.on_finish()


def create_app(coord: Coordinator) -> Flask:
    app = Flask(__name__, static_folder=None)

    def client_id() -> int:
        return int(request.args["client_id"])

    @app.post("/register")
    def register():
        cid = coord.register()
        if cid is None:
            return jsonify(error="no free client slot (more clients than num_clients?)"), 409
        return jsonify(client_id=cid, spec=coord.spec)

    @app.post("/ready")
    def ready():
        ok = coord.ready(client_id(), int(request.args["num_samples"]))
        return (jsonify(ok=True), 200) if ok else (jsonify(error="unknown client"), 404)

    @app.get("/task")
    def task():
        t = coord.next_task(client_id())
        return jsonify(t) if t else (jsonify(error="unknown client"), 404)

    @app.get("/weights")
    def weights():
        got = coord.get_weights(client_id())
        if got is None:
            return jsonify(error="unknown client"), 404
        rnd, data = got
        return Response(data, mimetype="application/octet-stream", headers={"X-Round": str(rnd)})

    @app.post("/heartbeat")
    def heartbeat():
        return (jsonify(ok=True), 200) if coord.heartbeat(client_id()) else (jsonify(error="unknown client"), 404)

    @app.post("/update")
    def update():
        a = request.args
        result = coord.submit_update(client_id(), int(a["round"]), int(a["num_samples"]),
                                     float(a["loss"]), request.get_data())
        code = {"ok": 200, "stale": 409, "invalid": 400}[result]
        return jsonify(result=result), code

    @app.get("/status")
    def status():
        return jsonify(coord.status())

    @app.get("/")
    def dashboard():
        return send_from_directory(Path(__file__).parent / "static", "dashboard.html")

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Federated learning server")
    p.add_argument("--spec", required=True, help="run spec (.json from the runner, or a sweep-free .yaml)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto", help="where the server evaluates the global model")
    p.add_argument("--out", help="output directory (default: results/runs/<run>/seed<k>)")
    p.add_argument("--linger", action="store_true", help="keep serving the dashboard after training ends")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s server  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # no access log for every heartbeat

    spec = load_spec(args.spec)
    out = Path(args.out) if args.out else run_dir(spec)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", spec)

    coord = Coordinator(spec, out, get_device(args.device))
    httpd = make_server(args.host, args.port, create_app(coord), threaded=True)
    if not args.linger:
        coord.on_finish = httpd.shutdown
    threading.Thread(target=coord.run, daemon=True).start()
    log.info("listening on %s:%d (dashboard at http://localhost:%d/)", args.host, args.port, args.port)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
