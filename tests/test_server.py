"""Protocol tests: drive the Flask handlers directly, with the round opened by hand."""
import json
import time

import pytest
import torch

from fl.config import DEFAULTS, normalize
from fl.data import prepare
from server.app import Coordinator, create_app


@pytest.fixture
def coord(tmp_path):
    spec = normalize({**DEFAULTS, "dataset": "synthetic", "num_clients": 2, "rounds": 1,
                      "data_dir": str(tmp_path / "data")})
    prepare(spec)
    c = Coordinator(spec, tmp_path / "out", torch.device("cpu"))
    yield c
    c.metrics.close()


def open_round(coord, selected):
    """What Coordinator._run_round does before it starts waiting."""
    with coord.cond:
        coord.round += 1
        coord.selected, coord.pending = set(selected), set(selected)
        coord.dispatched, coord.dropped, coord.updates = set(), set(), {}
        for cid in selected:
            coord.last_beat[cid] = time.time()


def test_registration_hands_out_each_shard_once(coord):
    app = create_app(coord).test_client()
    assert app.post("/register").json["client_id"] == 0
    assert app.post("/register").json["client_id"] == 1
    assert app.post("/register").status_code == 409          # only two shards exist


def test_round_protocol(coord):
    app = create_app(coord).test_client()
    for _ in range(2):
        app.post("/register")
    open_round(coord, [0])

    task = app.get("/task?client_id=0").json
    assert task["action"] == "train" and task["round"] == 1
    w = app.get("/weights?client_id=0")
    assert w.headers["X-Round"] == "1"
    weights = w.data

    url = "/update?client_id={}&round={}&num_samples=50&loss=1.0"
    assert app.post(url.format(1, 1), data=weights).status_code == 409    # client 1 was not selected
    assert app.post(url.format(0, 1), data=b"not weights").status_code == 400
    assert app.post(url.format(0, 2), data=weights).status_code == 409    # wrong round
    assert app.post(url.format(0, 1), data=weights).status_code == 200
    assert app.post(url.format(0, 1), data=weights).status_code == 409    # already reported
    assert coord.pending == set() and set(coord.updates) == {0}


def test_task_is_handed_out_once(coord, monkeypatch):
    """A client that silently skipped its task (crashed, or simulated dropout) gets
    "wait" when it polls again, and stays silent until the server drops it."""
    monkeypatch.setattr("server.app.POLL_SECONDS", 0.2)
    app = create_app(coord).test_client()
    app.post("/register")
    open_round(coord, [0])
    assert app.get("/task?client_id=0").json["action"] == "train"
    assert app.get("/task?client_id=0").json["action"] == "wait"


def test_status_reports_client_states(coord):
    app = create_app(coord).test_client()
    app.post("/register")
    s = app.get("/status").json
    assert [c["state"] for c in s["clients"]] == ["loading", "absent"]
    json.dumps(s)


def test_crashed_client_id_can_be_reclaimed(coord):
    """A restarted container gets the id (and shard) of a client that went silent."""
    import server.app as srv
    app = create_app(coord).test_client()
    app.post("/register")
    app.post("/register")
    assert app.post("/register").status_code == 409
    coord.clients[1]["last_contact"] -= srv.RECLAIM_SECONDS + 1     # client 1 has been silent too long
    assert app.post("/register").json["client_id"] == 1
