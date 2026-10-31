"""Start a real server and real client processes over HTTP, on CPU, with synthetic data."""
import json

from fl.config import expand, run_dir
from fl.metrics import read_jsonl
from scripts.run_experiments import run_one


def _spec(tmp_path, **kw):
    (spec,) = expand({"base": {"dataset": "synthetic", "num_clients": 3, "alpha": 0.5, "rounds": 3,
                               "local_epochs": 1, "augment": False, "heartbeat_timeout": 1.5,
                               "data_dir": str(tmp_path / "data"), "results_dir": str(tmp_path / "results"),
                               **kw}})
    return spec


def test_fedprox_run_completes(tmp_path):
    spec = _spec(tmp_path, algorithm="fedprox", mu=0.01)
    assert run_one(spec, gpu_clients=0)
    out = run_dir(spec)
    rows = read_jsonl(out / "metrics.jsonl")
    assert [r["round"] for r in rows] == [1, 2, 3]
    assert all(r["received"] == 3 and r["dropped"] == 0 for r in rows)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["ok"] and 0.0 <= summary["final_acc"] <= 1.0


def test_dropouts_are_detected_and_survived(tmp_path):
    spec = _spec(tmp_path, algorithm="fedavg", dropout=0.5, rounds=4)
    assert run_one(spec, gpu_clients=0)
    rows = read_jsonl(run_dir(spec) / "metrics.jsonl")
    assert len(rows) == 4
    assert sum(r["dropped"] for r in rows) > 0                 # some clients went silent...
    assert all(r["received"] + r["dropped"] == r["selected"] for r in rows)   # ...and were accounted for
