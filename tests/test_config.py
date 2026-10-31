import pytest

from fl.config import expand, run_name


def test_sweep_times_seeds():
    specs = expand({"name": "x", "seeds": [0, 1, 2], "base": {"algorithm": "fedavg"},
                    "sweep": {"alpha": [1, 0.1], "num_clients": [5, 10]}})
    assert len(specs) == 2 * 2 * 3


def test_runs_list_overrides_base():
    specs = expand({"base": {"alpha": 0.1}, "runs": [{"algorithm": "fedavg"}, {"algorithm": "fedprox", "mu": 0.1}]})
    assert [s["algorithm"] for s in specs] == ["fedavg", "fedprox"]
    assert specs[1]["mu"] == 0.1


def test_fedavg_ignores_mu():
    (spec,) = expand({"base": {"algorithm": "fedavg", "mu": 0.5}})
    assert spec["mu"] == 0.0


def test_run_names():
    (a,) = expand({"base": {"algorithm": "fedprox", "mu": 0.01, "alpha": 0.1, "dropout": 0.2}})
    assert run_name(a) == "fedprox_mu0.01_a0.1_n10_p1_d0.2"
    (b,) = expand({"base": {"algorithm": "centralized"}})
    assert run_name(b) == "centralized"


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        expand({"base": {"learning_rate": 0.1}})
