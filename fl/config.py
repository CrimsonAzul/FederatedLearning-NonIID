"""Experiment configs.

An experiment YAML has a `base` block, an optional `runs` list (each item
overrides `base`), an optional `sweep` block (every combination is run) and a
list of `seeds`. Anything not set falls back to DEFAULTS below. `expand()`
turns the file into a list of *run specs*: flat dicts with one value per
hyper-parameter. A run spec is the only thing the server, the clients and the
baseline scripts ever read.

Results live in results/runs/<run_name>/seed<k>/, keyed by the run name and
not by the experiment, so the same run requested by two experiments (e.g.
FedAvg α=0.1 with 10 clients appears in experiment 3 and in the scaling sweep)
is trained once and reused.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "algorithm": "fedavg",     # fedavg | fedprox | centralized | local
    "dataset": "cifar10",      # cifar10 | synthetic (tiny, for tests and CI)
    "num_clients": 10,
    "alpha": 100.0,            # Dirichlet concentration: 100 ≈ IID, 0.1 = very skewed
    "participation": 1.0,      # fraction of clients asked to train each round
    "dropout": 0.0,            # probability that a selected client disconnects mid-round
    "rounds": 100,             # federated rounds
    "local_epochs": 2,         # epochs each selected client runs on its shard per round
    "epochs": 50,              # length of the centralized / local-only baselines
    "batch_size": 64,
    "lr": 0.01,                # 0.02 diverges sometimes without BatchNorm; NIID-Bench uses 0.01
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "mu": 0.0,                 # FedProx proximal coefficient (only used by fedprox)
    "augment": True,           # random crop + horizontal flip
    "eval_every": 1,           # server evaluates the global model every N rounds
    "target_acc": 0.60,        # for the "rounds to reach target" column
    "heartbeat_timeout": 2.0,  # seconds of silence before a selected client counts as dropped
    "seed": 0,
    "data_dir": "data",
    "results_dir": "results",
}

# Keys that change the result of a run, per algorithm. Two specs with the same
# run name must agree on these, otherwise the cached result would be wrong.
_COMMON = ["dataset", "batch_size", "lr", "momentum", "weight_decay", "augment", "seed"]
RELEVANT = {
    "centralized": _COMMON + ["epochs"],
    "local": _COMMON + ["num_clients", "alpha", "epochs"],
    "fedavg": _COMMON + ["num_clients", "alpha", "participation", "dropout", "rounds", "local_epochs"],
    "fedprox": _COMMON + ["num_clients", "alpha", "participation", "dropout", "rounds", "local_epochs", "mu"],
}


def fmt(x) -> str:
    """Compact number formatting for run names: 100.0 -> '100', 0.10 -> '0.1'."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return f"{x:g}" if isinstance(x, float) else str(x)


def run_name(spec: dict) -> str:
    a = spec["algorithm"]
    prefix = "" if spec["dataset"] == "cifar10" else f"{spec['dataset']}_"
    if a == "centralized":
        return f"{prefix}centralized"
    if a == "local":
        return f"{prefix}local_a{fmt(spec['alpha'])}_n{spec['num_clients']}"
    name = f"{prefix}{a}"
    if a == "fedprox":
        name += f"_mu{fmt(spec['mu'])}"
    name += f"_a{fmt(spec['alpha'])}_n{spec['num_clients']}_p{fmt(spec['participation'])}"
    if spec["dropout"] > 0:
        name += f"_d{fmt(spec['dropout'])}"
    return name


def normalize(spec: dict) -> dict:
    """Validate a spec and neutralise fields the algorithm ignores."""
    unknown = set(spec) - set(DEFAULTS) - {"experiment"}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if spec["algorithm"] not in RELEVANT:
        raise ValueError(f"algorithm must be one of {sorted(RELEVANT)}")
    spec = dict(spec)
    if spec["algorithm"] != "fedprox":
        spec["mu"] = 0.0           # FedAvg is FedProx with mu = 0
    if not 0 < spec["participation"] <= 1:
        raise ValueError("participation must be in (0, 1]")
    if not 0 <= spec["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    for key in ("alpha", "lr", "momentum", "weight_decay", "mu", "participation",
                "dropout", "target_acc", "heartbeat_timeout"):
        spec[key] = float(spec[key])
    return spec


def expand(experiment: dict) -> list[dict]:
    """Experiment dict -> list of run specs (runs × sweep product × seeds)."""
    base = {**DEFAULTS, **(experiment.get("base") or {})}
    variants = experiment.get("runs") or [{}]
    sweep = experiment.get("sweep") or {}
    keys = list(sweep)
    specs = []
    for variant in variants:
        for combo in itertools.product(*(sweep[k] for k in keys)):
            for seed in experiment.get("seeds", [0]):
                spec = {**base, **variant, **dict(zip(keys, combo)), "seed": seed}
                spec["experiment"] = experiment.get("name", "unnamed")
                specs.append(normalize(spec))
    return specs


def load_experiment(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        experiment = yaml.safe_load(f)
    experiment.setdefault("name", Path(path).stem)
    return expand(experiment)


def load_spec(path: str | Path) -> dict:
    """Load exactly one run (JSON written by the runner, or a sweep-free YAML)."""
    path = Path(path)
    if path.suffix == ".json":
        return normalize({**DEFAULTS, **json.loads(path.read_text(encoding="utf-8"))})
    specs = load_experiment(path)
    if len(specs) != 1:
        raise ValueError(f"{path} describes {len(specs)} runs; use scripts.run_experiments for sweeps")
    return specs[0]


def run_dir(spec: dict) -> Path:
    return Path(spec["results_dir"]) / "runs" / run_name(spec) / f"seed{spec['seed']}"


def relevant(spec: dict) -> dict:
    return {k: spec[k] for k in RELEVANT[spec["algorithm"]]}
