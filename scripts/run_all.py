"""The whole study in one resumable command, for an unattended overnight run.

    python -m scripts.run_all --gpu-clients 10
    python -m scripts.run_all --dry-run           # what is done, what is left, which mu

1. every experiment except dropout (00-05), skipping finished runs;
2. pick FedProx's mu for the dropout experiment: the mu in 04_fedprox_mu.yaml
   with the best mean final accuracy over its seeds, written into
   06_dropout.yaml so that file keeps describing exactly what was run;
3. the dropout experiment (06);
4. the results table and plots, pasted into README.md.

Stops at the first failing step. Ctrl-C is safe: re-run the same command and
finished runs are skipped. Leave the laptop alone while it runs; the 20-client
runs put half their clients on the CPU and take noticeably longer.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from fl.config import load_experiment, run_dir

CONFIGS = Path("configs")
MAIN = ["00_local_only.yaml", "01_centralized.yaml", "02_fedavg_iid.yaml",
        "03_fedavg_noniid.yaml", "04_fedprox_mu.yaml", "05_scaling.yaml"]
MU_SWEEP, DROPOUT = CONFIGS / "04_fedprox_mu.yaml", CONFIGS / "06_dropout.yaml"
FEDPROX_LINE = re.compile(r"^(\s*-\s*\{algorithm:\s*fedprox,\s*mu:\s*)([^}\s]+)(\s*\}.*)$", re.M)


def best_mu() -> tuple[float, dict[float, float]] | None:
    """Best mu from the finished 04 runs, or None while any seed is missing."""
    by_mu: dict[float, list[float]] = {}
    for spec in load_experiment(MU_SWEEP):
        summary = run_dir(spec) / "summary.json"
        if not summary.exists() or not json.loads(summary.read_text()).get("ok"):
            return None
        by_mu.setdefault(spec["mu"], []).append(json.loads(summary.read_text())["final_acc_last5"])
    means = {mu: sum(v) / len(v) for mu, v in by_mu.items()}
    return max(means, key=means.get), means


def set_dropout_mu(mu: float, means: dict[float, float]) -> None:
    text = DROPOUT.read_text(encoding="utf-8")
    if not FEDPROX_LINE.search(text):
        raise SystemExit(f"no '- {{algorithm: fedprox, mu: ...}}' line found in {DROPOUT}")
    note = "  # best in 04: " + ", ".join(f"mu {m:g} -> {a:.2%}" for m, a in sorted(means.items()))
    text = FEDPROX_LINE.sub(lambda m: f"{m.group(1)}{mu:g}}}{note}", text, count=1)
    DROPOUT.write_text(text, encoding="utf-8")


def step(*args: str) -> None:
    print("\n$ python -m", " ".join(args), flush=True)
    if subprocess.call([sys.executable, "-m", *args]) != 0:
        raise SystemExit(f"step failed: {' '.join(args)}; fix it and re-run, finished runs are kept")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gpu-clients", type=int, default=10)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    gpu = ["--gpu-clients", str(args.gpu_clients)]

    if args.dry_run:
        step("scripts.run_experiments", *(str(CONFIGS / c) for c in MAIN), str(DROPOUT), "--dry-run")
        picked = best_mu()
        print("\nmu for dropout:", f"{picked[0]:g}" if picked else "decided after 04 finishes")
        return

    step("scripts.run_experiments", *(str(CONFIGS / c) for c in MAIN), *gpu)
    picked = best_mu()
    if picked is None:
        raise SystemExit(f"{MU_SWEEP} has unfinished runs; cannot choose mu")
    set_dropout_mu(*picked)
    print(f"\ndropout experiment uses FedProx mu = {picked[0]:g}")
    step("scripts.run_experiments", str(DROPOUT), *gpu)
    step("scripts.report", "--readme")


if __name__ == "__main__":
    main()
