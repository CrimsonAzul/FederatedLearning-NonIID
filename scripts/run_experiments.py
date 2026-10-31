"""Run every (run, seed) described by one or more experiment YAML files.

    python -m scripts.run_experiments configs/smoke.yaml
    python -m scripts.run_experiments configs/0*.yaml            # everything
    python -m scripts.run_experiments configs/04_fedprox_mu.yaml --dry-run

Each (run, seed) writes results/runs/<run>/seed<k>/: config.json,
metrics.jsonl, summary.json and logs/. Finished runs are skipped, so an
interrupted sweep picks up where it stopped (Ctrl-C is safe).

Federated runs start the real system: one server process plus num_clients
client processes talking HTTP on localhost. It is the same code the Docker
Compose setup runs, only launched as local processes so a sweep of dozens of
runs does not pay container start-up each time.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from fl.config import load_experiment, relevant, run_dir, run_name
from fl.data import prepare
from fl.metrics import write_json


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def is_done(spec: dict) -> bool:
    """True if this run already finished with the same settings."""
    out = run_dir(spec)
    summary, config = out / "summary.json", out / "config.json"
    if not (summary.exists() and config.exists()):
        return False
    old = json.loads(config.read_text())
    if {k: old.get(k) for k in relevant(spec)} != relevant(spec):
        raise SystemExit(f"{out} was produced with different settings than requested.\n"
                         f"  saved:     {relevant({**spec, **old})}\n  requested: {relevant(spec)}\n"
                         "Delete that directory to re-run it.")
    return bool(json.loads(summary.read_text()).get("ok"))


def tee(proc: subprocess.Popen, log_path: Path, show) -> threading.Thread:
    """Copy a child's output to a log file, echoing the lines `show` accepts."""
    def pump():
        with open(log_path, "w", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                if show(line):
                    print("   " + line.rstrip(), flush=True)
    t = threading.Thread(target=pump, daemon=True)
    t.start()
    return t


def wait_for_server(port: int, proc: subprocess.Popen, timeout: float = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited during start-up (see logs/server.log)")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=2)
            return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("server did not start")


def stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + 10
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()


def run_federated(spec: dict, out: Path, gpu_clients: int, show_every: int = 10) -> None:
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    port = free_port()
    py = sys.executable
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    n = spec["num_clients"]
    n_cpu_clients = n - min(gpu_clients, n)
    cpu_threads = max(1, (os.cpu_count() or 2) // max(1, n_cpu_clients))

    def show(line: str) -> bool:   # echo every `show_every`-th round, drops and problems
        if "round" in line and "/" in line and "updates" in line:
            r = int(line.split("round")[1].split("/")[0])
            return r % show_every == 0 or r == 1
        return any(w in line for w in ("done:", "Error", "error", "crashed", "Traceback"))

    procs: list[subprocess.Popen] = []
    log_files = []
    try:
        server = subprocess.Popen([py, "-m", "server.app", "--spec", str(out / "config.json"),
                                   "--port", str(port), "--host", "127.0.0.1", "--out", str(out)],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace", env=env)
        procs.append(server)
        pump = tee(server, logs / "server.log", show)
        wait_for_server(port, server)
        for i in range(n):
            device = "cuda" if i < gpu_clients else "cpu"
            log_file = open(logs / f"client_{i:02d}.log", "w", encoding="utf-8")
            log_files.append(log_file)
            procs.append(subprocess.Popen(
                [py, "-m", "client.client", "--server", f"http://127.0.0.1:{port}",
                 "--device", device, "--threads", str(cpu_threads)],
                stdout=log_file, stderr=subprocess.STDOUT, env=env))
        server.wait(timeout=spec["rounds"] * 600)
        pump.join(timeout=5)
        for p in procs[1:]:                      # clients exit on "stop"; give them a moment
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
    finally:
        stop(procs)
        for f in log_files:
            f.close()


def run_script(module: str, out: Path) -> None:
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, "-m", module, "--spec", str(out / "config.json"), "--out", str(out)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env)
    try:
        def show(line: str) -> bool:     # every 10th epoch, the final line, and errors
            if line.startswith("epoch"):
                return int(line.split()[1].split("/")[0]) % 10 == 0
            return line.startswith(("mean", "Traceback")) or "Error" in line
        pump = tee(proc, logs / "run.log", show)
        proc.wait()
        pump.join(timeout=5)
    finally:
        stop([proc])


def run_one(spec: dict, gpu_clients: int) -> bool:
    out = run_dir(spec)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").unlink(missing_ok=True)
    write_json(out / "config.json", spec)
    prepare(spec)
    t0 = time.time()
    if spec["algorithm"] == "centralized":
        run_script("scripts.centralized", out)
    elif spec["algorithm"] == "local":
        run_script("scripts.local_only", out)
    else:
        run_federated(spec, out, gpu_clients)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if not summary.get("ok"):
        print(f"   FAILED after {time.time() - t0:.0f}s, see {out / 'logs'}")
        return False
    print(f"   final acc (mean of last 5) {summary['final_acc_last5']:.4f}, "
          f"rounds to {summary['target_acc']:.0%}: {summary['rounds_to_target']}, {time.time() - t0:.0f}s")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Run experiment sweeps from YAML configs")
    p.add_argument("configs", nargs="+", help="experiment YAML files")
    p.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    p.add_argument("--only", help="only runs whose name contains this text")
    p.add_argument("--gpu-clients", type=int, default=None,
                   help="how many client processes may use the GPU (default: all if CUDA is "
                        "available). Each CUDA process costs ~0.3-0.5 GB of GPU memory for its "
                        "context, so a 6 GB card fits about 10; the rest train on the CPU.")
    args = p.parse_args()
    gpu_clients = args.gpu_clients if args.gpu_clients is not None else (10**6 if cuda_available() else 0)

    specs, seen = [], {}
    for path in args.configs:
        for spec in load_experiment(path):
            key = run_dir(spec)
            if key in seen:                      # same run requested twice: must be identical
                if relevant(seen[key]) != relevant(spec):
                    raise SystemExit(f"{key} is requested twice with different settings "
                                     f"({seen[key]['experiment']} vs {spec['experiment']})")
                continue
            seen[key] = spec
            if not args.only or args.only in run_name(spec):
                specs.append(spec)

    todo = [s for s in specs if not is_done(s)]
    print(f"{len(specs)} runs, {len(specs) - len(todo)} already done, {len(todo)} to go")
    if args.dry_run:
        for s in specs:
            print(f"  {'done' if s not in todo else 'todo'}  {run_name(s)}  seed {s['seed']}")
        return

    failed = []
    for i, spec in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {run_name(spec)}  seed {spec['seed']}", flush=True)
        try:
            if not run_one(spec, gpu_clients):
                failed.append(spec)
        except KeyboardInterrupt:
            print("\ninterrupted; finished runs are kept, re-run the same command to resume")
            sys.exit(130)
        except Exception as e:                       # keep the sweep going; report at the end
            print(f"   FAILED: {e}")
            failed.append(spec)
    if failed:
        print(f"{len(failed)} run(s) failed:", *(f"  {run_name(s)} seed {s['seed']}" for s in failed), sep="\n")
        sys.exit(1)
    print("all runs finished. Next: python -m scripts.report")


if __name__ == "__main__":
    main()
