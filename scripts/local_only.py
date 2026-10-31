"""Lower bound: every client trains alone on its own shard and never shares anything.

Each client's model is scored on the full, balanced test set; the reported
accuracy is the mean over clients. Under strong label skew a client that has
seen only two or three classes cannot do much better than guessing on the rest,
which is exactly the gap federation is supposed to close.

    python -m scripts.local_only --spec <run config.json>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from fl.config import load_spec, run_dir
from fl.data import load_npz, prepare, shard_path, test_path
from fl.metrics import MetricsLogger, summarize, write_json
from fl.model import SmallCNN
from fl.train import evaluate, get_device, make_optimizer, seed_everything, to_tensors, train_local


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec", required=True)
    p.add_argument("--out")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    spec = load_spec(args.spec)
    out = Path(args.out) if args.out else run_dir(spec)
    write_json(out / "config.json", spec)
    device = get_device(args.device)
    prepare(spec)
    xt, yt = to_tensors(*load_npz(test_path(spec)), device)

    epochs = spec["epochs"]
    checkpoints = sorted(set(range(max(1, epochs // 10), epochs + 1, max(1, epochs // 10))) | {epochs})
    acc = np.zeros((spec["num_clients"], len(checkpoints)))
    t_start = time.time()
    for cid in range(spec["num_clients"]):
        x, y = to_tensors(*load_npz(shard_path(spec, cid)), device)
        seed_everything(spec["seed"])                 # same initial model as the federated runs
        model = SmallCNN().to(device)
        opt = make_optimizer(model, spec["lr"], spec["momentum"], spec["weight_decay"])
        done = 0
        for j, ep in enumerate(checkpoints):
            train_local(model, x, y, epochs=ep - done, batch_size=spec["batch_size"], lr=spec["lr"],
                        momentum=spec["momentum"], weight_decay=spec["weight_decay"],
                        use_augment=spec["augment"], opt=opt)
            done = ep
            acc[cid, j], _ = evaluate(model, xt, yt)
        print(f"client {cid:2d}: {y.numel():5d} samples, "
              f"{int(torch.unique(y).numel())} classes, test acc {acc[cid, -1]:.4f}", flush=True)

    metrics = MetricsLogger(out / "metrics.jsonl")
    for j, ep in enumerate(checkpoints):
        metrics.log(round=ep, test_acc=float(acc[:, j].mean()), test_acc_std=float(acc[:, j].std()),
                    test_acc_min=float(acc[:, j].min()), test_acc_max=float(acc[:, j].max()))
    metrics.close()
    summary = summarize(metrics.rows, spec, ok=True, per_client_acc=acc[:, -1].round(4).tolist())
    summary["total_time"] = round(time.time() - t_start, 1)
    write_json(out / "summary.json", summary)
    print(f"mean local-only accuracy {acc[:, -1].mean():.4f} (min {acc[:, -1].min():.4f}, max {acc[:, -1].max():.4f})")


if __name__ == "__main__":
    main()
