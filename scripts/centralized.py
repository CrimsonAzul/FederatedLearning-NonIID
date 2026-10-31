"""Upper bound: one model trained on all the training data in one place.

Same model, optimizer, batch size and augmentation as the federated runs, so
the only thing that differs is where the data lives.

    python -m scripts.centralized --spec configs/01_centralized.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from fl.config import load_spec, run_dir
from fl.data import load_dataset
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

    seed_everything(spec["seed"])
    x_train, y_train, x_test, y_test = load_dataset(spec)     # the pooled training set
    x, y = to_tensors(x_train, y_train, device)
    xt, yt = to_tensors(x_test, y_test, device)
    model = SmallCNN().to(device)
    opt = make_optimizer(model, spec["lr"], spec["momentum"], spec["weight_decay"])
    metrics = MetricsLogger(out / "metrics.jsonl")

    for epoch in range(1, spec["epochs"] + 1):
        t0 = time.time()
        stats = train_local(model, x, y, epochs=1, batch_size=spec["batch_size"], lr=spec["lr"],
                            momentum=spec["momentum"], weight_decay=spec["weight_decay"],
                            use_augment=spec["augment"], opt=opt)
        acc, loss = evaluate(model, xt, yt)
        metrics.log(round=epoch, test_acc=acc, test_loss=loss, train_loss=stats["train_loss"],
                    round_time=round(time.time() - t0, 3))
        print(f"epoch {epoch:3d}/{spec['epochs']}  acc {acc:.4f}  train loss {stats['train_loss']:.3f}", flush=True)

    metrics.close()
    write_json(out / "summary.json", summarize(metrics.rows, spec, ok=True))


if __name__ == "__main__":
    main()
