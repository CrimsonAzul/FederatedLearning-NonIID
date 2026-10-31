"""Download CIFAR-10 (once) and write the test set and client shards for every run in the configs.

    python -m scripts.prepare_data configs/demo.yaml
    python -m scripts.prepare_data configs/0*.yaml

The runner calls prepare() itself, so this is only needed before starting the
server and clients by hand, and as the one-shot `prepare` service in Docker.
"""
import argparse

from fl.config import load_experiment
from fl.data import prepare


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("configs", nargs="+")
    args = p.parse_args()
    for path in args.configs:
        for spec in load_experiment(path):
            prepare(spec)
    print("data ready")


if __name__ == "__main__":
    main()
