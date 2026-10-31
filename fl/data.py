"""CIFAR-10 loading, the Dirichlet partition and per-client shard files.

`prepare(spec)` writes everything a run needs, once, and is safe to call again:

    data/raw/cifar-10-batches-bin/            the download (only `prepare` reads it)
    data/test/<dataset>.npz                   held-out test set: the server reads only this
    data/shards/<key>/client_007.npz          one private shard per client: a client reads only its own
    data/shards/<key>/partition.json          label counts per client (for the partition plot)

In Docker the server container mounts data/test and the client containers
mount data/shards, so the server physically cannot open training data.
"""
from __future__ import annotations

import hashlib
import json
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

CIFAR_URLS = [
    "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
    "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz",
]
CIFAR_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
NUM_CLASSES = 10

# Per-channel statistics of the CIFAR-10 training set, used to normalise inputs.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


# --------------------------------------------------------------------------- download / read

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_cifar10(raw_dir: Path) -> Path:
    """Download and extract the binary CIFAR-10 archive (no pickle involved)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted = raw_dir / "cifar-10-batches-bin"
    if (extracted / "test_batch.bin").exists():
        return extracted
    archive = raw_dir / "cifar-10-binary.tar.gz"
    if not archive.exists() or _md5(archive) != CIFAR_MD5:
        for url in CIFAR_URLS:
            try:
                print(f"downloading {url} (162 MB)")
                urllib.request.urlretrieve(url, archive)
                break
            except OSError as e:
                print(f"  failed: {e}")
        else:
            raise RuntimeError("could not download CIFAR-10; place cifar-10-binary.tar.gz in " + str(raw_dir))
        if _md5(archive) != CIFAR_MD5:
            raise RuntimeError("CIFAR-10 archive has the wrong checksum; delete it and retry")
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):   # Python >= 3.10.12 / 3.11.4: refuse unsafe paths
            tar.extractall(raw_dir, filter="data")
        else:
            tar.extractall(raw_dir)
    return extracted


def _read_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Each record is 1 label byte + 3072 pixel bytes (all red, then green, then blue)."""
    raw = np.fromfile(path, dtype=np.uint8).reshape(-1, 1 + 3 * 32 * 32)
    labels = raw[:, 0].astype(np.int64)
    images = raw[:, 1:].reshape(-1, 3, 32, 32)  # already channels-first, as PyTorch wants
    return images, labels


def load_cifar10(data_dir: str | Path):
    folder = download_cifar10(Path(data_dir) / "raw")
    train = [_read_bin(folder / f"data_batch_{i}.bin") for i in range(1, 6)]
    x_train = np.concatenate([x for x, _ in train])
    y_train = np.concatenate([y for _, y in train])
    x_test, y_test = _read_bin(folder / "test_batch.bin")
    return x_train, y_train, x_test, y_test


def make_synthetic(n_train: int = 2000, n_test: int = 500, seed: int = 1234):
    """A tiny CIFAR-shaped dataset (blocky class templates + noise) for tests and CI.

    It has the same shapes and dtypes as CIFAR-10, so every code path is
    exercised without a 162 MB download.
    """
    rng = np.random.default_rng(seed)
    templates = rng.normal(size=(NUM_CLASSES, 3, 8, 8)).repeat(4, axis=2).repeat(4, axis=3)

    def sample(n):
        y = rng.integers(0, NUM_CLASSES, n)
        x = templates[y] + rng.normal(scale=2.0, size=(n, 3, 32, 32))
        return np.clip(127.5 + 30 * x, 0, 255).astype(np.uint8), y.astype(np.int64)

    return (*sample(n_train), *sample(n_test))


def load_dataset(spec: dict):
    """(x_train, y_train, x_test, y_test) as uint8 NCHW images and int64 labels."""
    if spec["dataset"] == "cifar10":
        return load_cifar10(spec["data_dir"])
    if spec["dataset"] == "synthetic":
        return make_synthetic()
    raise ValueError(f"unknown dataset {spec['dataset']!r}")


# --------------------------------------------------------------------------- partition

def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int,
                        min_size: int = 10, max_tries: int = 1000) -> list[np.ndarray]:
    """Split sample indices across clients with label skew controlled by `alpha`.

    For every class c we draw proportions p ~ Dirichlet(alpha, ..., alpha) over
    the clients and deal out class c's samples in those proportions. Large
    alpha gives every client roughly the same mix (IID); small alpha gives each
    client a few dominant classes.

    Two details from the usual implementation (Hsu et al. 2019; NIID-Bench):
    a client that already holds its even share (N / num_clients) gets no more
    samples, which keeps shard sizes from exploding, and the draw is repeated
    until every client has at least `min_size` samples.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    n = len(labels)
    for _ in range(max_tries):
        shards: list[list[int]] = [[] for _ in range(num_clients)]
        for c in np.unique(labels):
            idx = rng.permutation(np.flatnonzero(labels == c))
            p = rng.dirichlet(np.full(num_clients, alpha))
            not_full = np.array([len(s) < n / num_clients for s in shards])
            p = p * not_full
            # With tiny alpha every non-full client can draw p == 0.0 exactly.
            p = p / p.sum() if p.sum() > 0 else not_full / not_full.sum()
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for shard, part in zip(shards, np.split(idx, cuts)):
                shard.extend(part.tolist())
        if min(len(s) for s in shards) >= min_size:
            return [np.array(s, dtype=np.int64) for s in shards]
    raise RuntimeError(f"no partition with >= {min_size} samples per client after {max_tries} tries")


def label_counts(labels: np.ndarray, parts: list[np.ndarray]) -> list[list[int]]:
    return [np.bincount(labels[idx], minlength=NUM_CLASSES).tolist() for idx in parts]


# --------------------------------------------------------------------------- files

def shard_dir(spec: dict) -> Path:
    from fl.config import fmt
    key = f"{spec['dataset']}_n{spec['num_clients']}_a{fmt(spec['alpha'])}_s{spec['seed']}"
    return Path(spec["data_dir"]) / "shards" / key


def shard_path(spec: dict, client_id: int) -> Path:
    return shard_dir(spec) / f"client_{client_id:03d}.npz"


def test_path(spec: dict) -> Path:
    return Path(spec["data_dir"]) / "test" / f"{spec['dataset']}.npz"


def _save_npz(path: Path, **arrays) -> None:
    """Write to a temp file then rename, so a crash never leaves half a shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as f:
        return f["x"], f["y"]


def prepare(spec: dict) -> None:
    """Create the test file and (for client-based runs) the shards for this spec."""
    need_test = not test_path(spec).exists()
    needs_shards = spec["algorithm"] != "centralized"
    marker = shard_dir(spec) / "partition.json"
    need_shards = needs_shards and not marker.exists()
    if not (need_test or need_shards):
        return
    x_train, y_train, x_test, y_test = load_dataset(spec)
    if need_test:
        _save_npz(test_path(spec), x=x_test, y=y_test)
    if need_shards:
        parts = dirichlet_partition(y_train, spec["num_clients"], spec["alpha"], spec["seed"])
        for cid, idx in enumerate(parts):
            _save_npz(shard_path(spec, cid), x=x_train[idx], y=y_train[idx])
        info = {"num_clients": spec["num_clients"], "alpha": spec["alpha"], "seed": spec["seed"],
                "dataset": spec["dataset"], "counts": label_counts(y_train, parts)}
        marker.write_text(json.dumps(info))  # written last: its presence means "complete"
        sizes = [len(p) for p in parts]
        print(f"wrote {len(parts)} shards to {shard_dir(spec)} (sizes {min(sizes)}..{max(sizes)})")
