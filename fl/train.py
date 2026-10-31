"""Local training and evaluation, shared by clients, the server and the baselines.

Data stays as a uint8 tensor on the training device for the whole run; each
batch is converted to float, normalised and augmented on the fly. That avoids
DataLoader workers entirely (they are slow to start, awkward on Windows, and
pointless when a whole shard fits on the GPU).
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from fl.data import MEAN, STD


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)        # seeds the CPU and every CUDA device


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def to_tensors(x: np.ndarray, y: np.ndarray, device: torch.device):
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=x_uint8.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x_uint8.device).view(1, 3, 1, 1)
    return (x_uint8.float() / 255.0 - mean) / std


def augment(x: torch.Tensor) -> torch.Tensor:
    """Per-sample random horizontal flip and random 32x32 crop from a 4-pixel padded image."""
    b, dev = x.size(0), x.device
    flip = torch.rand(b, device=dev) < 0.5
    x = torch.where(flip.view(b, 1, 1, 1), x.flip(3), x)
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    top = torch.randint(0, 9, (b,), device=dev)
    left = torch.randint(0, 9, (b,), device=dev)
    ar = torch.arange(32, device=dev)
    rows = (top.view(b, 1) + ar).view(b, 1, 32, 1)          # which padded row each output row reads
    cols = (left.view(b, 1) + ar).view(b, 1, 1, 32)
    batch = torch.arange(b, device=dev).view(b, 1, 1, 1)
    chan = torch.arange(3, device=dev).view(1, 3, 1, 1)
    return padded[batch, chan, rows, cols]


def make_optimizer(model, lr, momentum, weight_decay):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def train_local(model, x, y, *, epochs, batch_size, lr, momentum, weight_decay,
                mu: float = 0.0, use_augment: bool = True, opt=None) -> dict:
    """Run `epochs` of SGD on (x, y). With mu > 0 this is FedProx's local objective:

        loss = cross_entropy + (mu / 2) * || w - w_global ||^2

    where w_global are the weights the model had when training started (the
    global model the server just sent). The extra term pulls each client back
    towards the global model, limiting how far it drifts on its skewed data.
    With mu = 0 the term vanishes and this is plain FedAvg local training.
    The optimizer is created fresh every call, so momentum is not carried over
    between rounds, as in the FedAvg and FedProx papers. The baselines pass
    their own `opt` so momentum persists across epochs.
    """
    model.train()
    if opt is None:
        opt = make_optimizer(model, lr, momentum, weight_decay)
    global_params = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
    n = y.numel()
    loss_sum = torch.zeros((), device=y.device)
    steps = 0
    for _ in range(epochs):
        order = torch.randperm(n, device=y.device)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb = normalize(x[idx])
            if use_augment:
                xb = augment(xb)
            ce = F.cross_entropy(model(xb), y[idx])
            loss = ce
            if global_params is not None:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
                loss = ce + 0.5 * mu * prox
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += ce.detach() * idx.numel()    # stays on the GPU: no sync per step
            steps += 1
    return {"train_loss": (loss_sum / (n * epochs)).item(), "steps": steps}


@torch.no_grad()
def evaluate(model, x, y, batch_size: int = 1000) -> tuple[float, float]:
    """(accuracy, mean cross-entropy) of `model` on (x, y)."""
    model.eval()
    correct = torch.zeros((), device=y.device)
    loss_sum = torch.zeros((), device=y.device)
    for start in range(0, y.numel(), batch_size):
        xb, yb = normalize(x[start:start + batch_size]), y[start:start + batch_size]
        logits = model(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum")
        correct += (logits.argmax(1) == yb).sum()
    n = y.numel()
    return correct.item() / n, loss_sum.item() / n
