"""FedAvg aggregation (McMahan et al. 2017).

The new global model is the average of the client models, each weighted by
how many training samples that client has:

    w_global = sum_k (n_k / n) * w_k,      n = sum_k n_k

FedProx uses exactly the same aggregation; it only changes local training.
"""
from __future__ import annotations

import torch


def fedavg(updates: list[tuple[int, dict[str, torch.Tensor]]]) -> dict[str, torch.Tensor]:
    """updates: [(num_samples, state_dict), ...] -> averaged state_dict.

    Callers pass updates in a fixed order (sorted by client id): floating-point
    addition is not associative, so a fixed order keeps runs reproducible.
    """
    if not updates:
        raise ValueError("nothing to aggregate")
    total = sum(n for n, _ in updates)
    keys = updates[0][1].keys()
    averaged = {}
    for key in keys:
        if not updates[0][1][key].is_floating_point():
            averaged[key] = updates[0][1][key].clone()   # integer buffers (none in SmallCNN)
            continue
        averaged[key] = sum(state[key].double() * (n / total) for n, state in updates).float()
    return averaged
