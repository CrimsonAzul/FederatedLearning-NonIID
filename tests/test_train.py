import torch

from fl.data import make_synthetic
from fl.model import SmallCNN
from fl.train import seed_everything, to_tensors, train_local

X, Y, _, _ = make_synthetic(n_train=256, n_test=10)
HP = dict(epochs=1, batch_size=64, lr=0.01, momentum=0.9, weight_decay=5e-4, use_augment=True)


def _train(mu):
    seed_everything(0)
    model = SmallCNN()
    start = [p.detach().clone() for p in model.parameters()]
    x, y = to_tensors(X, Y, torch.device("cpu"))
    seed_everything(1)
    train_local(model, x, y, mu=mu, **HP)
    return start, list(model.parameters())


def _distance(a, b):
    return sum(((p - q) ** 2).sum() for p, q in zip(a, b)).sqrt().item()


def test_mu_zero_is_fedavg():
    """FedProx with mu = 0 must give exactly the FedAvg update."""
    _, w_avg = _train(0.0)
    _, w_prox = _train(0.0)
    assert all(torch.equal(a, b) for a, b in zip(w_avg, w_prox))


def test_proximal_term_keeps_weights_closer_to_global():
    start, w_free = _train(0.0)
    _, w_prox = _train(10.0)
    assert _distance(start, w_prox) < _distance(start, w_free)


def test_training_reduces_loss():
    seed_everything(0)
    model = SmallCNN()
    x, y = to_tensors(X, Y, torch.device("cpu"))
    first = train_local(model, x, y, **{**HP, "use_augment": False})["train_loss"]
    for _ in range(3):
        last = train_local(model, x, y, **{**HP, "use_augment": False})["train_loss"]
    assert last < first
