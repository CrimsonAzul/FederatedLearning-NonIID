import numpy as np
import pytest

from fl.data import dirichlet_partition, label_counts

LABELS = np.repeat(np.arange(10), 500)   # 5,000 samples, 500 per class


@pytest.mark.parametrize("alpha", [100, 1, 0.1])
def test_every_sample_assigned_exactly_once(alpha):
    parts = dirichlet_partition(LABELS, 10, alpha, seed=0)
    allidx = np.concatenate(parts)
    assert len(allidx) == len(LABELS)
    assert len(np.unique(allidx)) == len(LABELS)
    assert min(len(p) for p in parts) >= 10


def test_deterministic_per_seed():
    a = dirichlet_partition(LABELS, 10, 0.1, seed=3)
    b = dirichlet_partition(LABELS, 10, 0.1, seed=3)
    c = dirichlet_partition(LABELS, 10, 0.1, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def _top_class_share(alpha):
    counts = np.array(label_counts(LABELS, dirichlet_partition(LABELS, 10, alpha, seed=0)))
    return (counts.max(axis=1) / counts.sum(axis=1)).mean()


def test_alpha_controls_skew():
    iid, mid, skewed = _top_class_share(100), _top_class_share(1), _top_class_share(0.1)
    assert iid < 0.2                  # ~10% of each class when near-IID
    assert iid < mid < skewed
    assert skewed > 0.5               # most clients dominated by one class at alpha = 0.1
