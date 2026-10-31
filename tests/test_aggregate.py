import torch

from fl.aggregate import fedavg
from fl.model import SmallCNN
from fl.serialization import bytes_to_state, state_to_bytes


def test_weighted_by_sample_count():
    a = {"w": torch.tensor([0.0, 0.0])}
    b = {"w": torch.tensor([4.0, 8.0])}
    out = fedavg([(1, a), (3, b)])        # 1/4 * a + 3/4 * b
    assert torch.allclose(out["w"], torch.tensor([3.0, 6.0]))


def test_single_update_is_identity():
    state = SmallCNN().state_dict()
    out = fedavg([(123, state)])
    assert all(torch.allclose(out[k], state[k]) for k in state)


def test_equal_models_average_to_themselves():
    state = SmallCNN().state_dict()
    out = fedavg([(10, state), (500, state), (7, state)])
    assert all(torch.allclose(out[k], state[k], atol=1e-6) for k in state)


def test_serialization_roundtrip():
    state = SmallCNN().state_dict()
    back = bytes_to_state(state_to_bytes(state))
    assert back.keys() == state.keys()
    assert all(torch.equal(back[k], state[k]) for k in state)
