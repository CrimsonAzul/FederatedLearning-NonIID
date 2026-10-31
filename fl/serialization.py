"""Model weights <-> bytes for HTTP transfer.

torch.load unpickles by default, and unpickling bytes received over the
network can execute arbitrary code. weights_only=True restricts loading to
tensors and plain containers, so a malicious client cannot attack the server
(or vice versa) through the weights it sends.
"""
import io

import torch


def state_to_bytes(state: dict) -> bytes:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in state.items()}, buf)
    return buf.getvalue()


def bytes_to_state(data: bytes) -> dict:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
