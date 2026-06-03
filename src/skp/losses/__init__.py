import torch


def _check_shapes_equal(x: torch.Tensor, y: torch.Tensor) -> None:
    assert x.shape == y.shape, f"x.shape [{x.shape}] does not equal y.shape [{y.shape}]"