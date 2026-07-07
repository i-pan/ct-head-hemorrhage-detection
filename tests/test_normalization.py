import pytest
import torch

from skp.configs import Config
from skp.models.normalization import normalize_input


def test_linear_normalization_uses_explicit_input_and_output_bounds():
    x = torch.tensor([0.0, 0.5, 1.0])
    cfg = Config(
        normalization="linear",
        normalization_params={
            "input_min": 0.0,
            "input_max": 1.0,
            "output_min": -1.0,
            "output_max": 1.0,
        },
    )

    assert torch.allclose(normalize_input(x, cfg), torch.tensor([-1.0, 0.0, 1.0]))


def test_legacy_minmax_normalization_aliases_still_work():
    x = torch.tensor([0.0, 0.5, 1.0])
    cfg = Config(
        normalization="-1_1",
        normalization_params={"min": 0.0, "max": 1.0},
    )

    assert torch.allclose(normalize_input(x, cfg), torch.tensor([-1.0, 0.0, 1.0]))


def test_unknown_normalization_mode_raises():
    cfg = Config(normalization="mystery", normalization_params={})

    with pytest.raises(ValueError, match="Unknown normalization mode"):
        normalize_input(torch.zeros(1), cfg)
