"""Shared input normalization helpers for model wrappers."""

from __future__ import annotations

import torch

from skp.configs.base import Config


def normalize_input(x: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Normalize an input tensor according to ``cfg.normalization``.

    Prefer ``normalization="linear"`` with explicit input and output bounds:
    ``{"input_min": 0, "input_max": 255, "output_min": 0, "output_max": 1}``.
    Older names such as ``0_1``, ``-1_1``, ``minmax_0_1``, and ``minmax_-1_1``
    remain supported as aliases.
    """
    mode = cfg.normalization
    params = cfg.normalization_params

    if mode == "linear":
        return _linear_scale(
            x,
            input_min=params["input_min"],
            input_max=params["input_max"],
            output_min=params["output_min"],
            output_max=params["output_max"],
        )
    if mode in {"minmax_-1_1", "-1_1"}:
        return _linear_scale(
            x,
            input_min=params["min"],
            input_max=params["max"],
            output_min=-1.0,
            output_max=1.0,
        )
    if mode in {"minmax_0_1", "0_1"}:
        return _linear_scale(
            x,
            input_min=params["min"],
            input_max=params["max"],
            output_min=0.0,
            output_max=1.0,
        )
    if mode == "mean_sd":
        return (x - params["mean"]) / params["sd"]
    if mode == "per_channel_mean_sd":
        mean, sd = params["mean"], params["sd"]
        assert len(mean) == len(sd) == x.size(1)
        shape = (1, x.size(1), *([1] * (x.ndim - 2)))
        mean = x.new_tensor(mean).view(shape)
        sd = x.new_tensor(sd).view(shape)
        return (x - mean) / sd
    if mode == "none":
        return x
    raise ValueError(f"Unknown normalization mode: {mode}")


def _linear_scale(
    x: torch.Tensor,
    *,
    input_min: float,
    input_max: float,
    output_min: float,
    output_max: float,
) -> torch.Tensor:
    x = (x - input_min) / (input_max - input_min)
    return x * (output_max - output_min) + output_min
