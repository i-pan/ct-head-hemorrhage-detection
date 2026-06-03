"""Inference helpers for combining model predictions."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class Ensemble(nn.Module):
    """Mean-ensemble multiple SKP models during inference.

    Args:
        model_list: Models that accept a batch dictionary and return a dictionary.
        output_name: Output key to ensemble, usually ``"logits"``.
        activation_fn: Optional activation to apply before averaging. Supported
            values are ``"sigmoid"`` and ``"softmax"``.
    """

    def __init__(
        self,
        model_list: nn.ModuleList,
        output_name: str = "logits",
        activation_fn: Optional[str] = None,
    ):
        super().__init__()
        if not isinstance(model_list, nn.ModuleList):
            raise TypeError("model_list must be an nn.ModuleList.")
        if activation_fn not in {None, "sigmoid", "softmax"}:
            raise ValueError("activation_fn must be None, 'sigmoid', or 'softmax'.")
        self.models = model_list
        self.output_name = output_name
        self.activation_fn = activation_fn

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = []
        for model in self.models:
            out = model(batch, return_loss=False)[self.output_name]
            if self.activation_fn == "sigmoid":
                out = out.sigmoid()
            elif self.activation_fn == "softmax":
                out = out.softmax(dim=1)
            outputs.append(out)
        return torch.stack(outputs, dim=0).mean(dim=0)
