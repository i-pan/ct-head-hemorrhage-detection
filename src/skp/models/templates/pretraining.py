"""
Template for a model that computes its own pretraining loss.

Use this when the loss is naturally part of the model forward pass, such as
masked reconstruction.
"""

import torch
import torch.nn as nn

from typing import Dict

from skp.configs import Config


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        self.criterion = None

    def forward(self, batch: Dict, return_loss: bool = False) -> Dict[str, torch.Tensor]:
        x = batch["x"].float()
        pred = self.decoder(self.encoder(x))
        out = {"logits": pred}
        if return_loss:
            loss = torch.mean((pred - x) ** 2)
            out.update({"loss": loss, "loss_per_sample": (pred - x).flatten(1).pow(2).mean(1)})
        return out

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
