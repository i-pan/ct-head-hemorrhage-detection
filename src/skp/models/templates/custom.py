"""
Template for a custom SKP model.

Copy this file into `src/skp/models/`, rename it, and set `cfg.model` to the
module path. For example, `src/skp/models/my_model.py` with class `Net` can be
used as:

    cfg.model = "my_model"
"""

import torch
import torch.nn as nn

from typing import Dict

from skp.configs import Config


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.model = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(cfg.num_input_channels * cfg.image_height * cfg.image_width, cfg.num_classes),
        )
        self.criterion = None

    def forward(self, batch: Dict, return_loss: bool = False) -> Dict[str, torch.Tensor]:
        logits = self.model(batch["x"].float())
        out = {"logits": logits}
        if return_loss:
            out.update(self.criterion(out, batch))
        return out

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
