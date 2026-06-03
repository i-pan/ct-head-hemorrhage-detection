"""
Template for a custom SKP loss.

Copy this file into `src/skp/losses/`, rename it, and point `cfg.loss` at it.
For example, `src/skp/losses/my_losses.py` with class `MyLoss` can be used as:

    cfg.loss = "my_losses.MyLoss"

Losses receive model outputs and input batches as dictionaries. They must return
a dictionary containing the primary key `"loss"`. Any additional keys ending in
`_loss` will be logged by the task wrappers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict


class CustomLoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.weight = params.get("weight", 1.0)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p = out["logits"]
        t = batch["y"]

        custom_loss = F.mse_loss(p.float(), t.float())
        return {
            "custom_loss": custom_loss,
            "loss": self.weight * custom_loss,
        }
