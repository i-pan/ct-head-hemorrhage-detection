import torch
import torch.nn as nn

from importlib import import_module
from typing import Dict


def get_loss(loss_name, loss_params) -> nn.Module:
    module, loss = loss_name.rsplit(".", 1)
    module = import_module(f"skp.losses.{module}")
    return getattr(module, loss)(loss_params or {})


class CombinedLoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        # params format:
        # {
        #     "losses": {
        #         loss_name: {
        #             params: {...},
        #             output_key: ...
        #             weight: ...
        #         }
        #     }
        # }
        params = params["losses"] if "losses" in params else params
        self.loss_names = list(params)
        self.losses = nn.ModuleList(
            [get_loss(name, params[name].get("params", {})) for name in self.loss_names]
        )
        self.loss_name2key = {k: v["output_key"] for k, v in params.items()}
        self.loss_weights = {k: v.get("weight", 1.0) for k, v in params.items()}

    def forward(
        self, out: Dict[str, Dict], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # example out format
        # {
        #     cls: {
        #         logits
        #     },
        #     seg: {
        #         logits
        #     }
        # }
        # batch is similar except with expected contents of batch
        loss_dict = {}
        for loss_name, loss in zip(self.loss_names, self.losses):
            tmp_key = self.loss_name2key[loss_name]
            if tmp_key not in out:
                raise KeyError(f"CombinedLoss expected out['{tmp_key}']")
            if tmp_key not in batch:
                raise KeyError(f"CombinedLoss expected batch['{tmp_key}']")
            tmp_loss = loss(out[tmp_key], batch[tmp_key])
            if "loss" not in tmp_loss:
                raise KeyError(f"{loss_name} must return a loss dictionary with key 'loss'")
            tmp_loss = {f"{tmp_key}_{k}": v for k, v in tmp_loss.items()}
            loss_dict.update(tmp_loss)

        loss_dict["loss"] = 0
        for loss_name in self.loss_names:
            tmp_key = self.loss_name2key[loss_name]
            loss_dict["loss"] += (
                self.loss_weights[loss_name]
                * loss_dict[f"{tmp_key}_loss"]
            )

        return loss_dict
