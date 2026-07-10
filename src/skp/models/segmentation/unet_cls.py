import torch
import torch.nn as nn

from typing import Dict, List, Union
from skp.configs import Config
from skp.models.pooling import get_pool_layer
from skp.models.segmentation.base import Net as Segmenter
from skp.models.utils import filter_weights_by_prefix, torch_load_weights


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.segmenter = Segmenter(cfg)
        self.classifier = nn.Sequential(
            get_pool_layer(self.cfg, dim=2),
            nn.Dropout(p=self.cfg.get("cls_dropout", 0.0) or 0.0),
            # can specify different # of classes for segmentation and classification
            # otherwise, use same # of classes for both
            nn.Linear(
                self.segmenter.cfg.encoder_channels[-1],
                self.cfg.get("cls_num_classes") or self.cfg.num_classes,
            ),
        )

        if self.cfg.get("load_pretrained_segmenter"):
            print(
                f"Loading pretrained segmenter from {self.cfg.load_pretrained_segmenter} ..."
            )
            weights = torch_load_weights(self.cfg.load_pretrained_segmenter)
            weights = filter_weights_by_prefix(weights, "model.")
            self.segmenter.load_state_dict(weights)

        if self.cfg.get("load_pretrained_classifier_head"):
            self.load_pretrained_classifier_head()

        self.criterion = None

    def load_pretrained_classifier_head(self) -> None:
        print(
            "Loading pretrained classifier head from "
            f"{self.cfg.load_pretrained_classifier_head} ..."
        )
        weights = torch_load_weights(self.cfg.load_pretrained_classifier_head)
        weights = filter_weights_by_prefix(weights, "model.linear.")
        if len(weights) == 0:
            weights = filter_weights_by_prefix(weights, "model.classifier.2.")
        self.classifier[-1].load_state_dict(weights, strict=True)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_loss: bool = False,
        return_features: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        seg_out = self.segmenter(batch["seg"], return_loss=False, return_features=True)
        features = seg_out["features"] if return_features else seg_out.pop("features")
        out = {}
        out["seg"] = seg_out
        out["cls"] = {"logits": self.classifier(features[-1])}

        if return_loss:
            loss = self.criterion(out, batch)
            out.update(loss)

        return out

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
