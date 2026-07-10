"""
Generic 2D segmentation wrapper, supporting commonly used decoders:
    - Unet
    - UperNet
    - DeepLabV3+

Deep supervision with auxiliary segmentation heads is also supported.
"""

import torch
import torch.nn as nn

from timm import create_model
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Union

from skp.configs import Config
from skp.models.segmentation import decoders
from skp.models.normalization import normalize_input
from skp.models.utils import torch_load_weights, filter_weights_by_prefix


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.drop = nn.Dropout2d(p=dropout)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        if isinstance(size, (tuple, list)):
            self.up = nn.Upsample(size=size, mode="bilinear")
        else:
            self.up = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.conv(self.drop(x)))


class Net(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        encoder_args = {
            "model_name": self.cfg.backbone,
            "pretrained": self.cfg.pretrained,
            "features_only": True,
            "in_chans": self.cfg.num_input_channels,
        }
        if self.cfg.get("backbone_img_size", False):
            encoder_args["img_size"] = self.cfg.image_height, self.cfg.image_width
        self.encoder = create_model(**encoder_args)

        if self.cfg.get("enable_gradient_checkpointing", False):
            print("Enabling gradient checkpointing ...")
            self.encoder.set_grad_checkpointing()

        with torch.no_grad():
            size = [
                2,
                self.cfg.num_input_channels,
                self.cfg.image_height,
                self.cfg.image_width,
            ]

            out = self.encoder(torch.randn(size))
            self.cfg.encoder_channels = [o.shape[1] for o in out]
            del out

        self.decoder = getattr(decoders, self.cfg.decoder_type)(self.cfg)
        if self.cfg.get("decoder_out_channels") is None:
            decoder_out_channels = self.decoder.decoder_out_channels
        else:
            decoder_out_channels = (
                self.cfg.decoder_out_channels[-1]
                if isinstance(self.cfg.decoder_out_channels, list)
                else self.cfg.decoder_out_channels
            )

        output_size = (
            (self.cfg.image_height, self.cfg.image_width)
            if self.cfg.get("output_size") is None
            else self.cfg.output_size
        )
        self.segmentation_head = SegmentationHead(
            decoder_out_channels,
            self.cfg.num_classes,
            size=output_size,
            dropout=self.cfg.get("seg_dropout", 0.0) or 0,
        )

        self.criterion = None

        if self.cfg.get("deep_supervision", False):
            self.aux_segmentation_heads = nn.ModuleList()
            for idx in range(self.cfg.get("deep_supervision_num_levels", 1) or 1):
                aux_decoder_out_channels = (
                    self.cfg.decoder_out_channels[-(idx + 2)]
                    if isinstance(self.cfg.decoder_out_channels, list)
                    else self.cfg.decoder_out_channels
                )
                self.aux_segmentation_heads.append(
                    SegmentationHead(
                        aux_decoder_out_channels, self.cfg.num_classes, size=None
                    )
                )

        if self.cfg.get("enable_gradient_checkpointing", False):
            # Checkpointed backward passes require non-inplace activations.
            for module in self.encoder.modules():
                if hasattr(module, "inplace"):
                    module.inplace = False

        if self.cfg.get("load_pretrained_encoder"):
            self.load_pretrained_encoder()

        if self.cfg.get("load_pretrained_decoder"):
            self.load_pretrained_decoder()

        if self.cfg.get("load_pretrained_model"):
            self.load_pretrained_model()

        if self.cfg.get("freeze_encoder", False):
            print("Freezing encoder ...")
            self.freeze_encoder()

        if self.cfg.get("freeze_decoder", False):
            print("Freezing decoder ...")
            self.freeze_decoder()

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_loss: bool = False,
        return_features: bool = False,
        return_decoder_output: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        x = batch["x"]
        y = batch["y"] if "y" in batch else None

        if return_loss:
            assert y is not None

        x = self.normalize(x)
        feature_maps = self.encoder(x)
        if self.cfg.get("enable_gradient_checkpointing", False) and self.training:
            decoder_output = checkpoint(self.decoder, feature_maps, use_reentrant=False)
        else:
            decoder_output = self.decoder(feature_maps)
        logits = self.segmentation_head(decoder_output[-1])

        out = {"logits": logits}

        if return_features:
            out["features"] = feature_maps

        if return_decoder_output:
            out["decoder_output"] = decoder_output

        if self.cfg.get("deep_supervision", False):
            aux_logits_list = []
            for idx, each_head in enumerate(self.aux_segmentation_heads):
                aux_logits_list.append(each_head(decoder_output[-(idx + 2)]))
            out["aux_logits"] = aux_logits_list

        if return_loss:
            loss = self.criterion(out, batch)
            out.update(loss)

        return out

    def normalize(self, x):
        return normalize_input(x, self.cfg)

    def load_pretrained_encoder(self) -> None:
        print(f"Loading pretrained encoder from {self.cfg.load_pretrained_encoder} ...")
        weights = torch_load_weights(self.cfg.load_pretrained_encoder)
        weights = filter_weights_by_prefix(weights, "model.")
        # Get backbone only
        weights = filter_weights_by_prefix(weights, "backbone.")
        if len(weights) == 0:
            # If no backbone weights, check to see if encoder weights exist
            weights = filter_weights_by_prefix(weights, "encoder.")
        # seems weight names are not necessarily the same if features_only=True ...
        # for maxvit
        if self.cfg.backbone.startswith("maxvit_"):
            weights = {
                k.replace("stages.", "stages_"): v
                for k, v in weights.items()
                if not k.startswith("head.")
            }
        missing_keys, unexpected_keys = self.encoder.load_state_dict(
            weights, strict=False
        )
        if len(missing_keys) > 0:
            raise Exception(
                f"Error in loading state_dict, missing keys: {missing_keys}"
            )

    def load_pretrained_decoder(self) -> None:
        print(f"Loading pretrained decoder from {self.cfg.load_pretrained_decoder} ...")
        weights = torch_load_weights(self.cfg.load_pretrained_decoder)
        weights = filter_weights_by_prefix(weights, "model.")
        weights = filter_weights_by_prefix(weights, "decoder.")
        self.decoder.load_state_dict(weights, strict=True)

    def load_pretrained_segmentation_heads(
        self, weights: Dict[str, torch.Tensor]
    ) -> None:
        segmentation_head_weights = filter_weights_by_prefix(
            weights, "model.segmentation_head."
        )
        unequal_classes = (
            segmentation_head_weights["conv.weight"].shape[0] != self.cfg.num_classes
        )
        if self.cfg.get("load_segmentation_head_classes") is not None:
            class_indices = self.cfg.load_segmentation_head_classes
            if isinstance(class_indices, int):
                class_indices = [class_indices]
            # load head weights only for specified classes
            # should just be conv.weight and conv.bias
            print(f"Loading segmentation head weights for classes: {class_indices} ...")
            segmentation_head_weights = {
                k: v[class_indices] for k, v in segmentation_head_weights.items()
            }
        elif unequal_classes:
            print(
                "Pretrained segmentation head weights contain",
                segmentation_head_weights["conv.weight"].shape[0],
                "classes, but the current model has",
                f"{self.cfg.num_classes} classes ...",
            )
            print(f"Reshaping head weights to {self.cfg.num_classes} classes ...")
            for k, v in segmentation_head_weights.items():
                v = torch.stack([v.mean(0)] * self.cfg.num_classes)
                segmentation_head_weights[k] = v
        self.segmentation_head.load_state_dict(segmentation_head_weights)

        if self.cfg.get("deep_supervision", False):
            # load auxiliary heads for deep supervision, if available
            aux_segmentation_heads_weights = filter_weights_by_prefix(
                weights, "model.aux_segmentation_heads."
            )
            if len(aux_segmentation_heads_weights) > 0:
                if self.cfg.get("load_segmentation_head_classes"):
                    # load head weights only for specified classes
                    aux_segmentation_heads_weights = {
                        k: v[class_indices]
                        for k, v in aux_segmentation_heads_weights.items()
                    }
                elif unequal_classes:
                    for k, v in aux_segmentation_heads_weights.items():
                        v = torch.stack([v.mean(0)] * self.cfg.num_classes)
                        aux_segmentation_heads_weights[k] = v
                self.aux_segmentation_heads.load_state_dict(
                    aux_segmentation_heads_weights
                )

    def load_pretrained_model(self) -> None:
        print(f"Loading pretrained model from {self.cfg.load_pretrained_model} ...")
        weights = torch_load_weights(self.cfg.load_pretrained_model)
        segmenter_keys = [k for k in [*weights] if "segmenter" in k]
        if len(segmenter_keys) > 0:
            # if loading segmenter part of seg_cls model
            for k in segmenter_keys:
                weights[k.replace("segmenter.", "")] = weights.pop(k)
        encoder_weights = filter_weights_by_prefix(weights, "model.encoder.")
        decoder_weights = filter_weights_by_prefix(weights, "model.decoder.")
        self.encoder.load_state_dict(encoder_weights)
        self.decoder.load_state_dict(decoder_weights)
        self.load_pretrained_segmentation_heads(weights)

    def freeze_decoder(self) -> None:
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
