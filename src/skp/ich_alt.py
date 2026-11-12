"""Versioned inference interface for the stride-16 ICH auxiliary ensemble."""

from pathlib import Path
import hashlib
import json
import math

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from skp.configs import Config
from skp.configs.defaults import classification_2d_defaults
from skp.models.classification.ich_auxiliary import Net

REPO_ID = "ianpan/ct-head-hemorrhage-detection-alt"
FAMILIES = ("convnext_tiny", "maxvit_tiny", "coatnet0")
LABELS = (
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
)


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def model_config(family):
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    cfg = Config()
    classification_2d_defaults(cfg)
    cfg.backbone = dict(
        convnext_tiny="convnext_tiny.fb_in1k",
        maxvit_tiny="maxvit_tiny_tf_512.in1k",
        coatnet0="coatnet_0_rw_224.sw_in1k",
    )[family]
    cfg.auxiliary_base_model = (
        "classification.maxvit_logit_pool"
        if family == "maxvit_tiny"
        else "classification.spatial_logit_backbones"
    )
    cfg.pretrained = False
    cfg.num_input_channels = 9
    cfg.num_classes = 6
    cfg.pool = "avg"
    cfg.dropout = 0.2
    cfg.normalization = "linear"
    cfg.normalization_params = dict(
        input_min=0.0, input_max=1.0, output_min=-1.0, output_max=1.0
    )
    cfg.backbone_img_size = family == "coatnet0"
    cfg.image_height = cfg.image_width = 512
    cfg.logit_output_stride = 16
    cfg.label_columns = list(LABELS)
    cfg.segmentation_loss_weight = cfg.spatial_loss_weight = 0.5
    cfg.auxiliary_outside_roi_weight = 0.25
    return cfg


def load_member(directory, family, device="cpu"):
    """Strictly load one member, hash-verified, unfrozen and ready for fine-tuning."""
    if family not in FAMILIES:
        raise ValueError(family)
    directory = Path(directory)
    manifest = json.loads((directory / "release.json").read_text())
    if manifest["schema_version"] != 1 or manifest["labels"] != list(LABELS):
        raise ValueError("Unsupported release schema/label order")
    relative = f"weights/{family}.safetensors"
    entry = manifest["files"][relative]
    path = directory / relative
    if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError(f"Corrupt weights: {family}")
    model = Net(model_config(family))
    model.load_state_dict(load_file(path), strict=True)
    return model.to(device).eval()


def download_release(revision, families=FAMILIES, cache_dir=None):
    """Download only selected architecture weights. Use an immutable HF revision."""
    from huggingface_hub import snapshot_download

    families = tuple(families)
    if (
        not families
        or len(set(families)) != len(families)
        or set(families) - set(FAMILIES)
    ):
        raise ValueError("Select distinct supported families")
    return Path(
        snapshot_download(
            REPO_ID,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=[
                "release.json",
                "README.md",
                "config.json",
                *[f"weights/{f}.safetensors" for f in families],
            ],
        )
    )


def preprocess_slab(
    previous, center, following, *, rescale_slope=1.0, rescale_intercept=0.0
):
    """Stored-pixel 2.5D input; HU uses slope=1/intercept=0. Missing neighbors=None.

    Crop/pad stored pixels before rescaling, matching the training dataset.
    Returns (9,512,512), windowed [0,1]. Output maps use this processed grid.
    """
    if center is None or not np.isfinite([rescale_slope, rescale_intercept]).all():
        raise ValueError("A center image and finite rescale values are required")
    frames = []
    lower = np.array([0, -20, -800], np.float32)[:, None, None]
    width = np.array([80, 200, 2800], np.float32)[:, None, None]
    for image in [previous, center, following]:
        if image is None:
            frames.append(np.zeros((3, 512, 512), np.float32))
            continue
        x = np.asarray(image, dtype=np.float32)
        if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
            raise ValueError("Expected finite, nonempty 2D images")
        if max(x.shape) > 640:
            x = cv2.resize(x, (512, 512), interpolation=cv2.INTER_AREA)
        else:
            h, w = x.shape
            top = max((h - 512) // 2, 0)
            left = max((w - 512) // 2, 0)
            x = x[top : top + 512, left : left + 512]
            h, w = x.shape
            x = np.pad(
                x,
                (
                    ((512 - h) // 2, 512 - h - (512 - h) // 2),
                    ((512 - w) // 2, 512 - w - (512 - w) // 2),
                ),
            )
        hu = x * rescale_slope + rescale_intercept
        frames.append(np.clip(hu[None] - lower, 0, width) / (width + 1e-6))
    return np.ascontiguousarray(np.concatenate(frames), dtype=np.float32)


def pool_series(logits, temperature=0.25):
    if logits.ndim != 2 or len(logits) == 0 or not torch.isfinite(logits).all():
        raise ValueError("Expected nonempty finite (slices,classes) logits")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    return temperature * (
        torch.logsumexp(logits / temperature, dim=0) - math.log(len(logits))
    )


class ICHEnsemble:
    def __init__(self, directory, *, device="cpu", families=FAMILIES):
        self.families = tuple(families)
        if (
            not self.families
            or len(set(self.families)) != len(self.families)
            or set(self.families) - set(FAMILIES)
        ):
            raise ValueError("Select distinct supported families")
        self.device = torch.device(device)
        self.models = [load_member(directory, f, device) for f in self.families]

    @torch.inference_mode()
    def predict_series(self, slabs, *, batch_size=4, return_segmentation=False):
        """Mean member logits, normalized LSE .25; optional ungated mean mask/maps.

        Subtype series scores use the same pooling but have not been evaluated.
        No ground-truth or classifier gating is applied to localization outputs.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        x = torch.as_tensor(slabs)
        if (
            x.ndim != 4
            or x.shape[1:] != (9, 512, 512)
            or len(x) == 0
            or not torch.isfinite(x).all()
            or x.min() < 0
            or x.max() > 1
        ):
            raise ValueError("Expected finite [0,1] slabs of shape (N,9,512,512)")
        logits = []
        masks = []
        maps = []
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size].to(self.device, dtype=torch.float32)
            outputs = []
            for model in self.models:
                with torch.autocast(
                    self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    outputs.append(
                        model(
                            {"x": batch},
                            return_features=return_segmentation,
                            return_segmentation=return_segmentation,
                        )
                    )
            logits.append(
                torch.stack([o["logits"].float() for o in outputs]).mean(0).cpu()
            )
            if return_segmentation:
                masks.append(
                    torch.stack(
                        [o["segmentation_logits"].float().sigmoid() for o in outputs]
                    )
                    .mean(0)
                    .cpu()
                )
                maps.append(
                    torch.stack([o["logit_map"][:, -1:].float() for o in outputs])
                    .mean(0)
                    .cpu()
                )
        logits = torch.cat(logits)
        out = dict(
            slice_logits=logits.numpy(),
            slice_classification=logits.sigmoid().numpy(),
            series_classification=pool_series(logits).sigmoid().numpy(),
        )
        if return_segmentation:
            out.update(
                segmentation=torch.cat(masks).numpy(),
                any_logit_map=torch.cat(maps).numpy(),
            )
        return out
