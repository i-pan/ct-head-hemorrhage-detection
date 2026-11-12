import os
from skp.toolbox.ich_paths import path
"""Shared four-backbone auxiliary supervision experiment contract."""

import cv2
import albumentations as A


def configure(base):
    cfg = base.__deepcopy__()
    cfg.auxiliary_base_model = cfg.model
    cfg.model = "classification.ich_auxiliary"
    cfg.dataset = "rsna_ich_pseudolabel"
    cfg.pseudolabel_dir = str(path("ICH_PSEUDOLABELS", "./data/pseudolabels/rsna_three_family_final_20261110"))
    cfg.pseudolabel_supplement_dir = (
        "./data/pseudolabels/rsna_three_family_bhsd_related_20261111"
    )
    cfg.auxiliary_region = "none"
    cfg.auxiliary_roi_threshold = 0.07
    cfg.auxiliary_roi_radius = 3
    cfg.bhsd_manifest = "./data/bhsd/bhsd_segmentation_slices.csv"
    cfg.segmentation_loss_weight = 0.5
    cfg.spatial_loss_weight = 0.5
    cfg.pin_memory = False
    cfg.auxiliary_dice_positive_only = True

    cfg.pseudolabel_supplement_dir = os.environ.get("ICH_PSEUDOLABEL_SUPPLEMENT", cfg.pseudolabel_supplement_dir) or None
    cfg.bhsd_manifest = str(path("ICH_BHSD_MANIFEST", cfg.bhsd_manifest))

    # Preserve classification augmentations, align all spatial operations with masks.
    def align(transform):
        if isinstance(transform, A.CoarseDropout):
            transform.fill_mask = 0
        if hasattr(transform, "mask_interpolation"):
            transform.mask_interpolation = cv2.INTER_LINEAR
        for child in getattr(transform, "transforms", []):
            align(child)

    align(cfg.train_transforms)
    assert cfg.get("mix_proba") is None
    return cfg
