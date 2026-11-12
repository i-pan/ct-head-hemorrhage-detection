"""Auxiliary supervision must preserve image/mask geometry and small positives."""

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from skp.datasets.rsna_ich_pseudolabel import Dataset
from skp.models.classification.ich_auxiliary import auxiliary_losses


def test_soft_mask_geometry_and_negative_targets():
    ds = Dataset.__new__(Dataset)
    ds.mode = "train"
    ds.depth = 3
    ds.windows = [(40, 80)] * 3
    ds.image_size = (32, 32)
    ds.depth_flip_p = ds.horizontal_flip_p = ds.vertical_flip_p = 1.0
    ds.label_columns = ["any"]
    ds.series_index = {"a": 0}
    ds.df = pd.DataFrame(
        [
            dict(
                slice_path="positive",
                prev_slice_path="positive",
                next_slice_path="positive",
                rescale_slope=1.0,
                rescale_intercept=0.0,
                any=1.0,
                series_uid="a",
            ),
            dict(
                slice_path="negative",
                prev_slice_path="negative",
                next_slice_path="negative",
                rescale_slope=1.0,
                rescale_intercept=0.0,
                any=0.0,
                series_uid="a",
            ),
        ]
    )
    ds.pseudo_indices = np.array([0, -1])
    ds.probabilities = np.zeros((1, 32, 32), np.float16)
    ds.probabilities[0, 7:13, 10:17] = 0.75
    ds.transforms = A.Compose(
        [
            A.Affine(
                translate_px={"x": 3, "y": 2},
                scale=1,
                rotate=0,
                shear=0,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_LINEAR,
                p=1,
            )
        ],
        seed=88,
    )
    ds._load_windowed_slice = lambda path, **kwargs: (
        np.repeat(ds.probabilities.astype(np.float32), 3, axis=0)
        if path == "positive"
        else np.zeros((3, 32, 32), np.float32)
    )
    positive = ds[0]
    torch.testing.assert_close(positive["x"][0], positive["segmentation_target"][0])
    assert positive["segmentation_target"].max() == 0.75
    assert ds[1]["segmentation_target"].sum() == 0


def test_small_soft_target_and_all_losses_differentiate():
    y = torch.zeros(2, 1, 32, 32)
    y[0, 0, 1, 1] = 0.7
    logits = torch.zeros_like(y, requires_grad=True)
    spatial = torch.zeros(2, 1, 2, 2, requires_grad=True)
    bce, dice, mapping = auxiliary_losses(logits, spatial, y)
    (bce + dice + mapping).backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(spatial.grad).all()
    assert (
        spatial.grad[0, 0, 0, 0] < 0
    )  # .7 survives max pooling; averaging would reverse gradient
    assert spatial.grad[1].min() > 0


def test_missing_positive_mask_has_no_auxiliary_gradient():
    target = torch.zeros(2, 1, 32, 32)
    segmentation = torch.zeros_like(target, requires_grad=True)
    spatial = torch.zeros(2, 1, 2, 2, requires_grad=True)
    valid = torch.tensor([False, True])
    sum(auxiliary_losses(segmentation, spatial, target, valid)).backward()
    assert segmentation.grad[0].count_nonzero() == 0
    assert spatial.grad[0].count_nonzero() == 0
    assert segmentation.grad[1].abs().sum() > 0
    assert spatial.grad[1].abs().sum() > 0
    segmentation.grad = None
    spatial.grad = None
    sum(
        auxiliary_losses(
            segmentation, spatial, target, torch.zeros(2, dtype=torch.bool)
        )
    ).backward()
    assert segmentation.grad.count_nonzero() == spatial.grad.count_nonzero() == 0


def test_hu_teacher_union_inclusive_bounds_and_dilation():
    from skp.datasets.rsna_ich_pseudolabel import hu_teacher_union

    hu = np.array([-21, -20, -0.01, 0, 40, 100, 100.01, 180, 200], np.float32)[None]
    window = (np.clip(hu, -20, 180) + 20) / (np.float32(200) + np.float32(1e-6))
    roi = hu_teacher_union(window, np.zeros_like(window), radius=0)
    np.testing.assert_array_equal(roi, (hu >= 0) & (hu <= 100))
    p = np.zeros((32, 32), np.float32)
    p[16, 16] = 0.07
    roi = hu_teacher_union(np.zeros_like(p), p, radius=3)
    assert roi[16, 16] == 1 and roi[16, 19] == 1 and roi[16, 20] == 0


def test_roi_masks_both_losses_and_pool_targets():
    y = torch.zeros(1, 1, 32, 32)
    y[0, 0, 1, 1] = 0.7
    y[0, 0, 20, 20] = 1
    roi = torch.zeros_like(y)
    roi[:, :, 0:16, 0:16] = 1
    seg = torch.zeros_like(y, requires_grad=True)
    sp = torch.zeros(1, 1, 2, 2, requires_grad=True)
    sum(auxiliary_losses(seg, sp, y, roi=roi)).backward()
    assert seg.grad[0, 0, 20, 20] == 0 and seg.grad[0, 0, 1, 1] != 0
    assert sp.grad[0, 0, 0, 0] < 0 and sp.grad[0, 0, 1, 1] == 0
    assert sp.grad.count_nonzero() == 1
    empty = auxiliary_losses(seg, sp, y, roi=torch.zeros_like(y))
    assert all(loss == 0 for loss in empty)
    a = auxiliary_losses(seg, sp, y)
    b = auxiliary_losses(seg, sp, y, roi=torch.ones_like(y))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)


def test_positive_only_dice_excludes_negatives_but_bce_keeps_them():
    y = torch.full((2, 1, 8, 8), 0.06)
    seg = torch.zeros_like(y, requires_grad=True)
    spatial = torch.zeros(2, 1, 2, 2, requires_grad=True)
    positive = torch.tensor([True, False])
    bce, dice, mapping = auxiliary_losses(seg, spatial, y, positive=positive)
    expected = auxiliary_losses(seg[:1], spatial[:1], y[:1])[1]
    torch.testing.assert_close(dice, expected)
    dice.backward(retain_graph=True)
    assert seg.grad[0].abs().sum() > 0 and seg.grad[1].count_nonzero() == 0
    seg.grad = None
    (bce + mapping).backward()
    assert seg.grad[1].abs().sum() > 0 and spatial.grad[1].abs().sum() > 0
    empty = auxiliary_losses(
        seg, spatial, y, positive=torch.zeros(2, dtype=torch.bool)
    )[1]
    assert torch.isfinite(empty) and empty.item() == 0


def test_soft_roi_weight_preserves_targets_and_background_gradients():
    y = torch.zeros(1, 1, 4, 4)
    y[0, 0, 3, 3] = .8
    roi = torch.zeros_like(y)
    roi[:, :, :2, :2] = 1
    seg = torch.zeros_like(y, requires_grad=True)
    sp = torch.zeros(1, 1, 2, 2, requires_grad=True)
    bce, dice, spatial = auxiliary_losses(seg, sp, y, roi=roi, outside_weight=.25)
    bce.backward(retain_graph=True)
    torch.testing.assert_close(seg.grad[0,0,0,0] / seg.grad[0,0,0,3], torch.tensor(4.))
    torch.testing.assert_close(bce, torch.tensor(np.log(2), dtype=torch.float32))
    spatial.backward()
    assert sp.grad[0,0,1,1] < 0  # .8 target must not be attenuated to .2
    assert sp.grad[0,0,0,1] > 0
    for a,b in zip(auxiliary_losses(seg,sp,y,roi=roi,outside_weight=1), auxiliary_losses(seg,sp,y)):
        torch.testing.assert_close(a,b)
    assert auxiliary_losses(seg,sp,y,roi=roi,outside_weight=.25,positive=torch.tensor([False]))[1] == 0
