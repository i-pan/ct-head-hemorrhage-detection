import numpy as np
import pytest
import torch

from skp.toolbox.cropping import create_overlapping_chunks, crop_3d_with_center
from skp.toolbox.dicom import apply_windowing
from skp.toolbox.images import center_crop_or_pad_borders, convert_to_2dc, window_ct
from skp.toolbox.masks import mask2rle, rle2mask


def test_crop_3d_with_center_returns_requested_shape():
    image = np.ones((4, 5, 6), dtype=np.float32)
    mask = np.zeros((4, 5, 6), dtype=np.uint8)
    mask[2, 2, 3] = 1

    cropped_image, cropped_mask = crop_3d_with_center(
        image, mask, crop_size=(3, 3, 3), crop_mode="foreground"
    )

    assert cropped_image.shape == (3, 3, 3)
    assert cropped_mask.shape == (3, 3, 3)
    assert cropped_mask.sum() == 1


def test_create_overlapping_chunks_covers_small_volume():
    volume = np.ones((2, 2, 2), dtype=np.float32)
    chunks = create_overlapping_chunks(volume, crop_size=(4, 4, 4))

    assert chunks.shape == (1, 4, 4, 4)
    assert chunks[0, :2, :2, :2].sum() == pytest.approx(8.0)


def test_windowing_numpy_and_torch_agree():
    image = np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
    numpy_windowed = apply_windowing(image, (0, 1000), to_uint8=False)
    torch_windowed = apply_windowing(torch.from_numpy(image), (0, 1000), to_uint8=False)

    assert np.allclose(numpy_windowed, torch_windowed.numpy())


def test_mask_rle_roundtrip_preserves_values():
    mask = np.array([[[0, 1], [1, 2]]], dtype=np.uint8)
    rle, shape = mask2rle(mask)

    assert np.array_equal(rle2mask(rle, shape), mask)


def test_image_helpers_are_shape_stable():
    image = np.arange(9, dtype=np.float32).reshape(3, 3)

    assert center_crop_or_pad_borders(image, (5, 5)).shape == (5, 5)
    assert center_crop_or_pad_borders(image, (2, 2)).shape == (2, 2)
    assert window_ct(image, 4, 4).dtype == np.uint8
    assert convert_to_2dc(["a", "b", "c"], size=3) == [
        ["a", "a", "b"],
        ["a", "b", "c"],
        ["b", "c", "c"],
    ]
