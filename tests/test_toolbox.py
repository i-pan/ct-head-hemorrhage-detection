import numpy as np
import pytest
import torch

import skp.toolbox.dicom as dicom_utils
from skp.toolbox.cropping import create_overlapping_chunks, crop_3d_with_center
from skp.toolbox.dicom import _read_pixel_worker, apply_windowing
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


def test_dicom_pixel_worker_applies_rescale_values():
    class FakeDataset:
        pixel_array = np.asarray([[0, 100], [200, 300]], dtype=np.int16)

    class FakePydicom:
        @staticmethod
        def dcmread(path):
            return FakeDataset()

    metadata = {
        "path": "slice.dcm",
        "RescaleSlope": 2.0,
        "RescaleIntercept": -1000.0,
    }
    image = _read_pixel_worker(
        (metadata, "pydicom", FakePydicom(), None, True, (2, 2), "crop_pad")
    )

    assert np.array_equal(
        image,
        np.asarray([[-1000, -800], [-600, -400]], dtype=np.float32),
    )


def test_dicom_series_can_require_complete_rescale_metadata(tmp_path, monkeypatch):
    for index in range(2):
        (tmp_path / f"slice{index}.dcm").touch()

    def fake_metadata(path, backend, pydicom, dicomsdl):
        index = int(path[-5])
        return {
            "path": path,
            "InstanceNumber": index + 1,
            "ImagePositionPatient": [0, 0, index],
            "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
            "PixelSpacing": [1, 1],
            "Rows": 2,
            "Columns": 2,
            "SeriesInstanceUID": "1.2.3",
            "Modality": "CT",
            "ImageType": ["ORIGINAL", "PRIMARY", "AXIAL"],
            "RescaleSlope": 1.0,
            "RescaleIntercept": -1024.0 if index == 0 else None,
            "NumberOfFrames": None,
            "PixelRepresentation": 0,
        }

    monkeypatch.setattr(dicom_utils, "_read_metadata_worker", fake_metadata)

    with pytest.raises(ValueError, match="missing RescaleSlope or RescaleIntercept"):
        dicom_utils.load_dicom_series(
            str(tmp_path),
            require_rescale_values=True,
            max_workers=1,
        )


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
