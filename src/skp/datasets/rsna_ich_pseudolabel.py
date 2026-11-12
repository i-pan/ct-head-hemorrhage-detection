"""Original classification images plus aligned soft masks on training slices."""

import json
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from skp.datasets.rsna_ich_2p5d import Dataset as BaseDataset


def hu_teacher_union(
    windowed, teacher, lower=-20.0, width=200.0, threshold=0.07, radius=3
):
    assert lower < 0 and lower + width > 100
    denominator = np.float32(width) + np.float32(1e-6)
    hu_roi = (windowed >= (np.float32(0) - np.float32(lower)) / denominator) & (
        windowed <= (np.float32(100) - np.float32(lower)) / denominator
    )
    teacher_roi = (teacher >= threshold).astype(np.uint8)
    if radius:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        teacher_roi = cv2.dilate(teacher_roi, kernel)
    return (hu_roi | teacher_roi.astype(bool)).astype(np.float32)


class Dataset(BaseDataset):
    def __init__(self, cfg, mode):
        super().__init__(cfg, mode)
        self.probabilities = None
        self.supplemental_probabilities = None
        self.use_roi = cfg.get("auxiliary_region", "none") == "hu_teacher_union"
        assert cfg.get("auxiliary_region", "none") in ["none", "hu_teacher_union"]
        if mode == "train":
            root = Path(cfg.pseudolabel_dir)
            completion = json.loads((root / "completion.json").read_text())
            cache = json.loads((root / "training_cache.json").read_text())
            assert completion["status"] == "complete"
            assert cache["export_signature"] == completion["export_signature"]
            manifest = pd.read_csv(root / "manifest.csv")
            if cfg.get("pseudolabel_supplement_dir"):
                supplemental = Path(cfg.pseudolabel_supplement_dir)
                extra_complete = json.loads((supplemental / "completion.json").read_text())
                extra_cache = json.loads((supplemental / "training_cache.json").read_text())
                assert extra_complete["status"] == "complete"
                assert extra_cache["export_signature"] == extra_complete["export_signature"]
                assert (
                    json.loads((root / "provenance.json").read_text())["teachers"]
                    == json.loads((supplemental / "provenance.json").read_text())[
                        "teachers"
                    ]
                )
                extra = pd.read_csv(supplemental / "manifest.csv")
            else:
                extra = manifest.iloc[:0].copy()
            self.primary_count = len(manifest)
            extra["target_index"] += self.primary_count
            combined = pd.concat([manifest, extra], ignore_index=True)
            assert combined.slice_path.is_unique
            mapping = combined.set_index("slice_path").target_index
            self.pseudo_indices = (
                self.df.slice_path.map(mapping).fillna(-1).to_numpy(np.int64)
            )
            assert np.array_equal(
                self.pseudo_indices >= 0, self.df["any"].to_numpy() > 0
            )
            assert len(combined) == (self.pseudo_indices >= 0).sum()

    def __getitem__(self, index):
        if self.mode != "train":
            return super().__getitem__(index)
        if self.probabilities is None:
            self.probabilities = np.load(
                Path(self.cfg.pseudolabel_dir) / "training_probabilities.npy",
                mmap_mode="r",
            )
        row = self.df.iloc[index]
        frames = [
            self._load_windowed_slice(
                path, slope=row.rescale_slope, intercept=row.rescale_intercept
            )
            for path in [row.prev_slice_path, row.slice_path, row.next_slice_path]
        ]
        x = np.concatenate(frames, axis=0).transpose(1, 2, 0)
        pi = self.pseudo_indices[index]
        if pi < 0:
            mask = np.zeros(self.image_size, np.float32)
        elif pi < len(self.probabilities):
            mask = self.probabilities[pi].astype(np.float32)
        else:
            if self.supplemental_probabilities is None:
                self.supplemental_probabilities = np.load(
                    Path(self.cfg.pseudolabel_supplement_dir)
                    / "training_probabilities.npy",
                    mmap_mode="r",
                )
            mask = self.supplemental_probabilities[pi - len(self.probabilities)].astype(
                np.float32
            )
        if getattr(self, "use_roi", False):
            roi = hu_teacher_union(
                x[..., len(self.windows) + 1],
                mask,
                lower=self.window_lowers[1],
                width=self.window_widths[1],
                threshold=self.cfg.auxiliary_roi_threshold,
                radius=self.cfg.auxiliary_roi_radius,
            )
        else:
            roi = np.ones(self.image_size, np.float32)
        masks = np.stack([mask, roi], axis=-1)
        if self.transforms is not None:
            transformed = self.transforms(image=x, mask=masks)
            x, masks = transformed["image"], transformed["mask"]
        mask, roi = masks[..., 0], (masks[..., 1] >= 0.5).astype(np.float32)
        if not getattr(self, "use_roi", False):
            roi = np.ones_like(mask)
        x = x.reshape(*x.shape[:2], self.depth, len(self.windows))
        if np.random.random() < self.depth_flip_p:
            x = x[:, :, ::-1, :]
        if np.random.random() < self.horizontal_flip_p:
            x, mask, roi = x[:, ::-1], mask[:, ::-1], roi[:, ::-1]
        if np.random.random() < self.vertical_flip_p:
            x, mask, roi = x[::-1], mask[::-1], roi[::-1]
        x = x.transpose(2, 3, 0, 1).reshape(9, *x.shape[:2])
        return dict(
            x=torch.from_numpy(np.ascontiguousarray(x)).float(),
            y=torch.from_numpy(row[self.label_columns].to_numpy(np.float32)),
            segmentation_target=torch.from_numpy(
                np.ascontiguousarray(mask[None])
            ).float(),
            auxiliary_roi=torch.from_numpy(np.ascontiguousarray(roi[None])).float(),
            index=torch.tensor(index),
            group_index=torch.tensor(self.series_index[row.series_uid]).long(),
        )
