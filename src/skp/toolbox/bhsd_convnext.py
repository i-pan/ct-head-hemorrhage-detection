from skp.toolbox.ich_paths import path as configured_path, classifier, assert_teacher_isolation
"""Binary BHSD segmentation: classified ConvNeXt-S16 encoder and U-Net decoder."""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import timm
import albumentations as A
from skp.configs.rsna_ich_convnext_tiny_logitmax_s16_series_sampling import cfg
from skp.models.segmentation.decoders.unet import DecoderBlock

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = configured_path("ICH_BHSD_MANIFEST", ROOT / "data/bhsd/bhsd_segmentation_slices.csv")
CACHE = configured_path("ICH_BHSD_HU_CACHE", ROOT / "data/bhsd/hu_volumes")
CHECKPOINT = (
    ROOT
    / "experiments/rsna_ich_convnext_tiny_logitmax_s16_series_sampling/series_e050_p0.5_seed88_20261109_convnext_tiny_spatial_s16/fold0/checkpoints/last.ckpt"
)
CHECKPOINT = classifier("convnext_tiny", CHECKPOINT)
THRESHOLDS = np.arange(1, 10) / 10


def window(hu):
    windows = np.asarray(cfg.ct_windows, dtype=np.float32)
    low = windows[:, 0] - windows[:, 1] / 2
    return (
        np.clip(hu[None], low[:, None, None], (low + windows[:, 1])[:, None, None])
        - low[:, None, None]
    ) / (windows[:, 1, None, None] + 1e-6)


def prepare_cache():
    import nibabel as nib
    import cv2
    import json

    df = pd.read_csv(MANIFEST)
    assert df.groupby("patient_id").fold.nunique().max() == 1
    splits = pd.read_csv(configured_path("ICH_SPLITS", ROOT / "data/folds/rsna_ich_fixed_val_splits.csv"))
    assert_teacher_isolation(df, splits)
    excluded = set(
        zip(
            splits.loc[splits.excluded_bhsd, "patient_id"],
            splits.loc[splits.excluded_bhsd, "study_id"],
        )
    )
    assert all((r.patient_id, r.study_id) in excluded for r in df.itertuples())
    CACHE.mkdir(exist_ok=True, parents=True)
    checks = []
    for uid, rows in df.groupby("series_uid"):
        path = CACHE / f"{uid}.npy"
        if not path.exists():
            raw = nib.load(
                str(configured_path("ICH_BHSD_ROOT", "/mnt/stor/datasets/BHSD") / "label_192/images" / f"{uid}.nii.gz")
            ).get_fdata(dtype=np.float32)
            raw = np.rot90(raw[:, :, ::-1], axes=(0, 1))[:, ::-1]
            volume = np.ascontiguousarray(raw.transpose(2, 0, 1))
            assert volume.shape == (len(rows), 512, 512)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, volume)
            tmp.replace(path)
        volume = np.load(path, mmap_mode="r")
        mask = np.load(rows.iloc[0].mask_volume_path, mmap_mode="r")
        assert mask.shape == (512, 512, len(volume))
        for r in rows.itertuples():
            assert bool((mask[..., r.slice_index] > 0).any()) == bool(r.has_mask)
        row = rows.iloc[len(rows) // 2]
        png = cv2.imread(row.image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        error = np.abs(
            window(volume[int(row.slice_index)]).transpose(1, 2, 0) - png
        ).mean(axis=(0, 1))
        assert error.max() < 0.02, (uid, error)
        checks.append(
            dict(series_uid=uid, png_window_mae=error.tolist(), slices=len(rows))
        )
    (CACHE / "checks.json").write_text(json.dumps(checks, indent=2))
    return checks


class Dataset(torch.utils.data.Dataset):
    def __init__(self, fold, train, include_negatives=False):
        df = pd.read_csv(MANIFEST)
        self.df = df[
            (df.fold != fold) & (True if include_negatives else df.has_mask)
            if train
            else df.fold == fold
        ].reset_index(drop=True)
        self.train = train
        # Match classification augmentations except label-inconsistent cutout.
        choices = [
            t
            for t in cfg.train_transforms.transforms[0].transforms
            if not isinstance(t, A.CoarseDropout)
        ]
        self.augment = (
            A.Compose(
                [
                    A.OneOf(choices, p=1),
                    A.HorizontalFlip(p=0.25),
                    A.VerticalFlip(p=0.25),
                ],
                seed=88,
            )
            if train
            else None
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        hu = np.load(CACHE / f"{r.series_uid}.npy", mmap_mode="r")
        z = int(r.slice_index)
        frames = [
            window(hu[j]) if 0 <= j < len(hu) else np.zeros((3, 512, 512), np.float32)
            for j in (z - 1, z, z + 1)
        ]
        if self.train and np.random.random() < 0.25:
            frames.reverse()
        x = np.concatenate(frames).transpose(1, 2, 0)
        y = (np.load(r.mask_volume_path, mmap_mode="r")[..., z] > 0).astype(np.float32)[
            ..., None
        ]
        if self.augment:
            out = self.augment(image=x, mask=y)
            x, y = out["image"], out["mask"]
        return (
            torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1))),
            i,
        )


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            cfg.backbone,
            pretrained=False,
            num_classes=0,
            global_pool="",
            in_chans=9,
            output_stride=16,
        )
        checkpoint = torch.load(
            CHECKPOINT, map_location="cpu", weights_only=False, mmap=True
        )
        assert checkpoint["epoch"] == 49 and checkpoint["global_step"] > 0
        state = {
            k.removeprefix("model.backbone."): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.backbone.")
        }
        self.encoder.load_state_dict(state, strict=True)
        # Actual skips are 128,64,32,32: first decoder fusion stays at 32.
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(768, 384, 256),
                DecoderBlock(256, 192, 128),
                DecoderBlock(128, 96, 64),
                DecoderBlock(64, 0, 32),
                DecoderBlock(32, 0, 16),
            ]
        )
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        last, skips = self.encoder.forward_intermediates(x * 2 - 1)
        x = self.encoder.forward_head(last)  # retain trained spatial head normalization
        for j, block in enumerate(self.decoder):
            x = block(x, skips[-2 - j] if j < 3 else None)
        return self.head(x)


def loss_parts(logits, target, dice_reduction="slice"):
    if dice_reduction not in ("slice", "batch"):
        raise ValueError(dice_reduction)
    logits, target = logits.float(), target.float()
    prob = logits.sigmoid()
    axes = (1, 2, 3) if dice_reduction == "slice" else (0, 1, 2, 3)
    dice = (
        1
        - (2 * (prob * target).sum(axes) + 1e-5)
        / (prob.sum(axes) + target.sum(axes) + 1e-5)
    ).mean()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return dice + bce, dice, bce


def learning_rates(step, total):
    """One-based update number; reach peak after 5% warmup and floor on last update."""
    warmup = max(1, math.ceil(total * 0.05))
    if step <= warmup:
        lr = 3e-4 * step / warmup
    else:
        progress = (step - warmup) / (total - warmup)
        lr = 1e-6 + (3e-4 - 1e-6) * 0.5 * (1 + math.cos(math.pi * progress))
    return lr / 10, lr


def summarize_counts(counts, groups):
    # N x thresholds x [intersection,predicted,target], retains all negative slices.
    target = counts[:, 0, 2]
    positive = target > 0
    dice = 2 * counts[:, :, 0] / np.maximum(counts[:, :, 1] + counts[:, :, 2], 1)
    slice_dice = dice[positive].mean(0)
    volume = []
    for uid in np.unique(groups):
        c = counts[np.asarray(groups) == uid].sum(0)
        volume.append(2 * c[:, 0] / np.maximum(c[:, 1] + c[:, 2], 1))
    volume = np.mean(volume, axis=0)
    best = int(slice_dice.argmax())
    result = dict(
        positive_slice_dice_05=float(slice_dice[4]),
        positive_slice_dice_best=float(slice_dice[best]),
        best_threshold=float(THRESHOLDS[best]),
        volume_dice_05=float(volume[4]),
        volume_dice_at_slice_best=float(volume[best]),
    )
    for j, t in enumerate(THRESHOLDS):
        result[f"positive_slice_dice_{t:.1f}"] = float(slice_dice[j])
        result[f"volume_dice_{t:.1f}"] = float(volume[j])
        result[f"negative_slice_fp_fraction_{t:.1f}"] = (
            float((counts[~positive, j, 1] > 0).mean()) if (~positive).any() else 0.0
        )
    return result


class MixedSliceBatchSampler:
    """Fixed baseline-sized epochs with independently cycling pos/neg bins.

    Full batches have exact requested ratios. The last partial batch keeps the
    baseline sample count, rounding its positive count to the nearest integer.
    Bins persist across epochs and reshuffle only after exhaustion.
    """

    def __init__(self, labels, positive_fraction, samples_per_epoch, seed=88):
        if positive_fraction not in (0.5, 0.75):
            raise ValueError(positive_fraction)
        labels = np.asarray(labels, dtype=bool)
        self.bins = [np.flatnonzero(~labels), np.flatnonzero(labels)]
        if any(len(b) == 0 for b in self.bins):
            raise ValueError("Both label bins must be nonempty")
        self.fraction = positive_fraction
        self.samples = samples_per_epoch
        self.rngs = [
            np.random.default_rng(seed + 100),
            np.random.default_rng(seed + 200),
        ]
        self.pools = [np.array([], dtype=int), np.array([], dtype=int)]
        self.positions = [0, 0]

    def __len__(self):
        return math.ceil(self.samples / 32)

    def draw(self, bin_id, count):
        values = []
        while count:
            if self.positions[bin_id] == len(self.pools[bin_id]):
                self.pools[bin_id] = self.rngs[bin_id].permutation(self.bins[bin_id])
                self.positions[bin_id] = 0
            start = self.positions[bin_id]
            take = min(count, len(self.pools[bin_id]) - start)
            values.extend(self.pools[bin_id][start : start + take].tolist())
            self.positions[bin_id] += take
            count -= take
        return values

    def __iter__(self):
        for start in range(0, self.samples, 32):
            size = min(32, self.samples - start)
            positive = int(math.floor(size * self.fraction + 0.5))
            batch = self.draw(1, positive) + self.draw(0, size - positive)
            # Preserve independent-bin visitation order for cycle auditing.
            yield batch

    def state_dict(self):
        return dict(
            pools=self.pools,
            positions=self.positions,
            rngs=[r.bit_generator.state for r in self.rngs],
        )
