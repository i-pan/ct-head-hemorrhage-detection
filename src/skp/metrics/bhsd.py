"""BHSD segmentation metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from scipy.ndimage import binary_erosion, distance_transform_edt

from skp.configs import Config


DEFAULT_LABEL_COLUMNS = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]


def _distributed_sum(x: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return x
    device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else x.device
    reduced = x.to(device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced.cpu()


def _all_gather_objects(obj):
    if not dist.is_available() or not dist.is_initialized():
        return [obj]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def _binary_dice(pred: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(pred, target).sum()
    denominator = pred.sum() + target.sum()
    return float((2.0 * intersection) / denominator) if denominator else math.nan


def _empty_case_hd95(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> float:
    if not pred.any() and not target.any():
        return math.nan
    extents = [(size - 1) * spacing[idx] for idx, size in enumerate(pred.shape)]
    return float(math.sqrt(sum(extent * extent for extent in extents)))


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    eroded = binary_erosion(mask, border_value=0)
    surface = np.logical_xor(mask, eroded)
    return surface if surface.any() else mask


def _hd95(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]) -> float:
    if not pred.any() or not target.any():
        return _empty_case_hd95(pred, target, spacing)

    pred_surface = _surface(pred)
    target_surface = _surface(target)
    target_distance = distance_transform_edt(~target_surface, sampling=spacing)
    pred_distance = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate(
        [
            target_distance[pred_surface],
            pred_distance[target_surface],
        ]
    )
    if distances.size == 0:
        return math.nan
    return float(np.percentile(distances, 95))


def _mean_or_nan(total: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(total, torch.nan, dtype=torch.float32)
    valid = count > 0
    out[valid] = total[valid].float() / count[valid].float()
    return out


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}".rstrip("0").rstrip(".").replace(".", "_")


def _best_value_and_threshold(
    values: torch.Tensor,
    thresholds: list[float],
    *,
    mode: str = "max",
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = ~torch.isnan(values)
    if not bool(valid.any()):
        nan = torch.tensor(math.nan, dtype=torch.float32)
        return nan, nan.clone()
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if mode == "min":
        best_valid_idx = torch.argmin(values[valid_indices])
    else:
        best_valid_idx = torch.argmax(values[valid_indices])
    best_idx = valid_indices[best_valid_idx]
    return values[best_idx], torch.tensor(thresholds[int(best_idx)], dtype=torch.float32)


class SliceDiceHD95(nn.Module):
    """Per-slice Dice and HD95, excluding empty/empty class samples."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.class_names = cfg.get("label_columns") or DEFAULT_LABEL_COLUMNS
        self.threshold = float(cfg.get("metric_threshold") or 0.5)
        self.thresholds = [float(th) for th in (cfg.get("metric_thresholds") or [])]
        self.any_idx = self.class_names.index("any")
        self.reset()

    def reset(self) -> None:
        num_classes = len(self.class_names)
        self.dice_sum = torch.zeros(num_classes, dtype=torch.float64)
        self.dice_count = torch.zeros(num_classes, dtype=torch.float64)
        self.hd95_sum = torch.zeros(num_classes, dtype=torch.float64)
        self.hd95_count = torch.zeros(num_classes, dtype=torch.float64)
        self.any_sweep_dice_sum = torch.zeros(len(self.thresholds), dtype=torch.float64)
        self.any_sweep_dice_count = torch.zeros(len(self.thresholds), dtype=torch.float64)

    def update(self, out: Dict, batch: Dict) -> None:
        prob = out["logits"].detach().sigmoid().cpu().numpy()
        pred = prob >= self.threshold
        target = (batch["y"].detach() >= 0.5).cpu().numpy()
        spacing = batch["spacing_2d"].detach().cpu().numpy()

        for sample_idx in range(pred.shape[0]):
            sample_spacing = tuple(float(v) for v in spacing[sample_idx])
            for class_idx in range(pred.shape[1]):
                p = pred[sample_idx, class_idx].astype(bool)
                t = target[sample_idx, class_idx].astype(bool)
                if not p.any() and not t.any():
                    continue
                dice = _binary_dice(p, t)
                if not math.isnan(dice):
                    self.dice_sum[class_idx] += dice
                    self.dice_count[class_idx] += 1
                hd95 = _hd95(p, t, sample_spacing)
                if not math.isnan(hd95):
                    self.hd95_sum[class_idx] += hd95
                    self.hd95_count[class_idx] += 1
            any_target = target[sample_idx, self.any_idx].astype(bool)
            for threshold_idx, threshold in enumerate(self.thresholds):
                any_pred = prob[sample_idx, self.any_idx] >= threshold
                if not any_pred.any() and not any_target.any():
                    continue
                dice = _binary_dice(any_pred, any_target)
                if not math.isnan(dice):
                    self.any_sweep_dice_sum[threshold_idx] += dice
                    self.any_sweep_dice_count[threshold_idx] += 1

    def compute(self) -> Dict[str, torch.Tensor]:
        dice = _mean_or_nan(
            _distributed_sum(self.dice_sum),
            _distributed_sum(self.dice_count),
        )
        hd95 = _mean_or_nan(
            _distributed_sum(self.hd95_sum),
            _distributed_sum(self.hd95_count),
        )

        metrics = {}
        for idx, name in enumerate(self.class_names):
            metrics[f"slice_dice_{name}"] = dice[idx]
            metrics[f"slice_hd95_{name}"] = hd95[idx]
        metrics["slice_dice_mean"] = torch.nanmean(dice)
        metrics["slice_hd95_mean"] = torch.nanmean(hd95)
        if self.thresholds:
            any_sweep_dice = _mean_or_nan(
                _distributed_sum(self.any_sweep_dice_sum),
                _distributed_sum(self.any_sweep_dice_count),
            )
            for idx, threshold in enumerate(self.thresholds):
                threshold_key = _threshold_key(threshold)
                metrics[f"slice_dice_any_t{threshold_key}"] = any_sweep_dice[idx]
            best_dice, best_threshold = _best_value_and_threshold(
                any_sweep_dice,
                self.thresholds,
            )
            metrics["slice_dice_any_best"] = best_dice
            metrics["slice_dice_any_best_threshold"] = best_threshold
        return metrics


class VolumeDiceHD95(nn.Module):
    """Per-series 3D Dice and HD95 from packed validation slice predictions."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.class_names = cfg.get("label_columns") or DEFAULT_LABEL_COLUMNS
        self.threshold = float(cfg.get("metric_threshold") or 0.5)
        self.thresholds = [float(th) for th in (cfg.get("metric_thresholds") or [])]
        self.any_idx = self.class_names.index("any")
        self.reset()

    def reset(self) -> None:
        self.entries = []

    def update(self, out: Dict, batch: Dict) -> None:
        prob = out["logits"].detach().sigmoid().cpu().numpy()
        pred = prob >= self.threshold
        target = (batch["y"].detach() >= 0.5).cpu().numpy()
        group_index = batch["group_index"].detach().cpu().numpy()
        slice_index = batch["slice_index"].detach().cpu().numpy()
        spacing = batch["spacing_3d"].detach().cpu().numpy()
        batch_size, num_classes, height, width = pred.shape

        for sample_idx in range(batch_size):
            pred_pack = np.packbits(pred[sample_idx].reshape(num_classes, -1), axis=1)
            target_pack = np.packbits(
                target[sample_idx].reshape(num_classes, -1), axis=1
            )
            any_sweep_pack = None
            if self.thresholds:
                any_sweep_pred = np.stack(
                    [
                        prob[sample_idx, self.any_idx] >= threshold
                        for threshold in self.thresholds
                    ],
                    axis=0,
                )
                any_sweep_pack = np.packbits(
                    any_sweep_pred.reshape(len(self.thresholds), -1),
                    axis=1,
                ).tobytes()
            self.entries.append(
                {
                    "group_index": int(group_index[sample_idx]),
                    "slice_index": int(slice_index[sample_idx]),
                    "height": int(height),
                    "width": int(width),
                    "num_classes": int(num_classes),
                    "spacing": tuple(float(v) for v in spacing[sample_idx]),
                    "pred": pred_pack.tobytes(),
                    "target": target_pack.tobytes(),
                    "any_sweep_pred": any_sweep_pack,
                }
            )

    def _unpack(self, entry: dict, key: str) -> np.ndarray:
        packed = np.frombuffer(entry[key], dtype=np.uint8)
        count = entry["num_classes"] * entry["height"] * entry["width"]
        return np.unpackbits(packed, count=count).reshape(
            entry["num_classes"], entry["height"], entry["width"]
        ).astype(bool)

    def _unpack_any_sweep(self, entry: dict) -> np.ndarray:
        packed = np.frombuffer(entry["any_sweep_pred"], dtype=np.uint8)
        count = len(self.thresholds) * entry["height"] * entry["width"]
        return np.unpackbits(packed, count=count).reshape(
            len(self.thresholds), entry["height"], entry["width"]
        ).astype(bool)

    def compute(self) -> Dict[str, torch.Tensor]:
        gathered = _all_gather_objects(self.entries)
        entries = [entry for rank_entries in gathered for entry in rank_entries]
        by_group = defaultdict(list)
        for entry in entries:
            by_group[entry["group_index"]].append(entry)

        num_classes = len(self.class_names)
        dice_values = [[] for _ in range(num_classes)]
        hd95_values = [[] for _ in range(num_classes)]
        any_sweep_dice_values = [[] for _ in self.thresholds]

        for group_entries in by_group.values():
            group_entries = sorted(group_entries, key=lambda item: item["slice_index"])
            spacing = group_entries[0]["spacing"]
            pred = np.stack([self._unpack(entry, "pred") for entry in group_entries], axis=-1)
            target = np.stack(
                [self._unpack(entry, "target") for entry in group_entries], axis=-1
            )
            for class_idx in range(num_classes):
                p = pred[class_idx]
                t = target[class_idx]
                if not p.any() and not t.any():
                    continue
                dice = _binary_dice(p, t)
                if not math.isnan(dice):
                    dice_values[class_idx].append(dice)
                hd95 = _hd95(p, t, spacing)
                if not math.isnan(hd95):
                    hd95_values[class_idx].append(hd95)
            if self.thresholds:
                any_target = target[self.any_idx]
                any_sweep_pred = np.stack(
                    [self._unpack_any_sweep(entry) for entry in group_entries],
                    axis=-1,
                )
                for threshold_idx in range(len(self.thresholds)):
                    any_pred = any_sweep_pred[threshold_idx]
                    if not any_pred.any() and not any_target.any():
                        continue
                    dice = _binary_dice(any_pred, any_target)
                    if not math.isnan(dice):
                        any_sweep_dice_values[threshold_idx].append(dice)

        dice = torch.tensor(
            [
                float(np.mean(values)) if values else math.nan
                for values in dice_values
            ],
            dtype=torch.float32,
        )
        hd95 = torch.tensor(
            [
                float(np.mean(values)) if values else math.nan
                for values in hd95_values
            ],
            dtype=torch.float32,
        )
        metrics = {}
        for idx, name in enumerate(self.class_names):
            metrics[f"volume_dice_{name}"] = dice[idx]
            metrics[f"volume_hd95_{name}"] = hd95[idx]
        metrics["volume_dice_mean"] = torch.nanmean(dice)
        metrics["volume_hd95_mean"] = torch.nanmean(hd95)
        if self.thresholds:
            any_sweep_dice = torch.tensor(
                [
                    float(np.mean(values)) if values else math.nan
                    for values in any_sweep_dice_values
                ],
                dtype=torch.float32,
            )
            for idx, threshold in enumerate(self.thresholds):
                threshold_key = _threshold_key(threshold)
                metrics[f"volume_dice_any_t{threshold_key}"] = any_sweep_dice[idx]
            best_dice, best_threshold = _best_value_and_threshold(
                any_sweep_dice,
                self.thresholds,
            )
            metrics["volume_dice_any_best"] = best_dice
            metrics["volume_dice_any_best_threshold"] = best_threshold
        return metrics
