"""General image and array helpers used during exploration."""

from __future__ import annotations

import re
from typing import Sequence

import cv2
import numpy as np
import torch


def window_ct(x: np.ndarray, window_level: float, window_width: float) -> np.ndarray:
    """Apply CT windowing and return an 8-bit image."""
    lower = window_level - window_width / 2
    upper = window_level + window_width / 2
    x = np.clip(x, lower, upper)
    x = (x - lower) / max(window_width, 1e-6)
    return (x * 255.0).astype(np.uint8)


def center_crop_or_pad_borders(
    image: np.ndarray, size: tuple[int, int], pad_val: int | float = 0
) -> np.ndarray:
    """Center crop or pad a 2D image to ``size``."""
    height, width = image.shape[:2]
    new_height, new_width = size
    if new_height < height:
        top = (height - new_height) // 2
        image = image[top : top + new_height]
    elif new_height > height:
        pad_top = (new_height - height) // 2
        pad_bottom = new_height - height - pad_top
        image = np.pad(
            image,
            ((pad_top, pad_bottom), (0, 0)),
            mode="constant",
            constant_values=pad_val,
        )

    if new_width < width:
        left = (width - new_width) // 2
        image = image[:, left : left + new_width]
    elif new_width > width:
        pad_left = (new_width - width) // 2
        pad_right = new_width - width - pad_left
        image = np.pad(
            image,
            ((0, 0), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_val,
        )
    return image


def draw_bounding_boxes(
    img: np.ndarray,
    bboxes: Sequence[Sequence[int]],
    mode: str = "xyxy",
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Draw bounding boxes on a copy of ``img``."""
    if mode not in {"xyxy", "xywh"}:
        raise ValueError("mode must be 'xyxy' or 'xywh'.")
    out = img.copy()
    if out.ndim == 2 or out.shape[2] == 1:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    thickness = max(1, int(0.005 * max(out.shape[:2])))
    for box in bboxes:
        if mode == "xyxy":
            x1, y1, x2, y2 = box
        else:
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
        out = cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def overlay_images(image: np.ndarray, overlay: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """Blend two image arrays with ``alpha`` weight on ``image``."""
    if image.shape != overlay.shape:
        raise ValueError(f"image shape {image.shape} must match overlay {overlay.shape}.")
    return (alpha * image + (1 - alpha) * overlay).astype(np.uint8)


def convert_to_2dc(list_of_files: Sequence[str], size: int = 3) -> list[list[str]]:
    """Convert sorted slice paths to adjacent 2.5D context windows."""
    if size % 2 != 1:
        raise ValueError("size must be odd.")
    if len(list_of_files) == 0:
        return []
    files = list(list_of_files)
    original_length = len(files)
    pad = size // 2
    files = [files[0]] * pad + files + [files[-1]] * pad
    return [files[i : i + size] for i in range(original_length)]


def count_parameters(module: torch.nn.Module, subset: str | None = None) -> int:
    """Count trainable parameters, optionally filtered by parameter name regex."""
    if subset is None:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(
        p.numel()
        for name, p in module.named_parameters()
        if re.search(subset, name) and p.requires_grad
    )
