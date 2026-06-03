"""Mask serialization helpers."""

from __future__ import annotations

import numpy as np


def mask2rle(mask: np.ndarray) -> tuple[str, tuple[int, ...]]:
    """Encode a dense mask as value-length run-length encoding.

    Unlike binary Kaggle RLE helpers, this preserves foreground class values by
    storing ``value length value length ...`` pairs over the flattened array.
    """
    if mask.size == 0:
        return "", mask.shape

    pixels = mask.flatten(order="C")
    run_starts = np.concatenate(([0], np.where(pixels[:-1] != pixels[1:])[0] + 1))
    run_lengths = np.diff(np.concatenate((run_starts, [pixels.size])))
    run_values = pixels[run_starts]
    rle = np.array([run_values, run_lengths], dtype=np.int64).T.flatten()
    return " ".join(rle.astype(str)), mask.shape


def rle2mask(rle: str, shape: tuple[int, ...]) -> np.ndarray:
    """Decode value-length RLE produced by :func:`mask2rle`."""
    if not rle:
        return np.zeros(shape, dtype=np.uint8)
    values_and_lengths = [int(x) for x in rle.split()]
    if len(values_and_lengths) % 2 != 0:
        raise ValueError("RLE must contain value-length pairs.")
    values = values_and_lengths[0::2]
    lengths = values_and_lengths[1::2]
    pixels = np.repeat(values, lengths)
    expected = int(np.prod(shape))
    if pixels.size != expected:
        raise ValueError(f"Decoded RLE has {pixels.size} values, expected {expected}.")
    return pixels.reshape(shape, order="C").astype(np.uint8)
