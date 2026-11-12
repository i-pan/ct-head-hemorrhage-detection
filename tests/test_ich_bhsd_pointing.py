"""Pointing distinguishes strict pixel hits from coarse-cell and distance tolerance."""

import importlib.util
from pathlib import Path
import numpy as np
import torch


def test_pointing_geometry_and_negative_slice():
    source = (
        Path(__file__).resolve().parents[1] / "scripts/evaluate_ich_auxiliary_bhsd.py"
    )
    import sys

    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("bhsd_eval", source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    scores = torch.zeros(3, 1, 2, 2)
    scores[:, :, 0, 0] = 10
    target = torch.zeros(3, 1, 32, 32, dtype=torch.bool)
    target[0, 0, 0, 0] = True  # first pixel in tied maximum plateau
    target[1, 0, 0, 3] = True  # same coarse cell, 3mm away at unit spacing
    rows = m.pointing(scores, target, np.ones((3, 2)))
    assert rows[0]["point_hit"] and rows[0]["coarse_cell_hit"]
    assert not rows[1]["point_hit"] and rows[1]["coarse_cell_hit"]
    assert rows[1]["distance_mm"] == 3 and rows[1]["within_5mm"]
    assert rows[2]["point_hit"] is None and rows[2]["distance_mm"] is None
