import math
import numpy as np
import pandas as pd
import torch
from skp.toolbox.bhsd_convnext import (
    MANIFEST,
    learning_rates,
    loss_parts,
    summarize_counts,
    window,
)


def test_schedule_and_folds():
    df = pd.read_csv(MANIFEST)
    assert df.groupby("patient_id").fold.nunique().max() == 1
    for fold in range(5):
        positive = df[(df.fold != fold) & df.has_mask]
        for passes in (10, 25, 50):
            steps = passes * math.ceil(len(positive) / 32)
            rates = np.array([learning_rates(s, steps) for s in range(1, steps + 1)])
            np.testing.assert_allclose(rates[:, 1], rates[:, 0] * 10)
            np.testing.assert_allclose(rates[-1], [1e-7, 1e-6])
            assert np.isclose(rates[:, 1].max(), 3e-4)
            peak = math.ceil(steps * 0.05) - 1
            assert (np.diff(rates[: peak + 1, 1]) >= 0).all()
            assert (np.diff(rates[peak:, 1]) <= 0).all()
            assert steps * 32 >= len(positive) * passes


def test_positive_only_dice_and_volume_includes_negatives():
    counts = np.zeros((3, 9, 3))
    counts[0, :, 0] = 1
    counts[0, :, 1:] = 2  # positive dice .5
    counts[1, :, 1] = (
        100  # false positive same volume must penalize volume, not primary
    )
    counts[2, :, 0] = 2
    counts[2, :, 1:] = 2  # positive dice1
    m = summarize_counts(counts, np.array(["a", "a", "b"]))
    assert m["positive_slice_dice_05"] == 0.75
    assert np.isclose(m["volume_dice_05"], (2 / 104 + 1) / 2)
    assert m["best_threshold"] == 0.1
    assert m["negative_slice_fp_fraction_0.5"] == 1


def test_loss_and_window():
    logits = torch.zeros((2, 1, 2, 2), requires_grad=True)
    y = torch.tensor([[[[1.0, 0], [0, 0]]], [[[1.0, 1], [1, 1]]]])
    total, dice, bce = loss_parts(logits, y)
    expected = 1 - ((1 + 1e-5) / (3 + 1e-5) + (4 + 1e-5) / (6 + 1e-5)) / 2
    assert np.isclose(dice.item(), expected)
    assert np.isclose(bce.item(), np.log(2))
    total.backward()
    assert torch.isfinite(logits.grad).all()
    result = window(np.array([[0.0, 40.0, 80.0]], np.float32))
    np.testing.assert_allclose(result[0, 0], [0, 0.5, 1], atol=1e-6)


def test_batch_dice_changes_only_dice_reduction():
    logits = torch.zeros((2, 1, 2, 2), requires_grad=True)
    y = torch.tensor([[[[1.0, 0], [0, 0]]], [[[1.0, 1], [1, 1]]]])
    from skp.toolbox.bhsd_convnext import loss_parts

    _, slice_dice, slice_bce = loss_parts(logits, y, "slice")
    total, batch_dice, batch_bce = loss_parts(logits, y, "batch")
    assert torch.equal(slice_bce, batch_bce)
    assert np.isclose(batch_dice.item(), 1 - (5 + 1e-5) / (9 + 1e-5))
    assert not torch.isclose(slice_dice, batch_dice)
    total.backward()
    assert torch.isfinite(logits.grad).all()


def test_mixed_bins_cycle_independently_and_match_baseline():
    from skp.toolbox.bhsd_convnext import MixedSliceBatchSampler

    labels = np.array([True] * 37 + [False] * 81)
    for fraction in (0.5, 0.75):
        a = MixedSliceBatchSampler(labels, fraction, 101)
        b = MixedSliceBatchSampler(labels, fraction, 101)
        histories = {True: [], False: []}
        for epoch in range(5):
            batches = list(a)
            assert batches == list(b)
            assert len(batches) == 4 and [len(x) for x in batches] == [32, 32, 32, 5]
            for batch in batches:
                assert labels[batch].sum() == math.floor(len(batch) * fraction + 0.5)
                for i in batch:
                    histories[bool(labels[i])].append(i)
        for positive, stream in histories.items():
            size = int((labels == positive).sum())
            for start in range(0, len(stream) - size + 1, size):
                assert len(set(stream[start : start + size])) == size


def test_all_folds_match_saved_baseline_budget_and_validation():
    import json
    from skp.toolbox.bhsd_convnext import Dataset, ROOT, MixedSliceBatchSampler

    for fold in range(5):
        base = (
            ROOT
            / "experiments/bhsd_convnext_s16_binary_schedules_20261110/passes_25"
            / f"fold{fold}"
        )
        config = json.loads((base / "config.json").read_text())
        data, val = Dataset(fold, True, include_negatives=True), Dataset(fold, False)
        pd.testing.assert_frame_equal(
            val.df, pd.read_csv(base / "validation_manifest.csv")
        )
        n = int(data.df.has_mask.sum())
        assert n == config["train_positive_slices"]
        assert set(data.df.patient_id).isdisjoint(val.df.patient_id)
        for fraction in (0.5, 0.75):
            sampler = MixedSliceBatchSampler(data.df.has_mask, fraction, n)
            assert len(sampler) == config["steps_per_pass"]
            assert len(sampler) * 25 == config["total_steps"]


def test_evaluation_and_architecture_unchanged_from_baseline_snapshot():
    import ast
    from skp.toolbox.bhsd_convnext import ROOT

    base = (
        ROOT / "experiments/bhsd_convnext_s16_binary_schedules_20261110/passes_25/fold0"
    )

    if not (base / "model_data_source.py").exists():
        import pytest
        pytest.skip("Optional historical snapshot unavailable")

    def definition(path, name):
        tree = ast.parse(path.read_text().replace('checkpoint["global_step"] == 14950', 'checkpoint["global_step"] > 0'))
        return ast.dump(next(n for n in tree.body if getattr(n, "name", None) == name))

    for name in ("Net", "window", "learning_rates", "summarize_counts"):
        assert definition(
            ROOT / "src/skp/toolbox/bhsd_convnext.py", name
        ) == definition(base / "model_data_source.py", name)
    assert definition(
        ROOT / "scripts/train_bhsd_convnext_schedule.py", "validate"
    ) == definition(base / "training_source.py", "validate")
