from pathlib import Path
import sys
import json
import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from export_ich_checkpoint_predictions import export_both  # noqa: E402


def test_both_and_latest_not_filename_or_best(tmp_path, monkeypatch):
    import export_ich_checkpoint_predictions as module

    directory = tmp_path / "fold0/checkpoints"
    directory.mkdir(parents=True)
    for name, epoch, step in [("best", 1, 20), ("last", 4, 50), ("last-v1", 2, 30)]:
        torch.save(
            dict(epoch=epoch, global_step=step, state_dict={}),
            directory / f"{name}.ckpt",
        )
    calls = []

    def export(config, checkpoint, output, args):
        calls.append(checkpoint.name)
        np.savez_compressed(
            output,
            logits=np.zeros((2, 6)),
            targets=np.zeros((2, 6)),
            groups=np.arange(2),
        )

    monkeypatch.setattr(module, "export_predictions", export)
    export_both("cfg", tmp_path, None)
    assert calls == ["best.ckpt", "last.ckpt"]
    for role, step in [("best", 20), ("final", 50)]:
        metadata = json.loads(
            (tmp_path / f"validation_predictions_{role}.json").read_text()
        )
        assert metadata["global_step"] == step
        assert (tmp_path / f"validation_predictions_{role}.npz").exists()
    export_both("cfg", tmp_path, None)
    assert len(calls) == 2
    torch.save(dict(epoch=5, global_step=60, state_dict={}), directory / "last.ckpt")
    export_both("cfg", tmp_path, None)
    assert len(calls) == 3
    assert (
        json.loads((tmp_path / "validation_predictions_final.json").read_text())[
            "global_step"
        ]
        == 60
    )


def test_missing_latest_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="No latest checkpoint"):
        export_both("cfg", tmp_path, None)
