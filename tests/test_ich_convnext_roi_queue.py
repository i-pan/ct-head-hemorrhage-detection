"""Matched settings and mandatory post-training evaluations for both ROI arms."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def load_queue(tmp_path, monkeypatch):
    source = (
        Path(__file__).resolve().parents[1] / "scripts/run_ich_convnext_roi_ablation.py"
    )
    spec = importlib.util.spec_from_file_location("roi_queue", source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "OUT", tmp_path / "logs")
    (tmp_path / "scripts").mkdir()
    for name in [
        "run_ich_convnext_roi_ablation.py",
        "check_ich_auxiliary.py",
        "check_ich_auxiliary_data.py",
        "evaluate_ich_auxiliary_bhsd.py",
        "export_ich_checkpoint_predictions.py",
        "analyze_ich_backbone_ensemble.py",
    ]:
        (tmp_path / "scripts" / name).write_text("")
    monkeypatch.setattr(
        m, "__file__", str(tmp_path / "scripts/run_ich_convnext_roi_ablation.py")
    )
    for folder, n in [
        ("rsna_three_family_final_20261110", 91744),
        ("rsna_three_family_bhsd_related_20261111", 709),
    ]:
        p = tmp_path / "data/pseudolabels" / folder
        p.mkdir(parents=True)
        (p / "completion.json").write_text(
            json.dumps(
                dict(status="complete", target_slices=n, export_signature="test")
            )
        )
        (p / "training_cache.json").write_text(
            json.dumps(dict(export_signature="test"))
        )
    return m


def test_queue_preserves_baseline_and_evaluates_each_run(tmp_path, monkeypatch):
    from skp.train import parse_args, validate_trainer_args
    import torch

    m = load_queue(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: True)

    def run(command, **kwargs):
        calls.append(command)
        if "-m" in command:
            monkeypatch.setattr("sys.argv", ["train", *command[3:]])
            args, _ = parse_args()
            validate_trainer_args(args)
            assert (
                args.strategy == "ddp"
                and args.devices == 2
                and not args.no_sync_batchnorm
            )
            assert args.check_val_every_n_epoch == 5
            assert (
                "--accumulate_grad_batches" not in command
                and "--batch_size" not in command
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(m.subprocess, "run", run)
    m.main()
    state = json.loads((m.OUT / "queue.json").read_text())
    assert state["status"] == "complete" and len(state["records"]) == 2
    assert all(r["status"] == "complete" for r in state["records"])
    assert sum("-m" in c for c in calls) == 2
    assert sum("scripts/export_ich_checkpoint_predictions.py" in c for c in calls) == 2
    assert sum("scripts/evaluate_ich_auxiliary_bhsd.py" in c for c in calls) == 2
    assert [c[1] for c in calls][-1] == "scripts/evaluate_ich_auxiliary_bhsd.py"


def test_missing_supplement_blocks_training(tmp_path, monkeypatch):
    m = load_queue(tmp_path, monkeypatch)
    (
        tmp_path
        / "data/pseudolabels/rsna_three_family_bhsd_related_20261111/completion.json"
    ).unlink()
    monkeypatch.setattr(
        m.subprocess, "run", lambda *a, **k: pytest.fail("Must not launch any job")
    )
    with pytest.raises(FileNotFoundError):
        m.main()
