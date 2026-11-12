"""The handoff must gate training on completion and finish both checkpoint exports."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def queue_module(tmp_path, monkeypatch):
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_ich_auxiliary_after_export.py"
    )
    spec = importlib.util.spec_from_file_location("auxqueue", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path / "logs")
    monkeypatch.setattr(module, "PSEUDO", tmp_path / "pseudo")
    module.PSEUDO.mkdir()
    (tmp_path / "scripts").mkdir()
    for name in [
        "run_ich_auxiliary_after_export.py",
        "check_ich_auxiliary.py",
        "check_ich_auxiliary_data.py",
        "prepare_ich_pseudolabel_training_cache.py",
        "export_ich_checkpoint_predictions.py",
    ]:
        (tmp_path / "scripts" / name).write_text("")
    monkeypatch.setattr(
        module, "__file__", str(tmp_path / "scripts/run_ich_auxiliary_after_export.py")
    )
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return module


def test_no_training_after_failed_export(tmp_path, monkeypatch):
    m = queue_module(tmp_path, monkeypatch)
    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="failed", returncode=3),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Training must not start")

    monkeypatch.setattr(m.subprocess, "Popen", forbidden)
    with pytest.raises(RuntimeError, match="Export stopped"):
        m.main()
    assert json.loads((m.OUT / "queue.json").read_text())["status"] == "failed"


def test_completion_preflights_training_and_both_exports(tmp_path, monkeypatch):
    from skp.train import parse_args, validate_trainer_args
    import torch

    m = queue_module(tmp_path, monkeypatch)
    commands = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: True)

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == "systemctl":
            (m.PSEUDO / "completion.json").write_text(
                json.dumps(dict(status="complete", target_slices=91744, teachers=15))
            )
            return SimpleNamespace(
                stdout="inactive" if len(commands) > 1 else "active", returncode=0
            )
        assert (m.PSEUDO / "completion.json").exists()
        if "-m" in command:
            assert any("check_ich_auxiliary_data.py" in str(c) for c in commands)
            monkeypatch.setattr("sys.argv", ["train", *command[3:]])
            args, _ = parse_args()
            validate_trainer_args(args)
            assert args.devices == 2 and args.strategy == "ddp"
            assert not args.no_sync_batchnorm
            assert "--batch_size" not in command
            assert "--accumulate_grad_batches" not in command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(m.subprocess, "run", run)
    m.main()
    state = json.loads((m.OUT / "queue.json").read_text())
    assert state["status"] == "complete"
    assert all(r["status"] == "complete" for r in state["records"])
    assert state["effective_batch_size"] == 64
    assert state["accumulate_grad_batches"] == 1
    assert state["world_size"] == 2
    assert sum("-m" in c for c in commands) == 4
    assert (
        sum("scripts/export_ich_checkpoint_predictions.py" in c for c in commands) == 4
    )
