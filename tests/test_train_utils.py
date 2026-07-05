import os
import re
from argparse import Namespace

import pytest

from skp.configs import Config
from skp.train import (
    format_tracking_uri_for_log,
    get_mlflow_run_name,
    generate_random_run_id,
    get_split_save_name,
    parse_limit_batches,
    validate_trainer_args,
)


def test_generate_random_run_id_uses_petname_format():
    run_id = generate_random_run_id()

    assert re.fullmatch(r"[a-z]+-[a-z]+-\d{4}", run_id)


def test_get_split_save_name_supports_fold_and_fixed_splits():
    assert get_split_save_name(Config(fold=2)) == "fold2"
    assert get_split_save_name(Config()) == "fixed_split"


def test_get_mlflow_run_name_includes_config():
    cfg = Config(config="ich_baseline", run_id="silver-lake-0427")

    assert get_mlflow_run_name(cfg) == "ich_baseline/silver-lake-0427"


def test_parse_limit_batches_preserves_int_counts_and_float_percentages():
    assert parse_limit_batches("1") == 1
    assert parse_limit_batches("5") == 5
    assert parse_limit_batches("0.25") == 0.25
    assert parse_limit_batches("1.0") == 1.0
    assert parse_limit_batches("5.0") == 5
    with pytest.raises(Exception, match="floats must be in"):
        parse_limit_batches("1.5")


def _trainer_args(**overrides):
    args = Namespace(
        accelerator="cuda",
        devices=1,
        strategy="ddp",
        no_sync_batchnorm=False,
    )
    args.__dict__.update(overrides)
    return args


def test_validate_trainer_args_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA training was requested"):
        validate_trainer_args(_trainer_args())


def test_validate_trainer_args_rejects_too_many_cuda_devices(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr("torch.distributed.is_nccl_available", lambda: True)

    with pytest.raises(ValueError, match="only 1 CUDA"):
        validate_trainer_args(_trainer_args(devices=2))


def test_validate_trainer_args_resolves_all_visible_cuda_devices(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)
    monkeypatch.setattr("torch.distributed.is_nccl_available", lambda: True)
    args = _trainer_args(devices=-1)

    validate_trainer_args(args)

    assert args.devices == 4


def test_validate_trainer_args_resolves_non_cuda_all_devices_to_one():
    args = _trainer_args(
        accelerator="cpu",
        devices=-1,
        no_sync_batchnorm=True,
    )

    validate_trainer_args(args)

    assert args.devices == 1


def test_validate_trainer_args_requires_no_sync_batchnorm_for_cpu():
    with pytest.raises(ValueError, match="--no_sync_batchnorm"):
        validate_trainer_args(_trainer_args(accelerator="cpu"))


def test_validate_trainer_args_allows_cpu_ddp_without_sync_batchnorm():
    validate_trainer_args(_trainer_args(accelerator="cpu", no_sync_batchnorm=True))


def test_format_tracking_uri_for_log_strips_embedded_credentials():
    assert (
        format_tracking_uri_for_log("https://user:pass@mlflow.example.com:5000/path")
        == "https://mlflow.example.com:5000/path"
    )
    assert format_tracking_uri_for_log("sqlite:///mlflow.db") == "sqlite:///mlflow.db"
    assert format_tracking_uri_for_log(None) is None


def test_environment_file_is_gitignored():
    assert os.path.exists(".env.example")
