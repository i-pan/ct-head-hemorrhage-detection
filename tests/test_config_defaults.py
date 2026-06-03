import pytest

from skp.configs import Config
from skp.configs.defaults import (
    CONFIG_FIELD_DOCS,
    classification_2d_defaults,
    dataloader_defaults,
    runtime_defaults,
)


def test_config_is_strict_but_get_supports_optionals():
    cfg = Config()

    with pytest.raises(AttributeError):
        _ = cfg.missing

    assert cfg.get("missing") is None
    assert cfg.get("missing", "fallback") == "fallback"


def test_config_require_reports_missing_values():
    cfg = Config(existing=1)

    assert cfg.require("existing") == 1
    with pytest.raises(AttributeError, match="missing"):
        cfg.require("existing", "missing")


def test_default_groups_define_documented_core_values():
    cfg = Config()
    runtime_defaults(cfg)
    dataloader_defaults(cfg)
    classification_2d_defaults(cfg)

    assert cfg.save_dir == "./experiments"
    assert cfg.persistent_workers is True
    assert cfg.mixup is None
    assert "save_dir" in CONFIG_FIELD_DOCS["runtime"]
    assert "num_workers" in CONFIG_FIELD_DOCS["dataloader"]
