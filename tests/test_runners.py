from skp.configs import Config
from skp.runners import _apply_trial_to_config, _validate_progressive_resizing
from skp.search import generate_search


def test_apply_trial_to_config_updates_nested_dicts():
    cfg = Config(optimizer_params={"lr": 1e-3})
    trial = generate_search(
        [{"optimizer_params__lr": 3e-4, "batch_size": 16}], num_trials=1
    )[0]

    _apply_trial_to_config(cfg, trial, print_fn=lambda _: None)

    assert cfg.optimizer_params["lr"] == 3e-4
    assert cfg.batch_size == 16


def test_validate_progressive_resizing_counts_stages():
    assert _validate_progressive_resizing({"image_height": [128, 256]}) == 2
