import numpy as np
import pytest
import torch
from skp.ich_alt import preprocess_slab, pool_series, model_config


def test_preprocessing_boundaries_windows_and_channel_order():
    p = np.full((512, 512), 1024, np.uint16)
    c = np.full_like(p, 1064)
    n = np.full_like(p, 1104)
    x = preprocess_slab(p, c, n, rescale_intercept=-1024)
    np.testing.assert_allclose(x[[0, 3, 6], 100, 100], [0, 0.5, 1], atol=1e-6)
    edge = preprocess_slab(None, c, None, rescale_intercept=-1024)
    assert not edge[:3].any() and not edge[6:].any()
    assert edge.shape == (9, 512, 512)


def test_lse_independent_oracle_and_invalid_inputs():
    z = torch.tensor([[1.0, -2.0], [2.0, 3.0], [0.0, 1.0]], dtype=torch.float64)
    expected = np.log(np.exp(z.numpy() / 0.25).mean(axis=0)) * 0.25
    np.testing.assert_allclose(pool_series(z).numpy(), expected)
    assert torch.all(pool_series(z) <= z.max(0).values)
    for t in [0, -1, float("nan")]:
        with pytest.raises(ValueError):
            pool_series(z, t)
    with pytest.raises(ValueError):
        pool_series(z[:0])


def test_runtime_config_has_no_training_paths_or_pretrained_downloads():
    for family in ["convnext_tiny", "maxvit_tiny", "coatnet0"]:
        cfg = model_config(family)
        assert cfg.pretrained is False and cfg.logit_output_stride == 16
        assert not cfg.get("load_pretrained_model")
        assert not cfg.get("annotations_file")
