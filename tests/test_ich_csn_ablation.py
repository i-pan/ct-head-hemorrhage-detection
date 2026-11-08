import numpy as np
import pytest
import torch
from skp.configs.rsna_ich_csn_r101_series_sampling import cfg
from skp.configs.rsna_ich_maxvit_tiny_series_sampling import cfg as baseline
from skp.datasets.rsna_ich_2p5d import Dataset
from test_rsna_ich_training_components import _dataset_cfg


def test_protocol_changes_only_backbone_and_tensor_layout():
    differences = {
        k for k in vars(baseline) if repr(getattr(cfg, k)) != repr(getattr(baseline, k))
    }
    assert differences == {
        "model",
        "backbone",
        "num_input_channels",
        "flatten_depth_to_channels",
    }
    assert cfg.num_epochs == 50
    assert cfg.image_height == cfg.image_width == 512
    assert cfg.val_metric == "series_auc_any_max"


def test_same_contiguous_slices_and_center_targets(tmp_path):
    config = _dataset_cfg(tmp_path)
    config.flatten_depth_to_channels = False
    dataset = Dataset(config, "train")
    center = dataset[1]
    assert center["x"].shape == (3, 3, 4, 4)
    assert center["y"].tolist() == [0, 0, 0, 0, 1, 1]
    assert dataset[0]["y"][-1] == 0
    assert torch.count_nonzero(dataset[0]["x"][:, 0]) == 0
    assert torch.count_nonzero(dataset[2]["x"][:, 2]) == 0
    values = center["x"][0, :, 0, 0].numpy()
    assert np.all(np.diff(values) > 0)
    config.flatten_depth_to_channels = True
    flat = Dataset(config, "train")[1]
    torch.testing.assert_close(
        center["x"], flat["x"].reshape(3, 3, 4, 4).permute(1, 0, 2, 3)
    )
    torch.testing.assert_close(center["y"], flat["y"])


def test_csn_backward_and_checkpoint_roundtrip():
    pytest.importorskip("pytorchvideo")
    from skp.models.classification.csn3d import Net
    from skp.losses.ich import WeightedBCEWithLogitsLoss

    torch.set_num_threads(2)
    config = cfg.__deepcopy__()
    config.pretrained = False
    model = Net(config).train()
    model.set_criterion(WeightedBCEWithLogitsLoss(config.loss_params))
    batch = {"x": torch.rand(2, 3, 3, 64, 64), "y": torch.zeros(2, 6)}
    out = model(batch, return_loss=True)
    assert out["logits"].shape == (2, 6)
    out["loss"].backward()
    assert torch.isfinite(model.backbone[0].conv.weight.grad).all()
    assert model.backbone[0].conv.weight.grad.abs().sum() > 0
    restored = Net(config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch)["logits"], restored(batch)["logits"])
