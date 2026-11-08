import torch
from skp.configs.rsna_ich_maxvit_tiny_series_sampling import cfg as baseline
from skp.configs.rsna_ich_naflexvit_base_series_sampling import cfg
from skp.models.classification.naflexvit import Net
from skp.losses.ich import WeightedBCEWithLogitsLoss


def test_protocol_only_changes_backbone_and_adapter():
    differences = {
        k for k in vars(baseline) if repr(getattr(cfg, k)) != repr(getattr(baseline, k))
    }
    assert differences == {"backbone", "model"}
    assert cfg.num_epochs == 50 and cfg.num_input_channels == 9
    assert cfg.image_height == cfg.image_width == 512
    assert cfg.val_metric == "series_auc_any_max"


def test_native_pooling_backward_and_checkpoint():
    torch.set_num_threads(2)
    config = cfg.__deepcopy__()
    config.pretrained = False
    model = Net(config).eval()
    x = torch.rand(2, 9, 64, 64)
    with torch.no_grad():
        tokens = model.backbone.forward_features(model.normalize(x))
        expected = model.backbone.fc_norm(
            tokens[:, model.backbone.num_prefix_tokens :].mean(1)
        )
        out = model({"x": x}, return_features=True)
        torch.testing.assert_close(out["features"], expected)
    model.train()
    model.set_criterion(WeightedBCEWithLogitsLoss(config.loss_params))
    out = model({"x": x, "y": torch.zeros(2, 6)}, return_loss=True)
    assert out["logits"].shape == (2, 6)
    out["loss"].backward()
    grad = model.backbone.embeds.proj.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    model.eval()
    restored = Net(config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(
            model({"x": x})["logits"], restored({"x": x})["logits"]
        )


def test_patch_adaptation_preserves_repeated_rgb_inputs():
    from types import SimpleNamespace
    from torch import nn
    from skp.models.classification.naflexvit import adapt_patch_channels

    projection = nn.Linear(2 * 2 * 3, 7)
    backbone = SimpleNamespace(
        embeds=SimpleNamespace(
            proj=projection,
            norm_input=None,
            enable_patch_interpolator=False,
            patch_size=(2, 2),
            in_chans=3,
        ),
        in_chans=3,
    )
    patch = torch.randn(2, 4, 2, 2, 3)
    expected = projection(patch.flatten(-3))
    adapt_patch_channels(backbone, 9)
    actual = backbone.embeds.proj(patch.repeat(1, 1, 1, 1, 3).flatten(-3))
    torch.testing.assert_close(actual, expected)
