import torch

from skp.configs import Config
from skp.models.classification.efficientnet3d_stem import Conv3dTo2dStem, Net


def test_3d_stem_adapts_conv2d_weights_and_collapses_depth():
    conv2d = torch.nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1, bias=False)
    with torch.no_grad():
        conv2d.weight.fill_(3.0)
    stem = Conv3dTo2dStem(conv2d, depth=3)

    assert stem.conv.weight.shape == (2, 3, 3, 3, 3)
    assert torch.allclose(stem.conv.weight, torch.ones_like(stem.conv.weight))
    assert stem(torch.ones(1, 3, 3, 8, 8)).shape == (1, 2, 8, 8)


def test_efficientnet_3d_stem_model_forward_shape():
    cfg = Config(
        backbone="tf_efficientnetv2_b0",
        pretrained=False,
        num_input_channels=3,
        num_slices=3,
        image_height=64,
        image_width=64,
        num_classes=6,
        pool="avg",
        pool_params=None,
        dropout=0.2,
        normalization="linear",
        normalization_params={
            "input_min": 0.0,
            "input_max": 1.0,
            "output_min": -1.0,
            "output_max": 1.0,
        },
        enable_gradient_checkpointing=False,
        model_activation_fn=None,
        load_pretrained_backbone=None,
        load_pretrained_model=None,
    )
    model = Net(cfg)

    out = model({"x": torch.rand(2, 3, 3, 64, 64)})

    assert out["logits"].shape == (2, 6)
