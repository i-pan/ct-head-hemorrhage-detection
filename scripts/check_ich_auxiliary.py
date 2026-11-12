"""Verify backbone geometry, baseline parity, mask alignment, and auxiliary gradients."""

import argparse
from importlib import import_module
import json
import torch
import numpy as np
from skp.models.classification.ich_auxiliary import Net, auxiliary_losses
from skp.losses.ich import WeightedBCEWithLogitsLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family")
    parser.add_argument("--config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = import_module(
        f"skp.configs.{args.config or f'rsna_ich_{args.family}_s16_auxiliary'}"
    ).cfg.__deepcopy__()
    cfg.pretrained = False
    assert cfg.segmentation_loss_weight == cfg.spatial_loss_weight == 0.5
    torch.manual_seed(88)
    model = Net(cfg).to(args.device)
    model.set_criterion(WeightedBCEWithLogitsLoss(cfg.loss_params))
    x = torch.rand(args.batch_size, 9, 512, 512, device=args.device)
    target = torch.zeros(args.batch_size, 1, 512, 512, device=args.device)
    target[::2, :, 200:240, 210:250] = 0.8
    batch = dict(
        x=x,
        y=torch.zeros(args.batch_size, 6, device=args.device),
        segmentation_target=target,
    )
    batch["y"][::2, -1] = 1
    batch["auxiliary_roi"] = torch.ones_like(target)
    if cfg.auxiliary_region == "hu_teacher_union":
        batch["auxiliary_roi"].zero_()
        batch["auxiliary_roi"][:, :, 128:384, 128:384] = 1
    # Identical eval classification path; intermediate path must agree too.
    model.eval()
    with torch.no_grad():
        direct = model.backbone(model.normalize(x[:1]))
        last, skips = model.backbone.forward_intermediates(
            model.normalize(x[:1]), indices=model.feature_indices
        )
        intermediate = model.backbone.forward_head(last)
        torch.testing.assert_close(direct, intermediate)
        assert [tuple(s.shape[-2:]) for s in skips] == [
            (128, 128),
            (64, 64),
            (32, 32),
            (32, 32),
        ]
        out = model(dict(x=x[:1]), return_features=True)
        expected = model.linear(direct.permute(0, 2, 3, 1)).flatten(1, 2).max(1).values
        torch.testing.assert_close(out["logits"], expected)
        with_decoder = model(dict(x=x[:1]), return_segmentation=True)
        torch.testing.assert_close(with_decoder["logits"], expected)
        assert with_decoder["segmentation_logits"].shape == (1, 1, 512, 512)
    model.train()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
    ):
        out = model(batch, return_loss=True)
    torch.testing.assert_close(
        out["loss"],
        out["classification_loss"]
        + 0.5 * (out["segmentation_bce_loss"] + out["segmentation_dice_loss"])
        + 0.5 * out["spatial_loss"],
    )
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for name, module in [
        ("encoder", model.backbone),
        ("linear", model.linear),
        ("decoder", model.decoder),
        ("segmentation_head", model.segmentation_head),
    ]:
        gradients = [p.grad for p in module.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients), name
        assert sum(g.abs().sum().item() for g in gradients) > 0, name
    optimizer.step()
    # A single small soft target must survive spatial max pooling.
    t = torch.zeros(1, 1, 512, 512)
    t[0, 0, 17, 17] = 0.7
    coarse = torch.nn.functional.adaptive_max_pool2d(t, (32, 32))
    assert np.isclose(coarse[0, 0, 1, 1], 0.7) and (coarse > 0).sum() == 1
    for t in [torch.zeros(2, 1, 32, 32), torch.ones(2, 1, 32, 32)]:
        losses = auxiliary_losses(torch.zeros_like(t), torch.zeros(2, 1, 2, 2), t)
        assert all(torch.isfinite(loss) for loss in losses)
    print(
        json.dumps(
            dict(
                family=args.family,
                batch_size=args.batch_size,
                device=args.device,
                loss=float(out["loss"].detach()),
                gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30
                if args.device.startswith("cuda")
                else None,
                checks="geometry,classification parity,all-branch gradients,loss weighting,small-target pooling,empty/full masks passed",
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
