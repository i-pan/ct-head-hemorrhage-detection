"""Verify protocol equality and a full-size pretrained training step."""

import argparse
import importlib
import json
import time
import torch
from skp.configs.rsna_ich_maxvit_tiny_series_sampling import cfg as baseline
from skp.models.classification.net2d import Net
from skp.losses.ich import WeightedBCEWithLogitsLoss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = importlib.import_module("skp.configs." + args.config).cfg
    differences = [
        k for k in vars(baseline) if repr(getattr(cfg, k)) != repr(getattr(baseline, k))
    ]
    assert set(differences) <= {"backbone", "backbone_img_size"}, differences
    assert cfg.num_epochs == 50 and cfg.val_metric == "series_auc_any_max"
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed)
    start = time.monotonic()
    model = Net(cfg).cuda().train()
    model.set_criterion(WeightedBCEWithLogitsLoss(cfg.loss_params).cuda())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer_params["lr"],
        weight_decay=cfg.optimizer_params["weight_decay"],
    )
    batch = {
        "x": torch.rand(32, 9, 512, 512, device="cuda"),
        "y": torch.randint(0, 2, (32, 6), device="cuda").float(),
    }
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch, return_loss=True)
        assert out["logits"].shape == (32, 6)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        optimizer.step()
    torch.cuda.synchronize()
    print(
        json.dumps(
            dict(
                config=args.config,
                differences=differences,
                params=sum(p.numel() for p in model.parameters()),
                peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                seconds=time.monotonic() - start,
                passed=True,
            )
        ),
        flush=True,
    )
