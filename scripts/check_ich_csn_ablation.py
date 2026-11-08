"""Full-batch GPU preflight for the 3D CSN arm; run with GPUs idle."""

import importlib
import json
import time
import torch
from skp.configs.rsna_ich_csn_r101_series_sampling import cfg
from skp.configs.rsna_ich_maxvit_tiny_series_sampling import cfg as baseline
from skp.losses.ich import WeightedBCEWithLogitsLoss


def main():
    differences = {
        k for k in vars(baseline) if repr(getattr(cfg, k)) != repr(getattr(baseline, k))
    }
    assert differences == {
        "model",
        "backbone",
        "num_input_channels",
        "flatten_depth_to_channels",
    }, differences
    assert cfg.num_epochs == 50 and cfg.batch_size == 32
    assert cfg.image_height == cfg.image_width == 512
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed)
    started = time.monotonic()
    model = importlib.import_module("skp.models." + cfg.model).Net(cfg).cuda().train()
    model.set_criterion(WeightedBCEWithLogitsLoss(cfg.loss_params).cuda())
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer_params)
    batch = {
        "x": torch.rand(32, 3, 3, 512, 512, device="cuda"),
        "y": torch.randint(0, 2, (32, 6), device="cuda").float(),
    }
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch, return_loss=True)
        assert out["logits"].shape == (32, 6) and torch.isfinite(out["loss"])
        out["loss"].backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        optimizer.step()
    del batch
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert model({"x": torch.rand(64, 3, 3, 512, 512, device="cuda")})[
            "logits"
        ].shape == (64, 6)
    torch.cuda.synchronize()
    print(
        json.dumps(
            dict(
                passed=True,
                input_shape=[32, 3, 3, 512, 512],
                validation_batch=64,
                differences=sorted(differences),
                peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                seconds=time.monotonic() - started,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
