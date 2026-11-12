"""Full-size train and validation GPU preflight for stride-16 backbone ablations."""

import importlib
import json
import sys
import torch
from skp.configs.rsna_ich_maxvit_logitmax_s16_series_sampling import cfg as baseline
from skp.models.classification.spatial_logit_backbones import Net
from skp.losses.ich import WeightedBCEWithLogitsLoss

cfg = importlib.import_module("skp.configs." + sys.argv[1]).cfg
changes = {k for k in vars(cfg) if repr(cfg.get(k)) != repr(baseline.get(k))}
assert (
    changes <= {"model", "backbone", "backbone_img_size"}
    and {"model", "backbone"} <= changes
), changes
assert (
    cfg.num_epochs == 50
    and cfg.batch_size == 32
    and cfg.num_input_channels == 9
    and cfg.logit_output_stride == 16
)
torch.set_num_threads(4)
torch.manual_seed(cfg.seed)
model = Net(cfg).cuda().train()
model.set_criterion(WeightedBCEWithLogitsLoss(cfg.loss_params).cuda())
optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer_params)
batch = {
    "x": torch.rand(32, 9, 512, 512, device="cuda"),
    "y": torch.randint(0, 2, (32, 6), device="cuda").float(),
}
for _ in range(2):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch, return_loss=True, return_features=True)
    assert out["logit_map"].shape == (32, 6, 32, 32) and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    optimizer.step()
del out, batch
model.eval()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model({"x": torch.rand(64, 9, 512, 512, device="cuda")})
assert out["logits"].shape == (64, 6) and torch.isfinite(out["logits"]).all()
print(
    json.dumps(
        dict(
            passed=True,
            changes=sorted(changes),
            peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
    ),
    flush=True,
)
