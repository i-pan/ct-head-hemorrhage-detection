"""Protocol and full-batch GPU preflight, run only when preceding jobs finish."""

import importlib
import json
import sys
import torch
from skp.configs.rsna_ich_maxvit_tiny_series_sampling import cfg as baseline
from skp.models.classification.maxvit_logit_pool import Net
from skp.losses.ich import WeightedBCEWithLogitsLoss

cfg = importlib.import_module("skp.configs." + sys.argv[1]).cfg
changes = {k for k in vars(cfg) if repr(cfg.get(k)) != repr(baseline.get(k))}
assert changes == {"model", "logit_output_stride"}, changes
assert cfg.num_epochs == 50 and cfg.batch_size == 32 and cfg.num_input_channels == 9
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
    assert out["logit_map"].shape == (
        32,
        6,
        512 // cfg.logit_output_stride,
        512 // cfg.logit_output_stride,
    )
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    optimizer.step()
del batch, out
model.eval()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model({"x": torch.rand(64, 9, 512, 512, device="cuda")})
assert out["logits"].shape == (64, 6) and torch.isfinite(out["logits"]).all()
print(
    json.dumps(
        {
            "passed": True,
            "stride": cfg.logit_output_stride,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
    ),
    flush=True,
)
