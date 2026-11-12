"""Full two-rank read-only validation gate for the recovered checkpoint."""

import os, json, time
from pathlib import Path
import torch
import torch.distributed as dist
from skp.configs.rsna_ich_convnext_tiny_s16_auxiliary import cfg as source
from skp.train import get_task

root = Path(__file__).resolve().parents[1]
run = (
    root
    / "experiments/rsna_ich_convnext_tiny_s16_auxiliary/series_e050_p0.5_seed88_20261111_convnext_aux05_unrestricted"
)


def main():
    torch.set_num_threads(4)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cfg = source.__deepcopy__()
    cfg.pretrained = False
    cfg.pin_memory = False
    cfg.world_size = 2
    cfg.args = {"strategy": "ddp"}
    task, cfg = get_task(cfg)
    state = torch.load(
        run / "fold0/checkpoints/last.ckpt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    assert state["epoch"] == 3 and state["global_step"] == 1196
    task.model.load_state_dict(
        {
            k.removeprefix("model."): v
            for k, v in state["state_dict"].items()
            if k.startswith("model.")
        },
        strict=True,
    )
    del state
    task.cuda(rank).eval()
    loader = task.val_dataloader()
    start = time.monotonic()
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            batch = {k: v.cuda(rank) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                task.validation_step(batch, i)
            if rank == 0 and i % 25 == 0:
                print(
                    f"Validation {i + 1}/{len(loader)} elapsed {time.monotonic() - start:.1f}s",
                    flush=True,
                )
    metrics = task.compute_validation_metrics()
    if rank == 0:
        result = dict(
            status="complete",
            checkpoint_step=1196,
            batches_per_rank=len(loader),
            seconds=time.monotonic() - start,
            pin_memory=False,
            metrics={k: float(v) for k, v in metrics.items()},
        )
        path = root / "logs/ich_convnext_roi_nopin_20261111/validation_gate.json"
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
