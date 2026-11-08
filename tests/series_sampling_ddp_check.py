"""Run directly: actual two-rank CPU Lightning/accumulation/scheduler check."""

import json
import torch
import pandas as pd
import lightning as L
from torch.utils.data import Dataset
from skp.configs import Config
from skp.tasks.classification import Task
from skp.tasks.samplers import configure_series_sampling_schedule
from skp.optim import get_scheduler


class TinySeries(Dataset):
    collate_fn = None

    def __init__(self):
        self.df = pd.DataFrame(
            [
                {"series_uid": str(s), "any": int(i < 2)}
                for s in range(10)
                for i in range(20)
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return {
            "x": torch.tensor([i / 200.0]),
            "y": torch.tensor([float(self.df.iloc[i]["any"])]),
            "index": i,
            "group_index": i // 20,
        }


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, batch, return_loss=True):
        return {
            "loss": torch.nn.functional.binary_cross_entropy_with_logits(
                self.linear(batch["x"]), batch["y"]
            )
        }


class Audit(L.Callback):
    def __init__(self):
        self.groups = []
        self.lrs = []

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        self.groups.extend(batch["group_index"].tolist())

    def on_before_optimizer_step(self, trainer, module, optimizer):
        self.lrs.append(optimizer.param_groups[0]["lr"])

    def on_train_epoch_end(self, trainer, module):
        all_groups = [None] * 2
        torch.distributed.all_gather_object(all_groups, self.groups)
        groups = sum(all_groups, [])
        assert sorted(groups) == list(range(10)), groups
        assert trainer.global_step == (trainer.current_epoch + 1) * 2
        self.groups = []

    def on_train_end(self, trainer, module):
        assert trainer.global_step == 6
        assert module.scheduler.last_epoch == 6
        assert module.scheduler.total_steps == 6
        assert module.scheduler.warmup_steps == 2
        assert len(self.lrs) == 6
        assert abs(max(self.lrs) - 0.01) < 1e-9
        assert abs(module.optimizer.param_groups[0]["lr"] - 1e-6) < 1e-10
        if trainer.global_rank == 0:
            print(
                "DDP_SCHEDULE_CHECK_PASSED "
                + json.dumps(
                    {
                        "steps": 6,
                        "steps_per_epoch": 2,
                        "warmup_steps": 2,
                        "rank_batches": 3,
                        "accumulation": 2,
                        "series_per_epoch": 10,
                        "final_lr": module.optimizer.param_groups[0]["lr"],
                    }
                ),
                flush=True,
            )


def main():
    torch.set_num_threads(1)
    L.seed_everything(88)
    cfg = Config(
        seed=88,
        sampler="SeriesSliceSampler",
        series_positive_slice_probability=0.5,
        train_drop_last=False,
        world_size=2,
        batch_size=2,
        num_workers=0,
        args={"strategy": "ddp"},
        num_epochs=3,
        accumulate_grad_batches=2,
        scheduler="LinearWarmupCosineAnnealingLR",
        scheduler_interval="step",
        scheduler_params={"pct_start": 1 / 3, "init_lr": 0.0, "final_lr": 1e-6},
        val_metric="loss",
    )
    ds = TinySeries()
    configure_series_sampling_schedule(cfg, ds)
    task = Task(cfg)
    task.set("model", Model())
    task.set("datasets", [ds, ds])
    task.set("metrics", [])
    task.set("val_metric", "loss")
    opt = torch.optim.SGD(task.model.parameters(), lr=0.01)
    task.set("optimizer", opt)
    task.set("scheduler", get_scheduler(cfg, opt))
    trainer = L.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp",
        max_epochs=3,
        accumulate_grad_batches=2,
        use_distributed_sampler=False,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        callbacks=[Audit()],
    )
    trainer.fit(task)


if __name__ == "__main__":
    main()
