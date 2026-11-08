from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict

import lightning
import torch
import torch.nn as nn
from lightning.pytorch.loggers import MLFlowLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from skp.tasks.utils import build_dataloader


class BaseTask(lightning.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.val_loss = defaultdict(list)

    def set(self, name: str, attr: Any) -> None:
        if name == "metrics":
            attr = nn.ModuleList(attr)
        setattr(self, name, attr)

    def on_train_start(self) -> None:
        required = ["model", "datasets", "optimizer", "metrics", "val_metric"]
        missing = [obj for obj in required if not hasattr(self, obj)]
        if missing:
            raise AttributeError(f"Task missing required attributes: {missing}")

        if self.cfg.get("sampler") == "SeriesSliceSampler":
            expected = self.cfg.scheduler_total_steps
            actual = self.trainer.estimated_stepping_batches
            schedule_steps = getattr(
                self.scheduler, "total_steps", getattr(self.scheduler, "T_max", None)
            )
            if actual != expected or schedule_steps != expected:
                raise RuntimeError(
                    f"LR schedule mismatch: Lightning={actual}, planned={expected}, scheduler={schedule_steps}"
                )
            if (
                self.trainer.num_training_batches
                != self.cfg.train_batches_per_epoch_per_rank
            ):
                raise RuntimeError(
                    "Training loader batch count differs from the LR schedule plan."
                )
            self.print(
                f"VERIFIED LR SCHEDULE: {actual} optimizer steps, "
                f"warmup={getattr(self.scheduler, 'warmup_steps', 0)}, "
                f"steps/epoch={self.cfg.optimizer_steps_per_epoch}."
            )
            if self.global_rank == 0 and self.cfg.get("save_dir"):
                report = {
                    "series_per_epoch": self.cfg.train_samples_per_epoch,
                    "batches_per_rank": self.trainer.num_training_batches,
                    "optimizer_steps_per_epoch": self.cfg.optimizer_steps_per_epoch,
                    "total_optimizer_steps": actual,
                    "warmup_steps": getattr(self.scheduler, "warmup_steps", 0),
                    "epochs": self.cfg.num_epochs,
                    "world_size": self.trainer.world_size,
                    "batch_size_per_rank": self.cfg.batch_size,
                    "accumulate_grad_batches": self.trainer.accumulate_grad_batches,
                    "positive_bin_probability": self.cfg.series_positive_slice_probability,
                    "bin_policy": "Independent shuffled positive/negative cycles; resume by deterministic epoch replay.",
                }
                (Path(self.cfg.save_dir) / "series_sampling_schedule.json").write_text(
                    json.dumps(report, indent=2)
                )

        if isinstance(self.logger, MLFlowLogger):
            self.logger.log_hyperparams(self.cfg.__dict__)

    def on_train_epoch_end(self) -> None:
        if self.cfg.get("sampler") == "SeriesSliceSampler":
            if self.scheduler.last_epoch != self.global_step:
                raise RuntimeError(
                    "LR scheduler step count has diverged from optimizer steps."
                )
            if self.global_rank == 0 and self.cfg.get("save_dir"):
                record = {
                    "epoch": self.current_epoch,
                    "optimizer_step": self.global_step,
                    "scheduler_step": self.scheduler.last_epoch,
                    "lr": [g["lr"] for g in self.optimizer.param_groups],
                }
                with (Path(self.cfg.save_dir) / "series_sampling_lr.jsonl").open(
                    "a"
                ) as stream:
                    stream.write(json.dumps(record) + "\n")
        if self.global_rank == 0 and isinstance(self.logger, MLFlowLogger):
            self.logger.log_metrics(
                {"training/epoch": self.current_epoch}, step=self.global_step
            )

    def on_validation_epoch_end(self) -> None:
        metrics = self.compute_validation_metrics()
        self.reset_validation_state()
        self.print_validation_metrics(metrics)
        self.log_validation_metrics(metrics)

    def compute_validation_metrics(self) -> Dict:
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.compute())
        for key, values in self.val_loss.items():
            metrics[key] = torch.stack(values).mean()

        val_metric = self.val_metric
        if isinstance(val_metric, list):
            metrics["val_metric"] = torch.sum(
                torch.stack([metrics[name.lower()].cpu() for name in val_metric])
            ).item()
        else:
            metrics["val_metric"] = metrics[val_metric.lower()]
            if isinstance(metrics["val_metric"], torch.Tensor):
                metrics["val_metric"] = metrics["val_metric"].item()
        return metrics

    def reset_validation_state(self) -> None:
        self.val_loss = defaultdict(list)
        for metric in self.metrics:
            metric.reset()

    def print_validation_metrics(self, metrics: Dict) -> None:
        if self.global_rank != 0:
            return
        print("\n========")
        max_strlen = max(len(key) for key in metrics)
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            print(f"{key.ljust(max_strlen)} | {value:.4f}")

    def log_validation_metrics(self, metrics: Dict) -> None:
        if (
            self.trainer.state.stage
            == lightning.pytorch.trainer.states.RunningStage.SANITY_CHECKING
        ):
            return

        scalar_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            scalar_metrics[f"val/{key}"] = float(value)

        if isinstance(self.logger, MLFlowLogger):
            if self.global_rank == 0:
                self.logger.log_metrics(scalar_metrics, step=self.global_step)
        else:
            for key, value in scalar_metrics.items():
                self.log(key, value, sync_dist=False)

        # Custom metrics already gather validation predictions across ranks before
        # compute(), so another Lightning sync here can deadlock during epoch
        # teardown. Log on every rank without sync so callbacks can see val_metric.
        self.log(
            "val_metric",
            float(metrics["val_metric"]),
            sync_dist=False,
            logger=False,
        )

    def configure_optimizers(self) -> Dict:
        scheduler = getattr(self, "scheduler", None)
        if scheduler is None:
            return {"optimizer": self.optimizer}

        lr_scheduler = {
            "scheduler": scheduler,
            "interval": self.cfg.get("scheduler_interval", "epoch"),
        }
        if isinstance(scheduler, ReduceLROnPlateau):
            lr_scheduler["monitor"] = "val_metric"
        return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self) -> DataLoader:
        return build_dataloader(self.cfg, self.datasets[0], "train")

    def val_dataloader(self) -> DataLoader:
        return build_dataloader(self.cfg, self.datasets[1], "val")
