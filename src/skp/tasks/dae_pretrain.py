import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from lightning.pytorch.loggers import MLFlowLogger
from typing import Dict

from skp.tasks.base import BaseTask


class Task(BaseTask):
    def _preprocess_batch(self, batch: Dict) -> None:
        # 1. Cast to float (on GPU)
        if not batch["x"].is_floating_point():
            batch["x"] = batch["x"].float()

        # 2. CT Windowing or Clip/Normalize
        if getattr(self.cfg, "ct_windows", None) is not None:
            windows = torch.tensor(
                self.cfg.ct_windows, device=batch["x"].device, dtype=batch["x"].dtype
            )
            wl = windows[:, 0].view(1, -1, 1, 1, 1)
            ww = windows[:, 1].view(1, -1, 1, 1, 1)
            lower, upper = wl - ww / 2, wl + ww / 2
            # Broadcast (B, 1, D, H, W) vs (1, K, 1, 1, 1) -> (B, K, D, H, W)
            x = torch.clamp(batch["x"], min=lower, max=upper)
            batch["x"] = (x - lower) / (ww + 1e-6)

            if getattr(self.cfg, "normalized_range", None) is not None:
                min_out, max_out = self.cfg.normalized_range
                batch["x"] = batch["x"] * (max_out - min_out) + min_out
        else:
            assert self.cfg.clip_hu is not None
            batch["x"] = torch.clamp(
                batch["x"], self.cfg.clip_hu[0], self.cfg.clip_hu[1]
            )
            if getattr(self.cfg, "normalized_range", None) is not None:
                min_in, max_in = self.cfg.clip_hu
                min_out, max_out = self.cfg.normalized_range
                scale = (max_out - min_out) / (max_in - min_in)
                batch["x"] = (batch["x"] - min_in) * scale + min_out

        # 3. Resize
        if batch["x"].shape[-2:] != (self.cfg.H, self.cfg.W):
            # x: (B, C, D, H, W)
            B, C, D, H, W = batch["x"].shape
            # Optimization: Merge Depth into Channels to avoid transpose copy
            x = batch["x"].reshape(B, C * D, H, W)
            x = F.interpolate(
                x, size=(self.cfg.H, self.cfg.W), mode="bilinear", align_corners=False
            )
            batch["x"] = x.view(B, C, D, self.cfg.H, self.cfg.W)

            if "roi_mask" in batch and batch["roi_mask"] is not None:
                # roi_mask: (B, 1, D, H, W)
                m = batch["roi_mask"].reshape(B, D, H, W)
                m = F.interpolate(
                    m.float(), size=(self.cfg.H, self.cfg.W), mode="nearest"
                )
                batch["roi_mask"] = m.view(B, 1, D, self.cfg.H, self.cfg.W)

    def apply_dae_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Downsample
        downsample = getattr(self.cfg, "dae_downsample", 4)
        if downsample > 1:
            D, H, W = x.shape[-3:]
            # x: (B, C, D, H, W)
            x = F.interpolate(
                x,
                size=(D, H // downsample, W // downsample),
                mode="trilinear",
                align_corners=False,
            )
            x = F.interpolate(
                x, size=(D, H, W), mode="trilinear", align_corners=False
            )

        # 2. Noise
        noise_std = getattr(self.cfg, "dae_noise", 0.1)
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        # 3. Clip
        if getattr(self.cfg, "normalized_range", None) is not None:
            min_val, max_val = self.cfg.normalized_range
            x = torch.clamp(x, min=min_val, max=max_val)

        return x

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        self._preprocess_batch(batch)
        x_orig = batch["x"].clone()
        x_corrupted = self.apply_dae_augmentation(x_orig)
        out = self.model(
            x_corrupted, target=x_orig, roi_mask=batch.get("roi_mask", None)
        )
        for k, v in out.items():
            if "loss" in k and k != "loss_per_sample":
                self.log(k, v, prog_bar=k == "loss")

        log_interval = self.cfg.get("log_training_examples_interval")
        if (
            self.global_rank == 0
            and log_interval is not None
            and (self.global_step % log_interval == 0)
        ):
            self.log_visualization(out["target"], x_corrupted, out["pred"])

        return out["loss"]

    @torch.no_grad()
    def log_visualization(self, x, x_corrupted, pred):
        # x: (B, C, D, H, W)
        # x_corrupted: (B, C, D, H, W)
        # pred: (B, C, D, H, W)

        # Take first sample
        x = x[0].float().cpu()
        x_corrupted = x_corrupted[0].float().cpu()
        pred = pred[0].float().cpu()

        C, D, H, W = x.shape

        # Select middle slice
        d_idx = D // 2

        fig, axes = plt.subplots(C, 3, figsize=(12, 4 * C), squeeze=False)

        for c in range(C):
            for i, (img, title) in enumerate(
                zip([x, x_corrupted, pred], ["Original", "Corrupted", "Reconstruction"])
            ):
                ax = axes[c, i]
                t = img[c, d_idx]
                # Normalize to 0-1 for visualization
                t = t - t.min()
                t = t / (t.max() + 1e-6)

                ax.imshow(t.numpy(), cmap="gray")
                if c == 0:
                    ax.set_title(title)
                ax.axis("off")

        plt.tight_layout()

        if isinstance(self.logger, MLFlowLogger):
            if self.cfg.get("save_all_training_examples", False):
                self.logger.experiment.log_figure(
                    self.logger.run_id,
                    fig,
                    f"reconstruction_step_{self.global_step}.png",
                )
            else:
                self.logger.experiment.log_figure(
                    self.logger.run_id, fig, "reconstruction.png"
                )

        plt.close(fig)

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        self._preprocess_batch(batch)
        x_orig = batch["x"].clone()
        x_corrupted = self.apply_dae_augmentation(x_orig)
        out = self.model(
            x_corrupted, target=x_orig, roi_mask=batch.get("roi_mask", None)
        )
        for k, v in out.items():
            if "loss" in k and k != "loss_per_sample":
                self.val_loss[k].append(v.detach().clone())
        for m in self.metrics:
            # passing output and input dicts is the most flexible
            # then the metric can be customized to get the keys they need
            m.update(out, batch)
        return out["loss"]
