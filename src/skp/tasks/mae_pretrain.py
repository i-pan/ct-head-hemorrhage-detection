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

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        self._preprocess_batch(batch)
        out = self.model(
            batch["x"],
            random_masking=True,
            mask_ratio=self.cfg.mask_ratio,
            slice_mask_ratio=self.cfg.slice_mask_ratio or 0.0,
            norm_pix_loss=self.cfg.norm_pix_loss,
            roi_mask=batch.get("roi_mask", None),
            roi_weight=self.cfg.roi_weight or 1.0,
            ignore_background=self.cfg.ignore_background or False,
            mask_roi_only=self.cfg.mask_roi_only or False,
        )
        for k, v in out.items():
            if "loss" in k and k != "loss_per_sample":
                self.log(k, v, prog_bar=k == "loss")
            elif k == "slice_mask_ratio":
                self.log(k, v, prog_bar=False)

        log_interval = self.cfg.get("log_training_examples_interval")
        if (
            self.global_rank == 0
            and log_interval is not None
            and (self.global_step % log_interval == 0)
        ):
            self.log_visualization(
                batch["x"],
                out["pred"],
                out["mask"],
                out.get("patch_mean"),
                out.get("patch_var"),
            )

        return out["loss"]

    @torch.no_grad()
    def log_visualization(self, x, pred, mask, patch_mean=None, patch_var=None):
        # x: (B, C, D, H, W)
        # pred: (B, L, dim)
        # mask: (B, 1, D, H, W)

        # Take first sample
        x = x[0].float().cpu()
        pred = pred[0:1].float().cpu()  # Keep batch dim for unpatchify
        mask = mask[0].float().cpu() if mask is not None else torch.ones_like(x[:, 0:1])

        if patch_mean is not None and patch_var is not None:
            patch_mean = patch_mean[0:1].float().cpu()
            patch_var = patch_var[0:1].float().cpu()
            pred = pred * (patch_var + 1.0e-6) ** 0.5 + patch_mean

        C, D, H, W = x.shape

        # Unpatchify pred
        # dim = C * D * p * p
        dim = pred.shape[-1]
        p = int((dim / (C * D)) ** 0.5)
        h_grid = H // p
        w_grid = W // p

        rec = pred.reshape(1, h_grid, w_grid, C, D, p, p)
        rec = rec.permute(0, 3, 4, 1, 5, 2, 6)
        rec = rec.reshape(1, C, D, H, W)[0]

        masked_img = x * mask

        # Select middle slice
        d_idx = D // 2

        fig, axes = plt.subplots(C, 3, figsize=(12, 4 * C), squeeze=False)

        for c in range(C):
            for i, (img, title) in enumerate(
                zip([x, masked_img, rec], ["Original", "Masked", "Reconstruction"])
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
        out = self.model(
            batch["x"],
            random_masking=True,
            mask_ratio=self.cfg.mask_ratio,
            slice_mask_ratio=self.cfg.slice_mask_ratio or 0.0,
            norm_pix_loss=self.cfg.norm_pix_loss,
            roi_mask=batch.get("roi_mask", None),
            roi_weight=self.cfg.roi_weight or 1.0,
            ignore_background=self.cfg.ignore_background or False,
        )
        for k, v in out.items():
            if "loss" in k and k != "loss_per_sample":
                self.val_loss[k].append(v.detach().clone())
        for m in self.metrics:
            # passing output and input dicts is the most flexible
            # then the metric can be customized to get the keys they need
            m.update(out, batch)
        return out["loss"]
