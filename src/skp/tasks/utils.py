import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from skp.configs.base import Config
from skp.tasks import samplers


def _require_webdataset():
    try:
        import webdataset as wds
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "webdataset is required when cfg.wds = True. Install it with "
            "`uv add webdataset` or set cfg.wds = False."
        ) from e
    return wds


def _is_ddp(cfg: Config) -> bool:
    args = cfg.get("args", {}) or {}
    strategy = args.get("strategy")
    if strategy is None:
        return False
    return strategy == "ddp" or str(strategy).startswith("ddp")


def build_dataloader(cfg: Config, dataset: Dataset, mode: str) -> DataLoader:
    if cfg.get("wds", False):
        wds = _require_webdataset()
        num_workers = cfg.get("num_workers", 0) or 0
        persistent_workers = (cfg.get("persistent_workers", False) or False) and (
            num_workers > 0
        )
        if mode == "train":
            loader_params = {
                "batch_size": None,
                "num_workers": num_workers,
                "pin_memory": cfg.get("pin_memory", True) or False,
                "persistent_workers": persistent_workers,
            }
            if num_workers > 0:
                loader_params["prefetch_factor"] = 4
            loader = wds.WebLoader(dataset, **loader_params).with_epoch(
                cfg.num_iterations_per_epoch
            )
        else:
            val_num_workers = cfg.get("val_num_workers") or num_workers
            if cfg.get("num_val_batches_per_gpu") is not None:
                loader = wds.WebLoader(
                    dataset,
                    batch_size=None,
                    num_workers=val_num_workers,
                ).with_epoch(cfg.num_val_batches_per_gpu)
            else:
                loader = wds.WebLoader(
                    dataset,
                    batch_size=None,
                    num_workers=val_num_workers,
                )
        return loader

    def worker_init_fn(worker_id: int) -> None:
        try:
            import cv2

            cv2.setNumThreads(0)
            cv2.ocl.setUseOpenCL(False)
        except ImportError:
            pass
        np.random.seed(torch.initial_seed() % 2**32)

    num_workers = (
        cfg.num_workers
        if mode == "train"
        else cfg.get("val_num_workers", cfg.num_workers)
    )
    dataloader_params = {}
    dataloader_params["num_workers"] = num_workers
    dataloader_params["drop_last"] = mode == "train"
    dataloader_params["shuffle"] = mode == "train"
    dataloader_params["pin_memory"] = cfg.get("pin_memory", False) or False
    dataloader_params["collate_fn"] = dataset.collate_fn
    if num_workers > 0:
        if mode == "train":
            persistent_workers = cfg.get("persistent_workers", False) or False
            prefetch_factor = cfg.get("prefetch_factor", 2) or 2
        else:
            persistent_workers = cfg.get(
                "val_persistent_workers",
                cfg.get("persistent_workers", False) or False,
            )
            prefetch_factor = cfg.get(
                "val_prefetch_factor",
                cfg.get("prefetch_factor", 2) or 2,
            )
        dataloader_params["persistent_workers"] = persistent_workers
        dataloader_params["prefetch_factor"] = prefetch_factor

    if mode == "train":
        dataloader_params["batch_size"] = cfg.batch_size
    else:
        dataloader_params["batch_size"] = cfg.get("val_batch_size") or cfg.batch_size * 2

    sampler = None
    if cfg.get("sampler") and cfg.sampler != "" and mode == "train":
        sampler = getattr(samplers, cfg.sampler)(dataset=dataset, cfg=cfg)

    if sampler:
        dataloader_params["shuffle"] = False
        if _is_ddp(cfg):
            sampler = samplers.DistributedSamplerWrapper(sampler)
        print(f"Using sampler {sampler} for training ...")
        dataloader_params["sampler"] = sampler
    elif _is_ddp(cfg):
        dataloader_params["shuffle"] = False
        dataloader_params["sampler"] = DistributedSampler(
            dataset, shuffle=mode == "train"
        )

    loader = DataLoader(dataset, **dataloader_params, worker_init_fn=worker_init_fn)

    return loader
