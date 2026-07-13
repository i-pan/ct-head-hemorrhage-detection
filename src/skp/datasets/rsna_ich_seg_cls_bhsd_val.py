"""RSNA pseudolabel training with full-BHSD segmentation validation."""

from copy import deepcopy

from torch.utils.data import Dataset as TorchDataset

from skp.datasets.bhsd_seg import Dataset as BHSDDataset
from skp.datasets.rsna_ich_seg_cls import Dataset as RSNADataset


class Dataset(TorchDataset):
    def __init__(self, cfg, mode: str):
        self.mode = mode
        if mode == "train":
            self.dataset = RSNADataset(cfg, mode)
        elif mode == "val":
            bhsd_cfg = deepcopy(cfg)
            bhsd_cfg.annotations_file = cfg.bhsd_annotations_file
            bhsd_cfg.mask_cache_size = cfg.get("bhsd_mask_cache_size", 16)
            # Intentionally use all BHSD folds for model selection in this
            # decoder-pretraining experiment.
            self.dataset = BHSDDataset(bhsd_cfg, "inference")
        else:
            raise ValueError(f"Unsupported mode for mixed dataset: {mode!r}")
        self.collate_fn = self.dataset.collate_fn

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        if self.mode == "train":
            return sample
        return {"seg": sample, "cls": {}}
