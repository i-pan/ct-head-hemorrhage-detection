from copy import deepcopy

from skp.configs.rsna_ich_joint_effv2m_9ch_deeplab import cfg as base_cfg


cfg = deepcopy(base_cfg)
cfg.backbone = "maxvit_tiny_tf_512.in1k"
cfg.pseudolabel_dir = "./data/pseudolabels/rsna_maxvit_9ch_any2"
cfg.pseudolabel_manifest = f"{cfg.pseudolabel_dir}/manifest.csv"

segmentation_loss = cfg.loss_params["losses"][
    "segmentation.PositiveDiceNegativeFocalLoss"
]
segmentation_loss["params"]["class_weights"] = [1, 1, 1, 1, 1, 2]
