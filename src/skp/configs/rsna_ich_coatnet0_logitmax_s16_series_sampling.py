from skp.configs.rsna_ich_maxvit_logitmax_s16_series_sampling import cfg as baseline

cfg = baseline.__deepcopy__()
cfg.model = "classification.spatial_logit_backbones"
cfg.backbone = "coatnet_0_rw_224.sw_in1k"
cfg.backbone_img_size = True
