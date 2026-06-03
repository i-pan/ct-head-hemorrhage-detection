import torch
import torch.nn as nn
from typing import Dict


def torch_load_weights(path: str) -> Dict[str, torch.Tensor]:
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    return weights


def filter_weights_by_prefix(
    weights: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    # If model was compiled
    weights = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
    weights = {
        k.replace(prefix, ""): v for k, v in weights.items() if k.startswith(prefix)
    }
    weights = {k: v for k, v in weights.items() if not k.startswith("criterion.")}
    return weights


def change_num_input_channels(model, num_input_channels):
    # Assumes original number of input channels in model is 3
    for i, m in enumerate(model.modules()):
        if isinstance(m, (nn.Conv2d, nn.Conv3d)) and m.in_channels == 3:
            m.in_channels = num_input_channels
            # First, sum across channels
            W = m.weight.sum(1, keepdim=True)
            # Then, divide by number of channels
            W = W / num_input_channels
            # Then, repeat by number of channels
            size = [1] * W.ndim
            size[1] = num_input_channels
            W = W.repeat(size)
            m.weight = nn.Parameter(W)
            break
