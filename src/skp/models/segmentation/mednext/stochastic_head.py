import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticWeightSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_total_classes,
        num_sample_classes,
        ignore_background_class_for_sampling,
    ):
        """
        Args:
            in_channels (int): Number of input feature channels.
            num_total_classes (int): Total classes (e.g., 117).
            num_sample_classes (int): Number of classes to sample per forward pass.
            ignore_background_class_for_sampling (bool): Ignore background class for sampling (i.e., subtract 1 from label tensor)
        """
        super().__init__()
        self.num_total_classes = num_total_classes
        self.num_sample_classes = num_sample_classes
        self.ignore_background_class_for_sampling = ignore_background_class_for_sampling

        # We store the weights and biases as parameters for the full class set.
        # We use a 1x1x1 kernel for 3D segmentation classification.
        self.weight = nn.Parameter(torch.empty(num_total_classes, in_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.empty(num_total_classes))

        # Initialization
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.bias, 0)

    def forward(self, x, y=None, indices=None, seed=None):
        """
        Args:
            x (Tensor): Input tensor (N, C, D, H, W).
            y (Tensor, optional): Ground truth labels (N, D, H, W).
            indices (list/Tensor, optional): Explicit class indices to select.
            seed (int, optional): Random seed for sampling consistency.
        """
        assert not (
            y is not None and indices is not None
        ), "Cannot specify both y and indices."

        device = x.device
        all_class_indices = torch.arange(self.num_total_classes, device=device)

        # Manual Indices provided
        if indices is not None:
            # Make sure they are in ascending order
            assert all(
                indices[i] <= indices[i + 1] for i in range(len(indices) - 1)
            ), "Indices must be in ascending order."
            selected_classes = torch.as_tensor(indices, dtype=torch.long, device=device)
        # All classes specified
        elif self.num_total_classes == self.num_sample_classes:
            selected_classes = all_class_indices
        # Ground Truth provided (Stochastic Sampling with GT priority)
        elif y is not None:
            present_classes = torch.unique(y).long()
            if self.ignore_background_class_for_sampling:
                present_classes = present_classes[present_classes > 0]
                present_classes = present_classes - 1

            generator = torch.Generator(device=device)
            if seed is not None:
                generator.manual_seed(seed)

            if len(present_classes) >= self.num_sample_classes:
                # If too many classes are present, sample from the present ones
                perm = torch.randperm(
                    len(present_classes), generator=generator, device=device
                )
                selected_classes = present_classes[perm[: self.num_sample_classes]]
            else:
                # Include all present classes
                # Find non-present classes to fill the remaining slots
                mask = torch.ones(
                    self.num_total_classes, dtype=torch.bool, device=device
                )
                mask[present_classes] = False
                non_present_classes = all_class_indices[mask]

                num_needed = self.num_sample_classes - len(present_classes)
                perm = torch.randperm(
                    len(non_present_classes), generator=generator, device=device
                )
                sampled_non_present = non_present_classes[perm[:num_needed]]

                selected_classes = torch.cat([present_classes, sampled_non_present])

            selected_classes, _ = torch.sort(selected_classes)

        # Nothing provided - Output all classes
        else:
            selected_classes = all_class_indices

        # Slice the weights and biases to only the selected classes
        sampled_weight = self.weight[selected_classes]
        sampled_bias = self.bias[selected_classes]

        # Use F.conv3d to apply the 1x1x1 kernel.
        # This is memory efficient because the convolution is only computed
        # for the selected output channels.
        out = F.conv3d(x, sampled_weight, sampled_bias)

        # Make selected_classes torch.Tensor if not already
        if not isinstance(selected_classes, torch.Tensor):
            selected_classes = torch.as_tensor(selected_classes)

        return out, selected_classes
