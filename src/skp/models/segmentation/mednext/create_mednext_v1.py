# FROM: https://github.com/MIC-DKFZ/MedNeXt/

from .MedNeXtV1 import MedNeXt


def create_mednextv1_small(
    num_input_channels,
    num_classes,
    use_conv_transpose,
    checkpoint_style,
    kernel_size=3,
    ds=False,
    ds_levels=1,
):
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=2,
        kernel_size=kernel_size,
        deep_supervision=ds,
        ds_levels=ds_levels,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        checkpoint_style=checkpoint_style,
        use_conv_transpose=use_conv_transpose,
    )


def create_mednextv1_base(
    num_input_channels,
    num_classes,
    use_conv_transpose,
    checkpoint_style,
    kernel_size=3,
    ds=False,
    ds_levels=1,
):
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        kernel_size=kernel_size,
        deep_supervision=ds,
        ds_levels=ds_levels,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        checkpoint_style=checkpoint_style,
        use_conv_transpose=use_conv_transpose,
    )


def create_mednextv1_medium(
    num_input_channels,
    num_classes,
    use_conv_transpose,
    checkpoint_style,
    kernel_size=3,
    ds=False,
    ds_levels=1,
):
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
        kernel_size=kernel_size,
        deep_supervision=ds,
        ds_levels=ds_levels,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3],
        checkpoint_style=checkpoint_style,
        use_conv_transpose=use_conv_transpose,
    )


def create_mednextv1_large(
    num_input_channels,
    num_classes,
    use_conv_transpose,
    checkpoint_style,
    kernel_size=3,
    ds=False,
    ds_levels=1,
):
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        kernel_size=kernel_size,
        deep_supervision=ds,
        ds_levels=ds_levels,
        do_res=True,
        do_res_up_down=True,
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        checkpoint_style=checkpoint_style,
        use_conv_transpose=use_conv_transpose,
    )
