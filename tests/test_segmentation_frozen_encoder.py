import torch
import torch.nn as nn

from skp.configs import Config
from skp.models.segmentation.base import Net


def make_minimal_net() -> Net:
    net = Net.__new__(Net)
    nn.Module.__init__(net)
    net.cfg = Config(freeze_encoder=True, frozen_encoder_eval=True)
    net.encoder = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    net.decoder = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    return net


def test_frozen_encoder_stays_in_eval_when_model_trains():
    net = make_minimal_net()
    net.freeze_encoder()
    net.train()

    assert net.training
    assert net.decoder.training
    assert not net.encoder.training
    assert all(not parameter.requires_grad for parameter in net.encoder.parameters())


def test_frozen_encoder_batchnorm_state_does_not_update():
    net = make_minimal_net()
    net.freeze_encoder()
    net.train()
    batch_norm = net.encoder[1]
    running_mean = batch_norm.running_mean.clone()
    batches_tracked = batch_norm.num_batches_tracked.clone()

    with torch.no_grad():
        net.encoder(torch.randn(8, 2))

    torch.testing.assert_close(batch_norm.running_mean, running_mean)
    torch.testing.assert_close(batch_norm.num_batches_tracked, batches_tracked)
