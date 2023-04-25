import unittest

import torch

from inhomcorr.methods.unet import UNet
from inhomcorr.methods.unet_estimator import UNetHyperParameters


class TestMethodsUnet(unittest.TestCase):

    def create_unet(self):
        hparams = UNetHyperParameters()
        u = UNet(hparams)
        return u

    def test_apply_unet(self):
        u = self.create_unet()

        x = torch.rand(1, 2, 36, 36)
        xcnn = u(x)

        assert xcnn.shape[2:] == x.shape[2:]
