import unittest

import numpy as np
import torch

import inhomcorr.interfaces.mr_data_interface as interface


class TestImageData(unittest.TestCase):

    def getRandomPixelNumber(self, nmax=32):
        return np.random.randint(1, nmax)

    def getRandom2DShape(self):
        npix = self.getRandomPixelNumber()
        return (1, 1, npix, npix)

    def getRandom2DData(self, shape):
        return torch.rand(shape)

    def test_2Dshape(self):
        dummy = interface.ImageData()
        shape = self.getRandom2DShape()
        dummy.data = self.getRandom2DData(shape)
        self.assertEqual(dummy.shape, shape)
