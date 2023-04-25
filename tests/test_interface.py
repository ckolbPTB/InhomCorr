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

    def test_setMask(self):
        dummy = interface.ImageData()
        shape = self.getRandom2DShape()
        dummy_mask = torch.randint(1, shape)
        dummy.mask = dummy_mask
        torch.testing.assert_close(dummy.mask, dummy_mask)

    def test_setHeader(self):
        dummy = interface.ImageData()
        dummy_header = {'test': 8956}
        dummy.header = dummy_header
        self.assertEqual(dummy.header['test'], dummy_header['test'])
