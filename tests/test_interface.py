import unittest

import torch

import inhomcorr.interfaces.mr_data_interface as interface


class TestImageData(unittest.TestCase):

    def getRandom2DData(self):
        self.numPixels = 16
        self.dataShape = (1, 1, self.numPixels, self.numPixels)
        return torch.rand(self.dataShape)

    def test_2Dshape(self):
        dummy = interface.ImageData()
        dummy.data = self.getRandom2DData()
        self.assertEqual(dummy.shape, self.dataShape)
