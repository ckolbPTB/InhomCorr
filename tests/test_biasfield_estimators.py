import unittest

import numpy as np
import torch

import inhomcorr.methods as methods
from inhomcorr.interfaces.mr_data_interface import ImageData


class TestN4Estimator(unittest.TestCase):

    def getRandomPixelNumber(self, nmax=32):
        return np.random.randint(1, nmax)

    def getRandom2DShape(self):
        npix = self.getRandomPixelNumber()
        return (1, 1, npix, npix)

    def getRandom2DData(self, shape):
        return torch.rand(shape)

    def test_biasfield_estimation(self):
        testImage = ImageData()
        shape = self.getRandom2DShape()
        testImage.data = self.getRandom2DData(shape)

        bfe = methods.N4Estimator(hparams=None)
        bf = bfe(testImage)

        assert bf.shape == testImage.shape,\
            'The shapes of biasfield and image should match.'\
            f'You have {bf.shape} and {testImage.shape}'
