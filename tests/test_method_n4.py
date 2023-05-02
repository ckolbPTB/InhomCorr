"""N4 Estimator tests."""
import unittest

from inhomcorr.methods import N4Estimator
from tests.testdata import TestData


class TestN4Estimator(unittest.TestCase):
    def setUp(self):
        self.TestData = TestData(shape=(1, 1, 64, 64))

    def test_biasfield_estimation(self):
        testImage = self.TestData.get_random_image()

        bfe = N4Estimator(hparams=None)
        bf = bfe(testImage)

        assert bf.shape == testImage.shape,\
            'The shapes of biasfield and image should match.'\
            f'You have {bf.shape} and {testImage.shape}'
