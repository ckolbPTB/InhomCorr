"""Image data interface tests."""
import unittest

from inhomcorr.interfaces.mr_data_interface import ImageData

from .testdata import TestData


class TestImageData(unittest.TestCase):

    def setUp(self):
        self.data = TestData()

    def test_2D_shape(self):
        dummy = ImageData()
        dummy.data = self.data.get_random_tensor()
        self.assertEqual(dummy.shape, self.data.dim)
