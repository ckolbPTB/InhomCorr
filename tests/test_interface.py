"""Image data interface tests."""
import unittest

from testdata import TestData

import inhomcorr.interfaces.mr_data_interface as interface


class TestImageData(unittest.TestCase):

    def SetUp(self):
        self.data = TestData()

    def test_2D_shape(self):
        dummy = interface.ImageData()
        dummy.data = self.data.get_random_tensor()
        self.assertEqual(dummy.shape, self.data.dim)
