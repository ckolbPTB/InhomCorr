"""Image data interface tests."""
import unittest

from inhomcorr.interfaces.mr_data_interface import ImageData
from tests.testdata import TestData


class TestImageData(unittest.TestCase):

    def setUp(self):
        self.data = TestData()
        self.shape = self.data.shape

    def test_2D_shape(self):
        dummy = ImageData()
        dummy.data = self.data.get_random_tensor()
        self.assertEqual(dummy.shape, self.shape)

    def test_set_mask(self):
        dummy = ImageData()
        dummy_mask = torch.randint(1, self.shape)
        dummy.mask = dummy_mask
        torch.testing.assert_close(dummy.mask, dummy_mask)

    def test_set_header(self):
        dummy = ImageData()
        dummy_header = {'test': 8956}
        dummy.header = dummy_header
        self.assertEqual(dummy.header['test'], dummy_header['test'])
