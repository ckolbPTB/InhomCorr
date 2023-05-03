"""Image data interface tests."""
import unittest

import torch

from inhomcorr.mrdata import ImageData
from tests.testdata import TestData


class TestImageData(unittest.TestCase):

    def setUp(self):
        self.data = TestData()
        self.shape = self.data.shape

    def test_2D_shape(self):
        dummy = ImageData(self.data.get_random_tensor())
        self.assertEqual(dummy.shape, self.shape)

    def test_set_mask(self):
        img = self.data.get_random_image()
        mask = torch.randint(1, self.shape)
        img.mask = mask
        torch.testing.assert_close(img.mask, mask)

    def test_set_header(self):
        img = self.data.get_random_image()
        header = {'test': 8956}
        img.header = header
        self.assertEqual(img.header['test'], header['test'])
