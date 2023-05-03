"""Tests of bias field operation."""
import unittest

from inhomcorr.bias_creator.torchio_bias import BiasCreatorTorchio
from inhomcorr.mrdata import ImageData
from tests.testdata import TestData


class TestBiasCreationTorchio(unittest.TestCase):

    def setUp(self):
        self.shape = (1, 1, 40, 30)
        testdata = TestData(img_shape=self.shape)
        self.image = testdata.get_random_image()

    def test_bias_creation_torchio(self):

        bf_creator = BiasCreatorTorchio()
        bf = bf_creator.get_bias_field(self.image)

        # Test shape and dtype
        self.assertEqual(list(self.shape), list(bf.shape))
        self.assertIsInstance(bf, ImageData)

    def test_random_bias_creation_torchio(self):

        bf_creator = BiasCreatorTorchio()
        bf_list = bf_creator.get_random_bias_fields(self.image, number=6)

        # Test length and dtype
        self.assertEqual(len(bf_list), 6)
        self.assertIsInstance(bf_list[3], ImageData)
