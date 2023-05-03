"""Tests of mr signal T1 object."""
import unittest

import torch

from inhomcorr.mrdata import ImageData
from inhomcorr.mrsig.flash import MRSigFlash
from tests.testdata import TestData


class TestMRSigFlash(unittest.TestCase):

    def setUp(self):
        # Create two instances of testdata for image and qmri
        self.img_shape = (1, 1, 16, 16)
        self.qmri_shape = (1, 16, 16)
        self.testdata = TestData(img_shape=self.img_shape,
                                 qmri_shape=self.qmri_shape)
        self.rand_qmri = self.testdata.get_random_qmri()

    def test_rand_mr_sig_flash(self):

        mrsig = MRSigFlash()
        param = self.testdata.get_gre_param()
        img = mrsig(self.rand_qmri, param)

        # Test shape and dtype
        self.assertEqual(self.img_shape, img.shape)
        self.assertIsInstance(img, ImageData)

    def test_mr_sig_flash(self):

        # Create instance of MR FLASH signal and get test data
        mrsig = MRSigFlash()
        qmri, params, img_ref = self.testdata.get_test_data()

        # Apply mr signal method
        img_out = mrsig(qmri, params)

        # Check if output image is instance of ImageData
        self.assertIsInstance(img_out, ImageData)
        torch.testing.assert_close(img_out.data, img_ref.data)
