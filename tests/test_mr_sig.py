"""Tests of mr signal T1 object."""
import unittest

from inhomcorr.mrdata import ImageData
from inhomcorr.mrsig.flash import MRSigFlash
from tests.testdata import TestData


class TestMRSigFlash(unittest.TestCase):

    def setUp(self):
        self.rand_shape = (1, 1, 16, 16)
        self.testdata = TestData(shape=self.rand_shape)
        self.rand_qmri = self.testdata.get_random_qmri()

    def test_rand_mr_sig_flash(self):

        mrsig = MRSigFlash()
        param = self.testdata.get_gre_param()
        img = mrsig(self.rand_qmri, param)

        # Test shape and dtype
        self.assertEqual(self.rand_shape, img.shape)
        self.assertIsInstance(img, ImageData)

    def test_mr_sig_flash(self):

        # Create instance of MR FLASH signal and get test data
        mrsig = MRSigFlash()
        qmri, params, img_ref = self.testdata.get_test_data()

        # Apply mr signal method
        img_out = mrsig(qmri, params)

        # Check if output image is instance of ImageData
        self.assertIsInstance(img_out, ImageData)

        # Test MRSig values and dtype
        for io, ir in zip(img_out.data.flatten(), img_ref.data.flatten()):
            self.assertAlmostEqual(float(io), float(ir), delta=1e-4)
