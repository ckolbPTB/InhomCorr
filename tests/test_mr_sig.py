"""Tests of mr signal T1 object."""
import unittest

from inhomcorr.mrdata import ImageData
from inhomcorr.mrsig.flash import MRSigFlash
from tests.testdata import TestData


class TestMRSigFlash(unittest.TestCase):

    def setUp(self):
        testdata = TestData()
        self.shape = testdata.shape
        self.rand_qmri = testdata.get_random_qmri()
        self.test_qmri = testdata.get_test_qmri()
        self.test_image = testdata.get_test_image()
        self.param = testdata.get_mr_param_gre()

    def test_rand_mr_sig_flash(self):

        mrsig = MRSigFlash()
        img = mrsig(self.rand_qmri, self.param)

        # Test shape and dtype
        self.assertEqual(self.shape, img.shape)
        self.assertIsInstance(img, ImageData)

    def test_mr_sig_flash(self):

        mrsig = MRSigFlash()
        img = mrsig(self.test_qmri, self.param)
        test_img = self.test_image
        # Test MRSig values and dtype
        for t, r in zip(test_img.data.flatten(), img.data.flatten()):
            self.assertAlmostEqual(float(t), float(r), delta=1e-4)
        self.assertIsInstance(img, ImageData)
