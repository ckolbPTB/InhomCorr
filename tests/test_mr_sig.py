"""Tests of mr signal T1 object."""
import unittest

from testdata import TestData

from inhomcorr.interfaces.mr_data_interface import ImageData
from inhomcorr.mrsig.flash import MRSigFlash


class TestMRSigFlash(unittest.TestCase):

    def setUp(self):
        testdata = TestData()
        self.dim = testdata.dim
        self.qmri = testdata.get_random_qmri()
        self.param = testdata.get_mr_param_gre()

    def test_mr_sig_flash(self):

        mrsig = MRSigFlash()
        img = mrsig(self.qmri, self.param)

        # Test shape and dtype
        self.assertEqual(self.dim, img.shape)
        self.assertIsInstance(img, ImageData)
