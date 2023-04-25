"""Tests of mr signal T1 object."""
import unittest

from testdata import TestData

from inhomcorr.interfaces.mr_data_interface import ImageData
from inhomcorr.mrsig.flash_t1 import MRSigFlashT1


class TestMRSigFlashT1(unittest.TestCase):

    def SetUp(self):
        testdata = TestData()
        self.dim = testdata.dim
        self.qmri = testdata.get_random_qmri()
        self.param = testdata.get_mr_param_t1()

    def test_mr_sig_flash_t1(self):

        mrsig = MRSigFlashT1()
        img = mrsig(self.qmri, self.param)

        # Test shape and dtype
        self.assertEqual(self.dim, img.shape)
        self.assertIsInstance(img, ImageData)
