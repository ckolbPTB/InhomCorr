"""Image data interface tests."""
import unittest

import torch

from inhomcorr.mrdata import QMRIData
from tests.testdata import TestData


class TestQMRIData(unittest.TestCase):

    def setUp(self):
        self.qmri_shape = (1, 16, 16)
        self.data = TestData(qmri_shape=self.qmri_shape)
        self.t1_test = torch.rand(self.qmri_shape, dtype=torch.float)
        self.rho_test = torch.rand(self.qmri_shape, dtype=torch.float)

    def test_shape(self):
        qmri = QMRIData()
        # Check if shape of default values fits
        self.assertEqual(qmri.shape, (1, 1, 1))
        # Check if shape after setting only t1 fits
        qmri.t1 = self.t1_test
        self.assertEqual(qmri.shape, self.qmri_shape)
        # Check if shape after setting rho fits
        qmri.rho = self.rho_test
        self.assertEqual(qmri.shape, self.qmri_shape)

    def test_getter(self):
        # Use constructor here
        qmri = QMRIData(t1=self.t1_test,
                        rho=self.rho_test)
        # Calculate mse and check if mse equals 0
        t1_mse = float(((qmri.t1-self.t1_test)**2).mean())
        rho_mse = float(((qmri.rho-self.rho_test)**2).mean())
        self.assertEqual(t1_mse, 0)
        self.assertEqual(rho_mse, 0)

    def test_default_values(self):
        # Construct qmri with default values
        qmri = QMRIData()
        self.assertEqual(float(qmri.t1.squeeze()), float('inf'))
        self.assertEqual(float(qmri.rho.squeeze()), 1.)
