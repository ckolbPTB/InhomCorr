"""Tests for mrdata submodule."""
import unittest

import torch

from inhomcorr.mrdata import ImageData
from inhomcorr.mrdata import QMRIData
from tests.testdata import TestData


class TestImageData(unittest.TestCase):

    def setUp(self):
        self.shape = (1, 1, 8, 8)
        self.wrong_shape = (1, 1, 16, 16)
        self.data = TestData(img_shape=self.shape)

    def test_data_setter_exception(self):
        with self.assertRaises(RuntimeError):
            dummy = ImageData(torch.rand(self.shape, dtype=torch.float))
            dummy.data = torch.rand(self.wrong_shape, dtype=torch.float)

    def test_mask_setter_exception(self):
        with self.assertRaises(RuntimeError):
            dummy = ImageData(torch.rand(self.shape, dtype=torch.float))
            dummy.mask = torch.rand(self.wrong_shape, dtype=torch.float)

    def test_2D_shape(self):
        data = torch.rand(self.shape, dtype=torch.float)
        dummy = ImageData(data)
        self.assertEqual(dummy.shape, data.shape)

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


class TestQMRIData(unittest.TestCase):

    def setUp(self):
        self.qmri_shape = (1, 16, 16)
        self.wrong_shape = (1, 24, 24)
        self.data = TestData(qmri_shape=self.qmri_shape)
        self.t1_test = torch.rand(self.qmri_shape, dtype=torch.float)
        self.rho_test = torch.rand(self.qmri_shape, dtype=torch.float)

    def test_mask_setter_exception(self):
        with self.assertRaises(RuntimeError):
            dummy = QMRIData(t1=torch.rand(self.qmri_shape, dtype=torch.float))
            dummy.mask = torch.rand(self.wrong_shape, dtype=torch.float)

    def test_t1_setter_excecption(self):
        with self.assertRaises(RuntimeError):
            qmri = QMRIData(t1=self.t1_test,
                            rho=self.rho_test)
            qmri.t1 = torch.rand(self.wrong_shape, dtype=torch.float)

    def test_rho_setter_excecption(self):
        with self.assertRaises(RuntimeError):
            qmri = QMRIData(t1=self.t1_test,
                            rho=self.rho_test)
            qmri.rho = torch.rand(self.wrong_shape, dtype=torch.float)

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
        torch.testing.assert_close(qmri.t1, self.t1_test)
        torch.testing.assert_close(qmri.rho, self.rho_test)

    def test_default_values(self):
        # Construct qmri with default values
        qmri = QMRIData()
        self.assertEqual(float(qmri.t1.squeeze()), float('inf'))
        self.assertEqual(float(qmri.rho.squeeze()), 1.)
