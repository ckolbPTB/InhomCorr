"""Data loader tests."""
import os
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from inhomcorr.data_loader.data_loader_nii import QMRIDataLoaderNii
from inhomcorr.mrdata import QMRIData


class TestQMRIDataLoaderNii(unittest.TestCase):

    def setUp(self):
        # Get working dir
        self.tmp_path = os.getcwd()

        # Create nifti image containing t1 and rho and save as file
        self.shape = (20, 30, 1, 3)
        dat = np.random.randn(self.shape[0], self.shape[1],
                              self.shape[2], self.shape[3])
        nifti_im = nib.Nifti1Image(dat, affine=np.eye(4))
        nifti_im.header['descrip'] = 'testing'
        self.nii_file = Path(self.tmp_path + '/test_nii_rho_alpha_t1.nii')
        nib.save(nifti_im, self.nii_file)

    def tearDown(self):
        # Delete file
        os.remove(self.nii_file)

    def test_load_qmri_nii_file(self):
        qmri_dat_ld = QMRIDataLoaderNii()
        qmri_dat_ld.load_t1(self.nii_file)
        qmri_dat_ld.load_rho(self.nii_file)

        # Verfiy type, shape and header of data
        qmri_dat = qmri_dat_ld.get_data()
        self.assertIsInstance(qmri_dat, QMRIData)
        self.assertEqual(list(qmri_dat.t1.shape), list(self.shape[-2::-1]))
