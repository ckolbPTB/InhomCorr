"""Data loader tests."""
import os
import unittest

import nibabel as nib
import numpy as np

from inhomcorr.data_loader.data_loader_nii import QMRIDataLoaderNii
from inhomcorr.interfaces import QMRIData


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
        self.nii_file = self.tmp_path + '/test_nii_rho_alpha_t1.nii'
        nib.save(nifti_im, self.nii_file)

    def tearDown(self):
        # Delete file
        os.remove(self.nii_file)

    def test_load_qmri_nii_file(self):
        qmri_dat_ld = QMRIDataLoaderNii()
        self.assertEqual(len(qmri_dat_ld.qmri_data_list), 0)
        qmri_dat_ld.add_qmridata_from_folder(self.tmp_path)

        # Verify that the data was added correctly
        self.assertEqual(len(qmri_dat_ld.qmri_data_list), 1)
        self.assertEqual(len(qmri_dat_ld.get_all_data()), 1)

        # Verfiy type, shape and header of data
        qmri_dat = qmri_dat_ld.get_data(0)
        self.assertIsInstance(qmri_dat, QMRIData)
        self.assertEqual(list(qmri_dat.t1.shape), list(self.shape[-2::-1]))
