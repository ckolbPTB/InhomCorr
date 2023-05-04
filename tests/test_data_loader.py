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
        dat4d = np.random.randn(self.shape[0], self.shape[1],
                                self.shape[2], self.shape[3])
        nifti_im_4d = nib.Nifti1Image(dat4d, affine=np.eye(4))
        nifti_im_3d = nib.Nifti1Image(dat4d[..., 0], affine=np.eye(4))
        nifti_im_2d = nib.Nifti1Image(dat4d[..., 0, 0], affine=np.eye(4))
        nifti_im_list = [nifti_im_4d, nifti_im_3d, nifti_im_2d]

        nii_files = []
        for i, item in enumerate(nifti_im_list):
            item.header['descrip'] = 'testing'
            nii_files.append(
                Path(self.tmp_path + f'/test_nii_rho_alpha_t1_{4-i}d.nii'))
            nib.save(item, nii_files[i])

        self.nii_files = nii_files

    def tearDown(self):
        # Delete file
        for file in self.nii_files:
            os.remove(file)

    def test_load_qmri_nii_file(self):
        for nii_file in self.nii_files:
            qmri_dat_ld = QMRIDataLoaderNii()
            qmri_dat_ld.load_header(nii_file)
            qmri_dat_ld.load_t1(nii_file)

            # Only load rho if 4dim object
            if qmri_dat_ld.t1_nii_file_dim == 4:
                qmri_dat_ld.load_rho(nii_file)

            # Verfiy type, shape and header of data
            qmri_dat = qmri_dat_ld.get_data()
            self.assertIsInstance(qmri_dat, QMRIData)
            self.assertEqual(list(qmri_dat.t1.shape), list(self.shape[-2::-1]))
