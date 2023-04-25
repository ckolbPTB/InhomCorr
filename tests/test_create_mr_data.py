import os
import unittest

import nibabel as nib
import numpy as np

from inhomcorr.data_loading.create_qmri import get_hdr_from_nii_file
from inhomcorr.data_loading.create_qmri import get_m0_from_nii_file
from inhomcorr.data_loading.create_qmri import get_qpar_from_nii_file
from inhomcorr.data_loading.create_qmri import get_t1_from_nii_file


class TestCreateQMRIData(unittest.TestCase):

    def test_read_nii_header(self):
        # Get working dir
        tmp_path = os.getcwd()

        # Create nifti image and save in file
        dat = np.random.randn(20, 20, 30, 4)
        nifti_im = nib.Nifti1Image(dat, affine=np.eye(4))
        nifti_im.header['descrip'] = 'testing'
        nib.save(nifti_im, tmp_path + '/test_nii_2378580.nii')

        # Get header and verify
        nii_hdr = dict(get_hdr_from_nii_file(
            tmp_path + '/test_nii_2378580.nii'))
        self.assertEqual(nii_hdr['descrip'], nifti_im.header['descrip'])

        # Delete folder and content
        os.remove(tmp_path + '/test_nii_2378580.nii')

    def test_read_nii_data(self):
        # Get working dir
        tmp_path = os.getcwd()

        # Create nifti image and save in file
        dat = np.random.randn(20, 20, 30, 4)
        nifti_im = nib.Nifti1Image(dat, affine=np.eye(4))
        nib.save(nifti_im, tmp_path + '/test_nii_2390580.nii')

        # Load all maps
        q_map = get_qpar_from_nii_file(tmp_path + '/test_nii_2390580.nii')
        np.testing.assert_array_almost_equal(q_map, dat)

        # Load t1 map
        t1_map = get_t1_from_nii_file(tmp_path + '/test_nii_2390580.nii')
        np.testing.assert_array_almost_equal(t1_map, dat[:, :, :, 2])

        # Load m0 map
        m0_map = get_m0_from_nii_file(tmp_path + '/test_nii_2390580.nii')
        np.testing.assert_array_almost_equal(m0_map, dat[:, :, :, 0])

        # Delete folder and content
        os.remove(tmp_path + '/test_nii_2390580.nii')
