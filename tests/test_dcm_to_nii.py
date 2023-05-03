"""Data loader tests."""
import os
import unittest
from pathlib import Path
from inhomcorr.data_loader.dcm2nii import convert_dcm2nii_dir_single_slices
import nibabel

# from inhomcorr.data_loading.dcm2nii import convert_dcm2nii_dir


class TestDCM2NII(unittest.TestCase):

    def setUp(self):
        # Get working dir
        self.tmp_path_1dcm = Path('tests')/'TestDicoms_Phantom1Dicom'

    def test_1dcm2nii(self):
        # call DCM2NII  and save the results
        convert_dcm2nii_dir_single_slices(
            self.tmp_path_1dcm, self.tmp_path_1dcm)

        files = list(self.tmp_path_1dcm.glob('*.nii'))
        self.assertEqual(len(files), 1)
        nii = nibabel.load(files[0])
        self.assertEqual(nii.shape, (1, 512, 512))
