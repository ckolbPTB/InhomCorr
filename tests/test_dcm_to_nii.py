"""Data loader tests."""
import tempfile
import unittest
from pathlib import Path

import nibabel

from inhomcorr.data_loader.dcm2nii import convert_dcm2nii_dir


class TestDCM2NII(unittest.TestCase):

    def setUp(self):
        self.dicom_path = Path('tests')/'test_data'/'TestDicoms_Phantom1Dicom'

    def test_testdata_available(self):
        # check if ou testdata is available
        self.assertEqual(len(list(self.dicom_path.glob('*.IMA'))), 1)

    def test_dicom2nii_setpath(self):
        # call DCM2NII  and save the results
        with tempfile.TemporaryDirectory() as stroutpath:
            outpath = Path(stroutpath)
            returned_list = convert_dcm2nii_dir(
                self.dicom_path, outpath)

            files = list(outpath.glob('*.nii'))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].parent, outpath)
            nii = nibabel.load(files[0])
            self.assertEqual(nii.shape, (1, 512, 512))
            self.assertEqual(returned_list, files)

    def test_dicom2nii_tmppath(self):
        # call DCM2NII with no path and save the results
        returned_list = convert_dcm2nii_dir(
            self.dicom_path, None)

        self.assertEqual(Path(tempfile.gettempdir()),
                         returned_list[0].parent.parent)
        try:
            returned_list[0].unlink()
            returned_list[0].parent.rmdir()
        except Exception:
            raise AssertionError('Could not delete generated tempdirectory')
        self.assertEqual(len(returned_list), 1)

    def test_dicom2nii_empty(self):
        # call DCM2NII  with no dicoms
        with tempfile.TemporaryDirectory() as strtmpdir:
            tmpdir = Path(strtmpdir)
            returned_list = convert_dcm2nii_dir(
                tmpdir, tmpdir)
            self.assertEqual(len(returned_list), 0)
