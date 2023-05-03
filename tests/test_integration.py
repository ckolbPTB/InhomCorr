"""Integraton tests."""
import unittest
from pathlib import Path

from inhomcorr.data_loader.data_loader_nii import QMRIDataLoaderNii
from inhomcorr.mrsig.flash import MRSigFlash
from tests.testdata import TestData

TEST_DATA_DIR = Path(__file__).parent / 'test_data'


class TestImageData(unittest.TestCase):

    def setUp(self):
        self.nifti_files = list(Path(TEST_DATA_DIR).rglob('*.nii'))
        testdata = TestData()
        self.param = testdata.get_mr_param_gre()

    def integration_single_file(self, file):
        # Load data from nifiti file
        dataloader = QMRIDataLoaderNii()
        dataloader.load_header(file)
        dataloader.load_t1(file)

        # Here, we only load rh0 if nii is 4D
        if dataloader.t1_nii_file_dim == 4:
            dataloader.load_rho(file)
        qmridata = dataloader.get_data()

        # create MR signal for T1/rho
        mrsig = MRSigFlash()
        mrsig(qmridata, self.param)

    def test_integration(self):
        for file in self.nifti_files:
            self.integration_single_file(file)


if __name__ == '__main__':
    unittest.main()
