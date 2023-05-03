"""Data loader tests."""
import os
import unittest

from inhomcorr.data_loading.dcm2nii import convert_dcm2nii_dir_single_slices

# from inhomcorr.data_loading.dcm2nii import convert_dcm2nii_dir


class TestDCM2NII(unittest.TestCase):

    def setUp(self):
        # Get working dir
        self.tmp_path_1dcm = '.\\tests\\TestDicoms_Phantom1Dicom'
        # self.tmp_path_multidcm = R"C:\Users\aigner01\Documents\Kooperationen\
        # Hackathon\InhomCorr\tests\TestDicoms_PhantomMultiDicom"

    def test_1dcm2nii(self):
        # call DCM2NII  and save the results
        convert_dcm2nii_dir_single_slices(
            self.tmp_path_1dcm, self.tmp_path_1dcm)

        # get all the files in the tmp_path_1dcm
        print(os.getcwd())
        files = [f for f in os.listdir(self.tmp_path_1dcm)
                 if os.path.isfile(os.path.join(self.tmp_path_1dcm, f))]

        nii_file = 'None'
        # check if there is a nii file in the folder
        for file in files:
            fileName, fileExtension = os.path.splitext(file)
            if fileExtension == '.nii':
                nii_file = fileName + fileExtension

        # check if there is a nii file in there
        filepathhelper = self.tmp_path_1dcm + '\\'
        check_file = os.path.exists(filepathhelper + nii_file)
        self.assertEqual(check_file, True)

    # # this test would be used to test for entire directory
    # def test_multidcm2nii(self):
    #     # call DCM2NII  and save the results
    #     convert_dcm2nii_dir(self.tmp_path_multidcm, self.tmp_path_multidcm)

    #     # get all the files in the tmp_path_multidcm
    #     files = [f for f in os.listdir(self.tmp_path_multidcm)
    #              if os.path.isfile(os.path.join(self.tmp_path_multidcm, f))]

    #     nii_file = 'None'
    #     # check if there is a nii file in the folder
    #     for file in files:
    #         fileName, fileExtension = os.path.splitext(file)
    #         if fileExtension == '.nii':
    #             nii_file = fileName + fileExtension

    #     # check if there is a nii file in there
    #     filepathhelper = self.tmp_path_multidcm + "\\"
    #     check_file = os.path.exists(filepathhelper + nii_file)
    #     self.assertEqual(check_file, True)
