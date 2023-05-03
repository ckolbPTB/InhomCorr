# %%
"""File: dcm2nii.py."""
#   Routines to convert dicom images into nifti format
#   Input is the name of the dicom folder
#   Output is the name of the newly created nifti folder
#   Heavily relies on package dicom2nifti
#        https://github.com/icometrix/dicom2nifti

import dicom2nifti
import dicom2nifti.settings as settings


#   Converting a directory with dicom files to nifti files
#       (one nifti file for each series)
def convert_dcm2nii_dir(folder_in_dicom, folder_out_nifti):
    """_summary_.

    Parameters
    ----------
    folder_in_dicom
        _description_
    folder_out_nifti
        _description_
    """
    dicom2nifti.convert_directory(
        folder_in_dicom, folder_out_nifti, reorient=True, compression=False)


def convert_dcm2nii_dir_series2file(folder_in_dicom, file_out_nifti):
    """_summary_.

    Parameters
    ----------
    folder_in_dicom
        _description_
    file_out_nifti
        _description_
    """
    dicom2nifti.dicom_series_to_nifti(
        folder_in_dicom, file_out_nifti, reorient_nifti=True)


# converts all dicom images in a folder into nifti files,
#     individual slices are allowed
def convert_dcm2nii_dir_single_slices(folder_in_dicom, folder_out_nifti):
    """_summary_.

    Parameters
    ----------
    folder_in_dicom
        _description_
    folder_out_nifti
        _description_
    """
    settings.disable_validate_slicecount()
    dicom2nifti.convert_directory(
        folder_in_dicom, folder_out_nifti, compression=False)
# %%
