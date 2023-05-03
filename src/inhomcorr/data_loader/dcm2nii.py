"""Routines to convert dicom images into nifti format.

Heavily relies on package dicom2nifti
https://github.com/icometrix/dicom2nifti.
"""

import tempfile
from pathlib import Path

import dicom2nifti
import dicom2nifti.settings as settings


def convert_dcm2nii_dir(folder_in_dicom: Path,
                        folder_out_nifti: Path | None = None) -> list:
    """Convert dicoms in a directory to niftis.

    Parameters
    ----------
    folder_in_dicom
        folder to convert all dicoms in
    folder_out_nifti
        target folder, has to be writable to.
        None (default) will create a temp directory.

    Returns
    -------
        List of nifti files
    """
    settings.disable_validate_slicecount()
    if folder_out_nifti is None:
        folder_out_nifti = Path(tempfile.mkdtemp())

    existing_niftis = set(folder_out_nifti.glob('*.nii'))
    dicom2nifti.convert_directory(
        folder_in_dicom, folder_out_nifti, reorient=True, compression=False)
    current_nifits = set(folder_out_nifti.glob('*.nii'))
    new_nifits = current_nifits-existing_niftis
    return list(new_nifits)
