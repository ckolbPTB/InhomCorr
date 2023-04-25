"""File: create_qmri.py."""
import typing
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from inhomcorr.interfaces.mr_data_interface import QMRIData


# typing.Dict[str, typing.Any]:
def get_hdr_from_nii_file(nii_file_name: str | Path) -> typing.Any:
    """Get header from nifti file and return it as a dictionary.

    Parameters
    ----------
        nii_file_name (str | Path): Filename for nifti file

    Returns
    -------
        typing.Any: Nifti header as a dictionary
    """
    # Read in nii file
    nii_file = nib.load(nii_file_name)

    # Return header as dictionary
    return (nii_file.header)


def get_qpar_from_nii_file(nii_file_name: str | Path) -> np.ndarray:
    """Get 4d quantitative parameters from nifti file.

    Parameters
    ----------
        nii_file_name (str | Path): Filename for nifti file

    Returns
    -------
        np.ndarray: 4d quantitative parameters
    """
    # Read in nii file
    nii_file = nib.load(nii_file_name)

    # Get numpy data as float32
    nii_data = np.asarray(nii_file.dataobj, dtype=np.float32)

    # Verify size
    if nii_data.ndim != 4:
        raise NotImplementedError(
            'Currently only 4-dimensional nifti images can be read in')

    return (nii_data)


def get_t1_from_nii_file(nii_file_name: str | Path) -> np.ndarray:
    """Get t1 map from nifti file.

    Parameters
    ----------
        nii_file_name (str | Path): Filename for nifti file

    Returns
    -------
        np.ndarray: t1 map
    """
    return get_qpar_from_nii_file(nii_file_name)[:, :, :, 2]


def get_m0_from_nii_file(nii_file_name: str | Path) -> np.ndarray:
    """Get m0 map from nifti file.

    Parameters
    ----------
        nii_file_name (str | Path): Filename for nifti file

    Returns
    -------
        np.ndarray: m0 map
    """
    return get_qpar_from_nii_file(nii_file_name)[:, :, :, 0]


def create_qmri_from_nii_file(nii_file_name: str | Path) -> QMRIData:
    """Create QMRIData object based on nifti file.

    Parameters
    ----------
        nii_file_name (str | Path): Filename for nifti file

    Returns
    -------
        QMRIData: QMRIData object containing t1, rho and header
    """
    # Create empty QMRIData object
    qmri_data = QMRIData()

    # Add t1
    qmri_data.t1 = torch.FloatTensor(get_t1_from_nii_file(nii_file_name))

    # Get m0
    m0 = get_m0_from_nii_file(nii_file_name)

    # Calculate rho from m0
    rho = m0

    # Add rho
    qmri_data.rho = torch.FloatTensor(rho)

    # Add nifti header
    # qmri_data.header = get_hdr_from_nii_file(nii_file_name)

    return (qmri_data)
