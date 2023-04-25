"""File: create_qmri.py."""
import typing
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from inhomcorr.interfaces.mr_data_interface import QMRIData


def get_hdr_from_nii_file(nii_file_name: str | Path
                          ) -> nib.nifti1.Nifti1Header:
    """Get header from nifti file and return it.

    Parameters
    ----------
        nii_file_name (str | Path): Filename for nifti file

    Returns
    -------
        typing.Any: Nifti header
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
    qmri_data.header = dict(get_hdr_from_nii_file(nii_file_name))

    return (qmri_data)


def create_qmri_from_folder(nii_folder: Path) -> typing.List[QMRIData]:
    """Create list of QMRIData objects based on folder.

    Parameters
    ----------
        nii_folder (Path): Folder with nifti files

    Returns
    -------
        typing.List[QMRIData]: List of QMRIData objects
    """
    # Get all nifti files
    nii_file_list = nii_folder.glob('*.nii')

    # Create QMRIData
    qmri_data_list = []
    for nii_file in nii_file_list:
        qmri_data_list.append(create_qmri_from_nii_file(nii_file))

    return (qmri_data_list)


def create_qmri_hackathon() -> typing.List[QMRIData]:
    """Create list of QMRIData objects for hackathon.

    Returns
    -------
        typing.List[QMRIData]: List of QMRIData objects
    """
    # Define folders
    folder_nii_list = []
    folder_nii_list.append(
        '/Users/kolbit01/Documents/PTB/Data/Hackathon_InHomCorr/heart/t1')
    folder_nii_list.append(
        '/Users/kolbit01/Documents/PTB/Data/Hackathon_InHomCorr/liver/t1')

    qmri_data_list = []
    for folder_nii in folder_nii_list:
        qmri_data_list.extend(create_qmri_from_folder(Path(folder_nii)))

    return qmri_data_list
