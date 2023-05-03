"""Data loader for QMRIData objects from nifti files."""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage as ndi

from inhomcorr.data_loader.data_loader_interface import QMRIDataLoader
from inhomcorr.mrdata import QMRIData


class QMRIDataLoaderNii(QMRIDataLoader):
    """Load QMRIData objects from nifti file(s)."""

    def __init__(self) -> None:
        self.qmri_data_list: list[QMRIData] = []

    def add_qmridata_from_folder(self, foldername_nii: str) -> None:
        """Read in all nii files from folder and add them.

        Parameters
        ----------
            foldername_nii (str): Foldername where all nii files are
        """
        # Verify path
        path_nii = Path(foldername_nii)
        assert path_nii.is_dir(), f'Directory {str(path_nii)} not found.'

        # Get all nifti files
        nii_file_list = path_nii.glob('*.nii')

        # Create QMRIData
        for nii_file in nii_file_list:
            self.qmri_data_list.append(
                self.load_data_from_single_file(str(nii_file)))

    def load_data_from_single_file(self, filename_nii: str) -> QMRIData:
        """Load quantitative paramaters from file and create a QMRIData object.

        Parameters
        ----------
            Name of nifti file were parameters are stored

        Returns
        -------
            QMRIData object containting t1, rho and nifti header
        """
        # Read in nii file
        nii_file = nib.load(filename_nii)

        # Get numpy data as float32
        nii_data = np.asarray(nii_file.dataobj, dtype=np.float32)

        # Verify size
        if nii_data.ndim != 4:
            raise NotImplementedError(
                'Currently only 4-dimensional nifti images can be read in')

        # Prepare t1
        t1 = (nii_data[:, :, :, 2])
        t1 = torch.as_tensor(t1, dtype=torch.float32)

        # Nifti is [x,y,z], QMRIData is [z,y,x]
        t1 = torch.moveaxis(t1, (0, 1, 2), (2, 1, 0))

        # Calculate rho from m0
        m0 = nii_data[:, :, :, 0]

        # Calculate mask
        m0 = m0 / m0.max()
        rho = np.zeros(m0.shape)
        rho[m0 > 0.02] = 1

        rho = ndi.morphology.binary_opening(
            rho.astype(int), np.ones((2, 2, 2)).astype(int), iterations=10)

        rho = torch.as_tensor(rho, dtype=torch.float32)

        # Nifti is [x,y,z], QMRIData is [z,y,x]
        rho = torch.moveaxis(rho, (0, 1, 2), (2, 1, 0))

        # Create QMRData
        qmri_data = QMRIData(t1=t1, rho=rho)

        # Add nifti header
        qmri_data.header = dict(nii_file.header)

        return (qmri_data)

    def get_data(self, index: int) -> QMRIData:
        """Return single QMRIData object.

        Parameters
        ----------
        index
            Index of the object

        Returns
        -------
            QMRIData object
        """
        assert index < len(self.qmri_data_list)
        return (self.qmri_data_list[index])

    def get_all_data(self) -> list[QMRIData]:
        """Return all available QMRIData objects.

        Returns
        -------
            List of QMRIData objects
        """
        return (self.qmri_data_list)
