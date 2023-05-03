"""Data loader for QMRIData objects from nifti files."""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage as ndi

from inhomcorr.data_loader.data_loader_interface import QMRIDataLoader
from inhomcorr.mrdata import QMRIData


class QMRIDataLoaderNii(QMRIDataLoader):
    """Load QMRIData object from nifti file(s)."""

    def __init__(self) -> None:
        self.qmri_data = QMRIData()

    def load_header(self, file_nii: Path) -> None:
        """Load header from nifti file.

        Parameters
        ----------
        file_nii
            provide nii file used for header of QMRIData object
        """        """"""
        # Read in nii file
        nii_file = nib.load(file_nii)
        self.qmri_data.header = dict(nii_file.header)

    def load_t1(self, file_nii: Path, t1_dim: int = 2) -> None:
        """Load T1 from from nifti file.

        Parameters
        ----------
        file_nii
             provide nii file used for T1 of QMRIData object
        t1_dim, optional
            if file_nii is 4d, specifies the dimension where T1 is stored,
            by default 2
        """
        self.qmri_data.t1, self.t1_nii_file_dim = self._load_param_from_nii(
            file_nii, t1_dim)

    def load_rho(self, file_nii: Path, m0_dim: int = 0) -> None:
        """Load rho from from nifti file.

        Parameters
        ----------
        file_nii
             provide nii file used for rho of QMRIData object
        t1_dim, optional
            if file_nii is 4d, specifies the dimension where m0 is stored,
            by default 0
        """
        m0, self.m0_nii_file_dim = self._load_param_from_nii(file_nii, m0_dim)

        # Calculate rho from m0
        # Calculate mask
        m0 = m0 / m0.max()
        rho = np.zeros(m0.shape)
        rho[m0 > 0.02] = 1

        rho = ndi.morphology.binary_opening(
            rho.astype(int), np.ones((2, 2, 2)).astype(int), iterations=10)

        self.qmri_data.rho = torch.as_tensor(rho, dtype=torch.float32)

    def _load_param_from_nii(self, filename_nii: Path,
                             dim: int = None) -> tuple[torch.Tensor, int]:
        """Gerneral function to load parameter from nifti file.

        Parameters
        ----------
        filename_nii
            nii file
        dim, optional
            dimension of specific parameter in case of 4d nifit object,
            by default None

        Returns
        -------
            torch.Tensor with parameter data
            int with number of dimensions of nii file

        Raises
        ------
        ValueError
            _description_
        """
        # Read in nii file
        nii_file = nib.load(filename_nii)

        # Get numpy data as float32
        nii_data = np.asarray(nii_file.dataobj, dtype=np.float32)

        # revert oder of axis
        nii_data = np.transpose(nii_data)

        # Verify size of nii file
        ndim = nii_data.ndim
        if ndim == 4:
            nii_data = nii_data[dim]

        elif ndim == 2:
            nii_data = nii_data[np.newaxis]

        elif ndim != 3:
            raise ValueError(f'Wrong number of dimensions in nii file: {ndim}.\
                            Needs to be 2, 3 or 4 dimensional.')

        data = torch.as_tensor(nii_data, dtype=torch.float32)

        return data, ndim

    def get_data(self) -> QMRIData:
        """Return QMRIData object.

        Returns
        -------
            QMRIData object
        """
        return self.qmri_data
