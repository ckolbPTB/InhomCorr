"""File containing a test data object."""
import torch

from inhomcorr.interfaces.mr_data_interface import QMRIData
from inhomcorr.mrsig.flash import MRParamGRE


class TestData:
    """TestData."""

    def __init__(self, dim: tuple = (1, 1, 8, 8)) -> None:
        self.dim = dim

    def get_mr_param_gre(self, tr: float = 100e-3,
                         alpha: int = 35) -> MRParamGRE:
        """Get an MRParamT1 Object for Testing.

        Parameters
        ----------
        tr, optional
            repetition time, by default 100e-3
        alpha, optional
            flip angle, by default 35

        Returns
        -------
            MR T1 parameter object
        """
        return MRParamGRE(
            tr=tr,
            alpha=alpha,
        )

    def get_random_qmri(self) -> QMRIData:
        """Generate a QMRI Object.

        Parameters
        ----------
        dim
            dimension of maps

        Returns
        -------
            QMRI data object
        """
        qmri = QMRIData()
        qmri.t1 = torch.rand(self.dim, dtype=torch.float)
        qmri.rho = torch.rand(self.dim, dtype=torch.float)
        return qmri

    def get_random_tensor(self) -> torch.Tensor:
        """Creates a random float tensor.

        Returns
        -------
            Random tensor
        """
        return torch.rand(self.dim, dtype=torch.float)
