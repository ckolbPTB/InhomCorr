"""File containing a test data object."""
import torch

from inhomcorr.interfaces.mr_data_interface import ImageData
from inhomcorr.interfaces.mr_data_interface import QMRIData
from inhomcorr.mrsig.flash import MRParamGRE


class TestData:
    """TestData."""

    def __init__(self,
                 shape: tuple[int, int, int, int] = (1, 1, 8, 8)
                 ) -> None:
        self.shape = shape

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

        Returns
        -------
            QMRI data object
        """
        qmri = QMRIData()
        qmri.t1 = torch.rand(self.shape, dtype=torch.float)
        qmri.rho = torch.rand(self.shape, dtype=torch.float)
        return qmri

    def get_random_image(self) -> ImageData:
        """Generate a Image Object.

        Returns
        -------
            Image data object
        """
        image = ImageData()
        image.data = torch.rand(self.shape, dtype=torch.float)
        return image

    def get_random_tensor(self) -> torch.Tensor:
        """Create a random float tensor.

        Returns
        -------
            Random tensor
        """
        return torch.rand(self.shape, dtype=torch.float)
