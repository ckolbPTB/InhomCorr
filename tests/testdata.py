"""File containing a test data object."""
import torch

from inhomcorr.mrdata import ImageData
from inhomcorr.mrdata import QMRIData
from inhomcorr.mrsig.flash import MRParamGRE


class TestData:
    """TestData."""

    def __init__(self,
                 shape: tuple[int, int, int, int] = (1, 1, 8, 8)) -> None:
        self.shape = shape

    def get_gre_param(self,
                      tr: float = 100e-3,
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
        image = ImageData(torch.rand(self.shape, dtype=torch.float))
        return image

    def get_random_tensor(self, datatype: torch.dtype = torch.float
                          ) -> torch.Tensor:
        """Create a random float tensor.

        Returns
        -------
            Random tensor
        """
        return torch.rand(self.shape, dtype=datatype)

    def get_test_data(self) -> tuple[QMRIData, MRParamGRE, ImageData]:
        """Generate Test QMRI and Image Objects.

        Returns
        -------
            Tuple of qmri, mr gre params and image data object
        """
        # Create qmri input data
        qmri = QMRIData()
        qmri.t1 = torch.tensor(
            [[1e-3, 1], [1e-3, 1]], dtype=torch.float)
        qmri.rho = torch.tensor(
            [[1, 1], [0.1, 0.1]], dtype=torch.float)

        params = self.get_gre_param()

        # Create reference image data
        image = ImageData(
            torch.tensor([[-0.4282, -0.0224],
                          [-0.0428, -0.0022]], dtype=torch.float)
        )

        return (qmri, params, image)
