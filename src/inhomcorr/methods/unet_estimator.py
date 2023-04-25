"""Implementaion of bias field estimators using a NN."""
from dataclasses import dataclass

import torch

from inhomcorr.interfaces.bias_estimator_interface import BiasEstimator
from inhomcorr.interfaces.bias_estimator_interface import HyperParameters
from inhomcorr.interfaces.mr_data_interface import ImageData

from .unet import UNet


@dataclass
class UNetHyperParameters(HyperParameters):
    """Hyperparameters for the UNet Estimator."""

    dim: torch.int = 2
    nChIn: torch.int = 2
    nChOut: torch.int = 2
    KernelSize: torch.int = 3
    nEncStages: torch.int = 3
    nConvsPerStage: torch.int = 2
    nFilters: torch.int = 16
    ResConnection: torch.bool = True
    Bias: torch.bool = True


class UNetEstimator(BiasEstimator):
    """UNet Estimator which predicts a bias fiel from an input image."""

    def __init__(self, hparams: UNetHyperParameters | None) -> None:
        """Generate a UNet bias estimator.

        Parameters
        ----------
        hparams
            UNetHyperParameters.
        """
        if hparams is None:
            hparams = UNetHyperParameters()
        self.hparams = hparams

        self.unet = UNet(hparams)

    def __call__(self, image: ImageData) -> ImageData:
        """Compute the bias field from an input image.

        Parameters
        ----------
        image
            ImageData object

        Returns
        -------
            _description_
        """
        return self.unet_estimation(image)

    def unet_estimation(self, image: ImageData) -> ImageData:
        """Estimate the bias field from an input image a UNet.

        Parameters
        ----------
        image
            Image with bias field correction

        Returns
        -------
            Biasfield
        """
        data = image.data
        if data is None:
            return ImageData()

        assert data.ndim < 4 and data.ndim > 1,\
            f'Your data must be 3D or 2D. You gave {data.ndim}'

        biasfield_img = 1.

        return biasfield_img
