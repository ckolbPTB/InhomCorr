"""Implementaion of N4 bias field estimator."""
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
import torch

from inhomcorr.interfaces import BiasEstimator
from inhomcorr.interfaces import HyperParameters
from inhomcorr.interfaces import ImageData


@dataclass
class N4Hyperparameters(HyperParameters):
    """Hyperparameters containing setup for N4 algorithm.

    Parameters
    ----------
    Hyperparameters
        maxNumberIterations: Number of iterations inside N4
        numberFittingLevels: Levels of Spline Approximation
    """

    maxNumberIterations: int = 50
    numberFittingLevels: int = 4


class N4Estimator(BiasEstimator):
    """Biasifeld Estimator based on the SITK N4 method.

    Parameters
    ----------
    BiasEstimator
        Class of type BiasEstimator
    """

    def __init__(self, hparams: N4Hyperparameters | None) -> None:
        """Generate an N4 Biasfield Corrector.

        Parameters
        ----------
        hparams
            N4Hyperparameters.
        """
        if hparams is None:
            hparams = N4Hyperparameters()
        self.hparams = hparams

    def __call__(self, image: ImageData) -> ImageData:
        """Compute the bias field based on SITK N4 method.

        Parameters
        ----------
        image
            ImageData object

        Returns
        -------
            _description_
        """
        return self.sitk_n4_estimation(image)

    def format_input_data(self, image: ImageData) -> sitk.Image:
        """Extract data from image and put into sitk.Image.

        Parameters
        ----------
        image
            ImageData containting image

        Returns
        -------
            sitk.Image
        """
        data = image.numpy

        if data is None:  # TODO: Remove once data is forced not to be None
            raise RuntimeError('ImageData.data should not be None.')

        data = np.squeeze(data)
        assert data.ndim < 4 and data.ndim > 1,\
            f'Your data must be 3D or 2D. You gave {data.ndim}'

        return sitk.GetImageFromArray(data)

    def sitk_n4_estimation(self, image: ImageData) -> ImageData:
        """Estimate the bias field from iamge using the N4 method of SITK.

        Parameters
        ----------
        image
            Image with bias field correction

        Returns
        -------
            Biasfield
        """
        sitk_img = self.format_input_data(image)
        # TODO: use mask of Image data
        maskImage = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        corrector.SetMaximumNumberOfIterations(
            [self.hparams.maxNumberIterations]
            * self.hparams.numberFittingLevels
        )
        corrector.Execute(sitk_img, maskImage)

        logbiasfield = corrector.GetLogBiasFieldAsImage(sitk_img)
        logbiasfield = sitk.GetArrayFromImage(logbiasfield)
        biasfield = np.exp(logbiasfield)
        biasfield = torch.tensor(biasfield)

        while biasfield.ndim < 4:
            biasfield = torch.unsqueeze(biasfield, 0)
        # TODO: adjust once we force setting the data in the constructor
        biasfield_img = ImageData()
        biasfield_img.data = biasfield

        return biasfield_img
