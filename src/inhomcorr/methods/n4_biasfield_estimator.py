"""Implementaion of bias field estimators."""
import numpy as np
import SimpleITK as sitk
import torch

from inhomcorr.interfaces.bias_estimator_interface import BiasEstimator
from inhomcorr.interfaces.bias_estimator_interface import HyperParameters
from inhomcorr.interfaces.mr_data_interface import ImageData


class N4Hyperparameters(HyperParameters):
    """Hyperparameters containing setup for N4 algorithm.

    Parameters
    ----------
    Hyperparameters
        maxNumberIterations: int
        numberFittingLevels: int
    """

    def __init__(self):
        """Create an instance of N4 Hyperparameters."""
        self._maxNumberIterations = 50
        self._numberFittingLevels = 4

    @property
    def maxNumberIterations(self) -> int:
        """Getter for maximum number of iterations.

        Returns
        -------
            Maximum number of iterations
        """
        return self._maxNumberIterations

    @maxNumberIterations.setter
    def maxNumberIterations(self, value: int) -> None:
        """Setter for maximum number of iterations.

        Parameters
        ----------
        value
            int
        """
        self._maxNumberIterations = value

    @property
    def numFittingLevels(self) -> int:
        """Getter for number of fitting levels.

        Returns
        -------
            Number of fitting levels
        """
        return self._numberFittingLevels

    @numFittingLevels.setter
    def numFittingLevels(self, value: int) -> None:
        """Setter for number of fitting levels.

        Parameters
        ----------
        value
            int
        """
        self._numberFittingLevels = value


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

        if data is None:
            return sitk.GetImageFromArray(np.ndarray([]))

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

        maskImage = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        corrector.SetMaximumNumberOfIterations(
            [self.hparams.maxNumberIterations] * self.hparams.numFittingLevels)
        corrector.Execute(sitk_img, maskImage)

        logbiasfield = corrector.GetLogBiasFieldAsImage(sitk_img)
        logbiasfield = sitk.GetArrayFromImage(logbiasfield)
        biasfield = np.exp(logbiasfield)
        biasfield = torch.FloatTensor(biasfield)

        while biasfield.ndim < 4:
            biasfield = torch.unsqueeze(biasfield, 0)

        biasfield_img = ImageData()
        biasfield_img.data = biasfield

        return biasfield_img
