"""Implementaion of bias field estimators using a NN."""
from inhomcorr.interfaces.bias_estimator_interface import BiasEstimator
from inhomcorr.interfaces.mr_data_interface import ImageData
from inhomcorr.methods.unet import UNet
from inhomcorr.methods.unet import UNetHyperParameters


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

        self.nn = UNet(hparams)

    def __call__(self, image: ImageData) -> ImageData:
        """Compute the bias field from an input image.

        Parameters
        ----------
        image
            Image with bias field

        Returns
        -------
            Biasfield
        """
        biasfield = ImageData()
        biasfield.data = self.nn(image)

        return biasfield
