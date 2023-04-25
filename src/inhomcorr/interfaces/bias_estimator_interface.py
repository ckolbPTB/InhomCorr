"""Interfaces for bias field estimators."""
from abc import ABC
from abc import abstractmethod

from inhomcorr.interfaces.mr_data_interface import ImageData


class HyperParameters(ABC):
    """Hyper Parameters Interface."""

    pass


class BiasEstimator(ABC):
    """Bias Estimator Interface."""

    @abstractmethod
    def __init__(self, hparams: HyperParameters | None) -> None:
        pass

    @abstractmethod
    def __call__(self, image: ImageData) -> ImageData:
        """Inferface of a bias field estimator.

        Parameters
        ----------
        image
            Bias corrupted Image

        Returns
        -------
            Bias field
        """
        pass
