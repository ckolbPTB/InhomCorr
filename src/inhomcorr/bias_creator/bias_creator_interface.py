"""Interfaces for bias field creators."""
from abc import ABC
from abc import abstractmethod

from inhomcorr.mrdata import ImageData


class BiasCreator(ABC):
    """Bias Creator Interface."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_bias_field(self, image: ImageData) -> ImageData:
        """Inferface of a bias field creator.

        Parameters
        ----------
        image
            Reference image

        Returns
        -------
            Bias field
        """
        pass

    @abstractmethod
    def get_random_bias_fields(self, image: ImageData,
                               number: int) -> list[ImageData]:
        """Inferface of a bias field creator provding multiple bias fields.

        Parameters
        ----------
        image
            Reference image
        number
            Number of bias fields

        Returns
        -------
            List of bias field
        """
        pass
