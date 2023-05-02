"""Wrapper to use torchio to simulate bias fields."""

import torch
import torchio as tio

from inhomcorr.interfaces import BiasCreator
from inhomcorr.interfaces import ImageData


class BiasCreatorTorchio(BiasCreator):
    """Bias field creation using torchio."""

    def __init__(self, coefficient_range: float = 0.2, order: int = 3) -> None:
        self.coefficient_range: float = coefficient_range
        self.order: int = order

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
        # Create torchio random bias field transform
        bft_rnd = tio.RandomBiasField(
            coefficients=self.coefficient_range, order=self.order)

        # Sample from random parameters
        rnd_coeff = bft_rnd.get_params(
            bft_rnd.order, bft_rnd.coefficients_range)

        # Create bias field
        bft = tio.BiasField(coefficients=rnd_coeff, order=bft_rnd.order)
        bf = bft.generate_bias_field(data=tio.ScalarImage(tensor=image.data),
                                     order=bft_rnd.order,
                                     coefficients=rnd_coeff)

        # Return as Image object
        bf_image = ImageData()
        bf_image.data = torch.tensor(bf).unsqueeze(0)
        return bf_image

    def get_random_bias_fields(self, image: ImageData,
                               number: int) -> list[ImageData]:
        """Inferface of a bias field creator providing multiple bias fields.

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
        bf_image_list = []
        for ind in range(number):
            bf_image_list.append(self.get_bias_field(image))
        return (bf_image_list)
