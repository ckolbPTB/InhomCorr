"""data_manip.py contains functions for data processing and manipulation."""

import numpy as np


class DummyClass:
    """
     This is a simple dummy class for documentation purposes.
    """

    def __init__(self, data: np.ndarray) -> None:
        """
        __init__ is the constructor for the DummyClass

        Parameters
        ----------
        data : np.ndarray
            input image data
        """
        self.data: np.ndarray = data
        self.id: int = 0


def normalise_image(im: np.ndarray) -> np.ndarray:
    """
    normalise_image normalises an image array to the range [0, 1]

    Parameters
    ----------
    im : np.ndarray
        input image

    Returns
    -------
    np.ndarray
        normalised image
    """

    assert isinstance(im, np.ndarray), (
        f'Image should be numpy array. Got {type(im)}'
    )
    return(im - np.min(im))/(np.max(im) - np.min(im))
