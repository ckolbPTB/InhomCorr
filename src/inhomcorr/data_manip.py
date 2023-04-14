"""data_manip.py contains functions for data processing and manipulation."""

import numpy as np


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
    return (im - np.min(im))/(np.max(im) - np.min(im))
