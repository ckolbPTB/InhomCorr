""" This module contains the DummyClass class. """
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

    def increase_id(self) -> None:
        """
        increase_id increases the id of the DummyClass by 1
        """
        self.id += 1
