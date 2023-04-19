"""Contains the DummyClass class."""

from numpy.typing import NDArray


class DummyClass:
    """Dummy class for documentation purposes."""

    def __init__(self, data: NDArray) -> None:
        """__init__ is the constructor for the DummyClass.

        Parameters
        ----------
        data :
            input image data
        """
        self.data: NDArray = data
        self.id: int = 0

    def increase_id(self) -> None:
        """increase_id increases the id of the DummyClass by 1."""
        self.id += 1
