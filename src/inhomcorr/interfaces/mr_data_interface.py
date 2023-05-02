"""MRData Interface."""
import numpy as np
import torch


class MRData():
    """Basic MR Data Class."""

    def __init__(self) -> None:
        self._header: dict = {}
        self._mask: torch.Tensor | None = None

    @property
    def header(self) -> dict:
        """Header getter function.

        Returns
        -------
            Header dictionary
        """
        return self._header

    @header.setter
    def header(self, value: dict):
        """Setter for header.

        Parameters
        ----------
        value
            dictionary

        Returns
        -------
            None
        """
        self._header = value

    @property
    def mask(self) -> torch.Tensor | None:
        """Mask getter function.

        Returns
        -------
            Mask tensor
        """
        return self._mask

    @mask.setter
    def mask(self, value: torch.Tensor):
        """Setter for mask.

        Parameters
        ----------
        value
            torch.Tensor

        Returns
        -------
            None
        """
        self._mask = value


class ImageData(MRData):
    """Image Data Class."""

    def __init__(self) -> None:
        super().__init__()
        self._data: torch.Tensor | None = None

    @property
    def data(self) -> torch.Tensor | None:
        """Getter for data.

        Returns
        -------
            torch.Tensor
        """
        return self._data

    @data.setter
    def data(self, value: torch.Tensor):
        """Setter for data.

        Parameters
        ----------
        value
            torch.Tensor

        Returns
        -------
            None
        """
        self._data = value

    @property
    def numpy(self) -> np.ndarray | None:
        """Get the data as numpy array.

        The function forces the conversion to cpu and detaches
        from autograd.

        Returns
        -------
            numpy nd array or None
        """
        if self._data is None:
            return None

        return self._data.numpy(force=True)

    @property
    def shape(self) -> tuple | None:
        """Getter for shape of data.

        Returns
        -------
            Shape of _data tensor or None.
        """
        if self._data is None:
            return None
        return self._data.shape


class QMRIData(MRData):
    """QMRI Data Class."""

    def __init__(self) -> None:
        super().__init__()
        self._t1: torch.Tensor | None = None
        self._rho: torch.Tensor | None = None

        # To be added in the future
        # self._t2: torch.Tensor[torch.float] | None = None
        # self._db0: torch.Tensor[torch.float] | None = None

    @property
    def t1(self) -> torch.Tensor | None:
        """Getter of T1 map.

        Returns
        -------
            T1 map tensor
        """
        return self._t1

    @t1.setter
    def t1(self, value: torch.Tensor) -> None:
        """Setter for t1.

        Parameters
        ----------
        var
            T1 map tensor
        """
        self._t1 = value

    @property
    def rho(self) -> torch.Tensor | None:
        """Getter of rho.

        Returns
        -------
            rho tensor
        """
        return self._rho

    @rho.setter
    def rho(self, value: torch.Tensor) -> None:
        """Setter of rho.

        Returns
        -------
            None
        """
        self._rho = value
