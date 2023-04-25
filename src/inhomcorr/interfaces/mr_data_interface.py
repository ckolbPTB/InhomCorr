"""File: mr_data_interfaces.py."""
import numpy as np
import torch


class MRData():
    """Basic MR Data Class."""

    def __init__(self) -> None:
        self._header: dict = {}
        self._mask: torch.IntTensor | None = None
        self._shape: torch.Size | None = None

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
    def mask(self) -> torch.IntTensor | None:
        """Mask getter function.

        Returns
        -------
            Mask tensor
        """
        return self._mask

    @mask.setter
    def mask(self, value: torch.IntTensor):
        """Setter for mask.

        Parameters
        ----------
        value
            torch.IntTensor

        Returns
        -------
            None
        """
        if self._shape is None:
            raise Exception(
                'Mask cannot be set for an otherwise empty MRData object.')
        elif self._shape != value.shape:
            raise Exception(
                'Mask has to have the same shape as the other MRData'
                'data objects.')
        self._mask = value

    @property
    def shape(self) -> tuple | None:
        """Getter for shape of tensors.

        Returns
        -------
            Shape of tensors.
        """
        return self._shape

    @shape.setter
    def shape(self, value: torch.Size):
        """Setter for shape.

        Parameters
        ----------
        value
            torch.Size

        Returns
        -------
            None
        """
        if self._shape is None:
            self._shape = value
        elif self._shape != value:
            raise Exception(
                f'Shape ({value}) has to be the same as existing'
                f'_shape ({self._shape}).')
        self._shape = value


class ImageData(MRData):
    """Image Data Class."""

    def __init__(self) -> None:
        super().__init__()
        self._data: torch.FloatTensor | None = None

    @property
    def data(self) -> torch.FloatTensor | None:
        """Getter for data.

        Returns
        -------
            torch.FloatTensor
        """
        return self._data

    @data.setter
    def data(self, value: torch.FloatTensor):
        """Setter for data.

        Parameters
        ----------
        value
            torch.FloatTensor

        Returns
        -------
            None
        """
        self.shape = value.shape
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


class QMRIData(MRData):
    """QMRI Data Class."""

    def __init__(self) -> None:
        super().__init__()
        self._t1: torch.FloatTensor | None = None
        self._t1_default_value: torch.float = torch.inf
        self._rho: torch.FloatTensor | None = None
        self._rho_default_value: torch.float = 0.0

        # To be added in the future
        # self._t2: torch.Tensor[torch.float] | None = None
        # self._db0: torch.Tensor[torch.float] | None = None

    @property
    def t1(self) -> torch.FloatTensor | None:
        """Getter of T1 map.

        Returns
        -------
            T1 map tensor
        """
        if self._shape is None and self._t1 is None:
            raise Exception(
                'At least one parameter (e.g. t1) of QMRIData has to be set.')
        elif self._t1 is None:
            return (torch.FloatTensor(self._shape)*self._t1_default_value)
        else:
            return self._t1

    @t1.setter
    def t1(self, value: torch.FloatTensor) -> None:
        """Setter for t1.

        Parameters
        ----------
        var
            T1 map tensor
        """
        self.shape = value.shape
        self._t1 = value

    @property
    def rho(self) -> torch.FloatTensor | None:
        """Getter of rho.

        Returns
        -------
            rho tensor
        """
        if self._shape is None and self._rho is None:
            raise Exception(
                'At least one parameter (e.g. rho) of QMRIData has to be set.')
        elif self._rho is None:
            return (torch.FloatTensor(self._shape)*self._rho_default_value)
        else:
            return self._rho

    @rho.setter
    def rho(self, value: torch.FloatTensor) -> None:
        """Setter of rho.

        Returns
        -------
            None
        """
        self.shape = value.shape
        self._rho = value
