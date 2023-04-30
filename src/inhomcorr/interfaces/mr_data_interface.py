"""MRData Interface."""
import numpy as np
import torch


class MRData():
    """Basic MR Data Class."""

    def __init__(self) -> None:
        self._header: dict = {}
        self._mask: torch.Tensor | None = None
        self._shape: list[int] | None = None

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
        if self._shape is None:
            raise Exception(
                'Mask cannot be set for an otherwise empty MRData object.')
        elif self._shape != value.shape:
            raise Exception(
                'Mask has to have the same shape as the other MRData'
                'data objects.')
        self._mask = value

    @property
    def shape(self) -> list[int] | None:
        """Getter for shape of tensors.

        Returns
        -------
            Shape of tensors.
        """
        return self._shape

    @shape.setter
    def shape(self, value: list):
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
        self.shape = list(value.shape)
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
        self._t1: torch.Tensor | None = None
        self._t1_default_value: torch.Tensor = torch.Tensor(torch.inf)
        self._rho: torch.Tensor | None = None
        self._rho_default_value: torch.Tensor = torch.Tensor(0.0)

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
        if self._shape is None and self._t1 is None:
            raise Exception(
                'At least one parameter (e.g. t1) of QMRIData has to be set.')
        elif self._t1 is None:
            # TODO FIX THIS MESS
            assert isinstance(self._shape, list)
            return (torch.ones(self._shape, dtype=torch.float) *
                    self._t1_default_value)
        else:
            return self._t1

    @t1.setter
    def t1(self, value: torch.Tensor) -> None:
        """Setter for t1.

        Parameters
        ----------
        var
            T1 map tensor
        """
        self.shape = list(value.shape)
        self._t1 = value

    @property
    def rho(self) -> torch.Tensor | None:
        """Getter of rho.

        Returns
        -------
            rho tensor
        """
        if self._shape is None and self._rho is None:
            raise Exception(
                'At least one parameter (e.g. rho) of QMRIData has to be set.')
        elif self._rho is None:
            # TODO FIX THIS MESS
            assert isinstance(self._shape, list)
            return (torch.ones(self._shape, dtype=torch.float) *
                    self._rho_default_value)
        else:
            return self._rho

    @rho.setter
    def rho(self, value: torch.Tensor) -> None:
        """Setter of rho.

        Returns
        -------
            None
        """
        self.shape = list(value.shape)
        self._rho = value
