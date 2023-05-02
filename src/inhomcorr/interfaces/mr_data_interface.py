"""MRData Interface."""
import numpy as np
import torch


class MRData():
    """Basic MR Data Class."""

    def __init__(self) -> None:
        self._header: dict = {}
        self._mask: torch.Tensor | None = None
        self._shape: tuple[int, int, int, int] = (1, 1, 1, 1)

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

    def shape(self) -> tuple:
        """Getter for shape of data.

        Returns
        -------
            Shape of _data tensor or None.
        """
        return self._shape

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
        try:
            self._shape = torch.broadcast_shapes(value.shape, self._shape)
        except RuntimeError:
            raise RuntimeError(
                f'Shapes do not match for the parameter map got {value.shape}'
                'for the parameter which is not broadcastable to current'
                f'shape {self._shape}'
            )
        self._mask = value


class ImageData(MRData):
    """Image Data Class."""

    def __init__(self, data: torch.Tensor) -> None:
        super().__init__()
        self._data: torch.Tensor | None = None
        self.data = data

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
        try:
            self._shape = torch.broadcast_shapes(value.shape, self._shape)
        except RuntimeError:
            raise RuntimeError(
                f'Shapes do not match for the parameter map got {value.shape}'
                'for the parameter which is not broadcastable to current'
                f'shape {self._shape}'
            )
        self._data = value

    @property
    def numpy(self) -> np.ndarray:
        """Get the data as numpy array.

        The function forces the conversion to cpu and detaches
        from autograd.

        Returns
        -------
            numpy nd array or None
        """
        if not self._data:
            raise AttributeError('Data not defined.')
        return self._data.numpy(force=True)


class QMRIData(MRData):
    """QMRI Data Class.

    All data should be given in SI units.
    """

    def __init__(self,
                 t1: torch.Tensor | None = None,
                 rho: torch.Tensor | None = None) -> None:
        super().__init__()

        # Set initial class attributes
        self._t1: torch.Tensor | None = None
        self._rho: torch.Tensor | None = None

        # Use setters to set t1 and rho
        self.t1 = t1
        self.rho = rho

        # To be added in the future
        # self._t2: torch.Tensor[torch.float] | None = None
        # self._db0: torch.Tensor[torch.float] | None = None

    @property
    def t1(self) -> torch.Tensor:
        """Getter of T1 map.

        Returns
        -------
            T1 map tensor [s]
        """
        return torch.broadcast_to(self._t1, self._shape)

    @t1.setter
    def t1(self, value: torch.Tensor | None) -> None:
        """Setter for t1.

        Parameters
        ----------
        value
            T1 map tensor [s]
        """
        if value is None:
            value = torch.tensor(float('inf')).reshape((1, 1, 1, 1))
        try:
            self._shape = torch.broadcast_shapes(value.shape, self._shape)
        except RuntimeError:
            raise RuntimeError(
                f'Shapes do not match for the parameter map got {value.shape}'
                'for the parameter which is not broadcastable to current'
                f'shape {self._shape}'
            )
        self._t1 = value

    @property
    def rho(self) -> torch.Tensor:
        """Getter of rho.

        Returns
        -------
            rho tensor [au]
        """
        return torch.broadcast_to(self._rho, self._shape)

    @rho.setter
    def rho(self, value: torch.Tensor) -> None:
        """Setter of rho.

        Parameters
        ----------
        value
            Rho map tensor [au]
        """
        if value is None:
            # Defaults to (1, 1, 1, 1) Tensor with value 1.
            value = torch.tensor(1.).reshape((1, 1, 1, 1))
        try:
            self._shape = torch.broadcast_shapes(value.shape, self._shape)
        except RuntimeError:
            raise RuntimeError(
                f'Shapes do not match for the parameter map got {value.shape}'
                'for the parameter which is not broadcastable to current'
                f'shape {self._shape}'
            )
        self._rho = value
