"""MRData Interface."""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TypeVar

import numpy as np
import torch

TMRData = TypeVar('TMRData', bound='MRData')


class MRData(ABC):
    """Base MR Data Class."""

    def __init__(self) -> None:
        self._header: dict = {}
        self._mask: torch.Tensor | None = None
        self._shape: tuple[int, int, int, int] = (1, 1, 1, 1)
        self._device: torch.device | None = None

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
        """Setter of the mask tensor.

        Parameters
        ----------
        value
            Torch tensor to be set.

        Raises
        ------
        RuntimeError
            Raises an error if shape does not match.
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

    @property
    def device(self) -> torch.device:
        """Get device the object resides on.

        Returns
        -------
            the device
        """
        if self._device is None:
            return torch.device('cpu')
        else:
            return self._device

    @abstractmethod
    def to(self: TMRData, *, device: str | torch.device) -> TMRData:
        """Move the object to a device."""
        pass

    def cpu(self: TMRData) -> TMRData:
        """Move Object to CPU. Returns a copy.

        Returns
        -------
            A copy of the object on the cpu
        """
        return self.to(device='cpu')

    def cuda(self: TMRData, device=None) -> TMRData:
        """Move Object to GPU. Returns a copy.

        Parameters
        ----------
            device (Optional): The device to move to. Must be a GPU.

        Returns
        -------
            A copy of the object on the cpu
        """
        if device is None:
            device = torch.device('cuda')
        else:
            if device.type != 'cuda':
                raise ValueError('device must be a cuda device')
        return self.to(device=device)

    def clone(self: TMRData) -> TMRData:
        """Create a copy of the object.

        Returns
        -------
            Copy of the object
        """
        return self.to(device=self.device)


class ImageData(MRData):
    """Image Data Class."""

    def __init__(self, data: torch.Tensor) -> None:
        super().__init__()
        self._data: torch.Tensor = torch.tensor([])
        self._device = data.device
        self.data = data

    @property
    def data(self) -> torch.Tensor:
        """Getter for data.

        Returns
        -------
            torch.Tensor
        """
        return self._data

    @data.setter
    def data(self, value: torch.Tensor):
        """Setter of the data attribute.

        Parameters
        ----------
        value
            Tensor to be set.

        Raises
        ------
        RuntimeError
            Error is raised if shape does not match.
        """
        try:
            self._shape = torch.broadcast_shapes(value.shape, self._shape)
        except RuntimeError:
            raise RuntimeError(
                f'Shapes do not match for the parameter map got {value.shape}'
                'for the parameter which is not broadcastable to current'
                f'shape {self._shape}'
            )
        self._data = value.to(device=self.device)

    @property
    def numpy(self) -> np.ndarray:
        """Torch tensor to numpy array function.

        The function forces the conversion to cpu and detaches
        from autograd.

        TODO: Check if really required.

        Returns
        -------
            Numpy array representation of torch data tensor.

        Raises
        ------
        AttributeError
            Raises an error if data is none.
        """
        if self._data is None:
            raise AttributeError('Data not defined.')
        return self._data.numpy(force=True)

    def to(self, device: torch.device | str) -> ImageData:
        """Move to a device, always returns a copy.

        Parameters
        ----------
        device:
            Target device


        Returns
        -------
            a copy of the ImageData object moved to the device
        """
        new = ImageData(self.data.to(device=device))
        if self._mask is not None:
            new._mask = self._mask.to(device)
        return new


class QMRIData(MRData):
    """QMRI Data Class.

    All data should be given in SI units.
    """

    def __init__(self,
                 t1: torch.Tensor | None = None,
                 rho: torch.Tensor | None = None) -> None:
        super().__init__()

        # Set initial class attributes
        self._t1 = torch.tensor([])
        self._rho = torch.tensor([])

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
        """Setter of the t1 map.

        Parameters
        ----------
        value
            Torch tensor with t1 map.

        Raises
        ------
        RuntimeError
            Raises an error if shape does not match.
        """
        if value is None:
            value = torch.tensor(float('inf')).reshape((1, 1, 1, 1))
        else:
            if self._device is None:
                self._device = value.device
            else:
                value = value.to(self._device)
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
    def rho(self, value: torch.Tensor | None) -> None:
        """Setter of the rho map.

        Parameters
        ----------
        value
            Torch tensor with rho map.

        Raises
        ------
        RuntimeError
            Raises an error if shape does not match.
        """
        if value is None:
            # Defaults to (1, 1, 1, 1) Tensor with value 1.
            value = torch.tensor(1.).reshape(
                (1, 1, 1, 1)).to(device=self.device)
        else:
            if self._device is None:
                self._device = value.device
            else:
                value = value.to(self._device)
        try:
            self._shape = torch.broadcast_shapes(value.shape, self._shape)
        except RuntimeError:
            raise RuntimeError(
                f'Shapes do not match for the parameter map got {value.shape}'
                'for the parameter which is not broadcastable to current'
                f'shape {self._shape}'
            )
        self._rho = value

    def to(self, device: str | torch.device) -> QMRIData:
        """Return a copy on the specified device.

        Parameters
        ----------
        device
            Target device

        Returns
        -------
            a copy on the device
        """
        new = QMRIData(t1=self.t1.to(device=device),
                       rho=self.rho.to(device=device))
        return new
