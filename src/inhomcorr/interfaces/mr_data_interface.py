"""File: mr_data_interfaces.py."""
import torch


class MRData():
    """Basic MR Data Class."""

    def __init__(self) -> None:
        self._header: dict = {}
        self._mask: torch.Tensor[torch.int] | None = None

    @property
    def header(self) -> dict:
        """Header getter function.

        Returns
        -------
            Header dictionary
        """
        return self._header

    @property
    def mask(self) -> torch.Tensor[torch.int] | None:
        """Mask getter function.

        Returns
        -------
            Mask tensor
        """
        return self._mask

    # TODO: Implementation of setter functions


class ImageData(MRData):
    """MR Data Class for Quantitative Data."""

    def __init__(self) -> None:
        super().__init__()
        self._data: torch.Tensor[torch.cfloat] | None = None


class QMRIData(MRData):
    """Image Data Class for Image Data."""

    def __init__(self) -> None:
        super().__init__()
        self._t1: torch.Tensor[torch.float] | None = None
        self._rho: torch.Tensor[torch.float] | None = None

        # To be added in the future
        # self._t2: torch.Tensor[torch.float] | None = None
        # self._db0: torch.Tensor[torch.float] | None = None

    @property
    def t1(self) -> torch.Tensor[torch.float]:
        """Getter of T1 map.

        Returns
        -------
            T1 map tensor
        """
        return self._t1

    @t1.setter
    def t1(self, value: torch.Tensor[torch.float]) -> None:
        """Setter for t1.

        Parameters
        ----------
        var
            T1 map tensor
        """
        self._t1 = value

    @property
    def rho(self) -> torch.Tensor[torch.float]:
        """Getter of rho.

        Returns
        -------
            rho tensor
        """
        return self._rho

    @rho.setter
    def rho(self, value: torch.Tensor[torch.float]) -> None:
        """Setter of rho.

        Returns
        -------
            None
        """
        self._rho = value
