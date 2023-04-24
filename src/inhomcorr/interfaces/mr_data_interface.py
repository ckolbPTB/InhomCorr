"""File: mr_data_interfaces.py."""
import torch


class MRData():
    """Basic MR Data Class."""

    def __init__(self) -> None:
        pass


class QMRIData(MRData):
    """MR Data Class for Quantitative Data."""

    def __init__(self) -> None:
        super().__init__()


class ImageData(MRData):
    """Image Data Class for Image Data."""

    def __init__(self) -> None:
        super().__init__()
        self.__t1_map = torch.tensor()
        self.__t2_map = torch.tensor()
        self.__B0_map = torch.tensor()
        self.__rho = torch.tenor()

    def get_T1_map(self) -> torch.Tensor:
        """Getter of T1 map.

        Returns
        -------
            T1 map tensor
        """
        return self.__t1_map

    def get_T2_map(self) -> torch.Tensor:
        """Getter of T2 map.

        Returns
        -------
            T2 map tensor
        """
        return self.__t2_map

    def get_B0_map(self) -> torch.Tensor:
        """Getter of B0 map.

        Returns
        -------
            B0 map tensor
        """
        return self.__B0_map

    def get_rho_map(self) -> torch.Tensor:
        """Getter of density distribution.

        Returns
        -------
            Densiy map tensor
        """
        return self.__rho
