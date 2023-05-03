"""Interface for Data loader."""
from abc import ABC
from abc import abstractmethod

from inhomcorr.mrdata import QMRIData


class QMRIDataLoader(ABC):
    """Data loader for QMRI datasets."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_data(self, index: int) -> QMRIData:
        """Return single QMRIData object.

        Parameters
        ----------
        index
            Index of the object

        Returns
        -------
            QMRIData object
        """
        pass

    @abstractmethod
    def get_all_data(self) -> list[QMRIData]:
        """Return all available QMRIData objects.

        Returns
        -------
            List of QMRIData objects
        """
        pass
