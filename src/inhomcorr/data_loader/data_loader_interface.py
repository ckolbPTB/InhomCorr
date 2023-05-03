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
    def load_t1(self, index: int) -> QMRIData:
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
