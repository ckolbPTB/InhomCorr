"""File: mr_sig_interfaces.py."""
import abc
from dataclasses import dataclass
from typing import Any

from inhomcorr.mrdata import ImageData
from inhomcorr.mrdata import QMRIData


@dataclass
class MRParam():
    """MR parameter base class."""

    pass


class MRSig(abc.ABC):
    """Basic MR signal generator interface."""

    @abc.abstractmethod
    def __call__(self, qmap: QMRIData, param: Any) -> ImageData:
        """__call__ _summary_.

        Parameters
        ----------
        qmap
            _description_
        param
            _description_

        Returns
        -------
            _description_
        """
        pass
