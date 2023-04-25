"""flash_t1.py."""

from inhomcorr.interfaces import ImageData
from inhomcorr.interfaces import MRParam
from inhomcorr.interfaces import MRSig
from inhomcorr.interfaces import QMRIData


class MRParamT1(MRParam):
    """MRParamT1 _summary_.

    Parameters
    ----------
    MRParam
        _description_
    """

    tr: float = 0.0
    alpha: float = 0.0


class MRSigFlashT1(MRSig):
    """Flash T1 MR Sig Interface."""

    def __call__(self, qmap: QMRIData, param: MRParamT1) -> ImageData:
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
        return ImageData()
