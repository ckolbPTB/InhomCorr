"""flash_t1.py."""

import math

import torch

from inhomcorr.interfaces import ImageData
from inhomcorr.interfaces import MRParam
from inhomcorr.interfaces import MRSig
from inhomcorr.interfaces import QMRIData


class MRParamGRE(MRParam):
    """MRParamGRE _summary_.

    Parameters
    ----------
    MRParam
        _description_
    """

    tr: float = 0.0    # in s
    te: float = 0.0    # in s
    alpha: float = 0.0  # in radian


class MRSigFlash(MRSig):
    """Flash MR Sig Interface."""

    def __init__(self, with_t2s: bool = False) -> None:
        """Init."""
        super().__init__()
        self.with_t2s = with_t2s  # False: GRE signal without T2 star

    def __call__(self, qmap: QMRIData, param: MRParamGRE) -> ImageData:
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
        greimage = torch.tensor()

        # define GRE steady state signal equation
        e1 = math.exp(-param.tr / qmap.t1)
        greimage = qmap.rho * (1-e1) * math.sin(param.alpha) / \
            (1-math.cos(param.alpha * e1))

        if self.with_t2s:
            pass
            # qmridata needs attribute t2s
            # greimage = greimage * math.exp(param.te / qmap.t2s)

        # save the GRE image in ImageData and return it
        gre_id = ImageData()
        gre_id.data = greimage

        return gre_id
