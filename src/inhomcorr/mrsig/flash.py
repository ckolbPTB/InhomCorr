"""Flash T1 Simulation."""

from dataclasses import dataclass

import torch

from inhomcorr.mrdata import ImageData
from inhomcorr.mrdata import QMRIData
from inhomcorr.mrsig.mrsig_interface import MRParam
from inhomcorr.mrsig.mrsig_interface import MRSig


@dataclass
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
        # define GRE steady state signal equation
        if qmap.t1 is None:
            raise AttributeError('T1 map not defined')

        e1 = torch.exp(-param.tr / qmap.t1)
        greimage = qmap.rho * (1-e1) * torch.sin(torch.tensor(param.alpha)) / \
            (1 - torch.cos(torch.tensor(param.alpha)) * e1)

        if self.with_t2s:
            pass
            # qmridata needs attribute t2s
            # greimage = greimage * math.exp(param.te / qmap.t2s)

        # save the GRE image in ImageData and return it
        gre_id = ImageData(torch.Tensor(greimage))

        return gre_id
