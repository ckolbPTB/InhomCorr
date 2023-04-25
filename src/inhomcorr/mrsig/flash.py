"""Flash T1 Simulation."""

from dataclasses import dataclass

import torch

from inhomcorr.interfaces import ImageData
from inhomcorr.interfaces import MRParam
from inhomcorr.interfaces import MRSig
from inhomcorr.interfaces import QMRIData


@dataclass
class MRParamGRE(MRParam):
    """Parameters for GRE.

    Parameters
    ----------
    tr
        tr in s
    te
        te in s
    alpha
        flipangle in radian
    """

    tr: float = 0.0    # in s
    te: float = 0.0    # in s
    alpha: float = 0.0  # in radian


class MRSigFlash(MRSig):
    """Flash Simulator"""

    def __init__(self, with_t2s: bool = False) -> None:
        """Flash Simulator
        
        Parameters
        ----------
        with_t2s
            include T2* in simulation
       """
        self.with_t2s = with_t2s

    def __call__(self, qmap: QMRIData, param: MRParamGRE) -> ImageData:
        """Simulates an acquisition.

        Parameters
        ----------
        qmap
           quantitative maps. T1 and rho will be used.
        param
            Sequence Parameters

        Returns
        -------
            The image
        """
        if qmap.t1 is None:
            raise AttributeError('T1 map not defined')
        # define GRE steady state signal equation
        e1 = torch.exp(-param.tr / qmap.t1)
        greimage = qmap.rho * (1-e1) * torch.sin(torch.tensor(param.alpha)) / \
            (1 - torch.cos(torch.tensor(param.alpha)) * e1)

        if self.with_t2s:
            pass
            # qmridata needs attribute t2s
            # greimage = greimage * math.exp(param.te / qmap.t2s)

        # save the GRE image in ImageData and return it
        gre_id = ImageData()
        gre_id.data = torch.Tensor(greimage)

        return gre_id
