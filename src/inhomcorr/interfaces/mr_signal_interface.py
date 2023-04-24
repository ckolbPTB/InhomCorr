"""File: mr_signal_interface.py."""
from inhomcorr.interfaces.mr_data_interface import ImageData
from inhomcorr.interfaces.mr_data_interface import QMRIData


class MRSignalInterface:
    """MR Signal Interface Class."""

    def run(self, qmri: QMRIData) -> ImageData:
        """Run interface method to calculate MR signal.

        Parameters
        ----------
        qmri
            Input data

        Returns
        -------
            Image
        """
        pass
