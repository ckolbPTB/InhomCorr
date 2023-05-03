"""Trainer Interface."""
from abc import ABC
from abc import abstractmethod
from typing import Any

from inhomcorr.interfaces import BiasEstimator
from inhomcorr.interfaces import SupervisedTrainingDataset


class TrainingParameters(ABC):
    """Training Parameters Interface."""

    pass


class Trainer(ABC):
    """Bias field estimator Trainer Interface."""

    @abstractmethod
    def __init__(self, model: BiasEstimator | Any) -> None:
        pass

    @abstractmethod
    def _call__(self, dataset: SupervisedTrainingDataset) -> None:
        """Inferface of a bias field estimato Trainer.

        Parameters
        ----------
        dataset
            The dataset to train on.
        """
        pass
