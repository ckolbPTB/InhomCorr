"""Interface for Bias Field Dataset."""
from abc import ABC

from torch.utils.data import Dataset


class SupervisedTrainingDataset(ABC, Dataset):
    """Bias Field Dataset Interface."""

    pass
