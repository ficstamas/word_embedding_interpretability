import numpy as np
from abc import ABC, abstractmethod


class Transform(ABC):
    def __init__(self):
        self.coeff_ = None
        self.required_coeff = False

    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def apply(self, X: np.ndarray, **kwargs):
        pass
