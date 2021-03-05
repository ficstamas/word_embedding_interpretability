from sklearn.preprocessing import StandardScaler
import numpy as np
from src.modules.utilities.logging import Logger
from .transform import Transform


class Centering(Transform):
    def fit(self, X: np.ndarray, **kwargs):
        return self._centring(X, **kwargs)

    def apply(self, X: np.ndarray, **kwargs):
        self.fit(X, **kwargs)

    @staticmethod
    def _centring(X: np.ndarray, **kwargs):
        log = Logger().logger
        log.info("Centering the Embedding")
        try:
            R = X - np.mean(X, **kwargs)[:, np.newaxis]
        except ValueError:
            R = X - np.mean(X, **kwargs)[np.newaxis, :]
        return R


class Normalize(Transform):
    def fit(self, X: np.ndarray, **kwargs):
        return self._normalize(X, **kwargs)

    def apply(self, X: np.ndarray, **kwargs):
        self.fit(X, **kwargs)

    @staticmethod
    def _normalize(X: np.ndarray, **kwargs):
        log = Logger().logger
        log.info("Normalizing the Embedding")
        try:
            norm = X / np.linalg.norm(X, **kwargs)[:, np.newaxis]
        except ValueError:
            norm = X / np.linalg.norm(X, **kwargs)[np.newaxis, :]
        return norm


class Standardize(Transform):
    def fit(self, X: np.ndarray, **kwargs):
        return self._standardize(X, **kwargs)

    def apply(self, X: np.ndarray, **kwargs):
        self.fit(X, **kwargs)

    @staticmethod
    def _standardize(X: np.ndarray, **kwargs):
        log = Logger().logger
        log.info("Standardizing the Embedding")
        std = StandardScaler(**kwargs)
        return std.fit_transform(X)
