import numpy as np
from src.modules.utilities.logging import Logger
from .transform import Transform


class Whiten(Transform):
    def fit(self, X: np.ndarray, **kwargs):
        return self._whiten(X, **kwargs)

    def apply(self, X: np.ndarray, **kwargs):
        pass

    def _whiten(self, X: np.ndarray, **kwargs):
        """
        Source: https://gist.github.com/joelouismarino/ce239b5601fff2698895f48003f7464b
        Whitens the input matrix X using specified whitening method.
        Inputs:
            X:      Input data matrix with data examples along the first dimension
            method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                    'pca_cor', or 'cholesky'.
        """
        log = Logger().logger
        try:
            method = kwargs["method"]
        except KeyError:
            method = "zca"
            log.info("No whitening method was provided, defaulting to ZCA")

        log.info(f"Whitening Embedding Space with {method}")
        X = X.reshape((-1, np.prod(X.shape[1:])))
        X_centered = X - np.mean(X, axis=0)
        Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
        W = None

        if method in ['zca', 'pca', 'cholesky']:
            U, Lambda, _ = np.linalg.svd(Sigma)
            if method == 'zca':
                W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
            elif method == 'pca':
                W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
            elif method == 'cholesky':
                W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
        elif method in ['zca_cor', 'pca_cor']:
            V_sqrt = np.diag(np.std(X, axis=0))
            P = np.dot(np.dot(np.linalg.pinv(V_sqrt), Sigma), np.linalg.pinv(V_sqrt))
            G, Theta, _ = np.linalg.svd(P)
            if method == 'zca_cor':
                W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.pinv(V_sqrt))
            elif method == 'pca_cor':
                W = np.dot(np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T), np.linalg.pinv(V_sqrt))
        else:
            raise NotImplementedError('Whitening method not found.')
        log.debug(f"W shape: {W.shape}")
        self.coeff_ = W.T
        return X @ W.T
