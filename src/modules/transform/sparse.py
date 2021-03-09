import numpy as np
import spams
from src.modules.utilities.logging import Logger
from src.modules.transform.transform import Transform


class SparseCoding(Transform):
    def __init__(self):
        super(SparseCoding, self).__init__()
        self.required_coeff = True
        self.log = Logger().logger

    def fit(self, X: np.ndarray, **kwargs):
        normalize = kwargs['normalize'] if 'normalize' in kwargs else False
        try:
            params = {
                'K': kwargs['K'],
                'lambda1': kwargs['lda'],
                'numThreads': kwargs['numThreads'] if 'numThreads' in kwargs else 8,
                'batchsize': kwargs['batchSize'] if 'batchSize' in kwargs else 400,
                'iter': kwargs['iter'] if 'iter' in kwargs else 1000,
                'verbose': kwargs['verbose'] if 'verbose' in kwargs else False,
                'posAlpha': kwargs['posAlpha'] if 'posAlpha' in kwargs else True
            }
        except KeyError:
            raise KeyError(f"Config should include `K` and `lda` params.")
        self.log.info(f"Generating sparse embedding with K={params['K']} basis and "
                      f"lda={params['lambda1']} regularization. (normalization={normalize})")
        embedding = X
        if normalize:
            embedding = self.row_normalize(embedding)
        embedding = embedding.T
        if not np.isfortran(embedding):
            embedding = np.asfortranarray(embedding)

        self.coeff_ = spams.trainDL(embedding, **params)
        alphas = spams.lasso(X, D=self.coeff_, **params)
        return alphas.toarray()

    @staticmethod
    def row_normalize(embeddings):
        row_norms = np.sqrt((embeddings ** 2).sum(axis=1))[:, np.newaxis]
        row_norms[row_norms == 0] = 1  # we do not want to divide by 0
        return embeddings / row_norms

    @staticmethod
    def col_normalize(embeddings):
        col_norms = np.sqrt((embeddings ** 2).sum(axis=0))[np.newaxis, :]
        col_norms[col_norms == 0] = 1  # we do not want to divide by 0
        return embeddings / col_norms

    def apply(self, X: np.ndarray, **kwargs):
        normalize = kwargs['normalize'] if 'normalize' in kwargs else False
        try:
            params = {
                'K': kwargs['K'],
                'lambda1': kwargs['lda'],
                'numThreads': kwargs['numThreads'] if 'numThreads' in kwargs else 8,
                'batchsize': kwargs['batchSize'] if 'batchSize' in kwargs else 400,
                'iter': kwargs['iter'] if 'iter' in kwargs else 1000,
                'verbose': kwargs['verbose'] if 'verbose' in kwargs else False,
                'posAlpha': kwargs['posAlpha'] if 'posAlpha' in kwargs else True,
                'pos': kwargs['pos'] if 'pos' in kwargs else True,
                'ols': kwargs['ols'] if 'ols' in kwargs else False
            }
        except KeyError:
            raise KeyError(f"Config should include `K` and `lda` params.")

        self.log.info(f"Generating sparse embedding from existing Dictionary matrix with K={params['K']} basis and "
                      f"lda={params['lambda1']} regularization. (normalization={normalize})")
        D = self.coeff_
        if not np.isfortran(D):
            D = np.asfortranarray(D)

        embedding = X
        if normalize:
            embedding = self.row_normalize(embedding)
        embedding = embedding.T

        alphas = spams.lasso(embedding, D=D, **params)
        return alphas.toarray()
