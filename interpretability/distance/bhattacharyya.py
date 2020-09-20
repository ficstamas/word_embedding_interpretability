import numpy as np
from scipy.integrate import quad
from sklearn.neighbors import KernelDensity
from typing import Tuple, Literal

__all__ = ["closed_bhattacharyya_distance", "continuous_bhattacharyya_distance", "exponential_bhattacharyya_distance"]


def closed_bhattacharyya_distance(p: np.ndarray, q: np.ndarray, config) -> Tuple[int, int]:
    """
    Calculates Bhattacharyya distance between two vectors which assumed to come from normal distribution
    Parameters
    ----------
    p : np.ndarray
        Vector 1
    q : np.ndarray
        Vector 2
    config: Config
        Just to maintain uniform headers between distance functions
    Returns
    -------
    tuple
        Returns with the distance and the sign of the difference of means
    """
    # Variance of p and q
    var1 = np.std(p) ** 2
    var2 = np.std(q) ** 2

    if var1 == 0:
        var1 = 1e-5
    if var2 == 0:
        var2 = 1e-5

    # Mean of p and q
    mean1 = np.mean(p)
    mean2 = np.mean(q)

    # Formula
    x = np.log((var1 / var2 + var2 / var1 + 2) / 4) / 4
    bc = x + ((mean1 - mean2) ** 2 / (var1 + var2)) / 4
    sign = -1 if mean1 - mean2 < 0 else 1
    return bc, sign


def exponential_bhattacharyya_distance(p: np.ndarray, q: np.ndarray, config) -> Tuple[int, int]:
    """
    Calculates Bhattacharyya distance between two vectors which assumed to come from exponential distribution
    Parameters
    ----------
    p : np.ndarray
        Vector 1
    q : np.ndarray
        Vector 2
    config: Config
        Just to maintain uniform headers between distance functions
    Returns
    -------
    tuple
        Returns with the distance and the sign of the difference of means
    """
    # Mean of p and q
    mean1 = np.mean(p)
    mean2 = np.mean(q)

    # rate of change
    alpha = 1/mean1
    beta = 1/mean2

    # Formula
    bc = (2*np.sqrt(alpha*beta))/(alpha+beta)
    if alpha + beta == 0:
        bc = 1e-5
    sign = -1 if mean1 - mean2 < 0 else 1
    return -np.log(bc), sign


def _remove_zeros(x: np.ndarray):
    r = x[x != 0]
    if r.shape[0] == 0:
        return np.array([0])
    return r


def continuous_bhattacharyya_distance(p: np.ndarray, q: np.ndarray, config) -> Tuple[int, int]:
    """
    Calculates Bhattacharyya distance between two vectors
    Parameters
    ----------
    p
        Vector 1
    q
        Vector 2
    config: Config
        Contains parameters for KDE
    Returns
    -------
    tuple
        Returns with the distance and the sign of the difference of means
    """
    p = _remove_zeros(p)
    q = _remove_zeros(q)
    # Mean of p and q
    mean1 = np.mean(p)
    mean2 = np.mean(q)

    p_kde = KernelDensity(bandwidth=config.kde.bandwidth, kernel=config.kde.kernel)
    q_kde = KernelDensity(bandwidth=config.kde.bandwidth, kernel=config.kde.kernel)
    _p = p[:, np.newaxis]
    p_kde_fit = p_kde.fit(_p)

    _q = q[:, np.newaxis]
    q_kde_fit = q_kde.fit(_q)

    def g(x, __p, __q):
        p_kde_score = __p.score_samples(np.array([[x]]))

        q_kde_score = __q.score_samples(np.array([[x]]))

        e_p = np.exp(p_kde_score)
        e_q = np.exp(q_kde_score)

        r = np.sqrt(e_p * e_q)
        return r

    ig = quad(g, -np.inf, np.inf, args=(p_kde_fit, q_kde_fit))

    return -np.log(ig[0]), -1 if mean1 - mean2 < 0 else 1