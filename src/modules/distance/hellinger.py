import numpy as np
from scipy.integrate import quad
from sklearn.neighbors import KernelDensity
from typing import Tuple

__all__ = ["closed_hellinger_distance", "continuous_hellinger_distance", "exponential_hellinger_distance"]


def closed_hellinger_distance(p: np.ndarray, q: np.ndarray, **kwargs) -> Tuple[int, int]:
    """
    Calculates Hellinger distance between two vectors which assumes normal distribution
    Parameters
    ----------
    p : np.ndarray
        Vector 1
    q : np.ndarray
        Vector 2
    Returns
    -------
    tuple
        Returns with the distance and the sign of the difference of means
    """
    std_1 = np.std(p)
    std_2 = np.std(q)

    # Variance of p and q
    var_1 = std_1 ** 2
    var_2 = std_2 ** 2

    if var_1 == 0:
        var_1 = 1e-5
        std_1 = np.sqrt(1e-5)
    if var_2 == 0:
        var_2 = 1e-5
        std_2 = np.sqrt(1e-5)

    # Mean of p and q
    mu_1 = np.mean(p)
    mu_2 = np.mean(q)

    # Formula
    x = -0.25 * (((mu_1 - mu_2) ** 2) / (var_1 + var_2))
    h = 1 - np.sqrt((2 * std_1 * std_2) / (var_1 + var_2)) * np.exp(x)

    sign = -1 if mu_1 - mu_2 < 0 else 1
    return np.sqrt(h), sign


def exponential_hellinger_distance(p: np.ndarray, q: np.ndarray, **kwargs) -> Tuple[int, int]:
    """
    Calculates Hellinger distance between two vectors which assumed to come from exponential distribution
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
    mu_1 = np.mean(p)
    mu_2 = np.mean(q)

    # rate of change
    alpha = 1/mu_1
    beta = 1/mu_2

    # Formula
    h = 1 - (2*np.sqrt(alpha*beta)) / (alpha + beta)
    if alpha + beta == 0:
        h = 1

    sign = -1 if mu_1 - mu_2 < 0 else 1
    return np.sqrt(h), sign


def _remove_zeros(x: np.ndarray):
    r = x[x != 0]
    if r.shape[0] == 0:
        return np.array([0])
    return r


def continuous_hellinger_distance(p: np.ndarray, q: np.ndarray, **kwargs) -> Tuple[int, int]:
    """
    Calculates Hellinger distance between two vectors
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

    # modification begin
    # if np.std(q) == 0 and np.mean(q) == 0 and np.std(p) == 0 and np.mean(p) == 0:
    #     return 0, 1
    # if np.std(q) == 0 and np.mean(q) == 0:
    #     return 1, -1 if mean1 < 0 else 1
    #
    # if np.std(p) == 0 and np.mean(p) == 0:
    #     return 0, 1
    #
    # p = p[p != 0]
    # q = q[q != 0]
    # # modification end

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

    return np.sqrt(1-ig[0]), -1 if mean1 - mean2 < 0 else 1
