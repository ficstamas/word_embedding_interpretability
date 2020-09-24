import numpy as np


def remove_zeros(p: np.ndarray, q: np.ndarray, maximum_value=0):

    # modification begin
    if np.std(q) == 0 and np.mean(q) == 0:
        return -np.log(maximum_value), -1 if np.mean(p) < 0 else 1

    if np.std(p) == 0 and np.mean(p) == 0:
        return 0, 1

    p = p[p != 0]
    q = q[q != 0]
    # modification end
    return p, q