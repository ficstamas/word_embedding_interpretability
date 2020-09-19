import os
import numpy as np
import scipy.sparse as sp
from argparse import ArgumentParser
import scipy.stats as stats


def load(path: str) -> np.ndarray:
    if path.endswith("npz"):
        return sp.load_npz(path).toarray()
    elif path.endswith("npy"):
        return np.load(path)
    else:
        raise ValueError


def ks_test(x: np.ndarray):
    ks_test = np.zeros(x.shape[1])
    p_values = np.zeros(x.shape[1])

    for i in range(x.shape[1]):
        d, p = stats.kstest(x[:, i], 'norm')
        ks_test[i] = d
        p_values[i] = p
    return ks_test, p_values, "KS Test"


if __name__ == '__main__':
    parser = ArgumentParser(description='Gather results in workspace')
    # Embedding
    parser.add_argument("--path", type=str, required=True,
                        help="Path to embedding")
    args = parser.parse_args()

    w = load(args.path)

    _, p, _ = ks_test(w)

    acp = np.where(p >= 0.05/w.shape[1])
    print(p, acp, acp[0].shape[0])