import numpy as np
from sklearn import decomposition
from argparse import ArgumentParser
import scipy.sparse as sp


def pca(w) -> np.ndarray:
    model = decomposition.PCA(44)
    r = model.fit_transform(w)
    return r


if __name__ == '__main__':
    parser = ArgumentParser(description='')

    # Embedding
    parser.add_argument("--weights", type=str)

    args = parser.parse_args()

    if args.train.endswith(".npz"):
        weights = sp.load_npz(args.weights)
        weights = weights.toarray()
    else:
        weights = np.load(args.weights)

    r = pca(weights)

    np.save(f"{args.weights}.pca-44.npy", r)